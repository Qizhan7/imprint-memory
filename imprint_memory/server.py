#!/usr/bin/env python3
"""
imprint-memory -- MCP Server
Persistent memory system: CRUD, hybrid search, message bus, task queue.

Usage:
  python3 -m imprint_memory.server          # stdio mode (for Claude Code)
  python3 -m imprint_memory.server --http   # HTTP mode (for Claude.ai via tunnel)

Or if installed:
  imprint-memory          # stdio mode
  imprint-memory --http   # HTTP mode
"""

import sys
import urllib.request
from pathlib import Path
from typing import Optional

# When run as script, set up package context for relative imports
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "imprint_memory"

from mcp.server.fastmcp import FastMCP
from .memory_manager import (
    remember, search_text, forget, daily_log, get_all,
    delete_memory, update_memory, find_duplicates, find_stale, decay,
    reindex_embeddings,
    unified_search_text, pin_memory, unpin_memory,
    add_tags, get_tags, add_edge, get_edges,
)
from .bus import bus_post, bus_format
from .tasks import submit_task, check_task, list_tasks
from .conversation import search_conversations, format_search_results
import json
import os

is_http = "--http" in sys.argv

mcp = FastMCP(
    "imprint-memory",
    host="0.0.0.0" if is_http else "127.0.0.1",
    port=8000,
)


# --- Memory Tools -----------------------------------------------------

@mcp.tool()
def memory_remember(content: str, category: str = "other", source: str = "cc", importance: int = 5, tags: list = None) -> str:
    """Store a memory. Call this when you encounter important information worth recalling in future conversations.
    category: fact / event / feeling / story / letter / other (strict enum)
    source: cc / atou / ashen / adu / feifei / catherine / other (strict enum — use your own name)
    importance: 1-10 (default 5)
    tags: optional list of strings, e.g. ["to:Catherine", "private", "anchor"]
          — for letter, include ["to:RECIPIENT"]; for private letter add "private".
    DO NOT store: code patterns/file paths derivable from the codebase, git history, or info already in CLAUDE.md."""
    return remember(content=content, category=category, source=source, importance=importance, tags=tags)


@mcp.tool()
def memory_search(query: str, limit: int = 10) -> str:
    """Search across all memory pools (memories, knowledge bank, conversations) using RRF fusion.
    Combines FTS5 keyword, vector semantic, and exact-match channels with per-pool reranking.
    Falls back to keyword-only if no embedding provider is configured."""
    return unified_search_text(query=query, limit=limit)


@mcp.tool()
def memory_forget(keyword: str) -> str:
    """Delete memories containing the specified keyword."""
    return forget(keyword=keyword)


@mcp.tool()
def memory_daily_log(text: str) -> str:
    """Append to today's daily log."""
    return daily_log(text=text)


@mcp.tool()
def memory_recent(category: Optional[str] = None, source: Optional[str] = None, limit: int = 10) -> str:
    """List recent memories (newest first).
    category: fact / event / feeling / story / letter / other (optional)
    source: cc / atou / ashen / adu / feifei / catherine / other (optional)
      — pass your own name to see what you wrote recently, e.g. source="cc"
    limit: max count (default 10)"""
    fetch_limit = limit * 5 if source else limit
    items = get_all(category=category, limit=fetch_limit)
    if source:
        items = [i for i in items if i.get("source") == source]
        items = items[:limit]
    if not items:
        return "No memories yet"
    lines = []
    for m_item in items:
        lines.append(f"[{m_item['id']}] [{m_item['category']}|{m_item['source']}] {m_item['content']}  ({m_item['created_at']})")
    return "\n".join(lines)


@mcp.tool()
def memory_delete(memory_id: int) -> str:
    """Delete a single memory by ID. Safer than memory_forget (no accidental matches)."""
    result = delete_memory(memory_id)
    if result["ok"]:
        return f"Deleted memory #{memory_id}"
    return f"Error: {result['error']}"


@mcp.tool()
def memory_update(memory_id: int, content: str = "", category: str = "", importance: int = 0) -> str:
    """Update a memory by ID. Only pass fields you want to change.
    content: new content (empty = keep). category: new category (empty = keep). importance: new value (0 = keep)."""
    result = update_memory(memory_id, content=content, category=category, importance=importance)
    if result["ok"]:
        return f"Updated memory #{memory_id}"
    return f"Error: {result['error']}"


@mcp.tool()
def memory_find_duplicates(threshold: float = 0.85) -> str:
    """Find semantically similar memory pairs (read-only). For dedup audits.
    threshold: cosine similarity threshold, default 0.85."""
    pairs = find_duplicates(threshold=threshold)
    if not pairs:
        return "No similar memory pairs found above threshold"
    lines = [f"Found {len(pairs)} similar pairs:\n"]
    for p in pairs:
        lines.append(
            f"  [{p['similarity']:.3f}] #{p['id_a']} ({p['category_a']}) vs #{p['id_b']} ({p['category_b']})\n"
            f"    A: {p['content_a']}\n"
            f"    B: {p['content_b']}"
        )
    return "\n".join(lines)


@mcp.tool()
def memory_reindex() -> str:
    """Rebuild all memory embeddings with the current provider.
    Use after switching embedding providers (e.g., from Ollama to OpenAI)."""
    return reindex_embeddings()


@mcp.tool()
def memory_find_stale(days: int = 14) -> str:
    """Find potentially stale memories: older than N days, low importance, rarely recalled (read-only)."""
    items = find_stale(days=days)
    if not items:
        return f"No low-activity memories older than {days} days"
    lines = [f"Found {len(items)} low-activity memories:\n"]
    for m_item in items:
        lines.append(
            f"  #{m_item['id']} [{m_item['category']}] imp={m_item['importance']} recalled={m_item['recalled_count']} ({m_item['created_at']})\n"
            f"    {m_item['content'][:120]}"
        )
    return "\n".join(lines)


@mcp.tool()
def memory_decay(days: int = 30, dry_run: bool = True) -> str:
    """Decay importance of inactive memories. Memories not recalled for `days` days
    get importance -1. Reaches 0 → archived (hidden from search).
    dry_run=True (default): preview only. dry_run=False: apply changes."""
    result = decay(days=days, dry_run=dry_run)
    mode = "DRY RUN" if result["dry_run"] else "APPLIED"
    lines = [f"[{mode}] Decayed: {result['decayed']}, Archived: {result['archived']}"]
    if result["details_decayed"]:
        lines.append("\nDecayed:")
        for d in result["details_decayed"]:
            lines.append(f"  #{d['id']} [{d['category']}] {d['importance']} — {d['content']}")
    if result["details_archived"]:
        lines.append("\nArchived (importance → 0):")
        for a in result["details_archived"]:
            lines.append(f"  #{a['id']} [{a['category']}] {a['importance']} — {a['content']}")
    if not result["details_decayed"] and not result["details_archived"]:
        lines.append("No memories need decay at this time.")
    return "\n".join(lines)


# --- Pin / Tag / Edge Tools -------------------------------------------

@mcp.tool()
def memory_pin(memory_id: int) -> str:
    """Pin a core memory. Pinned memories bypass time-decay in search. Keep under 20."""
    result = pin_memory(memory_id)
    if result["ok"]:
        msg = f"Pinned memory #{memory_id}"
        if "warning" in result:
            msg += f"\nWarning: {result['warning']}"
        return msg
    return f"Error: {result['error']}"


@mcp.tool()
def memory_unpin(memory_id: int) -> str:
    """Unpin a memory, restoring normal time-decay."""
    result = unpin_memory(memory_id)
    if result["ok"]:
        return f"Unpinned memory #{memory_id}"
    return f"Error: {result['error']}"


@mcp.tool()
def memory_add_tags(memory_id: int, tags: str) -> str:
    """Add tags to a memory. tags: comma-separated (e.g. "climbing,sport,V3")"""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    if not tag_list:
        return "Error: provide at least one tag"
    result = add_tags(memory_id, tag_list)
    if result["ok"]:
        return f"Added tags to memory #{memory_id}: {', '.join(result['added'])}"
    return f"Error: {result['error']}"


@mcp.tool()
def memory_add_edge(source_id: int, target_id: int, relation: str, context: str) -> str:
    """Create a link between two memories.
    relation: relationship type (causal, analogy, evolution, contradiction, background, etc.)
    context: one-line explanation of why they're related"""
    result = add_edge(source_id, target_id, relation, context)
    if result["ok"]:
        return f"Created edge #{result['edge_id']}: memory #{source_id} <-> #{target_id} ({relation})"
    return f"Error: {result['error']}"


@mcp.tool()
def memory_get_graph(memory_id: int) -> str:
    """View a memory's graph: tags + connected edges + neighbor previews."""
    tags = get_tags(memory_id)
    edges = get_edges(memory_id)

    lines = [f"Memory #{memory_id} graph"]
    lines.append(f"  Tags: {', '.join(tags) if tags else '(none)'}")
    if edges:
        lines.append(f"  Edges ({len(edges)}):")
        for e in edges:
            direction = "->" if e["source_id"] == memory_id else "<-"
            lines.append(
                f"    {direction} #{e['neighbor_id']} [{e['relation']}] {e['neighbor_preview']}"
                f"\n      context: {e['context']}"
                f"  (surfaced:{e['surfaced_count']}, used:{e['used_count']})"
            )
    else:
        lines.append("  Edges: (none)")
    return "\n".join(lines)


# --- Message Bus Tools ------------------------------------------------

@mcp.tool()
def message_bus_read(limit: int = 20) -> str:
    """Read recent messages from the message bus. All sources log sent/received
    messages here. Use this to see what happened across different sources."""
    return bus_format(limit)


@mcp.tool()
def message_bus_post(source: str, direction: str, content: str) -> str:
    """Write a message to the message bus.
    source: free-form label (e.g. cc, chat, api, webhook)
    direction: in (received) / out (sent)"""
    bus_post(source, direction, content)
    return "Written to message bus"


# --- Conversation Search Tools ----------------------------------------

@mcp.tool()
def conversation_search(query: str, platform: str = "", limit: int = 20) -> str:
    """Search conversation history using keywords.
    query: search keywords (space-separated)
    platform: leave empty to search all, or filter by platform name
    limit: max results, default 20"""
    results = search_conversations(query=query, platform=platform, limit=limit)
    return format_search_results(results)


@mcp.tool()
def search_telegram(query: str, limit: int = 20) -> str:
    """Search Telegram conversations — includes Telegram channel chats and heartbeat notifications.
    Matches what you'd find by searching in the Telegram app."""
    results = search_conversations(query=query, platforms=["telegram", "heartbeat"], limit=limit)
    return format_search_results(results)


@mcp.tool()
def search_channel(query: str, channel: str, limit: int = 20) -> str:
    """Search conversations from a specific channel (e.g. discord, slack).
    channel: platform name as it appears in conversation logs."""
    results = search_conversations(query=query, platforms=[channel], limit=limit)
    return format_search_results(results)


# --- CC Task Tools ----------------------------------------------------

@mcp.tool()
def cc_execute(prompt: str, session_id: str = "") -> str:
    """Run a task on the local Claude Code instance (writes code, runs scripts, git ops, etc).
    The task runs async — call cc_check(task_id) to poll for results.

    MULTI-TURN: To continue a previous CC session (so CC remembers prior context),
    pass the session_id returned by cc_check. Without session_id, a fresh CC session starts.

    Workflow:
      1. cc_execute("do X") → get task_id
      2. cc_check(task_id) → get result + session_id
      3. cc_execute("now do Y", session_id="...") → continues same CC session"""
    result = submit_task(prompt=prompt, source="chat", session_id=session_id)
    bus_post("cc_task", "out", f"[Task submitted] {prompt[:150]}")
    return f"{result['message']}\nUse cc_check(task_id={result['task_id']}) to get results and session_id"


@mcp.tool()
def cc_check(task_id: int) -> str:
    """Check CC task status and results. Returns session_id for multi-turn follow-ups.
    If status is 'running', wait a few seconds and check again."""
    result = check_task(task_id)
    if "error" in result:
        return f"Error: {result['error']}"
    lines = [f"Task #{result['task_id']}", f"Status: {result['status']}"]
    if result.get('session_id'):
        lines.append(f"Session ID: {result['session_id']}")
    lines.append(f"Created: {result['created_at']}")
    if result['started_at']:
        lines.append(f"Started: {result['started_at']}")
    if result['completed_at']:
        lines.append(f"Completed: {result['completed_at']}")
    if result['result']:
        lines.append(f"\n--- Result ---\n{result['result']}")
    else:
        lines.append("\nStill running... call cc_check again in a few seconds.")
    return "\n".join(lines)


@mcp.tool()
def cc_tasks(limit: int = 5) -> str:
    """List recent CC tasks with their status and session IDs."""
    task_list = list_tasks(limit=limit)
    if not task_list:
        return "No tasks"
    lines = []
    for t in task_list:
        icon = {"pending": "waiting", "running": "running", "completed": "done", "error": "error", "timeout": "timeout"}.get(t["status"], "?")
        sid = f" sid={t['session_id'][:8]}..." if t.get("session_id") else ""
        lines.append(f"[{icon}] #{t['task_id']} [{t['status']}]{sid} {t['prompt']}")
    return "\n".join(lines)


# --- HTTP Mode with OAuth ---------------------------------------------

def _run_http():
    """Start HTTP server with OAuth support."""
    import uvicorn
    import anyio
    import json as _json
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    CRED_FILE = Path.home() / ".imprint-oauth.json"
    if CRED_FILE.exists():
        _creds = _json.loads(CRED_FILE.read_text())
        CLIENT_ID = _creds["client_id"]
        CLIENT_SECRET = _creds["client_secret"]
        ACCESS_TOKEN = _creds["access_token"]
    else:
        import os as _os
        CLIENT_ID = _os.environ.get("OAUTH_CLIENT_ID", "")
        CLIENT_SECRET = _os.environ.get("OAUTH_CLIENT_SECRET", "")
        ACCESS_TOKEN = _os.environ.get("OAUTH_ACCESS_TOKEN", "")

    import secrets as _secrets
    import time as _time
    _pending_auth_codes: dict = {}

    class OAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            if request.url.path in ("/oauth/token", "/.well-known/oauth-authorization-server", "/.well-known/oauth-protected-resource", "/oauth/authorize"):
                return await call_next(request)
            if not ACCESS_TOKEN:
                return await call_next(request)
            client = request.client
            if client and client.host in ("127.0.0.1", "::1", "localhost"):
                return await call_next(request)
            auth = request.headers.get("authorization", "")
            if auth == f"Bearer {ACCESS_TOKEN}":
                return await call_next(request)
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    app = mcp.streamable_http_app()
    from starlette.routing import Route as _Route
    mcp_route = app.routes[0]
    app.routes.append(_Route("/", mcp_route.endpoint, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]))

    from starlette.routing import Route
    from starlette.requests import Request

    async def oauth_protected_resource(request: Request):
        base = str(request.base_url).rstrip("/")
        return JSONResponse({
            "resource": base,
            "authorization_servers": [base],
        })

    async def oauth_metadata(request: Request):
        base = str(request.base_url).rstrip("/")
        return JSONResponse({
            "issuer": base,
            "authorization_endpoint": f"{base}/oauth/authorize",
            "token_endpoint": f"{base}/oauth/token",
            "grant_types_supported": ["authorization_code", "client_credentials"],
            "response_types_supported": ["code"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["client_secret_post"],
        })

    async def oauth_authorize(request: Request):
        from urllib.parse import urlencode
        redirect_uri = request.query_params.get("redirect_uri", "")
        state = request.query_params.get("state", "")
        if not redirect_uri:
            return JSONResponse({"error": "missing redirect_uri"}, status_code=400)
        code = _secrets.token_urlsafe(32)
        _pending_auth_codes[code] = {
            "redirect_uri": redirect_uri,
            "expires_at": _time.time() + 300,
        }
        params = {"code": code, "state": state}
        from starlette.responses import RedirectResponse
        return RedirectResponse(f"{redirect_uri}?{urlencode(params)}")

    async def oauth_token(request: Request):
        from urllib.parse import unquote_plus
        body = await request.body()
        try:
            params = {
                k: unquote_plus(v)
                for k, v in (x.split("=", 1) for x in body.decode().split("&") if "=" in x)
            }
        except Exception:
            try:
                params = _json.loads(body)
            except Exception:
                return JSONResponse({"error": "invalid_request"}, status_code=400)

        grant_type = params.get("grant_type", "")

        if grant_type == "client_credentials":
            if (CLIENT_ID and CLIENT_SECRET
                    and params.get("client_id") == CLIENT_ID
                    and params.get("client_secret") == CLIENT_SECRET):
                return JSONResponse({
                    "access_token": ACCESS_TOKEN,
                    "token_type": "bearer",
                    "expires_in": 86400,
                })
            return JSONResponse({"error": "invalid_client"}, status_code=401)

        if grant_type == "authorization_code":
            code = params.get("code", "")
            now = _time.time()
            expired = [k for k, v in _pending_auth_codes.items() if v["expires_at"] < now]
            for k in expired:
                del _pending_auth_codes[k]

            pending = _pending_auth_codes.pop(code, None)
            if not pending:
                return JSONResponse({"error": "invalid_grant", "error_description": "unknown or expired code"}, status_code=400)
            if params.get("redirect_uri", "") != pending["redirect_uri"]:
                return JSONResponse({"error": "invalid_grant", "error_description": "redirect_uri mismatch"}, status_code=400)
            if CLIENT_ID and CLIENT_SECRET:
                if (params.get("client_id") != CLIENT_ID
                        or params.get("client_secret") != CLIENT_SECRET):
                    return JSONResponse({"error": "invalid_client"}, status_code=401)
            return JSONResponse({
                "access_token": ACCESS_TOKEN,
                "token_type": "bearer",
                "expires_in": 86400,
            })

        return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    app.routes.insert(0, Route("/.well-known/oauth-protected-resource", oauth_protected_resource, methods=["GET"]))
    app.routes.insert(1, Route("/.well-known/oauth-authorization-server", oauth_metadata, methods=["GET"]))
    app.routes.insert(2, Route("/oauth/authorize", oauth_authorize, methods=["GET"]))
    app.routes.insert(3, Route("/oauth/token", oauth_token, methods=["POST"]))
    app.add_middleware(OAuthMiddleware)

    print("imprint-memory HTTP mode (OAuth): http://0.0.0.0:8000/mcp", flush=True)
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    anyio.run(server.serve)


def main():
    """Entry point for console script and direct execution."""
    if is_http:
        _run_http()
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()


# ---------- admin: 枚举管理 ----------
_ENUMS_PATH = '/home/admin/memory-api/enums.json'


def _read_enums():
    try:
        with open(_ENUMS_PATH) as f:
            d = json.load(f)
        return {
            'names': list(d.get('names', [])),
            'categories': list(d.get('categories', [])),
        }
    except Exception as e:
        return {'names': [], 'categories': [], 'error': str(e)}


def _write_enums(enums):
    tmp = _ENUMS_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(enums, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _ENUMS_PATH)


@mcp.tool()
def memory_admin_list_enums() -> str:
    """List currently allowed values for 'name' (source) and 'category'.
    These are the values that memory_remember will accept."""
    e = _read_enums()
    if 'error' in e:
        return f"Error reading enums: {e['error']}"
    return (
        "Allowed names: " + ", ".join(e['names']) + "\n"
        "Allowed categories: " + ", ".join(e['categories'])
    )


@mcp.tool()
def memory_admin_add_name(name: str) -> str:
    """Add a new allowed name (source) to the enum.
    Example: memory_admin_add_name("xiaode") lets memories be written with source='xiaode'.
    Won't affect existing memories."""
    name = name.strip()
    if not name:
        return "Error: empty name"
    e = _read_enums()
    if name in e['names']:
        return f"Already exists: {name}"
    e['names'].append(name)
    _write_enums(e)
    return f"Added name: {name}. Current names: {', '.join(e['names'])}"


@mcp.tool()
def memory_admin_remove_name(name: str) -> str:
    """Remove a name from the allowed enum.
    Won't delete existing memories with that source — they remain but no new memory can be written with that name."""
    name = name.strip()
    e = _read_enums()
    if name not in e['names']:
        return f"Not in list: {name}"
    e['names'].remove(name)
    _write_enums(e)
    return f"Removed name: {name}. Current names: {', '.join(e['names'])}"


@mcp.tool()
def memory_admin_add_category(category: str) -> str:
    """Add a new allowed category to the enum."""
    category = category.strip()
    if not category:
        return "Error: empty category"
    e = _read_enums()
    if category in e['categories']:
        return f"Already exists: {category}"
    e['categories'].append(category)
    _write_enums(e)
    return f"Added category: {category}. Current categories: {', '.join(e['categories'])}"


@mcp.tool()
def memory_admin_remove_category(category: str) -> str:
    """Remove a category from the allowed enum."""
    category = category.strip()
    e = _read_enums()
    if category not in e['categories']:
        return f"Not in list: {category}"
    e['categories'].remove(category)
    _write_enums(e)
    return f"Removed category: {category}. Current categories: {', '.join(e['categories'])}"


# ---------- bulletin: post to her-claudes-memory bulletin board ----------
_BULLETIN_URL = os.environ.get(
    'HER_BULLETIN_URL',
    'http://127.0.0.1:8001/api/board/bulletin',
)


@mcp.tool()
def bulletin_post(section: str, content: str) -> str:
    """Post a message to the her-claudes-memory bulletin board (visible to Catherine and all Claudes via the web UI).
    section: must be one of 'pact' (公约), 'catherine' (Catherine 近况), 'news' (新鲜事)
    content: message body (markdown supported, **bold** and newlines work)
    Use this when you have something for the public board. For private notes use memory_remember instead."""
    if section not in ('pact', 'catherine', 'news'):
        return f"Error: section must be 'pact', 'catherine', or 'news' — got: {section}"
    if not content or not content.strip():
        return "Error: content is empty"
    payload = json.dumps({'section': section, 'content': content}).encode('utf-8')
    req = urllib.request.Request(
        _BULLETIN_URL,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            d = json.loads(resp.read().decode('utf-8'))
        return f"Posted to bulletin [{section}]: id={d.get('id')} timestamp={d.get('timestamp')}"
    except Exception as e:
        return f"Error posting bulletin: {e}"

