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

import os
import sys
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
    unified_search_text, surfacing_search, pin_memory, unpin_memory,
    add_tags, get_tags, add_edge, get_edges,
    build_stopwords, list_stopwords, add_stopword, remove_stopword,
)
from .bus import bus_post, bus_format
# Task tools (cc_execute/cc_check/cc_tasks) removed from public release —
# they are an agent orchestration feature unrelated to memory/search.
from .conversation import search_conversations, format_search_results
from .db import DATA_DIR


def _arg_value(flag: str, default: str) -> str:
    """Read a simple CLI flag value without forcing argparse on MCP stdio."""
    try:
        idx = sys.argv.index(flag)
    except ValueError:
        return default
    try:
        return sys.argv[idx + 1]
    except IndexError:
        return default


is_http = "--http" in sys.argv
HTTP_HOST = os.environ.get("IMPRINT_HTTP_HOST", _arg_value("--host", "0.0.0.0" if is_http else "127.0.0.1"))
HTTP_PORT = int(os.environ.get("IMPRINT_HTTP_PORT", _arg_value("--port", "8000")))

mcp = FastMCP(
    "imprint-memory",
    host=HTTP_HOST,
    port=HTTP_PORT,
)


# --- Memory Tools -----------------------------------------------------

@mcp.tool()
def memory_remember(content: str, category: str = "facts", source: str = "cc", importance: int = 5) -> str:
    """Store a memory. Call this when you encounter important information worth recalling in future conversations.
    category: facts (persistent truths/preferences/commitments) / events (things that happened, with dates) / insights (reflections/observations/lessons)
    Tech experience goes to experience-index.md, NOT here.
    source: free-form label for where the info came from (e.g. cc, chat, api)
    DO NOT store: code patterns/file paths derivable from the codebase, git history, or info already in CLAUDE.md."""
    return remember(content=content, category=category, source=source, importance=importance)


@mcp.tool()
def memory_search(query: str, limit: int = 10, after: Optional[str] = None, before: Optional[str] = None) -> str:
    """Search across memory pools (memories, knowledge bank, conversation chunks) using RRF fusion.
    Default pools: ["memory", "bank", "chunk"]. Chunks are an index over the raw conversation log —
    chunk hits expand into their top-ranked raw messages, so you see originals not summaries.
    If this returns nothing relevant, fall back to conversation_search (raw conversation_log FTS).
    Combines FTS5 keyword, vector semantic, and exact-match channels with per-pool reranking.
    Falls back to keyword-only if no embedding provider is configured.
    after/before: ISO date strings to filter by time range (e.g. '2026-04-01' or '2026-04-01T10:00:00')."""
    return unified_search_text(query=query, limit=limit, after=after, before=before)


@mcp.tool()
def memory_forget(keyword: str) -> str:
    """Delete memories containing the specified keyword."""
    return forget(keyword=keyword)


@mcp.tool()
def memory_daily_log(text: str) -> str:
    """Append to today's daily log."""
    return daily_log(text=text)


@mcp.tool()
def memory_list(category: Optional[str] = None, limit: int = 20, after: Optional[str] = None, before: Optional[str] = None) -> str:
    """List memories (newest first).
    after/before: ISO date strings to filter by time range (e.g. '2026-04-01' or '2026-04-01T10:00:00')."""
    items = get_all(category=category, limit=limit, after=after, before=before)
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


# --- Stopwords Tools --------------------------------------------------

@mcp.tool()
def stopwords_build(threshold: float = 0.15) -> str:
    """Rebuild the auto-stopwords table from document frequency analysis.
    Words appearing in >threshold fraction of documents are auto-stopped.
    This prevents high-frequency low-information words from polluting search."""
    result = build_stopwords(threshold=threshold)
    if result["ok"]:
        return f"Built stopwords: {result['auto_stopwords']} auto-detected (threshold {result['threshold']:.0%}, {result['total_docs']} docs scanned)"
    return f"Error: {result.get('error', 'unknown')}"


@mcp.tool()
def stopwords_show(limit: int = 50) -> str:
    """Show current stopwords with their document frequency.
    User can review which words are being filtered from search queries."""
    items = list_stopwords()
    if not items:
        return "No stopwords. Run stopwords_build to detect high-frequency words."
    lines = [f"Stopwords ({len(items)} total, showing top {min(limit, len(items))})"]
    lines.append(f"{'word':>12} | {'freq':>6} | {'source':>6} | active")
    lines.append("-" * 42)
    for item in items[:limit]:
        active = "Y" if item["active"] else "N"
        lines.append(f"{item['word']:>12} | {item['doc_freq']:>5.1%} | {item['source']:>6} | {active}")
    return "\n".join(lines)


@mcp.tool()
def stopwords_add(word: str) -> str:
    """Manually add a word to the stopwords list."""
    result = add_stopword(word)
    return f"Added '{word}' to stopwords" if result["ok"] else f"Error: {result['error']}"


@mcp.tool()
def stopwords_remove(word: str) -> str:
    """Remove a word from stopwords, or keep an auto-detected word active in search."""
    result = remove_stopword(word)
    return f"Removed '{word}' from stopwords" if result["ok"] else f"Error: {result['error']}"


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
def conversation_search_semantic(query: str, limit: int = 10) -> str:
    """Search conversation history by vector similarity.
    Prefers summarized conversation chunks, then falls back to message-level vectors."""
    from .conversation_chunker import search_chunks, format_chunk_results

    chunk_results = search_chunks(query=query, limit=limit)
    if chunk_results:
        return format_chunk_results(chunk_results)

    from .conversation_search import search_conversations_semantic
    results = search_conversations_semantic(query=query, limit=limit)
    if not results:
        return "没有找到相关对话记录"

    lines = []
    for r in results:
        direction = "←" if r["direction"] == "in" else "→"
        content = " ".join(r["content"].split())
        if len(content) > 240:
            content = content[:240] + "..."
        lines.append(
            f"[{r['similarity']:.3f}] [{r['created_at']}] "
            f"#{r['id']} {r['platform']}{direction} {content}"
        )
    return "\n".join(lines)


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




@mcp.tool()
def experience_append(title: str, content: str) -> str:
    """Append a new experience entry to memory/bank/experience.md.
    title: section heading (e.g. '端口冲突排查')
    content: markdown body (bullet points recommended)"""
    exp_path = DATA_DIR / "memory" / "bank" / "experience.md"
    if not exp_path.exists():
        exp_path.parent.mkdir(parents=True, exist_ok=True)
        exp_path.write_text("# Experience Bank\n", encoding="utf-8")
    with open(exp_path, "a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n{content}\n")
    return f"Added experience: {title}"


# --- HTTP Mode with OAuth ---------------------------------------------

def _run_http():
    """Start HTTP server with OAuth support."""
    import uvicorn
    import anyio
    import json as _json
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    CRED_FILE = Path(os.environ.get("IMPRINT_OAUTH_FILE", str(Path.home() / ".imprint-oauth.json"))).expanduser()
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

    print(f"imprint-memory HTTP mode (OAuth): http://{HTTP_HOST}:{HTTP_PORT}/mcp", flush=True)
    config = uvicorn.Config(app, host=HTTP_HOST, port=HTTP_PORT, log_level="info")
    server = uvicorn.Server(config)
    anyio.run(server.serve)


def _check_optional_deps() -> None:
    """Warn about missing deps that silently degrade search. Always to stderr —
    stdout is the MCP stdio protocol channel and must stay clean."""
    from .db import _JIEBA_OK
    from .memory_manager import _JIO_OK
    try:
        import numpy  # noqa: F401
        numpy_ok = True
    except ImportError:
        numpy_ok = False
    if not _JIEBA_OK:
        print("[imprint-memory] WARNING: jieba missing (should auto-install) — "
              "Chinese (CJK) search degraded to char-level, lower recall. "
              "Fix: pip install jieba", file=sys.stderr)
    if not numpy_ok:
        print("[imprint-memory] WARNING: numpy missing (should auto-install) — "
              "graph-mode retrieval (timeline/causal/origin) disabled. "
              "Fix: pip install numpy", file=sys.stderr)
    if not _JIO_OK:
        print("[imprint-memory] note: jionlp not installed (optional) — "
              "natural-language time parsing falls back to regex. "
              "Enable: pip install jionlp", file=sys.stderr)


def main():
    """Entry point for console script and direct execution."""
    _check_optional_deps()
    if is_http:
        _run_http()
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
