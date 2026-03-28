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
from pathlib import Path
from typing import Optional

# When run as script, set up package context for relative imports
if __name__ == "__main__" or not __package__:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    __package__ = "imprint_memory"

from mcp.server.fastmcp import FastMCP
from .memory_manager import (
    remember, search_text, forget, daily_log, get_all,
)
from .bus import bus_post, bus_format
from .tasks import submit_task, check_task, list_tasks

is_http = "--http" in sys.argv

mcp = FastMCP(
    "imprint-memory",
    host="0.0.0.0" if is_http else "127.0.0.1",
    port=8000,
)


# --- Memory Tools -----------------------------------------------------

@mcp.tool()
def memory_remember(content: str, category: str = "general", source: str = "cc", importance: int = 5) -> str:
    """Store a memory. Call this when you encounter important information.
    category: facts/events/tasks/experience/general
    source: cc/telegram/wechat/chat"""
    return remember(content=content, category=category, source=source, importance=importance)


@mcp.tool()
def memory_search(query: str, limit: int = 10) -> str:
    """Search memories. Supports semantic search (natural language) when Ollama is running."""
    return search_text(query=query, limit=limit)


@mcp.tool()
def memory_forget(keyword: str) -> str:
    """Delete memories containing the specified keyword."""
    return forget(keyword=keyword)


@mcp.tool()
def memory_daily_log(text: str) -> str:
    """Append to today's daily log."""
    return daily_log(text=text)


@mcp.tool()
def memory_list(category: Optional[str] = None, limit: int = 20) -> str:
    """List memories (newest first)."""
    items = get_all(category=category, limit=limit)
    if not items:
        return "No memories yet"
    lines = []
    for m_item in items:
        lines.append(f"[{m_item['id']}] [{m_item['category']}|{m_item['source']}] {m_item['content']}  ({m_item['created_at']})")
    return "\n".join(lines)


# --- Message Bus Tools ------------------------------------------------

@mcp.tool()
def message_bus_read(limit: int = 20) -> str:
    """Read recent cross-channel messages. Every channel (Telegram/WeChat/Chat/CC) logs sent
    and received messages here. Use this to understand what happened in other channels."""
    return bus_format(limit)


@mcp.tool()
def message_bus_post(source: str, direction: str, content: str) -> str:
    """Write a message to the cross-channel message bus.
    source: telegram/wechat/chat/cc_task/scheduled
    direction: in (user sent) / out (Claude sent)"""
    bus_post(source, direction, content)
    return "Written to message bus"


# --- CC Task Tools ----------------------------------------------------

@mcp.tool()
def cc_execute(prompt: str) -> str:
    """Submit a task for local Claude Code to execute. For: writing code, running scripts, git ops, etc.
    Returns a task_id. Use cc_check(task_id) to check results later."""
    result = submit_task(prompt=prompt, source="chat")
    bus_post("cc_task", "out", f"[Task submitted] {prompt[:150]}")
    return f"{result['message']}\nUse cc_check(task_id={result['task_id']}) to check results"


@mcp.tool()
def cc_check(task_id: int) -> str:
    """Check CC task status and results."""
    result = check_task(task_id)
    if "error" in result:
        return f"Error: {result['error']}"
    lines = [f"Task #{result['task_id']}", f"Status: {result['status']}", f"Created: {result['created_at']}"]
    if result['started_at']:
        lines.append(f"Started: {result['started_at']}")
    if result['completed_at']:
        lines.append(f"Completed: {result['completed_at']}")
    if result['result']:
        lines.append(f"\n--- Result ---\n{result['result']}")
    else:
        lines.append("\nStill running...")
    return "\n".join(lines)


@mcp.tool()
def cc_tasks(limit: int = 5) -> str:
    """List recent CC tasks."""
    task_list = list_tasks(limit=limit)
    if not task_list:
        return "No tasks"
    lines = []
    for t in task_list:
        icon = {"pending": "waiting", "running": "running", "completed": "done", "error": "error", "timeout": "timeout"}.get(t["status"], "?")
        lines.append(f"[{icon}] #{t['task_id']} [{t['status']}] {t['prompt']}")
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
