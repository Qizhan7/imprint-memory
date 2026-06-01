"""
Chat Sync Receiver — HTTP endpoint that ingests conversations from the
imprint-chat-sync browser extension (or any client posting to /api/ingest).

Run with:
    imprint-memory-receiver               # listens on 127.0.0.1:8001
    PORT=9001 imprint-memory-receiver     # custom port

Pipeline per ingest call:
    POST /api/ingest                  (browser extension)
      ↓
    log_message()                     (writes to conversation_log)
      ↓ background task
    embed_new_messages()              (Gemini Embedding 2 per message)
      ↓
    detect_topic_shifts()             (adjacent user-msg cosine → topic edges)
      ↓
    incremental_chunk_update()        (chunk → summarize → embed → top-K edges)

The receiver shares the same SQLite database as the main imprint-memory server
(configured via IMPRINT_DATA_DIR / IMPRINT_DB).
"""

import math
import os
import re
import struct
import sys
import threading
import time

from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Route


# Load API keys from common conventional locations before anything imports the
# embedding helper. Without an API key, embed_new_messages() silently no-ops via
# the broad except — leaving conversation_vectors empty and search degraded with
# no visible error.
_KEY_FILES = [
    os.path.join(os.getcwd(), ".env"),
    os.path.expanduser("~/.imprint/.env"),
    os.path.expanduser("~/.imprint/keys.env"),
    os.path.expanduser("~/.config/imprint/keys.env"),
]
for _kf in _KEY_FILES:
    if not os.path.isfile(_kf):
        continue
    try:
        for _line in open(_kf, encoding="utf-8"):
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            if _line.startswith("export "):
                _line = _line[len("export "):]
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    except Exception:
        pass

_EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "google").lower()
_KEY_VAR = {"google": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}.get(_EMBED_PROVIDER, "")
if _KEY_VAR and not os.environ.get(_KEY_VAR) and not os.environ.get("GOOGLE_API_KEYS"):
    print(
        f"⚠️  WARNING: {_KEY_VAR} is not set in the environment. "
        f"Per-message embedding (conversation_vectors) will be silently skipped, "
        f"so chunk-expansion and message-level search will degrade to keyword-only.\n"
        f"   Fix: export {_KEY_VAR}=... before launching, or drop a keys.env in one of:\n"
        f"     " + "\n     ".join(_KEY_FILES),
        file=sys.stderr,
        flush=True,
    )


from .conversation import log_message, get_recent
from .db import _get_db as _get_app_db

DEFAULT_PORT = 8001
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
EMBED_DELAY = float(os.environ.get("IMPRINT_RECEIVER_EMBED_DELAY", "0.7"))
SHIFT_THRESHOLD = float(os.environ.get("IMPRINT_RECEIVER_SHIFT_THRESHOLD", "0.50"))
RECEIVER_HOST_ENV = os.environ.get("IMPRINT_RECEIVER_HOST", os.environ.get("HOST", "127.0.0.1"))
RECEIVER_PORT_ENV = int(os.environ.get("IMPRINT_RECEIVER_PORT", os.environ.get("PORT", DEFAULT_PORT)))
RECEIVER_CORS_ORIGIN_REGEX = os.environ.get(
    "IMPRINT_RECEIVER_CORS_ORIGIN_REGEX",
    r"^chrome-extension://.*$",
)


def _blob_to_vec(blob):
    """Decode a float32 embedding blob."""
    return list(struct.unpack(f"{len(blob)//4}f", blob))


def _cosine_sim(a, b):
    """Return cosine similarity for same-length vectors."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _get_db():
    """Open the receiver database connection and ensure receiver-owned tables."""
    db = _get_app_db()
    db.execute("""CREATE TABLE IF NOT EXISTS conversation_vectors (
        msg_id INTEGER PRIMARY KEY,
        embedding BLOB NOT NULL,
        model TEXT DEFAULT 'gemini-embedding-2'
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS conversation_edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        msg_before INTEGER NOT NULL,
        msg_after INTEGER NOT NULL,
        session_id TEXT DEFAULT '',
        similarity REAL NOT NULL,
        strength REAL DEFAULT 1.0,
        surfaced_count INTEGER DEFAULT 0,
        last_surfaced_at TEXT,
        status TEXT DEFAULT 'active',
        created_at TEXT NOT NULL,
        UNIQUE(msg_before, msg_after)
    )""")
    return db


def embed_new_messages(msg_ids):
    """Background: embed newly ingested messages with the configured embedder."""
    if not msg_ids:
        return
    try:
        from .memory_manager import EMBED_MODEL, _embed
    except Exception:
        return

    db = _get_db()
    try:
        for msg_id in msg_ids:
            existing = db.execute(
                "SELECT 1 FROM conversation_vectors WHERE msg_id=?", (msg_id,)
            ).fetchone()
            if existing:
                continue

            row = db.execute(
                "SELECT content, direction FROM conversation_log WHERE id=?", (msg_id,)
            ).fetchone()
            if not row:
                continue

            content = row["content"]
            if row["direction"] == "out":
                content = THINK_RE.sub("", content).strip()
            if len(content) < 10:
                continue

            try:
                vec = _embed(content[:2000])
                if vec:
                    db.execute(
                        "INSERT OR IGNORE INTO conversation_vectors (msg_id, embedding, model) VALUES (?, ?, ?)",
                        (msg_id, struct.pack(f"{len(vec)}f", *vec), EMBED_MODEL),
                    )
                    db.commit()
            except Exception:
                pass

            time.sleep(EMBED_DELAY)

        detect_topic_shifts(db, msg_ids)
    except Exception:
        pass
    finally:
        db.close()

    # Incremental chunk processing: split → summarize → embed → Top-K edges
    try:
        from .conversation_chunker import incremental_chunk_update
        result = incremental_chunk_update(batch_size=200)
        if result.get("chunks_created", 0) > 0:
            print(
                f"[chunk] Created {result['chunks_created']} chunks, "
                f"{result.get('edges_created', 0)} edges",
                flush=True,
            )
    except Exception as e:
        print(f"[chunk] Error: {e}", flush=True)


def detect_topic_shifts(db, msg_ids):
    """Detect topic shifts by cosine similarity between adjacent user messages."""
    if not msg_ids:
        return

    placeholders = ",".join("?" * len(msg_ids))
    sessions = db.execute(
        f"SELECT DISTINCT session_id FROM conversation_log WHERE id IN ({placeholders})",
        msg_ids,
    ).fetchall()

    shifts = 0
    for (session_id,) in sessions:
        if not session_id:
            continue

        rows = db.execute(
            """SELECT c.id, v.embedding
               FROM conversation_log c
               JOIN conversation_vectors v ON c.id = v.msg_id
               WHERE c.session_id = ? AND c.direction = 'in'
               ORDER BY c.id""",
            (session_id,),
        ).fetchall()

        if len(rows) < 2:
            continue

        for i in range(1, len(rows)):
            vec_a = _blob_to_vec(rows[i - 1]["embedding"])
            vec_b = _blob_to_vec(rows[i]["embedding"])
            sim = _cosine_sim(vec_a, vec_b)

            if sim < SHIFT_THRESHOLD:
                try:
                    db.execute(
                        """INSERT OR IGNORE INTO conversation_edges
                           (msg_before, msg_after, session_id, similarity, created_at)
                           VALUES (?, ?, ?, ?, datetime('now'))""",
                        (rows[i - 1]["id"], rows[i]["id"], session_id, sim),
                    )
                    shifts += 1
                except Exception:
                    pass

    if shifts:
        db.commit()


async def ingest(request):
    """Ingest a batch of conversation messages from the browser extension."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid JSON"}, status_code=400)

    conversation_id = data.get("conversation_id", "")
    messages = data.get("messages", [])

    if not messages:
        return JSONResponse({"ok": True, "ingested": 0, "skipped": 0, "errors": 0})

    results = {"ingested": 0, "skipped": 0, "errors": 0}
    ingested_ids = []

    for msg in messages:
        content = (msg.get("content") or "").strip()
        if not content:
            results["errors"] += 1
            continue

        result = log_message(
            platform=data.get("platform", "claude.ai"),
            direction=msg.get("direction", "in"),
            content=content,
            speaker=msg.get("speaker", ""),
            session_id=conversation_id,
            entrypoint="browser_extension",
            created_at=msg.get("created_at", ""),
            summary=msg.get("summary", ""),
            model=msg.get("model", data.get("model", "")),
            external_id=msg.get("external_id") or msg.get("uuid", ""),
        )

        if result.get("skipped"):
            results["skipped"] += 1
        elif result.get("ok"):
            results["ingested"] += 1
            if result.get("id"):
                ingested_ids.append(result["id"])
        else:
            results["errors"] += 1

    return JSONResponse(
        {"ok": True, **results},
        background=BackgroundTask(embed_new_messages, ingested_ids),
    )


async def health(request):
    """Return a lightweight liveness response."""
    return JSONResponse({"ok": True, "service": "imprint-chat-sync-receiver"})


async def status(request):
    """Return recent ingest and embedding status for the extension popup."""
    recent = get_recent(platform="claude.ai", limit=5)
    db = _get_db()
    try:
        vec_count = db.execute("SELECT count(*) FROM conversation_vectors").fetchone()[0]
    finally:
        db.close()
    return JSONResponse({
        "ok": True,
        "recent_count": len(recent),
        "last_message": recent[-1]["created_at"] if recent else None,
        "vectors": vec_count,
    })


app = Starlette(
    routes=[
        Route("/api/ingest", ingest, methods=["POST"]),
        Route("/api/health", health, methods=["GET"]),
        Route("/api/status", status, methods=["GET"]),
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=RECEIVER_CORS_ORIGIN_REGEX,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def _backfill_on_startup():
    """Background-embed any messages that don't have a vector yet."""

    def _run():
        db = _get_db()
        try:
            unembedded = db.execute("""
                SELECT COUNT(*) as c FROM conversation_log cl
                LEFT JOIN conversation_vectors cv ON cl.id = cv.msg_id
                WHERE cv.msg_id IS NULL AND cl.platform NOT IN ('cc')
                AND cl.content IS NOT NULL AND length(cl.content) > 0
            """).fetchone()[0]
        finally:
            db.close()

        if unembedded == 0:
            print("[backfill] No unembedded messages, skipping", flush=True)
            return

        print(
            f"[backfill] Found {unembedded} unembedded messages, processing...",
            flush=True,
        )

        db = _get_db()
        try:
            rows = db.execute("""
                SELECT cl.id FROM conversation_log cl
                LEFT JOIN conversation_vectors cv ON cl.id = cv.msg_id
                WHERE cv.msg_id IS NULL AND cl.platform NOT IN ('cc')
                AND cl.content IS NOT NULL AND length(cl.content) > 0
                ORDER BY cl.id
            """).fetchall()
            msg_ids = [r[0] for r in rows]
        finally:
            db.close()

        batch_size = 100
        for i in range(0, len(msg_ids), batch_size):
            batch = msg_ids[i:i + batch_size]
            embed_new_messages(batch)
            print(
                f"[backfill] Embedded {min(i + batch_size, len(msg_ids))}/{len(msg_ids)}",
                flush=True,
            )

        print("[backfill] Embedding complete. Running incremental chunk update...", flush=True)
        try:
            from .conversation_chunker import incremental_chunk_update
            result = incremental_chunk_update(batch_size=200)
            print(f"[backfill] Chunk update: {result}", flush=True)
        except Exception as e:
            print(f"[backfill] Chunk update error: {e}", flush=True)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def main():
    """Run the chat-sync receiver HTTP service."""
    import argparse

    parser = argparse.ArgumentParser(description="Imprint chat sync receiver")
    parser.add_argument(
        "--host",
        default=RECEIVER_HOST_ENV,
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=RECEIVER_PORT_ENV,
        help=f"Port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--no-backfill",
        action="store_true",
        help="Skip the startup backfill pass over unembedded messages",
    )
    args = parser.parse_args()

    try:
        from .memory_manager import check_vector_dimensions
        _dim_warn = check_vector_dimensions()
        if _dim_warn:
            print(_dim_warn, file=sys.stderr, flush=True)
    except Exception:
        pass

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is required. Install with: pip install 'imprint-memory[receiver]'",
            flush=True,
        )
        raise SystemExit(1)

    print(
        f"imprint-memory chat-sync receiver listening on {args.host}:{args.port}",
        flush=True,
    )
    if not args.no_backfill:
        _backfill_on_startup()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
