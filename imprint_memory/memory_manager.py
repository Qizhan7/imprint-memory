"""
Claude Imprint — Memory System
Pure memory operations: CRUD, hybrid search (FTS5 + vector), bank indexing, daily log.
Includes RRF unified retrieval across memory, bank, and conversation pools.
"""

import base64
import json
import math
import os
import re
import sqlite3
import struct
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .db import (
    _get_db, now_local, now_str,
    DAILY_LOG_DIR, BANK_DIR, MEMORY_INDEX, LOCAL_TZ,
    segment_cjk, sanitize_fts_query,
    _CJK_RE, _JIEBA_OK,
)

# ─── Embedding Config ────────────────────────────────────
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "google")  # "google", "ollama", or "openai"

# Ollama settings
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# OpenAI-compatible settings (also works with Voyage AI, Azure, etc.)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBED_API_BASE = os.environ.get("EMBED_API_BASE", "https://api.openai.com")

# Google Gemini Embedding settings
# Supports multiple keys for rotation: comma-separated in GOOGLE_API_KEYS
GOOGLE_API_KEYS: list[str] = [
    k.strip() for k in os.environ.get("GOOGLE_API_KEYS", "").split(",") if k.strip()
]
if not GOOGLE_API_KEYS:
    _single = os.environ.get("GOOGLE_API_KEY", "")
    if _single:
        GOOGLE_API_KEYS = [_single]
_google_key_index = 0

# Model defaults per provider
_DEFAULT_MODELS = {
    "ollama": "bge-m3",
    "openai": "text-embedding-3-small",
    "google": "gemini-embedding-2",
}
EMBED_MODEL = os.environ.get("EMBED_MODEL", _DEFAULT_MODELS.get(EMBED_PROVIDER, "bge-m3"))

BANK_INDEX_VERSION = 2

# Hybrid search weights
WEIGHT_VECTOR = 0.4
WEIGHT_FTS = 0.4
WEIGHT_RECENCY = 0.2


# ─── Vector Embeddings ───────────────────────────────────

def _embed_ollama(text: str) -> Optional[list[float]]:
    """Generate embedding via Ollama (local)."""
    try:
        payload = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            embeddings = data.get("embeddings", [])
            if embeddings and len(embeddings[0]) > 0:
                return embeddings[0]
    except Exception:
        pass
    return None


def _embed_openai(text: str) -> Optional[list[float]]:
    """Generate embedding via OpenAI-compatible API.
    Works with: OpenAI, Voyage AI, Azure OpenAI, any OpenAI-compatible service.
    Set EMBED_API_BASE to point to your provider."""
    if not OPENAI_API_KEY:
        return None
    try:
        url = f"{EMBED_API_BASE.rstrip('/')}/v1/embeddings"
        payload = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            items = data.get("data", [])
            if items and "embedding" in items[0]:
                return items[0]["embedding"]
    except Exception:
        pass
    return None


def _next_google_key() -> Optional[str]:
    """Round-robin through available Google API keys."""
    global _google_key_index
    if not GOOGLE_API_KEYS:
        return None
    key = GOOGLE_API_KEYS[_google_key_index % len(GOOGLE_API_KEYS)]
    _google_key_index += 1
    return key


def _embed_google(text: str, image_path: Optional[str] = None) -> Optional[list[float]]:
    """Generate embedding via Google Gemini Embedding 2 API.
    Supports text and optional image (multimodal)."""
    api_key = _next_google_key()
    if not api_key:
        return None
    try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{EMBED_MODEL}:embedContent?key={api_key}"
        )
        parts: list[dict] = [{"text": text}]
        if image_path:
            img_path = Path(image_path)
            if img_path.exists():
                img_bytes = img_path.read_bytes()
                ext = img_path.suffix.lower()
                mime_map = {".png": "image/png", ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg", ".gif": "image/gif",
                            ".webp": "image/webp"}
                mime = mime_map.get(ext, "image/png")
                parts.append({
                    "inline_data": {
                        "mime_type": mime,
                        "data": base64.b64encode(img_bytes).decode("ascii"),
                    }
                })
        payload = json.dumps({"content": {"parts": parts}}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            embedding = data.get("embedding", {})
            values = embedding.get("values", [])
            if values:
                return values
    except Exception:
        pass
    return None


def _embed(text: str, image_path: Optional[str] = None) -> Optional[list[float]]:
    """Generate embedding vector using configured provider.
    Returns None on failure (search falls back to FTS5 keyword only).
    image_path is only supported with the 'google' provider."""
    if EMBED_PROVIDER == "google":
        return _embed_google(text, image_path=image_path)
    if EMBED_PROVIDER == "openai":
        return _embed_openai(text)
    return _embed_ollama(text)


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_vec(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0  # Different embedding dimensions — incomparable
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _recency_score(created_at: str) -> float:
    """Time decay score: more recent = higher (0-1). 30-day half-life."""
    try:
        t = datetime_strptime(created_at)
        days_ago = (now_local() - t).total_seconds() / 86400
        return math.exp(-days_ago / 30)
    except (ValueError, TypeError):
        return 0.5


def datetime_strptime(s: str):
    from datetime import datetime
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=LOCAL_TZ)


# ─── Core API ────────────────────────────────────────────

def remember(content: str, category: str = "general", source: str = "cc",
             tags: Optional[list[str]] = None, importance: int = 5,
             created_at: str = "") -> str:
    """Store a memory with automatic dedup and conflict detection.
    - Exact duplicate content → skip
    - Semantic similarity ≥ 0.92 → skip (nearly identical)
    - Semantic similarity 0.85~0.92 → supersede: old memory marked historical, new one stored
    - Semantic similarity < 0.85 → new memory, stored directly
    """
    db = _get_db()

    existing = db.execute(
        "SELECT id FROM memories WHERE content = ?", (content,)
    ).fetchone()
    if existing:
        db.close()
        return "Duplicate memory, skipped"

    # Generate embedding early (reused for semantic dedup + storage)
    vec = _embed(content)

    # Semantic dedup: check active memories in same category
    DUPLICATE_THRESHOLD = 0.92   # Nearly identical, skip
    SUPERSEDE_THRESHOLD = 0.85   # Similar but updated, supersede old
    supersede_ids = []

    if vec:
        cat_rows = db.execute(
            """SELECT m.id, m.content, v.embedding FROM memories m
               JOIN memory_vectors v ON m.id = v.memory_id
               WHERE m.category = ? AND m.superseded_by IS NULL""",
            (category,),
        ).fetchall()
        for r in cat_rows:
            existing_vec = _blob_to_vec(r["embedding"])
            sim = _cosine_similarity(vec, existing_vec)
            if sim >= DUPLICATE_THRESHOLD:
                db.close()
                return f"Semantically similar memory exists (ID {r['id']}, similarity {sim:.3f}). Use update_memory to update it."
            elif sim >= SUPERSEDE_THRESHOLD:
                supersede_ids.append((r["id"], r["content"][:40], sim))

    tags_json = json.dumps(tags or [], ensure_ascii=False)
    ts = created_at if created_at else now_str()

    cursor = db.execute(
        """INSERT INTO memories (content, category, source, tags, importance, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (content, category, source, tags_json, importance, ts),
    )
    memory_id = cursor.lastrowid

    if vec:
        db.execute(
            "INSERT INTO memory_vectors (memory_id, embedding, model) VALUES (?, ?, ?)",
            (memory_id, _vec_to_blob(vec), EMBED_MODEL),
        )

    # Mark old memories as historical (not deleted, just superseded)
    supersede_notes = []
    for old_id, old_preview, sim in supersede_ids:
        db.execute(
            "UPDATE memories SET superseded_by = ?, updated_at = ? WHERE id = ?",
            (memory_id, now, old_id),
        )
        supersede_notes.append(f"  ↳ Superseded #{old_id} ({old_preview}… sim {sim:.3f})")

    db.commit()
    db.close()
    _rebuild_index()

    result = f"Remembered [{category}]: {content[:50]}..."
    if supersede_notes:
        result += "\n" + "\n".join(supersede_notes)
    return result


def forget(keyword: str) -> str:
    """Delete memories containing keyword."""
    db = _get_db()
    rows = db.execute(
        "SELECT id, content FROM memories WHERE content LIKE ?",
        (f"%{keyword}%",),
    ).fetchall()

    if not rows:
        db.close()
        return f"No memories found containing '{keyword}'"

    for row in rows:
        db.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (row["id"],))
        db.execute("DELETE FROM memories WHERE id = ?", (row["id"],))

    db.commit()
    db.close()
    _rebuild_index()
    return f"Deleted {len(rows)} memories containing '{keyword}'"


def delete_memory(memory_id: int) -> dict:
    """Delete a single memory by ID."""
    db = _get_db()
    row = db.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        db.close()
        return {"ok": False, "error": f"Memory {memory_id} not found"}

    db.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (memory_id,))
    db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    db.commit()
    db.close()
    _rebuild_index()
    return {"ok": True}


def update_memory(memory_id: int, content: str = "", category: str = "", importance: int = 0) -> dict:
    """Update a single memory by ID. Only non-empty/non-zero fields are changed."""
    db = _get_db()
    row = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        db.close()
        return {"ok": False, "error": f"Memory {memory_id} not found"}

    new_content = content.strip() if content.strip() else row["content"]
    new_category = category.strip() if category.strip() else row["category"]
    new_importance = importance if importance > 0 else row["importance"]

    db.execute(
        "UPDATE memories SET content = ?, category = ?, importance = ?, updated_at = ? WHERE id = ?",
        (new_content, new_category, new_importance, now_str(), memory_id),
    )
    # Only refresh embedding if content changed
    vec_refreshed = False
    if new_content != row["content"]:
        db.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (memory_id,))
        vec = _embed(new_content)
        if vec:
            db.execute(
                "INSERT INTO memory_vectors (memory_id, embedding, model) VALUES (?, ?, ?)",
                (memory_id, _vec_to_blob(vec), EMBED_MODEL),
            )
            vec_refreshed = True

    db.commit()
    db.close()
    _rebuild_index()
    return {"ok": True, "embedding_refreshed": vec_refreshed}


def search(query: str, limit: int = 10, category: Optional[str] = None) -> list[dict]:
    """Hybrid search: vector semantic + FTS5 keyword + time decay."""
    db = _get_db()
    results = {}

    # 1. FTS5 keyword search
    try:
        fts_query = segment_cjk(sanitize_fts_query(query))
        if not fts_query:
            fts_query = query.replace('"', '""')
        cat_filter = "AND m.category = ?" if category else ""
        params = [fts_query, category] if category else [fts_query]
        fts_rows = db.execute(f"""
            SELECT m.id, m.content, m.category, m.source, m.importance,
                   m.created_at, m.recalled_count, rank
            FROM memories_fts f
            JOIN memories m ON f.rowid = m.id
            WHERE memories_fts MATCH ? AND m.superseded_by IS NULL {cat_filter}
            ORDER BY rank LIMIT {limit * 2}
        """, params).fetchall()

        if fts_rows:
            max_rank = max(abs(r["rank"]) for r in fts_rows) or 1
            for r in fts_rows:
                mid = r["id"]
                fts_score = abs(r["rank"]) / max_rank
                results[mid] = {
                    "id": mid, "content": r["content"], "category": r["category"],
                    "source": r["source"], "importance": r["importance"],
                    "created_at": r["created_at"], "recalled_count": r["recalled_count"],
                    "fts_score": fts_score, "vec_score": 0.0,
                }
    except Exception:
        pass

    # 2. Vector semantic search
    query_vec = _embed(query)
    if query_vec:
        cat_filter = "AND m.category = ?" if category else ""
        params = [category] if category else []
        vec_rows = db.execute(f"""
            SELECT m.id, m.content, m.category, m.source, m.importance,
                   m.created_at, m.recalled_count, v.embedding
            FROM memories m
            JOIN memory_vectors v ON m.id = v.memory_id
            WHERE m.superseded_by IS NULL {cat_filter}
        """, params).fetchall()

        scored = []
        for r in vec_rows:
            mem_vec = _blob_to_vec(r["embedding"])
            sim = _cosine_similarity(query_vec, mem_vec)
            scored.append((r, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        for r, sim in scored[:limit * 2]:
            mid = r["id"]
            if mid in results:
                results[mid]["vec_score"] = sim
            else:
                results[mid] = {
                    "id": mid, "content": r["content"], "category": r["category"],
                    "source": r["source"], "importance": r["importance"],
                    "created_at": r["created_at"], "recalled_count": r["recalled_count"],
                    "fts_score": 0.0, "vec_score": sim,
                }

    # 3. Combined scoring
    for mid, info in results.items():
        recency = _recency_score(info["created_at"])
        # recalled_count as tiny tiebreaker (max 0.05, prevents snowball)
        recall_bonus = min(0.05, 0.01 * math.log1p(info.get("recalled_count", 0)))
        info["final_score"] = (
            WEIGHT_VECTOR * info["vec_score"]
            + WEIGHT_FTS * info["fts_score"]
            + WEIGHT_RECENCY * recency
            + recall_bonus
        )

    MIN_SCORE = 0.40
    ranked = [r for r in results.values() if r["final_score"] >= MIN_SCORE]
    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    ranked = ranked[:limit]

    for r in ranked:
        if "id" in r:
            db.execute(
                "UPDATE memories SET recalled_count = recalled_count + 1 WHERE id = ?",
                (r["id"],),
            )
    db.commit()
    db.close()

    bank_results = _search_bank(query_vec, query, limit=5)
    ranked.extend(bank_results)
    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    return ranked[:limit]


def search_text(query: str, limit: int = 10) -> str:
    """Search and return formatted text. Adds staleness warning for old memories."""
    results = search(query, limit)
    if not results:
        return "No matching memories found"
    lines = []
    now = now_local()
    for r in results:
        score = f"{r['final_score']:.2f}"
        created = r.get('created_at', '')
        line = f"[{r['category']}|{r['source']}|{created}] (relevance:{score}) {r['content'][:200]}"
        # Staleness warning for old memories
        if created:
            try:
                from datetime import datetime
                created_dt = datetime.strptime(created[:10], "%Y-%m-%d")
                days_old = (now.replace(tzinfo=None) - created_dt).days
                if days_old > 14:
                    line += f"\n  ⚠ {days_old}天前的记忆，涉及代码/配置/状态请先验证再使用"
            except (ValueError, TypeError):
                pass
        lines.append(line)
    return "\n".join(lines)


def get_all(category: Optional[str] = None, limit: int = 50, after: Optional[str] = None, before: Optional[str] = None) -> list[dict]:
    """Get all active memories (by time desc). Excludes superseded memories.
    after: ISO date string, only memories created on or after this date (e.g. '2026-04-01').
    before: ISO date string, only memories created on or before this date."""
    db = _get_db()
    filters = []
    params: list = []
    if category:
        filters.append("AND category = ?")
        params.append(category)
    if after:
        filters.append("AND created_at >= ?")
        params.append(after)
    if before:
        filters.append("AND created_at <= ?")
        params.append(before)
    filter_sql = " ".join(filters)
    rows = db.execute(
        f"SELECT * FROM memories WHERE superseded_by IS NULL {filter_sql} ORDER BY created_at DESC LIMIT ?",
        (*params, limit),
    ).fetchall()
    db.close()
    return [dict(r) for r in rows]


# ─── Daily Log ───────────────────────────────────────────

def daily_log(text: str) -> str:
    """Append to today's daily log."""
    today = now_local().strftime("%Y-%m-%d")
    DAILY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = DAILY_LOG_DIR / f"{today}.md"

    now_time = now_local().strftime("%H:%M")
    entry = f"- [{now_time}] {text}\n"

    needs_header = not log_file.exists() or log_file.stat().st_size == 0
    with open(log_file, "a", encoding="utf-8") as f:
        if needs_header:
            f.write(f"# {today} Log\n\n")
        f.write(entry)

    db = _get_db()
    existing = db.execute("SELECT content FROM daily_logs WHERE date = ?", (today,)).fetchone()
    if existing:
        new_content = existing["content"] + entry
        db.execute("UPDATE daily_logs SET content = ? WHERE date = ?", (new_content, today))
    else:
        db.execute("INSERT INTO daily_logs (date, content) VALUES (?, ?)", (today, entry))
    db.commit()
    db.close()

    return f"Logged to {today}"


# ─── Notification Dedup ──────────────────────────────────

def was_notified(content_key: str, hours: int = 24) -> bool:
    """Check if already notified in the past N hours."""
    db = _get_db()
    cutoff = (now_local() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M")
    row = db.execute(
        "SELECT 1 FROM notifications WHERE content LIKE ? AND created_at > ? LIMIT 1",
        (f"%{content_key}%", cutoff),
    ).fetchone()
    db.close()
    return row is not None


def record_notification(content: str):
    """Record a sent notification."""
    db = _get_db()
    db.execute(
        "INSERT INTO notifications (content, created_at) VALUES (?, ?)",
        (content, now_str()),
    )
    db.commit()
    db.close()


# ─── Memory Health Tools ────────────────────────────────

def find_duplicates(threshold: float = 0.85) -> list[dict]:
    """Find memory pairs with cosine similarity above threshold. Read-only."""
    db = _get_db()
    rows = db.execute("""
        SELECT m.id, m.content, m.category, v.embedding
        FROM memories m
        JOIN memory_vectors v ON m.id = v.memory_id
    """).fetchall()
    db.close()

    pairs = []
    for i in range(len(rows)):
        vec_i = _blob_to_vec(rows[i]["embedding"])
        for j in range(i + 1, len(rows)):
            vec_j = _blob_to_vec(rows[j]["embedding"])
            sim = _cosine_similarity(vec_i, vec_j)
            if sim >= threshold:
                pairs.append({
                    "id_a": rows[i]["id"],
                    "content_a": rows[i]["content"][:100],
                    "category_a": rows[i]["category"],
                    "id_b": rows[j]["id"],
                    "content_b": rows[j]["content"][:100],
                    "category_b": rows[j]["category"],
                    "similarity": round(sim, 4),
                })
    pairs.sort(key=lambda x: x["similarity"], reverse=True)
    return pairs


def reindex_embeddings() -> str:
    """Rebuild all memory + bank embeddings using the current provider.
    Useful after switching embedding providers (e.g., ollama → google)."""
    db = _get_db()

    # Reindex memories
    rows = db.execute("SELECT id, content FROM memories").fetchall()
    mem_total = len(rows)
    mem_ok = 0
    mem_fail = 0
    for r in rows:
        vec = _embed(r["content"])
        db.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (r["id"],))
        if vec:
            db.execute(
                "INSERT INTO memory_vectors (memory_id, embedding, model) VALUES (?, ?, ?)",
                (r["id"], _vec_to_blob(vec), EMBED_MODEL),
            )
            mem_ok += 1
        else:
            mem_fail += 1

    # Reindex bank chunks
    bank_rows = db.execute("SELECT id, chunk_text FROM bank_chunks").fetchall()
    bank_total = len(bank_rows)
    bank_ok = 0
    for br in bank_rows:
        vec = _embed(br["chunk_text"])
        blob = _vec_to_blob(vec) if vec else None
        db.execute("UPDATE bank_chunks SET embedding = ? WHERE id = ?", (blob, br["id"]))
        if vec:
            bank_ok += 1

    db.commit()
    db.close()
    return (
        f"Reindexed memories: {mem_ok}/{mem_total} ({mem_fail} failed), "
        f"bank chunks: {bank_ok}/{bank_total}. "
        f"Provider: {EMBED_PROVIDER}, model: {EMBED_MODEL}"
    )


def find_stale(days: int = 14) -> list[dict]:
    """Find potentially stale memories: older than N days, importance < 7, recalled < 3. Read-only."""
    db = _get_db()
    cutoff = (now_local() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
    rows = db.execute("""
        SELECT id, content, category, importance, recalled_count, created_at
        FROM memories
        WHERE created_at < ? AND importance < 7 AND recalled_count < 3
            AND superseded_by IS NULL
        ORDER BY created_at ASC
    """, (cutoff,)).fetchall()
    db.close()
    return [dict(r) for r in rows]


def decay(days: int = 30, dry_run: bool = True) -> dict:
    """Decay importance of inactive memories. Memories older than `days` with
    recalled_count < 2 get importance decremented by 1 (minimum 0).
    Memories that reach importance=0 are marked as archived (superseded_by=-1).
    Returns summary of what was (or would be) changed.

    dry_run=True: preview only (default). dry_run=False: apply changes."""
    db = _get_db()
    cutoff = (now_local() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
    now = now_str()

    # Find candidates: old, rarely recalled, not already superseded/archived
    rows = db.execute("""
        SELECT id, content, category, importance, recalled_count, created_at
        FROM memories
        WHERE COALESCE(updated_at, created_at) < ? AND recalled_count < 2
            AND superseded_by IS NULL AND importance > 0
        ORDER BY importance ASC, created_at ASC
    """, (cutoff,)).fetchall()

    decayed = []
    archived = []
    for r in rows:
        new_imp = r["importance"] - 1
        entry = {"id": r["id"], "category": r["category"],
                 "content": r["content"][:100],
                 "importance": f"{r['importance']} → {new_imp}"}
        if new_imp <= 0:
            archived.append(entry)
            if not dry_run:
                db.execute(
                    "UPDATE memories SET importance = 0, superseded_by = -1, updated_at = ? WHERE id = ?",
                    (now, r["id"]),
                )
        else:
            decayed.append(entry)
            if not dry_run:
                db.execute(
                    "UPDATE memories SET importance = ?, updated_at = ? WHERE id = ?",
                    (new_imp, now, r["id"]),
                )

    if not dry_run:
        db.commit()
    db.close()

    if not dry_run:
        _rebuild_index()

    return {
        "dry_run": dry_run,
        "decayed": len(decayed),
        "archived": len(archived),
        "details_decayed": decayed[:20],
        "details_archived": archived[:20],
    }


# ─── Memory Context ──────────────────────────────────────

def get_context(query: Optional[str] = None, max_chars: int = 3000) -> str:
    """Generate memory context summary."""
    if query:
        return search_text(query, limit=10)

    db = _get_db()
    rows = db.execute("""
        SELECT content, category, source, created_at, importance
        FROM memories
        ORDER BY
            CASE WHEN importance >= 7 THEN 0 ELSE 1 END,
            created_at DESC
        LIMIT 20
    """).fetchall()
    db.close()

    if not rows:
        return "(No memories yet)"

    lines = ["# Memory Summary\n"]
    total = 0
    for r in rows:
        line = f"- [{r['category']}] {r['content']}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)

    return "\n".join(lines)


# ─── Bank File Index ─────────────────────────────────────

def _clean_bank_chunk(chunk: str) -> Optional[str]:
    """Remove template comments from a bank chunk."""
    cleaned_lines = []
    substantive_lines = []
    in_comment = False

    for line in chunk.split("\n"):
        stripped = line.strip()
        if in_comment:
            if "-->" in stripped:
                in_comment = False
            continue
        if stripped.startswith("<!--"):
            if "-->" not in stripped:
                in_comment = True
            continue

        cleaned_lines.append(line.rstrip())
        if stripped and not stripped.startswith("#"):
            substantive_lines.append(stripped)

    cleaned = "\n".join(cleaned_lines).strip()
    if not cleaned or not substantive_lines:
        return None
    return cleaned


_BANK_EXCLUDE = {"north-todos.md", "backlog.md"}

def _index_bank_files():
    """Index markdown files in bank/ directory. Skip unchanged files."""
    if not BANK_DIR.exists():
        return
    db = _get_db()

    valid_paths = {
        str(p.resolve())
        for p in BANK_DIR.glob("*.md")
        if p.name not in _BANK_EXCLUDE
    }
    indexed = db.execute("SELECT DISTINCT file_path FROM bank_chunks").fetchall()
    for row in indexed:
        if row["file_path"] not in valid_paths:
            db.execute("DELETE FROM bank_chunks WHERE file_path = ?", (row["file_path"],))

    for md_file in BANK_DIR.glob("*.md"):
        if md_file.name in _BANK_EXCLUDE:
            continue
        md_file = md_file.resolve()
        mtime = md_file.stat().st_mtime
        existing = db.execute(
            "SELECT file_mtime, index_version FROM bank_chunks WHERE file_path = ? LIMIT 1",
            (str(md_file),),
        ).fetchone()
        if (
            existing
            and abs(existing["file_mtime"] - mtime) < 1
            and existing["index_version"] == BANK_INDEX_VERSION
        ):
            continue

        db.execute("DELETE FROM bank_chunks WHERE file_path = ?", (str(md_file),))

        text = md_file.read_text(encoding="utf-8")
        chunks = _split_into_chunks(text)

        for chunk in chunks:
            cleaned_chunk = _clean_bank_chunk(chunk)
            if not cleaned_chunk or len(cleaned_chunk) < 10:
                continue
            vec = _embed(cleaned_chunk)
            blob = _vec_to_blob(vec) if vec else None
            db.execute(
                """INSERT INTO bank_chunks
                   (file_path, chunk_text, embedding, file_mtime, index_version)
                   VALUES (?, ?, ?, ?, ?)""",
                (str(md_file), cleaned_chunk, blob, mtime, BANK_INDEX_VERSION),
            )
    db.commit()
    db.close()


def _split_into_chunks(text: str) -> list[str]:
    """Split by markdown ## headings."""
    chunks = []
    current = []
    for line in text.split("\n"):
        if line.startswith("## ") and current:
            chunks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _search_bank(query_vec, query_text: str, limit: int = 5) -> list[dict]:
    """Search bank/ file chunks."""
    _index_bank_files()
    db = _get_db()
    results = []

    if query_vec:
        rows = db.execute(
            "SELECT chunk_text, file_path, embedding FROM bank_chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        for r in rows:
            vec = _blob_to_vec(r["embedding"])
            sim = _cosine_similarity(query_vec, vec)
            if sim > 0.3:
                results.append({
                    "content": r["chunk_text"],
                    "source": Path(r["file_path"]).stem,
                    "category": "bank",
                    "final_score": sim,
                })

    # Keyword search — score no longer hardcoded, merges with vector results
    KEYWORD_BASE = 0.5
    KEYWORD_BONUS = 0.15
    DUAL_HIT_BONUS = 0.1
    query_lower = query_text.lower()
    rows = db.execute("SELECT chunk_text, file_path FROM bank_chunks").fetchall()
    for r in rows:
        if query_lower in r["chunk_text"].lower():
            kw_score = KEYWORD_BASE + KEYWORD_BONUS  # 0.65
            existing = next((x for x in results if x["content"] == r["chunk_text"]), None)
            if existing:
                existing["final_score"] = max(existing["final_score"], kw_score) + DUAL_HIT_BONUS
            else:
                results.append({
                    "content": r["chunk_text"],
                    "source": Path(r["file_path"]).stem,
                    "category": "bank",
                    "final_score": kw_score,
                })

    db.close()
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:limit]


# ─── MEMORY.md Index Rebuild ─────────────────────────────

def _summarize_for_index(content, max_len=50):
    """Truncate memory content to a short index pointer."""
    text = content.strip()
    for sep in ("：", "——", "—", "。", "，", "；"):
        idx = text.find(sep)
        if 0 < idx <= max_len:
            return text[:idx]
    for sep in (":", ", "):
        idx = text.find(sep)
        if 10 < idx <= max_len:
            return text[:idx]
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


def _rebuild_index():
    """Rebuild MEMORY.md as a lightweight index (date + keyword per line).
    Full content is available via memory_search."""
    db = _get_db()
    lines = ["# Memory Index\n", f"*Last updated: {now_str()}*\n"]

    total = db.execute("SELECT COUNT(*) as c FROM memories WHERE superseded_by IS NULL").fetchone()["c"]
    lines.append(f"*{total} memories — use memory_search for details*\n")

    categories = db.execute(
        "SELECT DISTINCT category FROM memories WHERE superseded_by IS NULL ORDER BY category"
    ).fetchall()

    for cat_row in categories:
        cat = cat_row["category"]
        rows = db.execute(
            """SELECT content, source, created_at, importance
               FROM memories WHERE category = ? AND superseded_by IS NULL
               ORDER BY importance DESC, created_at DESC""",
            (cat,),
        ).fetchall()
        if not rows:
            continue

        section = [f"\n## {cat}"]
        for r in rows:
            date = r["created_at"][:10] if r["created_at"] else ""
            short_date = date[5:].replace("-", "/") if date else ""
            summary = _summarize_for_index(r["content"])
            section.append(f"- [{short_date}] {summary}")

        lines.extend(section)

    db.close()

    with open(MEMORY_INDEX, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ═══════════════════════════════════════════════════════════════
# RRF Unified Retrieval — fusion across memory, bank, conversation
# ═══════════════════════════════════════════════════════════════

RRF_K = 60              # RRF ranking constant (standard value)
VEC_PRE_FILTER = 0.3    # Vector similarity pre-filter threshold
MIN_FINAL_SCORE = 0.003 # Pre-normalization floor (RRF-level)
POST_NORM_FLOOR = 0.15  # Post-normalization floor (after pool-confidence scaling)
RERANK_BLEND = 0.3      # How much rerank factors affect final score
LIKE_LIMIT = 50         # Max results from LIKE exact-match channel per pool
VEC_CONFIDENCE_NOISE = 0.40   # vec_sim below this → pool is noise
VEC_CONFIDENCE_GOOD = 0.55    # vec_sim above this → pool is clearly relevant


def _days_since(time_str: str, default: float = 30.0) -> float:
    """Days elapsed since a timestamp string."""
    if not time_str:
        return default
    try:
        t = datetime.strptime(time_str[:16], "%Y-%m-%d %H:%M").replace(tzinfo=LOCAL_TZ)
        return max(0.0, (now_local() - t).total_seconds() / 86400)
    except (ValueError, TypeError):
        return default


def _fts_query_cjk(query: str) -> str:
    """Build an FTS5 MATCH expression with proper CJK tokenization."""
    if not _CJK_RE.search(query):
        return query

    if _JIEBA_OK:
        return segment_cjk(query)

    parts = re.split(r'([\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df'
                     r'\U0002a700-\U0002ebef\u3000-\u303f\uff00-\uffef]+)', query)
    tokens = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if _CJK_RE.search(part):
            chars = [c for c in part if _CJK_RE.match(c)]
            if len(chars) >= 2:
                tokens.append('"' + ' '.join(chars) + '"')
            elif len(chars) == 1:
                tokens.append(chars[0])
        else:
            tokens.append(part)
    return ' '.join(tokens)


def _sanitize_fts(query: str) -> str:
    """Strip FTS5 operators and apply CJK segmentation."""
    cleaned = re.sub(r'["\(\)\*\:\^\{\}]', " ", query)
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        return cleaned
    return _fts_query_cjk(cleaned)


# ─── RRF Core ───────────────────────────────────────────

def _rrf_fuse(channel_rankings: list[list[tuple[str, int]]]) -> dict[str, float]:
    """Reciprocal Rank Fusion over N ranked lists."""
    scores: dict[str, float] = {}
    for ranking in channel_rankings:
        for key, rank in ranking:
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
    return scores


_RANK_BASELINE = 10

def _inject_default_ranks(
    fts_ranking: list[tuple[str, int]],
    vec_ranking: list[tuple[str, int]],
) -> None:
    """Give absent paired channel a default low rank so single-channel
    results aren't unfairly penalised. Mirrors rankings when one channel
    is completely empty (e.g. FTS can't tokenize the query)."""
    if not fts_ranking and not vec_ranking:
        return
    if not fts_ranking and vec_ranking:
        fts_ranking.extend(vec_ranking)
        return
    if not vec_ranking and fts_ranking:
        vec_ranking.extend(fts_ranking)
        return

    fts_keys = {k for k, _ in fts_ranking}
    vec_keys = {k for k, _ in vec_ranking}

    default_fts = max(len(fts_ranking), _RANK_BASELINE) + 1
    default_vec = max(len(vec_ranking), _RANK_BASELINE) + 1

    for k in vec_keys - fts_keys:
        fts_ranking.append((k, default_fts))
    for k in fts_keys - vec_keys:
        vec_ranking.append((k, default_vec))


# ─── Rerank Functions ───────────────────────────────────

def _rerank_memory(rrf_score: float, row: dict) -> float:
    """Memory rerank: time x activation x importance, blended with RRF."""
    if row.get("pinned"):
        return rrf_score

    importance = max(row.get("importance", 5), 1)
    recalled = row.get("recalled_count", 0)

    ref = row.get("last_accessed_at") or row.get("created_at", "")
    days = _days_since(ref, default=30)
    lam = 0.05 / (importance / 5)
    time_factor = 0.4 + 0.6 * math.exp(-lam * days)

    activation_factor = 0.8 + 0.2 * (math.log(recalled + 1) / math.log(51))
    importance_factor = 0.7 + 0.3 * (importance / 10)

    factor = time_factor * activation_factor * importance_factor
    return rrf_score * (1 - RERANK_BLEND + RERANK_BLEND * factor)


def _rerank_bank(rrf_score: float, row: dict) -> float:
    """Bank rerank: gentle file freshness (tiebreaker only)."""
    mtime = row.get("file_mtime")
    if mtime is not None:
        try:
            dt = datetime.fromtimestamp(float(mtime), tz=LOCAL_TZ)
            days = max(0.0, (now_local() - dt).total_seconds() / 86400)
        except (ValueError, TypeError, OSError):
            days = 7.0
    else:
        days = 7.0
    freshness = 0.90 + 0.10 * math.exp(-days / 90)
    return rrf_score * (1 - RERANK_BLEND + RERANK_BLEND * freshness)


def _rerank_conv(rrf_score: float, row: dict) -> float:
    """Conversation rerank: recency (7-day half-life)."""
    days = _days_since(row.get("created_at", ""), default=30)
    recency = 0.3 + 0.7 * math.exp(-days / 7)
    return rrf_score * (1 - RERANK_BLEND + RERANK_BLEND * recency)


# ─── Per-Pool Channel Search ────────────────────────────

def _search_memory_channels(query, query_vec, db, *, category=None, limit=50):
    """Return (fts_ranking, vec_ranking, like_ranking, details) for memory pool."""
    details = {}
    fts_ranking = []
    vec_ranking = []

    safe_q = _sanitize_fts(query)
    if safe_q:
        try:
            cat_sql = "AND m.category = ?" if category else ""
            params = [safe_q] + ([category] if category else []) + [limit]
            fts_rows = db.execute(
                f"""SELECT m.id, m.content, m.category, m.source, m.importance,
                           m.created_at, m.recalled_count,
                           m.last_accessed_at, m.pinned
                    FROM memories_fts f
                    JOIN memories m ON f.rowid = m.id
                    WHERE memories_fts MATCH ? AND m.superseded_by IS NULL {cat_sql}
                    ORDER BY f.rank
                    LIMIT ?""",
                params,
            ).fetchall()
            for idx, r in enumerate(fts_rows):
                key = f"mem_{r['id']}"
                fts_ranking.append((key, idx + 1))
                details[key] = dict(r)
        except Exception:
            pass

    if query_vec:
        cat_sql = "AND m.category = ?" if category else ""
        params = [category] if category else []
        vec_rows = db.execute(
            f"""SELECT m.id, m.content, m.category, m.source, m.importance,
                       m.created_at, m.recalled_count,
                       m.last_accessed_at, m.pinned,
                       v.embedding
                FROM memories m
                JOIN memory_vectors v ON m.id = v.memory_id
                WHERE m.superseded_by IS NULL {cat_sql}""",
            params,
        ).fetchall()

        scored = []
        for r in vec_rows:
            sim = _cosine_similarity(query_vec, _blob_to_vec(r["embedding"]))
            if sim >= VEC_PRE_FILTER:
                scored.append((r, sim))
        scored.sort(key=lambda x: x[1], reverse=True)

        for idx, (r, sim) in enumerate(scored[:limit]):
            key = f"mem_{r['id']}"
            vec_ranking.append((key, idx + 1))
            if key not in details:
                details[key] = dict(r)
            details[key]["vec_similarity"] = sim

    like_ranking = []
    q_lower = query.lower()
    if len(q_lower) >= 2:
        cat_sql = "AND category = ?" if category else ""
        params = [f"%{q_lower}%"] + ([category] if category else []) + [LIKE_LIMIT]
        like_rows = db.execute(
            f"""SELECT id, content, category, source, importance,
                       created_at, recalled_count,
                       last_accessed_at, pinned
                FROM memories
                WHERE LOWER(content) LIKE ? AND superseded_by IS NULL {cat_sql}
                ORDER BY created_at DESC
                LIMIT ?""",
            params,
        ).fetchall()
        for idx, r in enumerate(like_rows):
            key = f"mem_{r['id']}"
            like_ranking.append((key, idx + 1))
            if key not in details:
                details[key] = dict(r)

    return fts_ranking, vec_ranking, like_ranking, details


def _search_bank_channels(query, query_vec, db, *, limit=50):
    """Return (fts_ranking, vec_ranking, like_ranking, details) for bank pool."""
    _index_bank_files()
    details = {}
    fts_ranking = []
    vec_ranking = []

    q_lower = query.lower()
    kw_rows = db.execute(
        "SELECT id, chunk_text, file_path, file_mtime FROM bank_chunks"
    ).fetchall()
    matches = [r for r in kw_rows if q_lower in r["chunk_text"].lower()]
    for idx, r in enumerate(matches[:limit]):
        key = f"bank_{r['id']}"
        fts_ranking.append((key, idx + 1))
        details[key] = {
            "id": r["id"],
            "content": r["chunk_text"],
            "source": Path(r["file_path"]).stem,
            "file_path": r["file_path"],
            "file_mtime": r["file_mtime"],
            "category": "bank",
        }

    if query_vec:
        v_rows = db.execute(
            "SELECT id, chunk_text, file_path, file_mtime, embedding "
            "FROM bank_chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        scored = []
        for r in v_rows:
            sim = _cosine_similarity(query_vec, _blob_to_vec(r["embedding"]))
            if sim >= VEC_PRE_FILTER:
                scored.append((r, sim))
        scored.sort(key=lambda x: x[1], reverse=True)

        for idx, (r, sim) in enumerate(scored[:limit]):
            key = f"bank_{r['id']}"
            vec_ranking.append((key, idx + 1))
            if key not in details:
                details[key] = {
                    "id": r["id"],
                    "content": r["chunk_text"],
                    "source": Path(r["file_path"]).stem,
                    "file_path": r["file_path"],
                    "file_mtime": r["file_mtime"],
                    "category": "bank",
                }
            details[key]["vec_similarity"] = sim

    like_ranking = []
    return fts_ranking, vec_ranking, like_ranking, details


def _search_conv_channels(query, query_vec, db, *, platform="", limit=50):
    """Return (fts_ranking, vec_ranking, like_ranking, details) for conversation pool."""
    details = {}
    fts_ranking = []
    vec_ranking = []

    safe_q = _sanitize_fts(query)
    if safe_q:
        try:
            if platform:
                fts_rows = db.execute(
                    """SELECT c.id, c.platform, c.direction, c.speaker, c.content, c.created_at
                       FROM conversation_log_fts f
                       JOIN conversation_log c ON c.id = f.rowid
                       WHERE conversation_log_fts MATCH ? AND c.platform = ?
                       ORDER BY f.rank
                       LIMIT ?""",
                    (safe_q, platform, limit),
                ).fetchall()
            else:
                fts_rows = db.execute(
                    """SELECT c.id, c.platform, c.direction, c.speaker, c.content, c.created_at
                       FROM conversation_log_fts f
                       JOIN conversation_log c ON c.id = f.rowid
                       WHERE conversation_log_fts MATCH ?
                       ORDER BY f.rank
                       LIMIT ?""",
                    (safe_q, limit),
                ).fetchall()

            for idx, r in enumerate(fts_rows):
                key = f"conv_{r['id']}"
                fts_ranking.append((key, idx + 1))
                details[key] = dict(r)
        except Exception:
            pass

    like_ranking = []
    q_lower = query.lower()
    if len(q_lower) >= 2:
        if platform:
            like_rows = db.execute(
                """SELECT id, platform, direction, speaker, content, created_at
                   FROM conversation_log
                   WHERE LOWER(content) LIKE ? AND platform = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (f"%{q_lower}%", platform, LIKE_LIMIT),
            ).fetchall()
        else:
            like_rows = db.execute(
                """SELECT id, platform, direction, speaker, content, created_at
                   FROM conversation_log
                   WHERE LOWER(content) LIKE ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (f"%{q_lower}%", LIKE_LIMIT),
            ).fetchall()
        for idx, r in enumerate(like_rows):
            key = f"conv_{r['id']}"
            like_ranking.append((key, idx + 1))
            if key not in details:
                details[key] = dict(r)

    return fts_ranking, vec_ranking, like_ranking, details


# ─── Graph Expansion ───────────────────────────────────

def _expand_via_edges(results: list[dict], db, max_expand: int = 3) -> list[dict]:
    """Append edge-connected memories to search results."""
    existing_ids = {r["id"] for r in results if r.get("pool") == "memory"}
    expanded = []

    for r in results:
        if r.get("pool") != "memory" or len(expanded) >= max_expand:
            break
        rid = r.get("id")
        if not rid:
            continue

        try:
            edges = db.execute("""
                SELECT e.id, e.relation, e.context,
                       CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as neighbor_id
                FROM memory_edges e
                WHERE (e.source_id = ? OR e.target_id = ?)
            """, (rid, rid, rid)).fetchall()
        except Exception:
            continue

        for edge in edges:
            nid = edge["neighbor_id"]
            if nid in existing_ids or len(expanded) >= max_expand:
                continue
            neighbor = db.execute(
                "SELECT * FROM memories WHERE id = ? AND superseded_by IS NULL", (nid,)
            ).fetchone()
            if neighbor:
                existing_ids.add(nid)
                expanded.append({
                    "pool": "memory", "score": 0, "rrf_raw": 0,
                    "source": "edge",
                    "edge_relation": edge["relation"],
                    "edge_context": edge["context"],
                    **dict(neighbor),
                })
                db.execute(
                    """UPDATE memory_edges
                       SET surfaced_count = surfaced_count + 1,
                           strength = min(coalesce(strength, 1.0) + 0.1, 5.0),
                           last_surfaced_at = datetime('now'),
                           status = CASE WHEN status = 'dormant' THEN 'active' ELSE status END
                       WHERE id = ?""",
                    (edge["id"],),
                )

    if expanded:
        db.commit()

    return results + expanded


# ─── Unified Search ─────────────────────────────────────

def unified_search(
    query: str,
    limit: int = 10,
    pools: list[str] | None = None,
    category: str | None = None,
    platform: str = "",
    after: str | None = None,
    before: str | None = None,
    _internal: bool = False,
) -> list[dict]:
    """Search across all memory pools with RRF fusion and per-pool reranking.

    Args:
        query:    natural-language search query
        limit:    max results to return
        pools:    subset of ["memory", "bank", "conversation"]; None = all
        category: filter memory pool by category
        platform: filter conversation pool by platform
        _internal: skip side-effects (recalled_count, last_accessed_at) — for edge expansion

    Returns list of dicts sorted by final score, each containing:
        pool, score, rrf_raw, id, content, + pool-specific fields
    """
    if pools is None:
        pools = ["memory", "bank", "conversation"]

    if (after or before) and "bank" in pools:
        pools = [p for p in pools if p != "bank"]

    db = _get_db()
    query_vec = _embed(query)
    all_rankings: list[list[tuple[str, int]]] = []
    all_details: dict[str, dict] = {}

    if "memory" in pools:
        m_fts, m_vec, m_like, m_det = _search_memory_channels(
            query, query_vec, db, category=category
        )
        _inject_default_ranks(m_fts, m_vec)
        all_rankings += [m_fts, m_vec, m_like]
        all_details.update(m_det)

    if "bank" in pools:
        b_fts, b_vec, b_like, b_det = _search_bank_channels(query, query_vec, db)
        _inject_default_ranks(b_fts, b_vec)
        all_rankings += [b_fts, b_vec, b_like]
        all_details.update(b_det)

    if "conversation" in pools:
        c_fts, c_vec, c_like, c_det = _search_conv_channels(
            query, query_vec, db, platform=platform
        )
        if c_vec:
            _inject_default_ranks(c_fts, c_vec)
        all_rankings += [c_fts, c_vec, c_like]
        all_details.update(c_det)

    rrf_scores = _rrf_fuse(all_rankings)

    # Per-pool rerank + within-pool normalisation
    pool_items: dict[str, list[dict]] = {"memory": [], "bank": [], "conversation": []}

    for key, rrf in rrf_scores.items():
        detail = all_details.get(key, {})

        if key.startswith("mem_"):
            pool = "memory"
            reranked = _rerank_memory(rrf, detail)
        elif key.startswith("bank_"):
            pool = "bank"
            reranked = _rerank_bank(rrf, detail)
        elif key.startswith("conv_"):
            pool = "conversation"
            reranked = _rerank_conv(rrf, detail)
        else:
            continue

        if reranked < MIN_FINAL_SCORE:
            continue

        detail.pop("embedding", None)
        pool_items[pool].append({
            "pool": pool, "score": reranked, "rrf_raw": rrf, **detail
        })

    # Normalise within each pool, scaled by pool confidence.
    # Pool confidence uses best vec_similarity — the only signal with real relevance discrimination.
    results: list[dict] = []
    for pool, items in pool_items.items():
        if not items:
            continue
        max_score = max(r["score"] for r in items)
        max_vec = max((r.get("vec_similarity") or 0) for r in items)
        if max_vec >= VEC_CONFIDENCE_GOOD:
            pool_conf = 1.0
        elif max_vec <= VEC_CONFIDENCE_NOISE:
            pool_conf = 0.0
        else:
            pool_conf = (max_vec - VEC_CONFIDENCE_NOISE) / (VEC_CONFIDENCE_GOOD - VEC_CONFIDENCE_NOISE)
        for r in items:
            r["score"] = (r["score"] / max_score * pool_conf) if max_score > 0 else 0
        results.extend(items)

    # Keyword presence boost: results containing query terms get a bonus.
    # This prevents semantically vague matches from outranking exact keyword hits.
    _KEYWORD_BOOST = 0.3
    query_terms = query.split() if " " in query else [query]
    for r in results:
        content = r.get("content", "")
        matched = sum(1 for t in query_terms if t in content)
        if matched:
            r["score"] += _KEYWORD_BOOST * (matched / len(query_terms))

    results = [r for r in results if r["score"] >= POST_NORM_FLOOR]
    results.sort(key=lambda x: x["score"], reverse=True)

    # Time range filtering (after/before)
    if after or before:
        def _in_time_range(r):
            ts = r.get("created_at", "")
            if not ts:
                return True  # keep items without timestamp
            if after and ts < after:
                return False
            if before and ts > before:
                return False
            return True
        results = [r for r in results if _in_time_range(r)]

    results = results[:limit]

    # Graph expansion: append edge-connected memories
    if "memory" in pools:
        results = _expand_via_edges(results, db, max_expand=3)

    # Side-effect: update last_accessed_at + recalled_count
    if not _internal:
        mem_ids = [r["id"] for r in results if r.get("pool") == "memory"]
        if mem_ids:
            now = now_str()
            for mid in mem_ids:
                db.execute(
                    "UPDATE memories SET recalled_count = recalled_count + 1, "
                    "last_accessed_at = ? WHERE id = ?",
                    (now, mid),
                )
            db.commit()

    db.close()
    return results


_LOCALE_LABELS = {
    "en": {"memory": "Memory", "bank": "Bank", "conversation": "Conversation",
           "empty": "No matching results found"},
    "zh": {"memory": "记忆", "bank": "知识库", "conversation": "对话",
           "empty": "没有找到匹配的结果"},
}

def unified_search_text(
    query: str,
    limit: int = 10,
    pools: list[str] | None = None,
    platform: str = "",
    after: str | None = None,
    before: str | None = None,
) -> str:
    """Format unified search results as readable text.
    Set IMPRINT_LOCALE=zh for Chinese labels, default English.
    after/before: ISO date strings to filter by time range."""
    results = unified_search(query, limit=limit, pools=pools, platform=platform, after=after, before=before)
    locale = os.environ.get("IMPRINT_LOCALE", "en")
    loc = _LOCALE_LABELS.get(locale, _LOCALE_LABELS["en"])
    if not results:
        return loc["empty"]

    _labels = {k: loc[k] for k in ("memory", "bank", "conversation")}
    lines: list[str] = []

    for r in results:
        label = _labels.get(r["pool"], r["pool"])
        score = f"{r['score']:.4f}"
        content = r.get("content", "")[:200]

        if r["pool"] == "memory":
            cat = r.get("category", "")
            ts = r.get("created_at", "")
            pin = " [pinned]" if r.get("pinned") else ""
            if r.get("source") == "edge":
                rel = r.get("edge_relation", "")
                lines.append(f"[{label}|edge|{rel}] {content}")
            else:
                lines.append(f"[{label}|{cat}|{ts}]{pin} ({score}) {content}")

        elif r["pool"] == "bank":
            src = r.get("source", "")
            lines.append(f"[{label}|{src}] ({score}) {content}")

        elif r["pool"] == "conversation":
            plat = r.get("platform", "")
            dire = "<-" if r.get("direction") == "in" else "->"
            ts = r.get("created_at", "")
            lines.append(f"[{label}|{plat}{dire}|{ts}] ({score}) {content}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Pin / Tag / Edge operations
# ═══════════════════════════════════════════════════════════════

def pin_memory(memory_id: int) -> dict:
    """Pin a memory. Pinned memories bypass all time-decay in search."""
    db = _get_db()
    row = db.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        db.close()
        return {"ok": False, "error": f"Memory {memory_id} not found"}
    pinned_count = db.execute("SELECT COUNT(*) as c FROM memories WHERE pinned = 1").fetchone()["c"]
    db.execute("UPDATE memories SET pinned = 1, updated_at = ? WHERE id = ?", (now_str(), memory_id))
    db.commit()
    db.close()
    result = {"ok": True, "pinned": memory_id}
    if pinned_count >= 20:
        result["warning"] = f"Already {pinned_count} pinned memories — consider keeping under 20"
    return result


def unpin_memory(memory_id: int) -> dict:
    """Unpin a memory, restoring normal time-decay."""
    db = _get_db()
    row = db.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        db.close()
        return {"ok": False, "error": f"Memory {memory_id} not found"}
    db.execute("UPDATE memories SET pinned = 0, updated_at = ? WHERE id = ?", (now_str(), memory_id))
    db.commit()
    db.close()
    return {"ok": True, "unpinned": memory_id}


def add_tags(memory_id: int, tags: list[str]) -> dict:
    """Add tags to a memory (writes to memory_tags table and updates memories.tags JSON)."""
    db = _get_db()
    row = db.execute("SELECT id, tags FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        db.close()
        return {"ok": False, "error": f"Memory {memory_id} not found"}

    added = []
    for tag in tags:
        t = tag.strip()
        if t:
            try:
                db.execute("INSERT INTO memory_tags (memory_id, tag) VALUES (?, ?)", (memory_id, t))
                added.append(t)
            except sqlite3.IntegrityError:
                pass

    if added:
        existing_tags = json.loads(row["tags"] or "[]")
        merged = list(dict.fromkeys(existing_tags + added))
        db.execute("UPDATE memories SET tags = ? WHERE id = ?",
                   (json.dumps(merged, ensure_ascii=False), memory_id))

    db.commit()
    db.close()
    return {"ok": True, "memory_id": memory_id, "added": added}


def get_tags(memory_id: int) -> list[str]:
    """Get all tags for a memory."""
    db = _get_db()
    rows = db.execute("SELECT tag FROM memory_tags WHERE memory_id = ?", (memory_id,)).fetchall()
    db.close()
    return [r["tag"] for r in rows]


def add_edge(source_id: int, target_id: int, relation: str, context: str) -> dict:
    """Create a bidirectional edge between two memories."""
    if source_id == target_id:
        return {"ok": False, "error": "Cannot create edge to self"}

    db = _get_db()

    for mid in (source_id, target_id):
        row = db.execute("SELECT id FROM memories WHERE id = ?", (mid,)).fetchone()
        if not row:
            db.close()
            return {"ok": False, "error": f"Memory {mid} not found"}

    existing = db.execute("""
        SELECT id FROM memory_edges
        WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)
    """, (source_id, target_id, target_id, source_id)).fetchone()
    if existing:
        db.close()
        return {"ok": False, "error": f"Edge already exists (edge #{existing['id']})"}

    cursor = db.execute("""
        INSERT INTO memory_edges (source_id, target_id, relation, context, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (source_id, target_id, relation.strip(), context.strip(), now_str()))
    db.commit()
    db.close()
    return {"ok": True, "edge_id": cursor.lastrowid}


def get_edges(memory_id: int) -> list[dict]:
    """Get all edges for a memory, including neighbor previews."""
    db = _get_db()
    rows = db.execute("""
        SELECT e.id, e.source_id, e.target_id, e.relation, e.context,
               e.surfaced_count, e.used_count, e.created_at,
               CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as neighbor_id
        FROM memory_edges e
        WHERE e.source_id = ? OR e.target_id = ?
    """, (memory_id, memory_id, memory_id)).fetchall()

    edges = []
    for r in rows:
        neighbor = db.execute(
            "SELECT content, category FROM memories WHERE id = ?", (r["neighbor_id"],)
        ).fetchone()
        edges.append({
            "edge_id": r["id"],
            "source_id": r["source_id"],
            "target_id": r["target_id"],
            "neighbor_id": r["neighbor_id"],
            "neighbor_preview": neighbor["content"][:80] if neighbor else "(deleted)",
            "neighbor_category": neighbor["category"] if neighbor else "",
            "relation": r["relation"],
            "context": r["context"],
            "surfaced_count": r["surfaced_count"],
            "used_count": r["used_count"],
            "created_at": r["created_at"],
        })
    db.close()
    return edges


