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
    DATA_DIR, DAILY_LOG_DIR, BANK_DIR, MEMORY_INDEX, LOCAL_TZ,
    segment_cjk, sanitize_fts_query,
    _CJK_RE, _JIEBA_OK,
)

# ─── Speaker Config ──────────────────────────────────────
USER_NAME = os.environ.get("IMPRINT_USER_NAME", "User")
AGENT_NAME = os.environ.get("IMPRINT_AGENT_NAME", "Assistant")

# ─── Embedding Config ────────────────────────────────────
# Default is "google" (gemini-embedding-2, 3072-dim, multimodal text+image).
# Reason: the embed vector dimension MUST stay consistent across the lifetime
# of a single DB — mixing 1024-dim (ollama bge-m3) and 3072-dim (google)
# vectors makes cosine similarity return 0 across the boundary and the vec
# search channel goes silently dark. Google's free tier covers normal use
# without enabling billing; if you'd rather run fully offline, set
# EMBED_PROVIDER=ollama BEFORE first write — and never switch later.
EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "google").lower()  # "ollama", "openai", "google", "cloudflare"

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
    "cloudflare": "@cf/baai/bge-large-en-v1.5",
}
EMBED_MODEL = os.environ.get("EMBED_MODEL", _DEFAULT_MODELS.get(EMBED_PROVIDER, "gemini-embedding-2"))

BANK_INDEX_VERSION = 2

# Hybrid search weights
WEIGHT_VECTOR = 0.5
WEIGHT_FTS = 0.5
WEIGHT_RECENCY = 0.0  # time decay disabled — old hits rank by relevance, not age

# Stopwords config
STOPWORD_THRESHOLD = float(os.environ.get("STOPWORD_THRESHOLD", "0.15"))
STOPWORD_SKIP_PLATFORMS = {
    p.strip()
    for p in os.environ.get("IMPRINT_STOPWORD_SKIP_PLATFORMS", "cc").split(",")
    if p.strip()
}
_stopword_cache: set[str] | None = None
_stopword_cache_ts: float = 0

# Day boundary: hour at which the "day" rolls over for time-range search.
# 0 (default) = midnight. Set to e.g. 9 if the user's day starts at 9am
# (late-sleeper case: 5am-2pm sleep schedule treats early-morning hours as
# the previous day). Affects both _ts_to_day() and jionlp range expansion.
_DAY_BOUNDARY_HOUR = int(os.environ.get("IMPRINT_DAY_BOUNDARY_HOUR", "0"))


def _ts_to_day(ts: str) -> str:
    """Map any stored timestamp to a YYYY-MM-DD day string.

    Handles both 'YYYY-MM-DD HH:MM:SS' (memories/chunks) and ISO 8601 with
    timezone (conversation_log). When _DAY_BOUNDARY_HOUR > 0, shifts back
    by that many hours so early-morning timestamps map to the previous day.
    Returns '' for unparseable input — caller should treat as "unknown date".
    """
    if not ts:
        return ""
    try:
        s = ts.replace("T", " ", 1)
        dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
        if _DAY_BOUNDARY_HOUR:
            dt = dt - timedelta(hours=_DAY_BOUNDARY_HOUR)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ts[:10] if len(ts) >= 10 else ""


# ─── Stopwords ──────────────────────────────────────────

def build_stopwords(threshold: float | None = None) -> dict:
    """Scan all pools, compute document frequency, update stopwords table.
    Words appearing in >threshold fraction of documents are auto-stopped."""
    global _stopword_cache, _stopword_cache_ts
    if threshold is None:
        threshold = STOPWORD_THRESHOLD

    db = _get_db()
    try:
        from collections import Counter
        doc_freq: dict[str, Counter] = {}

        if STOPWORD_SKIP_PLATFORMS:
            skip_list = ",".join("?" for _ in STOPWORD_SKIP_PLATFORMS)
            conversation_sql = (
                "SELECT content FROM conversation_log "
                f"WHERE platform NOT IN ({skip_list}) AND content IS NOT NULL"
            )
            conversation_params = list(STOPWORD_SKIP_PLATFORMS)
        else:
            conversation_sql = "SELECT content FROM conversation_log WHERE content IS NOT NULL"
            conversation_params = []

        existing_tables = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}

        pools = [
            ("conversation_log", conversation_sql, conversation_params),
            ("memories", "SELECT content FROM memories WHERE content IS NOT NULL", []),
        ]
        if "conversation_chunks" in existing_tables:
            pools.append(("chunks", "SELECT summary AS content FROM conversation_chunks WHERE summary IS NOT NULL", []))

        pool_totals: dict[str, int] = {}
        word_pool_counts: Counter = Counter()

        for pool_name, sql, params in pools:
            rows = db.execute(sql, params).fetchall()
            pool_totals[pool_name] = len(rows)
            for r in rows:
                content = r["content"] or ""
                if _JIEBA_OK:
                    from jieba import cut
                    words = set(w.strip().lower() for w in cut(content) if w.strip())
                else:
                    words = set(re.findall(r'[一-鿿]+|[a-zA-Z]+', content.lower()))
                for w in words:
                    if len(w) >= 1:
                        word_pool_counts[w] += 1

        total_docs = sum(pool_totals.values())
        if total_docs == 0:
            return {"ok": True, "total": 0, "auto": 0}

        now = now_str()
        preserved = set()
        for r in db.execute("SELECT word FROM stopwords WHERE source IN ('manual', 'keep')").fetchall():
            preserved.add(r["word"])

        db.execute("DELETE FROM stopwords WHERE source = 'auto'")

        auto_count = 0
        all_freqs = []
        for word, count in word_pool_counts.items():
            freq = count / total_docs
            if freq >= threshold and word not in preserved:
                db.execute(
                    "INSERT OR REPLACE INTO stopwords (word, doc_freq, source, active, updated_at) VALUES (?, ?, 'auto', 1, ?)",
                    (word, freq, now))
                auto_count += 1
            all_freqs.append((word, freq))

        db.commit()

        _stopword_cache = None
        _stopword_cache_ts = 0

        return {"ok": True, "total_docs": total_docs, "auto_stopwords": auto_count, "threshold": threshold}
    finally:
        db.close()


def get_stopwords(force_refresh: bool = False) -> set[str]:
    """Get active stopwords set. Cached for 10 minutes."""
    global _stopword_cache, _stopword_cache_ts
    import time
    if not force_refresh and _stopword_cache is not None and (time.time() - _stopword_cache_ts) < 600:
        return _stopword_cache

    db = _get_db()
    try:
        rows = db.execute("SELECT word FROM stopwords WHERE active = 1").fetchall()
        _stopword_cache = {r["word"] for r in rows}
        _stopword_cache_ts = time.time()
        return _stopword_cache
    finally:
        db.close()


def list_stopwords() -> list[dict]:
    """List all stopwords with their metadata."""
    db = _get_db()
    try:
        rows = db.execute(
            "SELECT word, doc_freq, source, active FROM stopwords ORDER BY doc_freq DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        db.close()


def add_stopword(word: str) -> dict:
    """Manually add a stopword."""
    global _stopword_cache, _stopword_cache_ts
    word = word.strip().lower()
    if not word:
        return {"ok": False, "error": "empty word"}
    db = _get_db()
    try:
        db.execute(
            "INSERT OR REPLACE INTO stopwords (word, doc_freq, source, active, updated_at) VALUES (?, 0, 'manual', 1, ?)",
            (word, now_str()))
        db.commit()
        _stopword_cache = None
        _stopword_cache_ts = 0
        return {"ok": True, "added": word}
    finally:
        db.close()


def remove_stopword(word: str) -> dict:
    """Remove a stopword. Manual words are deleted; auto words are re-tagged as 'keep' so rebuilds don't re-add them."""
    global _stopword_cache, _stopword_cache_ts
    word = word.strip().lower()
    db = _get_db()
    try:
        row = db.execute("SELECT source FROM stopwords WHERE word = ?", (word,)).fetchone()
        if not row:
            return {"ok": False, "error": f"'{word}' not in stopwords"}
        if row["source"] == "manual":
            db.execute("DELETE FROM stopwords WHERE word = ?", (word,))
        else:
            db.execute("UPDATE stopwords SET source = 'keep', active = 0, updated_at = ? WHERE word = ?", (now_str(), word))
        db.commit()
        _stopword_cache = None
        _stopword_cache_ts = 0
        return {"ok": True, "removed": word}
    finally:
        db.close()


def filter_stopwords(terms: list[str]) -> list[str]:
    """Remove stopwords from a list of query terms."""
    stops = get_stopwords()
    if not stops:
        return terms
    filtered = [t for t in terms if t.lower() not in stops]
    return filtered if filtered else terms[:1]


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


def _embed_cloudflare(text: str) -> Optional[list[float]]:
    """Generate embedding via Cloudflare Workers AI."""
    account_id = os.environ.get("CF_ACCOUNT_ID", "")
    api_token = os.environ.get("CF_API_TOKEN", "")
    if not account_id or not api_token:
        return None
    try:
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{EMBED_MODEL}"
        payload = json.dumps({"text": [text]}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_token}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            result = data.get("result", {})
            vectors = result.get("data", [])
            if vectors and len(vectors) > 0:
                return vectors[0]
    except Exception:
        pass
    return None


def _embed(text: str, image_path: Optional[str] = None) -> Optional[list[float]]:
    """Generate an embedding vector, falling back to Ollama and then keyword-only search.

    Returns None when every provider is unavailable; callers keep working via
    FTS5/LIKE retrieval. image_path is only supported by the Google provider.
    """
    providers = [EMBED_PROVIDER]
    if EMBED_PROVIDER != "ollama":
        providers.append("ollama")

    for provider in providers:
        if provider == "google":
            vec = _embed_google(text, image_path=image_path)
        elif provider == "openai":
            vec = _embed_openai(text)
        elif provider == "cloudflare":
            vec = _embed_cloudflare(text)
        elif provider == "ollama":
            vec = _embed_ollama(text)
        else:
            vec = None
        if vec:
            return vec
    return None


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


def datetime_strptime(s: str) -> datetime:
    """Parse imprint timestamps with or without seconds."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s[:len("2026-01-01 00:00:00")], fmt).replace(tzinfo=LOCAL_TZ)
        except ValueError:
            continue
    parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.astimezone(LOCAL_TZ)


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
            (memory_id, now_str(), old_id),
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
        fts_terms = filter_stopwords(fts_query.split()) if fts_query else []
        fts_query = " ".join(fts_terms) if fts_terms else ""
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
MMR_SIM_THRESHOLD = 0.78      # chunk-pair cosine ≥ this counts as "same topic" for MMR dedup


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
    """Build an FTS5 MATCH expression with proper CJK tokenization.
    Uses OR so matching any term counts (not all terms required)."""
    if not _CJK_RE.search(query):
        terms = query.split()
        return " OR ".join(terms) if len(terms) > 1 else query

    if _JIEBA_OK:
        terms = [t for t in segment_cjk(query).split() if t.strip()]
        return " OR ".join(terms) if len(terms) > 1 else (terms[0] if terms else query)

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
    """Strip FTS5 operators, remove stopwords, and apply CJK segmentation."""
    cleaned = re.sub(r'["\(\)\*\:\^\{\}]', " ", query)
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        return cleaned
    # Tokenize first, drop stopwords, THEN OR-join. Doing it the other way
    # around (OR-join → filter_stopwords) leaves dangling OR operators
    # between dropped terms — FTS5 rejects "上次 OR OR OR p5js" as a syntax
    # error and the whole query silently returns nothing.
    segmented = _fts_query_cjk(cleaned)
    terms = [t for t in segmented.split() if t != "OR"]
    filtered = filter_stopwords(terms)
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0]
    return " OR ".join(filtered)


# ─── Chunk + Graph Search Channel ────────────────────────

def _search_chunk_channels(query_vec, db, query="", limit=20):
    """Return (fts_ranking, vec_ranking, details) for chunk pool with graph expansion."""
    fts_ranking = []
    vec_ranking = []
    details = {}

    # FTS5 search on chunk summaries
    try:
        fts_q = _fts_query_cjk(query) if query else ""
        if fts_q:
            fts_q = _sanitize_fts(fts_q)
        if fts_q:
            fts_rows = db.execute(
                """SELECT c.id, c.start_msg_id, c.end_msg_id, c.msg_count,
                          c.summary, c.keywords, c.embedding, c.start_time, c.end_time
                   FROM chunks_fts f
                   JOIN conversation_chunks c ON f.rowid = c.id
                   WHERE chunks_fts MATCH ?
                   ORDER BY f.rank
                   LIMIT ?""",
                (fts_q, limit),
            ).fetchall()
            for idx, row in enumerate(fts_rows):
                key = f"chunk_{row['id']}"
                fts_ranking.append((key, idx + 1))
                if key not in details:
                    details[key] = {
                        "id": row["id"],
                        "content": f"[回忆] {row['summary']}",
                        "created_at": row["start_time"],
                        "summary": row["summary"],
                        "keywords": row["keywords"],
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                        "msg_count": row["msg_count"],
                        "start_msg_id": row["start_msg_id"],
                        "end_msg_id": row["end_msg_id"],
                    }
    except Exception:
        pass

    if not query_vec:
        return fts_ranking, vec_ranking, details

    try:
        rows = db.execute(
            """SELECT id, start_msg_id, end_msg_id, msg_count, platforms,
                      summary, keywords, embedding, start_time, end_time
               FROM conversation_chunks WHERE embedding IS NOT NULL"""
        ).fetchall()
    except Exception:
        return fts_ranking, vec_ranking, details

    scored = []
    for row in rows:
        blob = row["embedding"]
        if not blob:
            continue
        vec = _blob_to_vec(blob)
        sim = _cosine_similarity(query_vec, vec)
        if sim > 0:
            scored.append((sim, row))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Seeds
    seed_ids = set()
    for idx, (sim, row) in enumerate(scored[:limit]):
        key = f"chunk_{row['id']}"
        vec_ranking.append((key, idx + 1))
        if key not in details:
            details[key] = {
                "id": row["id"],
                "content": f"[回忆] {row['summary']}",
                "created_at": row["start_time"],
                "vec_similarity": sim,
                "summary": row["summary"],
                "keywords": row["keywords"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "msg_count": row["msg_count"],
                "start_msg_id": row["start_msg_id"],
                "end_msg_id": row["end_msg_id"],
            }
        else:
            details[key]["vec_similarity"] = sim
        seed_ids.add(row["id"])

    # Graph expansion: follow edges from top-5 seeds
    try:
        for seed_id in list(seed_ids)[:5]:
            neighbors = db.execute(
                """SELECT target_id, similarity FROM chunk_edges
                   WHERE source_id = ? ORDER BY similarity DESC LIMIT 2""",
                (seed_id,),
            ).fetchall()
            for n in neighbors:
                tid = n["target_id"]
                nkey = f"chunk_{tid}"
                if nkey in details:
                    continue
                chunk = db.execute(
                    """SELECT id, summary, keywords, start_time, end_time, msg_count
                       FROM conversation_chunks WHERE id = ?""",
                    (tid,),
                ).fetchone()
                if chunk:
                    rank = len(vec_ranking) + 1
                    vec_ranking.append((nkey, rank))
                    details[nkey] = {
                        "id": chunk["id"],
                        "content": f"[回忆] {chunk['summary']}",
                        "created_at": chunk["start_time"],
                        "vec_similarity": n["similarity"] * 0.7,
                        "summary": chunk["summary"],
                        "keywords": chunk["keywords"],
                        "start_time": chunk["start_time"],
                        "end_time": chunk["end_time"],
                        "msg_count": chunk["msg_count"],
                    }
    except Exception:
        pass

    _inject_default_ranks(fts_ranking, vec_ranking)
    return fts_ranking, vec_ranking, details


def _search_fact_channels(query_vec, db, limit=30):
    """Search chunk_facts by vector, but return parent chunk as the result.
    Facts are search indices; chunks are the actual memories."""
    vec_ranking = []
    details = {}
    if not query_vec:
        return vec_ranking, details
    try:
        rows = db.execute(
            """SELECT f.id, f.chunk_id, f.content as fact_content, f.embedding,
                      c.summary, c.keywords, c.start_time, c.end_time, c.msg_count
               FROM chunk_facts f
               JOIN conversation_chunks c ON f.chunk_id = c.id
               WHERE f.embedding IS NOT NULL"""
        ).fetchall()
    except Exception:
        return vec_ranking, details

    scored = []
    for row in rows:
        blob = row["embedding"]
        if not blob:
            continue
        vec = _blob_to_vec(blob)
        sim = _cosine_similarity(query_vec, vec)
        if sim > 0.3:
            scored.append((sim, row))
    scored.sort(key=lambda x: x[0], reverse=True)

    seen_chunks = set()
    for idx, (sim, row) in enumerate(scored[:limit]):
        chunk_id = row["chunk_id"]
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        key = f"fact_{row['id']}"
        vec_ranking.append((key, idx + 1))
        details[key] = {
            "id": chunk_id,
            "content": f"[匹配] {row['fact_content']}\n[回忆] {row['summary']}",
            "matched_fact": row["fact_content"],
            "summary": row["summary"],
            "created_at": row["start_time"],
            "vec_similarity": sim,
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "msg_count": row["msg_count"],
        }

    return vec_ranking, details


# ─── LLM Rerank ─────────────────────────────────────────

_CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "")
_CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "")
_CF_RERANK_MODEL = os.environ.get("CF_RERANK_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")

_RERANK_PROMPT = """给每条候选结果打相关性分（0-10），只输出JSON数组。

查询：{query}

候选：
{candidates}

输出格式（严格JSON，不要其他内容）：
[{{"idx":0,"score":8}},{{"idx":1,"score":3}},...]

打分标准：
- 10=完全相关，直接回答查询
- 7-9=高度相关，包含关键信息
- 4-6=部分相关，有些关联但不直接
- 1-3=几乎不相关
- 0=完全无关"""


def _llm_rerank(query: str, candidates: list[dict]) -> list[dict]:
    if not _CF_ACCOUNT_ID or not _CF_API_TOKEN or not candidates:
        return candidates

    cand_text = "\n".join(
        f"[{i}] {r.get('content', '')[:200]}"
        for i, r in enumerate(candidates)
    )
    prompt = _RERANK_PROMPT.format(query=query, candidates=cand_text)

    url = f"https://api.cloudflare.com/client/v4/accounts/{_CF_ACCOUNT_ID}/ai/run/{_CF_RERANK_MODEL}"
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Authorization": f"Bearer {_CF_API_TOKEN}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            r = data.get("result", {}).get("response", "")
            if isinstance(r, list):
                scores = r
            else:
                raw = r if isinstance(r, str) else json.dumps(r)
                raw = re.sub(r'[\x00-\x1f\x7f]', ' ', raw)
                raw = re.sub(r'```(?:json)?\s*', '', raw)
                start = raw.find('[')
                end = raw.rfind(']')
                if start == -1 or end == -1:
                    return candidates
                scores = json.loads(raw[start:end + 1])

            score_map = {s["idx"]: s["score"] for s in scores if "idx" in s and "score" in s}
            for i, r in enumerate(candidates):
                llm_score = score_map.get(i, 0)
                r["llm_rerank_score"] = llm_score
                rrf_score = r.get("score", 0)
                r["score"] = rrf_score * 0.5 + (llm_score / 10.0) * 0.5

            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates

    except Exception as e:
        print(f"LLM rerank failed: {e}")
        return candidates


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
    """Memory rerank: activation x importance x specificity, blended with RRF.
    Time decay is disabled — a 3-month-old memory that matches the query
    well shouldn't lose to a recent irrelevant hit just because it's old."""
    if row.get("pinned"):
        return rrf_score

    importance = max(row.get("importance", 5), 1)
    recalled = row.get("recalled_count", 0)

    activation_factor = 0.8 + 0.2 * (math.log(recalled + 1) / math.log(51))
    importance_factor = 0.7 + 0.3 * (importance / 10)

    content_len = len(row.get("content", ""))
    specificity = min(1.0, 0.5 + 0.5 * (content_len / 40)) if content_len < 40 else 1.0

    factor = activation_factor * importance_factor * specificity
    return rrf_score * (1 - RERANK_BLEND + RERANK_BLEND * factor)


def _rerank_bank(rrf_score: float, row: dict) -> float:
    """Bank rerank: time decay disabled, pass RRF through."""
    return rrf_score


def _rerank_conv(rrf_score: float, row: dict) -> float:
    """Conversation/chunk rerank: time decay disabled, pass RRF through.
    Previously 7-day half-life crushed any chat older than two weeks; users
    searching for specific old conversations couldn't find them."""
    return rrf_score


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
    """Return (fts_ranking, vec_ranking, like_ranking, details) for conversation pool.

    The vec channel ranks conversation_vectors globally. Embeddings here may be
    multimodal (text+image, via Gemini Embedding 2) for messages that uploaded
    a file, so this channel is what makes "the red screenshot I sent" land on
    the image-bearing message even when no keyword matches.
    """
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

            rank_idx = 0
            for r in fts_rows:
                # Strip <think> blocks before deciding whether this row
                # carries actual conversation content. Assistant turns whose
                # bulk is internal reasoning ("<think>...</think>" with
                # almost nothing outside) used to win on FTS because the
                # reasoning matched the keywords — but the user sees
                # gibberish since reasoning is hidden in real chat. Drop
                # those entries here so they neither rank nor render.
                raw_content = r["content"] or ""
                visible = _THINK_RE.sub("", raw_content).strip()
                if len(visible) < 5:
                    continue
                key = f"conv_{r['id']}"
                rank_idx += 1
                fts_ranking.append((key, rank_idx))
                row_d = dict(r)
                row_d["content"] = visible  # downstream renderers see the clean version
                details[key] = row_d
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
        rank_idx = 0
        for r in like_rows:
            raw_content = r["content"] or ""
            visible = _THINK_RE.sub("", raw_content).strip()
            if len(visible) < 5:
                continue
            key = f"conv_{r['id']}"
            rank_idx += 1
            like_ranking.append((key, rank_idx))
            if key not in details:
                row_d = dict(r)
                row_d["content"] = visible
                details[key] = row_d

    # ── Vec ranking over conversation_vectors (may be multimodal) ──
    # Off by default. With many rows the matmul is cheap (~50ms via numpy) but
    # the SQL load is non-trivial and, more importantly, current embeddings
    # may have been generated without Gemini's task_type=RETRIEVAL_QUERY/
    # DOCUMENT so query↔document cosines don't align — short text messages
    # dominate the top ranks. Re-enable via IMPRINT_CONV_VEC_CHANNEL=1 once
    # vectors are rebuilt with proper task types.
    if query_vec and os.environ.get("IMPRINT_CONV_VEC_CHANNEL", "0") == "1":
        try:
            import numpy as _np
            if platform:
                v_rows = db.execute(
                    """SELECT c.id, c.platform, c.direction, c.speaker, c.content,
                              c.created_at, v.embedding
                       FROM conversation_log c
                       JOIN conversation_vectors v ON v.msg_id = c.id
                       WHERE v.embedding IS NOT NULL AND c.platform = ?""",
                    (platform,),
                ).fetchall()
            else:
                v_rows = db.execute(
                    """SELECT c.id, c.platform, c.direction, c.speaker, c.content,
                              c.created_at, v.embedding
                       FROM conversation_log c
                       JOIN conversation_vectors v ON v.msg_id = c.id
                       WHERE v.embedding IS NOT NULL"""
                ).fetchall()
            if v_rows:
                dim = len(query_vec)
                mat = _np.empty((len(v_rows), dim), dtype=_np.float32)
                valid = _np.ones(len(v_rows), dtype=bool)
                for i, r in enumerate(v_rows):
                    blob = r["embedding"]
                    if not blob or len(blob) // 4 != dim:
                        valid[i] = False
                        continue
                    mat[i] = _np.frombuffer(blob, dtype=_np.float32)
                q = _np.asarray(query_vec, dtype=_np.float32)
                row_norms = _np.linalg.norm(mat, axis=1)
                q_norm = float(_np.linalg.norm(q)) or 1.0
                sims = (mat @ q) / (row_norms * q_norm + 1e-8)
                sims[~valid] = -1.0
                sims[~_np.isfinite(sims)] = -1.0
                eligible = _np.where(sims >= VEC_PRE_FILTER)[0]
                if eligible.size:
                    order = eligible[_np.argsort(-sims[eligible])][:limit]
                    for idx, pos in enumerate(order):
                        r = v_rows[int(pos)]
                        key = f"conv_{r['id']}"
                        vec_ranking.append((key, idx + 1))
                        sim_val = float(sims[int(pos)])
                        if key not in details:
                            details[key] = dict(r)
                        details[key]["vec_similarity"] = max(
                            details[key].get("vec_similarity", 0) or 0,
                            sim_val,
                        )
                        details[key].pop("embedding", None)
        except ImportError:
            pass
        except Exception:
            pass

    return fts_ranking, vec_ranking, like_ranking, details


def _mmr_diversify(results: list[dict], db, limit_pool: int = 20,
                   sim_threshold: float = 0.78) -> list[dict]:
    """Greedy MMR pass over chunk results to break "echo amplification".

    When the same topic gets discussed repeatedly, each discussion becomes
    its own near-duplicate chunk, and the top-K starts crowding with copies
    of that one topic — pushing genuinely different related events out.

    For each candidate, walk in current score order. Compare against each
    already-kept chunk by cosine over their stored chunk embeddings; if
    any pair scores at or above sim_threshold, the candidate is "too
    similar" and gets shelved. If we run out of fresh candidates before
    reaching the caller's limit, the shelved ones fill the remaining
    slots so we never return fewer results than the unfiltered version.

    Only operates on the chunk pool. Memory / conversation rows pass
    through untouched (their dedup is handled elsewhere).
    """
    candidates = results[:limit_pool]
    tail = results[limit_pool:]

    chunk_ids = [r["id"] for r in candidates if r.get("pool") == "chunk" and r.get("id")]
    embeddings: dict[int, list[float]] = {}
    if chunk_ids:
        placeholders = ",".join("?" * len(chunk_ids))
        try:
            rows = db.execute(
                f"SELECT id, embedding FROM conversation_chunks "
                f"WHERE id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
            for row in rows:
                blob = row["embedding"]
                if blob:
                    embeddings[row["id"]] = _blob_to_vec(blob)
        except Exception:
            embeddings = {}

    kept: list[dict] = []
    shelved: list[dict] = []
    kept_chunk_vecs: list[list[float]] = []

    for r in candidates:
        if r.get("pool") != "chunk":
            kept.append(r)
            continue
        cid = r.get("id")
        cvec = embeddings.get(cid) if cid else None
        too_similar = False
        if cvec and kept_chunk_vecs:
            for kvec in kept_chunk_vecs:
                if _cosine_similarity(cvec, kvec) >= sim_threshold:
                    too_similar = True
                    break
        if too_similar:
            shelved.append(r)
        else:
            kept.append(r)
            if cvec:
                kept_chunk_vecs.append(cvec)

    return kept + shelved + tail


def _search_time_window(query_vec, db, after: str, before: str, limit: int = 20):
    """In-range vec recall channel.

    When the user pins a time range (e.g. "yesterday/last week/May 15"), the
    standard channels rank globally and may push the most relevant in-range
    content out of the top-K candidate pool — e.g. a chunk whose summary
    abstracted "drank tequila" into "vulnerability/testing the relationship"
    will have low cosine to the literal query "drank yesterday" and never
    survive a global limit, even though it's the right answer.

    This channel scopes vector ranking to candidates whose normalised day
    falls inside [after, before], guaranteeing the best in-range hits enter
    the candidate pool.

    Returns (chunk_ranking, conv_ranking, details).
    """
    chunk_ranking: list[tuple[str, int]] = []
    conv_ranking: list[tuple[str, int]] = []
    details: dict[str, dict] = {}
    if not query_vec or not after or not before:
        return chunk_ranking, conv_ranking, details

    # SQL pre-filter is intentionally loose: take a 1-day buffer on each side
    # so boundary-hour shifts (e.g. early-morning hours mapping to prev day)
    # don't drop edge timestamps. Python precise-filter via _ts_to_day applies
    # the actual day-boundary rule.
    try:
        sql_after = (datetime.strptime(after, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        sql_before = (datetime.strptime(before, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        return chunk_ranking, conv_ranking, details

    # ── Chunks in window ──
    try:
        rows = db.execute(
            """SELECT id, start_msg_id, end_msg_id, msg_count, summary, keywords,
                      embedding, start_time, end_time
               FROM conversation_chunks
               WHERE embedding IS NOT NULL
                 AND substr(start_time, 1, 10) BETWEEN ? AND ?""",
            (sql_after, sql_before),
        ).fetchall()
    except Exception:
        rows = []

    scored = []
    for row in rows:
        day = _ts_to_day(row["start_time"])
        if not day or not (after <= day <= before):
            continue
        sim = _cosine_similarity(query_vec, _blob_to_vec(row["embedding"]))
        scored.append((sim, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    for idx, (sim, row) in enumerate(scored[:limit]):
        key = f"chunk_{row['id']}"
        chunk_ranking.append((key, idx + 1))
        details[key] = {
            "id": row["id"],
            "content": f"[回忆] {row['summary']}",
            "created_at": row["start_time"],
            "vec_similarity": sim,
            "summary": row["summary"],
            "keywords": row["keywords"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "msg_count": row["msg_count"],
            "start_msg_id": row["start_msg_id"],
            "end_msg_id": row["end_msg_id"],
        }

    # ── Conversation messages in window ──
    try:
        rows = db.execute(
            """SELECT c.id, c.platform, c.direction, c.speaker, c.content,
                      c.created_at, v.embedding
               FROM conversation_log c
               JOIN conversation_vectors v ON v.msg_id = c.id
               WHERE v.embedding IS NOT NULL
                 AND substr(c.created_at, 1, 10) BETWEEN ? AND ?""",
            (sql_after, sql_before),
        ).fetchall()
    except Exception:
        rows = []

    scored = []
    for row in rows:
        day = _ts_to_day(row["created_at"])
        if not day or not (after <= day <= before):
            continue
        sim = _cosine_similarity(query_vec, _blob_to_vec(row["embedding"]))
        scored.append((sim, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    for idx, (sim, row) in enumerate(scored[:limit]):
        key = f"conv_{row['id']}"
        conv_ranking.append((key, idx + 1))
        if key not in details:
            details[key] = {
                "id": row["id"],
                "platform": row["platform"],
                "direction": row["direction"],
                "speaker": row["speaker"],
                "content": row["content"],
                "created_at": row["created_at"],
                "vec_similarity": sim,
            }
        else:
            details[key]["vec_similarity"] = max(details[key].get("vec_similarity", 0), sim)

    return chunk_ranking, conv_ranking, details


# ─── Chunk Keyword Expansion ─────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Match channel-agnostic "uploaded a file ... path=..." headers (Chinese
# channel-adapter convention; harmless no-op for messages without one).
_IMG_PATH_RE = re.compile(
    r"上传了一个文件:.*?路径=([^;\]]+?\.(?:jpg|jpeg|png|gif|webp))(?:\s|;|\])",
    re.IGNORECASE | re.DOTALL,
)

# Brackets that channel adapters inject as message prefix metadata. They're
# useful for routing/marking when the message is in transit, but on the way
# back out (e.g. in chunk-expansion surfacing) they're noise — they make a
# historical message look like it's the *current* conversation.
_ADAPTER_PREFIX_RES = [
    re.compile(r"\[当前对话窗口:[^\]]*\]"),
    re.compile(r"\[\d{4}-\d{2}-\d{2}[^\]]*\]"),
    re.compile(r"\[[^\]]*上传了一个文件:[^\]]*\]"),
]


def _extract_image_path(content: str) -> str:
    """Return the image file path embedded in a channel upload message, or
    empty string if none. Used to recognise messages that carry an image so
    chunk expansion can anchor on the "image + surrounding turns" unit."""
    if not content or "上传了一个文件" not in content:
        return ""
    m = _IMG_PATH_RE.search(content)
    return m.group(1).strip() if m else ""


_CC_SYSTEM_RE = re.compile(
    r"<(?:local-command-caveat|local-command-stdout|command-name|"
    r"command-message|command-args|system-reminder|north-home|"
    r"memory-check|recall)[^>]*>.*?</(?:local-command-caveat|"
    r"local-command-stdout|command-name|command-message|command-args|"
    r"system-reminder|north-home|memory-check|recall)>",
    re.DOTALL,
)
_CC_COMPACT_PREFIX = "This session is being continued from a previous conversation"


def _clean_msg_for_display(content: str) -> str:
    """Strip noise from a message before it's shown back to the user in a
    surfacing snippet: <think> blocks, CC system XML, channel-adapter
    brackets, and a leading "X 说:" lead-in. Returns the trimmed remainder."""
    if not content:
        return ""
    if content.lstrip().startswith(_CC_COMPACT_PREFIX):
        return ""
    text = _THINK_RE.sub("", content)
    text = _CC_SYSTEM_RE.sub("", text)
    for pat in _ADAPTER_PREFIX_RES:
        text = pat.sub("", text)
    text = re.sub(r"^\s*[\w]+说:\s*", "", text.strip())
    return text.strip()


def _expand_chunk_hybrid(query: str, query_vec, results: list[dict], db, max_msgs: int = 5) -> None:
    """Expand chunk results with hybrid keyword + embedding ranked messages.
    Short chunks (<=15 msgs): keyword-only (fast, sufficient).
    Long chunks: embedding baseline + keyword boost.

    Image-anchor rule: when the chunk contains an image-bearing message, pin
    that message + its immediate neighbour turns into the expansion so the
    "image + question + reply" unit shows up as one piece. Standalone image
    messages are useless ("[uploaded a file…]") — they only make sense in
    context."""
    if _JIEBA_OK:
        from jieba import cut
        query_terms = [w.strip().lower() for w in cut(query) if len(w.strip()) >= 2]
    else:
        query_terms = [w.lower() for w in query.split() if len(w) >= 2]
    query_terms = filter_stopwords(query_terms)
    if not query_terms:
        query_terms = [w.lower() for w in query.split() if len(w) >= 2]
    term_set = set(query_terms)

    for r in results:
        if r.get("pool") != "chunk":
            continue
        sid = r.get("start_msg_id")
        eid = r.get("end_msg_id")
        if not sid or not eid:
            continue

        msgs = db.execute(
            """SELECT id, direction, speaker, content
               FROM conversation_log
               WHERE id BETWEEN ? AND ?
                 AND (platform NOT IN ('cc') OR content LIKE '%上传了一个文件%')
               ORDER BY id""",
            (sid, eid),
        ).fetchall()

        use_embedding = bool(query_vec)

        scored = []
        for m in msgs:
            raw = m["content"] or ""
            clean = _THINK_RE.sub("", raw).strip()
            if len(clean) < 5:
                continue
            display = _clean_msg_for_display(raw) or clean

            lower = clean.lower()
            kw_matched = sum(1 for t in term_set if t in lower)
            kw_score = kw_matched / len(term_set) if term_set else 0

            density_bonus = 0.0
            if kw_matched > 0 and len(clean) < 60:
                density_bonus = min(0.4, 0.6 * kw_matched / max(len(clean), 5))

            emb_score = 0.0
            if use_embedding:
                vrow = db.execute(
                    "SELECT embedding FROM conversation_vectors WHERE msg_id = ?",
                    (m["id"],),
                ).fetchone()
                if vrow and vrow["embedding"]:
                    vec = _blob_to_vec(vrow["embedding"])
                    emb_score = _cosine_similarity(query_vec, vec)

            if use_embedding:
                score = emb_score + kw_score * 0.3 + density_bonus
            else:
                score = kw_score + density_bonus

            speaker = m["speaker"] or (USER_NAME if m["direction"] == "in" else AGENT_NAME)
            scored.append((score, m["id"], speaker, display[:300]))

        scored.sort(key=lambda x: (-x[0], x[1]))
        top = [s for s in scored if s[0] > 0][:max_msgs]

        if not top:
            chunk_kws = set(
                k.strip().lower()
                for k in (r.get("keywords") or "").split(",")
                if k.strip()
            )
            if chunk_kws:
                rescored = []
                msg_lookup = {m["id"]: m for m in msgs}
                for sc, mid, sp, disp in scored:
                    row = msg_lookup.get(mid)
                    if row:
                        lower = (row["content"] or "").lower()
                        hits = sum(1 for t in chunk_kws if t in lower)
                        rescored.append((hits, mid, sp, disp))
                    else:
                        rescored.append((0, mid, sp, disp))
                rescored.sort(key=lambda x: (-x[0], x[1]))
                top = [s for s in rescored if s[0] > 0][:max_msgs]
            if not top:
                top = scored[:max_msgs]

        # Image anchors: find any image-bearing msg inside the chunk and pin
        # it + its immediate predecessor and successor (as msg ids). Replaces
        # an equivalent number of low-scoring entries from `top` so the total
        # expansion size stays roughly bounded.
        anchor_ids = []
        for i, m in enumerate(msgs):
            if _extract_image_path(m["content"] or ""):
                anchor_ids.append(m["id"])
                if i > 0:
                    anchor_ids.append(msgs[i - 1]["id"])
                if i < len(msgs) - 1:
                    anchor_ids.append(msgs[i + 1]["id"])

        if anchor_ids:
            anchor_set = set(anchor_ids)
            non_anchor_top = [s for s in top if s[1] not in anchor_set]
            keep_n = max(0, max_msgs - len(anchor_set))
            non_anchor_top = non_anchor_top[:keep_n]
            anchor_top = []
            msg_by_id = {m["id"]: m for m in msgs}
            for mid in anchor_set:
                m = msg_by_id.get(mid)
                if not m:
                    continue
                raw = m["content"] or ""
                img_path = _extract_image_path(raw)
                display = _clean_msg_for_display(raw) or _THINK_RE.sub("", raw).strip()
                speaker = m["speaker"] or (USER_NAME if m["direction"] == "in" else AGENT_NAME)
                anchor_top.append((0.0, mid, speaker, display[:300], img_path))
            # msg_by_id values are sqlite3.Row, not dict — they don't support
            # .get(). Use direct indexing with a None-guard so a missing id
            # falls through to no-image instead of AttributeError.
            def _img_for(mid: int) -> str:
                row = msg_by_id.get(mid)
                if row is None:
                    return ""
                return _extract_image_path(row["content"] or "")
            top = [(s, i, sp, c, _img_for(i))
                   for (s, i, sp, c) in non_anchor_top] + anchor_top
        else:
            top = [(s, i, sp, c, "") for (s, i, sp, c) in top]

        # Pair each ranked message with the opposite-direction neighbour
        # so the expansion never shows one side of a turn in isolation.
        # 'in' msg → attach the next 'out' that follows it inside the chunk.
        # 'out' msg → attach the most recent 'in' that came before it.
        msg_by_id = {m["id"]: m for m in msgs}
        msg_idx_by_id = {m["id"]: i for i, m in enumerate(msgs)}
        chosen_ids = {t[1] for t in top}
        pair_entries = []
        for t in top:
            mid = t[1]
            if mid not in msg_idx_by_id:
                continue
            idx = msg_idx_by_id[mid]
            cur_dir = msgs[idx]["direction"]
            target = "out" if cur_dir == "in" else "in"
            pair_msg = None
            if target == "out":
                for j in range(idx + 1, len(msgs)):
                    if msgs[j]["direction"] == "out":
                        pair_msg = msgs[j]
                        break
            else:
                for j in range(idx - 1, -1, -1):
                    if msgs[j]["direction"] == "in":
                        pair_msg = msgs[j]
                        break
            if pair_msg is None or pair_msg["id"] in chosen_ids:
                continue
            raw = pair_msg["content"] or ""
            clean = _THINK_RE.sub("", raw).strip()
            if len(clean) < 5:
                continue
            display = _clean_msg_for_display(raw) or clean
            sp = pair_msg["speaker"] or (USER_NAME if pair_msg["direction"] == "in" else AGENT_NAME)
            img = _extract_image_path(raw)
            pair_entries.append((0.0, pair_msg["id"], sp, display[:300], img))
            chosen_ids.add(pair_msg["id"])
        top.extend(pair_entries)

        top.sort(key=lambda x: x[1])
        expanded_items = []
        for _, mid, speaker, content, img_path in top:
            item = {"speaker": speaker, "content": content}
            if img_path:
                item["image_path"] = img_path
            expanded_items.append(item)
        r["expanded"] = expanded_items


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


# ─── Query Expansion ─────────────────────────────────────

_EXPAND_PROMPT = """This is a personal chat history search. Query: {query}
Think of colloquial expressions and related context words that might appear in real conversations (not formal writing). Output 5 comma-separated words:"""


def _expand_query(query: str) -> str:
    """Use LLM to expand a query with synonyms for better recall."""
    if not _CF_ACCOUNT_ID or not _CF_API_TOKEN:
        return query
    if len(query) > 50:
        return query
    url = f"https://api.cloudflare.com/client/v4/accounts/{_CF_ACCOUNT_ID}/ai/run/{_CF_RERANK_MODEL}"
    payload = json.dumps({
        "messages": [{"role": "user", "content": _EXPAND_PROMPT.format(query=query)}],
        "max_tokens": 60,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={
            "Authorization": f"Bearer {_CF_API_TOKEN}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
            r = data.get("result", {}).get("response", "")
            if isinstance(r, str) and r.strip():
                expanded = query + " " + r.strip().replace("，", ",").replace(",", " ")
                return expanded
    except Exception:
        pass
    return query


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
    rerank: bool = True,
) -> list[dict]:
    """Search across memory pools with RRF fusion and per-pool reranking.

    Args:
        query:    natural-language search query
        limit:    max results to return
        pools:    subset of ["memory", "bank", "chunk", "conversation"]; None = ["memory", "bank", "chunk"]
                  - memory: curated facts extracted by the extractor
                  - bank:   markdown notes under memory/bank/
                  - chunk:  conversation chunks (index layer; hits expand to raw messages)
                  - conversation: raw conversation_log rows (opt-in; used by conversation_search)
        category: filter memory pool by category
        platform: filter conversation pool by platform
        _internal: skip side-effects (recalled_count, last_accessed_at) — for edge expansion

    Returns list of dicts sorted by final score, each containing:
        pool, score, rrf_raw, id, content, + pool-specific fields
    """
    if pools is None:
        # Default: chunks act as a navigation index over the raw chat log.
        # We search memory (curated facts), bank (markdown notes), and
        # chunks; for chunk hits the renderer expands each one into its
        # top-ranked raw messages rather than echoing the chunk summary —
        # so the user always sees originals while still benefitting from
        # chunk-level keyword / topic matching.
        pools = ["memory", "bank", "chunk"]

    if (after or before) and "bank" in pools:
        pools = [p for p in pools if p != "bank"]

    expanded_query = _expand_query(query)

    db = _get_db()
    query_vec = _embed(expanded_query if expanded_query != query else query)
    all_rankings: list[list[tuple[str, int]]] = []
    all_details: dict[str, dict] = {}

    if "memory" in pools:
        m_fts, m_vec, m_like, m_det = _search_memory_channels(
            expanded_query, query_vec, db, category=category
        )
        _inject_default_ranks(m_fts, m_vec)
        all_rankings += [m_fts, m_vec, m_like]
        all_details.update(m_det)

    if "bank" in pools:
        b_fts, b_vec, b_like, b_det = _search_bank_channels(expanded_query, query_vec, db)
        _inject_default_ranks(b_fts, b_vec)
        all_rankings += [b_fts, b_vec, b_like]
        all_details.update(b_det)

    if "conversation" in pools:
        c_fts, c_vec, c_like, c_det = _search_conv_channels(
            expanded_query, query_vec, db, platform=platform
        )
        if c_vec:
            _inject_default_ranks(c_fts, c_vec)
        all_rankings += [c_fts, c_vec, c_like]
        all_details.update(c_det)

    # Chunk search is now its own opt-in pool (was always tacked onto the
    # conversation branch). Auto-surfacing wants chunks; manual search
    # (memory_search MCP tool) wants raw originals only.
    if "chunk" in pools:
        ch_fts, ch_vec, ch_det = _search_chunk_channels(query_vec, db, query=expanded_query)
        if ch_fts:
            all_rankings.append(ch_fts)
        if ch_vec:
            all_rankings.append(ch_vec)
        all_details.update(ch_det)

        # Time-window channel: when the user pinned a precise time range,
        # rank chunks/messages scoped to that window so in-range hits can't be
        # crowded out of the candidate pool by globally-higher-but-irrelevant
        # results.
        if after and before:
            tw_ch, tw_conv, tw_det = _search_time_window(query_vec, db, after, before)
            if tw_ch:
                all_rankings.append(tw_ch)
            if tw_conv:
                all_rankings.append(tw_conv)
            for k, v in tw_det.items():
                if k in all_details:
                    if "vec_similarity" in v:
                        all_details[k]["vec_similarity"] = max(
                            all_details[k].get("vec_similarity", 0) or 0,
                            v["vec_similarity"],
                        )
                else:
                    all_details[k] = v

        # Chunk facts: disabled — quality too low from batch LLM extraction.
        # Facts will be manually curated by North during heartbeat instead.
        # f_vec, f_det = _search_fact_channels(query_vec, db)
        # if f_vec:
        #     all_rankings.append(f_vec)
        #     all_details.update(f_det)

    rrf_scores = _rrf_fuse(all_rankings)

    # Per-pool rerank + within-pool normalisation
    pool_items: dict[str, list[dict]] = {"memory": [], "bank": [], "conversation": [], "chunk": [], "fact": []}

    for key, rrf in rrf_scores.items():
        detail = all_details.get(key, {})

        if key.startswith("mem_"):
            pool = "memory"
            reranked = _rerank_memory(rrf, detail)
        elif key.startswith("bank_"):
            pool = "bank"
            reranked = _rerank_bank(rrf, detail)
        elif key.startswith("chunk_"):
            pool = "chunk"
            reranked = _rerank_conv(rrf, detail)
        elif key.startswith("fact_"):
            pool = "fact"
            reranked = _rerank_conv(rrf, detail)
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
    # Time-pinned queries get a chunk-pool boost so the relevant topic segment
    # surfaces even when its vec_sim is moderate — but conv pool stays on its
    # normal confidence curve. Image/multimodal messages are meant to appear
    # *inside* a chunk's expansion, not as standalone hits.
    time_pinned = bool(after and before)
    results: list[dict] = []
    for pool, items in pool_items.items():
        if not items:
            continue
        max_score = max(r["score"] for r in items)
        max_vec = max((r.get("vec_similarity") or 0) for r in items)
        # 0.01 covers FTS rank 1-40 (1/61 = 0.0164, 1/100 = 0.01). The old
        # 0.02 threshold only counted pools whose top hit also had vec or
        # like signal stacking onto FTS — conv pool, which currently runs
        # without a vec channel, never qualified and got the 0.15 noise
        # penalty even on strong keyword matches like "马桶".
        has_fts = any(r.get("rrf_raw", 0) > 0.01 for r in items)
        if time_pinned and pool == "chunk":
            pool_conf = 1.0
        elif max_vec >= VEC_CONFIDENCE_GOOD:
            pool_conf = 1.0
        elif has_fts:
            # FTS keyword match is strong evidence on its own — promote to
            # full confidence so the conversation pool (which currently
            # ships without a vec channel) can compete against memory hits
            # that have both FTS and vec signals.
            pool_conf = 1.0
        elif max_vec <= VEC_CONFIDENCE_NOISE:
            pool_conf = 0.15
        else:
            pool_conf = 0.15 + 0.85 * (max_vec - VEC_CONFIDENCE_NOISE) / (VEC_CONFIDENCE_GOOD - VEC_CONFIDENCE_NOISE)
        for r in items:
            r["score"] = (r["score"] / max_score * pool_conf) if max_score > 0 else 0
        results.extend(items)

    _KEYWORD_BOOST = 0.3
    if _JIEBA_OK:
        from jieba import cut
        query_terms = filter_stopwords([w for w in cut(query) if w.strip()])
    else:
        query_terms = filter_stopwords(query.split() if " " in query else [query])
    for r in results:
        content = r.get("content", "")
        matched = sum(1 for t in query_terms if t in content)
        if matched:
            r["score"] += _KEYWORD_BOOST * (matched / len(query_terms))

    floor = POST_NORM_FLOOR * 0.3 if rerank else POST_NORM_FLOOR
    results = [r for r in results if r["score"] >= floor]
    results.sort(key=lambda x: x["score"], reverse=True)

    # Noise filter: skip when reranker is on (reranker handles relevance)
    if not rerank and results:
        top_score = results[0]["score"]
        noise_floor = top_score * 0.5
        results = [r for r in results if r["score"] >= noise_floor]

    # Dedup: overlapping chunks keep only the higher-scored one
    kept_ranges = []
    deduped = []
    for r in results:
        if r.get("pool") == "chunk":
            sid = r.get("start_msg_id")
            eid = r.get("end_msg_id")
            if sid and eid:
                overlaps = any(not (eid < ks or sid > ke) for ks, ke in kept_ranges)
                if overlaps:
                    continue
                kept_ranges.append((sid, eid))
        deduped.append(r)
    results = deduped

    # Drop standalone image-upload messages ONLY when they're effectively
    # naked metadata — "[uploaded a file: 路径=...]" with no accompanying
    # caption. If the user typed something alongside the image
    # ("[uploaded ...] 看这张截图"), keep the row: the caption + multimodal
    # vec both carry signal, and waiting for the chunker to swallow it
    # before search can see it makes new images un-findable for hours.
    def _is_metadata_only_image(row: dict) -> bool:
        if row.get("pool") != "conversation":
            return False
        raw = row.get("content", "") or ""
        if not _extract_image_path(raw):
            return False
        # Strip the upload header + adapter wrappers; what's left is the
        # human caption. If <5 chars after stripping, treat as bare upload.
        stripped = _clean_msg_for_display(raw)
        stripped = re.sub(r"上传了一个文件:[^\]]*", "", stripped).strip()
        return len(stripped) < 5

    results = [r for r in results if not _is_metadata_only_image(r)]

    # Dedup: conversation messages covered by a chunk's range → drop
    chunk_ranges = [(ks, ke) for ks, ke in kept_ranges]
    if chunk_ranges:
        def _not_covered_by_chunk(r):
            if r.get("pool") != "conversation":
                return True
            msg_id = r.get("id")
            if not msg_id:
                return True
            return not any(s <= msg_id <= e for s, e in chunk_ranges)
        results = [r for r in results if _not_covered_by_chunk(r)]

    # Time range filtering:
    #   - both after & before → precise range (jionlp parsed "yesterday/last week/May 15"):
    #     hard filter; out-of-range entries are dropped, undated kept.
    #   - only one side → fuzzy range ("recent/last few days"): soft boost only.
    # Day comparison goes through _ts_to_day so stored timestamps in different
    # formats (memories/chunks/conversation_log) all normalise correctly and
    # the optional day-boundary shift applies.
    if after and before:
        kept = []
        for r in results:
            day = _ts_to_day(r.get("created_at", "") or r.get("start_time", ""))
            if not day or after <= day <= before:
                kept.append(r)
        results = kept
    elif after or before:
        _TIME_BOOST = 0.5
        for r in results:
            day = _ts_to_day(r.get("created_at", "") or r.get("start_time", ""))
            if not day:
                continue
            if after and day < after:
                continue
            if before and day > before:
                continue
            r["score"] += _TIME_BOOST
        results.sort(key=lambda x: x["score"], reverse=True)

    # Filter out PRIVATE-tagged memories
    results = [r for r in results if not (r.get("content", "").startswith("[PRIVATE]"))]

    # Filter out conversation-pool entries flagged is_test=1. Chunker already
    # skips test messages so they never become chunks, but conv-pool FTS/LIKE
    # paths can still surface raw test messages — drop them here in one pass.
    conv_ids = [r["id"] for r in results
                if r.get("pool") == "conversation" and r.get("id")]
    if conv_ids:
        try:
            cols = {x["name"] if hasattr(x, "keys") else x[1]
                    for x in db.execute("PRAGMA table_info(conversation_log)").fetchall()}
            if "is_test" in cols:
                placeholders = ",".join("?" for _ in conv_ids)
                test_ids = {
                    row["id"] if hasattr(row, "keys") else row[0]
                    for row in db.execute(
                        f"SELECT id FROM conversation_log "
                        f"WHERE id IN ({placeholders}) AND is_test = 1",
                        conv_ids,
                    ).fetchall()
                }
                if test_ids:
                    results = [r for r in results
                               if not (r.get("pool") == "conversation"
                                       and r.get("id") in test_ids)]
        except Exception:
            pass

    # LLM rerank: take top-20 candidates, ask LLM to score relevance
    if rerank and len(results) > 3:
        results = _llm_rerank(query, results[:20])

    # MMR diversification: prevent "echo amplification" where repeatedly
    # discussing event A produces many near-duplicate chunks that crowd the
    # top, pushing related-but-different events C/D out. Greedy: walk in
    # score order, skip a candidate if any already-kept chunk is too similar
    # to it (cosine >= MMR_SIM_THRESHOLD). Non-chunk results pass through
    # untouched. If everything is similar, falls back to taking by score.
    if len(results) > limit:
        results = _mmr_diversify(results, db, limit_pool=max(limit * 4, 20),
                                  sim_threshold=MMR_SIM_THRESHOLD)

    results = results[:limit]

    # Graph expansion: append edge-connected memories
    if "memory" in pools:
        results = _expand_via_edges(results, db, max_expand=3)

    # Expand chunk results with hybrid keyword + embedding ranked messages
    if any(r.get("pool") == "chunk" for r in results):
        _expand_chunk_hybrid(query, query_vec, results, db)

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

# jionlp prints a sponsor message to stdout on import — swallow it so it
# doesn't pollute hook output, MCP stdio, or scripts that pipe our stdout.
try:
    import contextlib
    import io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        import jionlp as _jio
    _JIO_OK = True
except ImportError:
    _jio = None
    _JIO_OK = False

_FUZZY_TIME_PATTERNS = [
    (r'最近几天|这几天', 7),
    (r'最近', 14),
    (r'上次|前几天|前两天', 14),
    (r'前段时间', 21),
]


def _extract_time_intent(query: str) -> tuple[str, str | None, str | None]:
    """Extract time intent from query, return (cleaned_query, after, before).
    Uses jionlp for precise parsing, falls back to regex for fuzzy terms."""
    today = now_local()

    # Fuzzy patterns that jionlp can't handle
    for pattern, days_back in _FUZZY_TIME_PATTERNS:
        if re.search(pattern, query):
            cleaned = re.sub(pattern, '', query).strip()
            if not cleaned:
                cleaned = query
            after_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
            return cleaned, after_date, None

    # jionlp NER: extracts the time expression *and* its surface text, so we
    # can strip it out of the query before passing to vector/FTS search.
    # Handles 昨天/三周前/4月初/去年冬天/上半年 etc.
    if _JIO_OK:
        try:
            ents = _jio.ner.extract_time(
                query,
                time_base={"year": today.year, "month": today.month, "day": today.day},
            )
            if ents:
                ent = ents[0]
                ent_text = (ent.get("text") or "").strip()
                # Bare time-of-day words ("凌晨", "下午", "晚上"...) by
                # themselves describe a recurring slice of day, not a
                # specific date. jionlp eagerly pins them to today, which
                # collapses every "凌晨 ... 拉屎" style query to zero hits
                # because today's log has nothing. Skip the parse unless
                # the matched surface text actually carries a date anchor.
                _has_date_anchor = bool(re.search(
                    r"\d|[今昨明前后]天|这?[周月年]|去年|前年|明年|"
                    r"[1-9]\d?[日号月]|周[一二三四五六日]",
                    ent_text,
                ))
                if not _has_date_anchor:
                    return query, None, None
                detail = ent.get("detail") or {}
                t = detail.get("time")
                if t and len(t) == 2:
                    after_date = t[0][:10]
                    before_date = t[1][:10]
                    cleaned = query.replace(ent_text, "", 1).strip()
                    if not cleaned:
                        cleaned = query
                    return cleaned, after_date, before_date
        except Exception:
            pass

    return query, None, None


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
    after/before: ISO date strings to filter by time range.
    Auto-detects time intent from query (最近/上次/昨天 etc.)."""
    if not after and not before:
        query, auto_after, auto_before = _extract_time_intent(query)
        if auto_after:
            after = auto_after
        if auto_before:
            before = auto_before
    results = unified_search(query, limit=limit, pools=pools, platform=platform, after=after, before=before)
    locale = os.environ.get("IMPRINT_LOCALE", "en")
    loc = _LOCALE_LABELS.get(locale, _LOCALE_LABELS["en"])
    if not results:
        return loc["empty"]

    _labels = {k: loc[k] for k in ("memory", "bank", "conversation")}
    lines: list[str] = []
    edge_lines: list[str] = []
    seen_em_sigs: set[tuple[str, str]] = set()

    top_score = max((r.get("score", 0) for r in results), default=0)
    display_floor = max(top_score * 0.55, 0.15)
    display_floor_strict = max(top_score * 0.72, 0.15)

    if _JIEBA_OK:
        from jieba import cut as _jcut
        _query_kw = set(w for w in _jcut(query) if len(w.strip()) >= 2 and len(set(w.strip())) > 1)
    else:
        _query_kw = set(query.split())

    for r in results:
        label = _labels.get(r["pool"], r["pool"])
        score = r.get("score", 0)
        content = r.get("content", "")[:120]

        # Memory edge expansions → collect for the "关联" section
        if r.get("source") == "edge":
            rel = r.get("edge_relation", "")
            edge_lines.append(f"[Memory|edge|{rel}] {content}")
            continue

        pool_floor = display_floor_strict if r["pool"] in ("memory", "bank") else display_floor
        if score <= 0 or score < pool_floor:
            continue

        # Filter system manuals from bank results — they match on example
        # code snippets (e.g. `memory_search(query="攀岩")`) and are never
        # what the user is looking for.
        if r["pool"] == "bank" and "guide" in (r.get("source", "") or ""):
            continue

        # For memory/bank: if no query keyword appears in the content,
        # require strong semantic signal (vec_sim > 0.55) to display.
        # Catches noise like "Telegram chat_id" showing up for "平安夜礼物".
        if r["pool"] in ("memory", "bank"):
            vs = r.get("vec_similarity") or 0
            r_content = ((r.get("content", "") or ""))[:150]
            _has_kw = any(t in r_content for t in _query_kw) if _query_kw else True
            if not _has_kw and vs < 0.55:
                continue

        if r["pool"] == "memory":
            cat = r.get("category", "")
            ts = r.get("created_at", "")
            pin = " [pinned]" if r.get("pinned") else ""
            lines.append(f"[{label}|{cat}|{ts}]{pin} ({score:.3f}) {content}")

        elif r["pool"] == "bank":
            src = r.get("source", "")
            lines.append(f"[{label}|{src}] ({score:.3f}) {content}")

        elif r["pool"] == "conversation":
            plat = r.get("platform", "")
            dire = "<-" if r.get("direction") == "in" else "->"
            ts = r.get("created_at", "")
            lines.append(f"[{label}|{plat}{dire}|{ts}] ({score:.3f}) {content}")

        elif r["pool"] == "chunk":
            # Chunk acts as a navigation index — surface the raw originals
            # the chunk points to, not the chunk's own summary. All
            # expansion messages from the same chunk share ONE header
            # (timestamp + score), with the messages indented underneath
            # so the caller can see at a glance "these turns belong to
            # the same conversation". Without the shared header each
            # line carried its own [原文|...] tag and it was impossible
            # to tell which two turns were consecutive vs which jumped
            # to a different chunk. Dedup by (speaker, content) still
            # applies — same line repeated 3-4x collapses to one.
            chunk_lines: list[str] = []
            for em in r.get("expanded", [])[:3]:
                raw = em.get("content", "") or ""
                em_text = _clean_msg_for_display(raw)
                if len(em_text) < 5:
                    continue
                em_sp = em.get("speaker") or ""
                em_text = " ".join(em_text.split())
                sig = (em_sp, em_text)
                if sig in seen_em_sigs:
                    continue
                seen_em_sigs.add(sig)
                chunk_lines.append(f"    {em_sp}: {em_text[:200]}")
            if chunk_lines:
                chunk_ts = (r.get("start_time") or "")[:16]
                lines.append("")  # blank line separates chunks
                lines.append(f"━━ 原文 {chunk_ts}  (score {score:.2f}) ━━")
                lines.extend(chunk_lines)

    # Extra section: intent-routed graph + memory edge expansion
    graph = _intent_graph_section(query, results, limit=2)
    if graph:
        lines.extend(graph)
    if edge_lines:
        lines.append("")
        lines.append("━━ 记忆关联 ━━")
        lines.extend(f"    {el}" for el in edge_lines)

    return "\n".join(lines)


def _cc_context_cutoff(session_id: str) -> str:
    """Find earliest timestamp still visible in the current cc session's
    JSONL transcript. Anything in conversation_log dated at-or-after this
    cutoff is already inside the cc prompt context (either authored in
    the live session, or carried over by a forge-reload), so surfacing
    it back is echo.

    Scans past metadata-only entries (e.g. {"type":"ai-title"} headers cc
    writes at the top of fresh sessions) until it finds the first entry
    that actually carries a `timestamp` field — that's the earliest real
    message and the true cutoff. Hard-limited to the first ~50 lines so a
    pathologically meta-heavy JSONL doesn't open the whole file.

    Returns "" when the session id can't be resolved to a JSONL file.
    """
    if not session_id:
        return ""
    try:
        import json as _json
        projects_dir = Path.home() / ".claude" / "projects"
        if not projects_dir.is_dir():
            return ""
        for jsonl in projects_dir.rglob(f"{session_id}.jsonl"):
            try:
                with jsonl.open(encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= 50:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = _json.loads(line)
                        except Exception:
                            continue
                        ts = obj.get("timestamp") or ""
                        if ts:
                            return ts
            except Exception:
                continue
    except Exception:
        pass
    return ""


def _parse_ts(ts: str):
    """Parse a stored timestamp into a tz-aware datetime in LOCAL_TZ.

    Handles three formats we see in conversation_log / conversation_chunks:
      - '2026-05-18 00:23:12'              (naive — assumed already local)
      - '2026-05-17T13:00:39.855Z'         (UTC ISO 8601)
      - '2026-05-18T00:23:12.345+12:00'    (offset ISO 8601)
    """
    if not ts:
        return None
    s = ts.strip()
    try:
        if "T" in s and (s.endswith("Z") or "+" in s[10:] or "-" in s[10:]):
            iso = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(iso)
            return dt.astimezone(LOCAL_TZ)
        s = s.replace("T", " ")[:19]
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=LOCAL_TZ)
    except Exception:
        return None


# CJK sentence enders + non-period ASCII enders always terminate a sentence.
# ASCII '.' is special-cased — it only counts when followed by whitespace
# or end-of-text so it won't break inside version numbers ("o4.7", "v1.2.3"),
# URLs ("example.com"), or abbreviations ("Dr.").
_SENTENCE_END_CHARS = "。！？!?\n"


def _smart_truncate(text: str, soft_limit: int = 150, hard_limit: int = 240) -> str:
    """Truncate around `soft_limit` chars but extend to the next sentence
    boundary so snippets don't cut mid-thought. Falls back to `hard_limit`
    when no sentence end shows up before then. Returns the unchanged text
    when it's already short enough.

    Note: char-based, not byte-based — `len("。")` is 1 in Python strings
    so the limits are character counts as humans see them.
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) <= soft_limit:
        return text
    upper = min(hard_limit, len(text))
    for i in range(soft_limit, upper):
        ch = text[i]
        if ch in _SENTENCE_END_CHARS:
            return text[: i + 1].rstrip()
        # ASCII '.' only counts as a sentence end when followed by whitespace
        # or end-of-text — otherwise it's likely a decimal, URL, or abbrev.
        if ch == "." and (i + 1 >= len(text) or text[i + 1] in " \t\n"):
            return text[: i + 1].rstrip()
    if len(text) > hard_limit:
        return text[:hard_limit].rstrip() + "…"
    return text


def surfacing_search(query: str, limit: int = 3) -> str:
    """Compact memory surfacing for auto-recall during conversation.
    Target ~400 chars: chunk summaries + top-1 expanded quote + 1 graph link.

    Same-context filter: when the caller exports IMPRINT_CURRENT_SESSION_ID
    (e.g. from a UserPromptSubmit hook that has access to the current cc
    session), surfacing skips anything already visible in cc's prompt
    context — both content authored in the live session AND content that
    a forge-reload carried over from a previous session. Detection: read
    the session's JSONL, take the earliest message timestamp as the
    context cutoff, drop cc/app messages and chunks dated at-or-after it.

    Other channels (claude.ai sync, telegram, wechat) the cc session can't
    actually see are NOT filtered — even if their timestamps are recent.

    Adapter-prefix stripping: removes "[当前对话窗口:...]" / timestamp /
    upload-header / "X 说:" lead-in from the query before searching, so
    they don't confuse vec/FTS or trick jionlp into time-pinning to today."""
    query = (query or "").strip()
    if not query:
        return ""
    if len(set(query)) <= 1:
        return ""
    if _JIEBA_OK:
        from jieba import cut
        meaningful = [w for w in cut(query) if len(w.strip()) >= 2 and len(set(w.strip())) > 1]
        if len(meaningful) < 1:
            return ""
    elif len(query) < 6:
        return ""

    current_session = (os.environ.get("IMPRINT_CURRENT_SESSION_ID") or "").strip()

    # Session-level dedup: don't re-surface the same memory/chunk twice.
    _surfaced_dir = DATA_DIR / ".surfaced"
    _surfaced_file = (_surfaced_dir / f"{current_session}.ids") if current_session else None
    _seen_surfaced: set[str] = set()
    if _surfaced_file and _surfaced_file.exists():
        _seen_surfaced = set(_surfaced_file.read_text().split())

    # Strip channel-adapter prefixes.
    query = re.sub(r"\[当前对话窗口:[^\]]*\]", "", query)
    query = re.sub(r"\[\d{4}-\d{2}-\d{2}[^\]]*\]", "", query)
    query = re.sub(r"\[[^\]]*上传了一个文件:[^\]]*\]", "", query)
    query = re.sub(r"^\s*[\w]+说:\s*", "", query.strip())
    query = query.strip()
    if not query:
        return ""

    # Social/status messages: short announcements of personal actions
    # or greetings — no recall intent, skip surfacing.
    if re.match(r"^(我去|我要去|去).{1,6}(了|啦|咯|去了)$", query):
        return ""
    if re.match(r"^(谢谢|感谢|多谢|辛苦|晚安|早安|拜拜|再见|回来了).{0,6}$", query):
        return ""

    # Surfacing ignores time extraction — when someone says "我今天又去
    # 攀岩了", the valuable recall is climbing history (past), not today's
    # messages. Time words provide context, not search constraints.
    search_query, _, _ = _extract_time_intent(query)

    inner_limit = max(limit, 12) if current_session else limit
    results = unified_search(
        search_query or query,
        limit=inner_limit,
        rerank=False,
        pools=["memory", "bank", "chunk"],
    )
    if not results:
        return ""

    # Same-context filter. Combines two signals:
    #   1) JSONL cutoff — anything dated at-or-after the cc session's earliest
    #      visible message is in cc's prompt context (live session OR
    #      forge-reload carry-over). Filter cc/app conv messages and chunks
    #      whose timestamp crosses the cutoff. Other channels are spared.
    #   2) session_id fallback — for entries whose timestamp doesn't parse
    #      cleanly, still catch ones whose session_id literally matches.
    if current_session:
        cc_cutoff_dt = _parse_ts(_cc_context_cutoff(current_session))
        db = _get_db()
        try:
            kept = []
            for r in results:
                pool = r.get("pool")
                if pool == "conversation":
                    platform = (r.get("platform") or "")
                    ts_dt = _parse_ts(r.get("created_at", "") or "")
                    if cc_cutoff_dt and ts_dt and ts_dt >= cc_cutoff_dt and platform in ("cc", "app"):
                        continue
                    msg_id = r.get("id")
                    if msg_id:
                        row = db.execute(
                            "SELECT 1 FROM conversation_log WHERE id = ? AND session_id = ? LIMIT 1",
                            (msg_id, current_session),
                        ).fetchone()
                        if row:
                            continue
                elif pool == "chunk":
                    ts_dt = _parse_ts(r.get("start_time", "") or "")
                    if cc_cutoff_dt and ts_dt and ts_dt >= cc_cutoff_dt:
                        continue
                    sid = r.get("start_msg_id")
                    eid = r.get("end_msg_id")
                    if sid and eid:
                        row = db.execute(
                            "SELECT 1 FROM conversation_log "
                            "WHERE id BETWEEN ? AND ? AND session_id = ? LIMIT 1",
                            (sid, eid, current_session),
                        ).fetchone()
                        if row:
                            continue
                kept.append(r)
            results = kept
        finally:
            db.close()
        if not results:
            return ""

    top = results[:limit]
    if not any(r.get("pool") == "chunk" for r in top):
        fallback_chunk = next(
            (r for r in results[limit:]
             if r.get("pool") == "chunk" and r.get("score", 0) > 0),
            None,
        )
        if fallback_chunk:
            non_chunk = [r for r in top if r.get("pool") != "chunk"]
            if non_chunk:
                to_drop = min(non_chunk, key=lambda r: r.get("score", 0))
                top = [r for r in top if r is not to_drop]
                top.append(fallback_chunk)
                top.sort(key=lambda r: r.get("score", 0), reverse=True)
    results = top

    # Task-instruction skip: queries asking for help/search/how-to are
    # not recall requests — skip surfacing entirely.
    _task_q = search_query or query
    if re.match(r"^(帮我|请|麻烦|你帮我).{0,6}(看|搜|查|找|写|改|做|修|算|翻译)", _task_q):
        return ""
    if re.search(r"怎么(写|做|用|弄|装|配|设)", _task_q):
        return ""

    # Keyword precision gate: require at least one meaningful query
    # keyword to appear in the top results.
    # Curated functional word list — words that never carry topic signal.
    # Organized by linguistic category, not by test case.
    # High-frequency corpus words are handled separately
    # by filter_stopwords() from the DB stopwords table.
    _SURFACING_STOPS = {
        # Connectives / adverbs
        "一下", "一些", "一样", "一点", "还是", "已经", "然后",
        "但是", "因为", "所以", "如果", "虽然", "比较", "应该",
        # Pronouns / demonstratives
        "这个", "那个", "这些", "那些",
        # Interrogative particles (question words, not topics)
        "怎么", "什么", "怎么样", "怎样", "如何", "多少", "有没有",
        # Interjections
        "哎呀", "哎", "呀", "唉", "嘛", "喔", "哇",
        # Modal / pragmatic particles (acknowledgments, giving up, etc.)
        "可以", "好的", "收到", "算了", "不想", "不要", "不用",
        # Action verbs without topic content
        "帮我", "看看", "想要", "不是",
        # English function words
        "ok", "let", "me", "try", "the", "is", "it", "do", "and",
        "yes", "no", "oh", "hi", "lol", "haha", "sure",
    }
    # Note: don't use filter_stopwords() here — its fallback (return
    # terms[:1] when all filtered) defeats the empty-keyword guard.
    _db_stops = get_stopwords()
    if _JIEBA_OK:
        from jieba import cut as _jcut
        _q_raw = [w for w in _jcut(_task_q) if len(w.strip()) >= 2 and len(set(w.strip())) > 1]
        _q_terms = set(w for w in _q_raw if w.lower() not in _SURFACING_STOPS and w.lower() not in _db_stops)
    else:
        # No jieba → segment_cjk falls back to char-level: the len>=2 word
        # filter would wipe every CJK term and the gate below returns "" for
        # any Chinese query. Keep single chars when word-level yields nothing.
        _seg = [w for w in segment_cjk(_task_q).split() if w.strip()]
        _q_terms = set(w for w in _seg if len(w) >= 2 and w.lower() not in _SURFACING_STOPS and w.lower() not in _db_stops)
        if not _q_terms:
            _q_terms = set(w for w in _seg if w.lower() not in _SURFACING_STOPS and w.lower() not in _db_stops)
    if not _q_terms:
        return ""
    _top_content = " ".join(
        (r.get("content", "") or "") + " " + (r.get("summary", "") or "")
        for r in results[:3]
    )
    _matched = sum(1 for t in _q_terms if t in _top_content)
    if _matched == 0:
        # Fallback: try expanded synonyms. "平安夜" → "圣诞节" bridging.
        _expanded = _expand_query(_task_q)
        if _expanded != _task_q:
            _exp_raw = [w for w in _jcut(_expanded) if len(w.strip()) >= 2 and len(set(w.strip())) > 1] if _JIEBA_OK else [w for w in segment_cjk(_expanded).split() if w.strip()]
            _exp_terms = set(w for w in _exp_raw if w.lower() not in _SURFACING_STOPS and w.lower() not in _db_stops)
            _matched = sum(1 for t in _exp_terms if t in _top_content)
        if _matched == 0:
            return ""

    lines = []
    locale = os.environ.get("IMPRINT_LOCALE", "en")
    mem_label = "记忆" if locale == "zh" else "Memory"
    conv_label = "对话" if locale == "zh" else "Chat"
    # Dedup expanded messages by (speaker, text) — same line repeated 3-4
    # times (e.g. "@gemini …" tapped over and over) should appear once.
    seen_em_sigs: set[tuple[str, str]] = set()

    def _oneline(s: str) -> str:
        # Collapse embedded newlines + runs of whitespace so each snippet
        # renders on a single line — multi-line raw messages otherwise
        # punched empty lines into the surfacing block.
        return " ".join((s or "").split())

    def _emit_expansion(e: dict) -> None:
        img = f" [📷 {e['image_path']}]" if e.get("image_path") else ""
        body = _oneline(_smart_truncate(e["content"]))
        sp = e.get("speaker") or ""
        sig = (sp, body)
        if sig in seen_em_sigs:
            return
        seen_em_sigs.add(sig)
        lines.append(f"   原文截取：{sp}: {body}{img}")

    idx = 0  # 1-based item counter for chunk/memory/conv hits
    _newly_surfaced: list[str] = []
    for r in results:
        if r.get("score", 0) <= 0 or r.get("source") == "edge":
            continue
        rid = str(r.get("id", ""))
        if rid and rid in _seen_surfaced:
            continue
        if rid:
            _newly_surfaced.append(rid)
        idx += 1

        if r["pool"] == "memory":
            content = _oneline(_smart_truncate(r.get("content", "") or ""))
            ts = (r.get("created_at", "") or "")[:10]
            lines.append(f"{idx}. [{mem_label}|{ts}] {content}")

        elif r["pool"] == "chunk":
            summary = _oneline(_smart_truncate(r.get("summary", r.get("content", "")) or ""))
            ts = (r.get("start_time", "") or "")[:10]
            lines.append(f"{idx}. [{ts}] {summary}")
            expanded = r.get("expanded", [])
            # When the chunk's expansion contains an image-anchored unit, show
            # that whole unit (image + neighbour turns); otherwise show the
            # single top expanded message as before.
            anchor_idx = next((i for i, e in enumerate(expanded) if e.get("image_path")), -1)
            if anchor_idx >= 0:
                lo = max(0, anchor_idx - 1)
                hi = min(len(expanded), anchor_idx + 2)
                for e in expanded[lo:hi]:
                    _emit_expansion(e)
            elif expanded:
                # Show top 3 expanded messages (already ranked by hybrid keyword
                # + per-message embedding inside _expand_chunk_hybrid). Earlier
                # this used expanded[0] only and wasted the ranking work.
                for e in expanded[:3]:
                    _emit_expansion(e)

        elif r["pool"] == "conversation":
            content = _oneline(_smart_truncate(_clean_msg_for_display(r.get("content", "") or "")))
            sp = r.get("speaker") or (USER_NAME if r.get("direction") == "in" else AGENT_NAME)
            ts = (r.get("created_at", "") or "")[:10]
            lines.append(f"{idx}. [{conv_label}|{ts}] {sp}: {content}")

    if not lines:
        return ""

    graph_lines = _intent_graph_surfacing(query, results)
    if graph_lines:
        lines.extend(graph_lines)

    # Persist newly surfaced IDs for session dedup.
    if _surfaced_file and _newly_surfaced:
        _surfaced_dir.mkdir(parents=True, exist_ok=True)
        with _surfaced_file.open("a") as f:
            for rid in _newly_surfaced:
                f.write(f"{rid}\n")

    return "\n".join(lines)


_GRAPH_EDGE_FLOOR = 0.75

# ─── Intent-Routed Graph ────────────────────────────────
# Default: graph section is empty. Only fires when query matches
# a specific intent signal (origin / evolution / causal / timeline).

_INTENT_CAUSAL = re.compile(
    r"为什么|为啥|为何|因为.{0,30}所以"
)
_INTENT_TEMPORAL_ORIGIN = re.compile(
    r"(?:怎么|什么时候).{0,2}(认识|遇见|碰上|开始|来到|找到我|在一起|相识)|"
    r"最早|第一次|起源|从哪.{0,3}开始"
)
_INTENT_TEMPORAL_EVOLUTION = re.compile(
    r"怎么(演进|演变|发展|变化|变成)|演进|演变|历程|过程"
)
_INTENT_TIMELINE = re.compile(
    r"我们都.{0,3}(聊|做|经历|讨论|想)|经历过|历史|都聊了什么|大事"
)


def _detect_intent(query: str, use_llm: bool = False) -> str | None:
    """Regex fast path, optional CF LLM fallback for ambiguous queries."""
    if _INTENT_CAUSAL.search(query):
        return "causal"
    if _INTENT_TEMPORAL_ORIGIN.search(query):
        return "temporal_origin"
    if _INTENT_TEMPORAL_EVOLUTION.search(query):
        return "temporal_evolution"
    if _INTENT_TIMELINE.search(query):
        return "timeline"
    if use_llm and _CF_ACCOUNT_ID and _CF_API_TOKEN:
        return _detect_intent_llm(query)
    return None


_INTENT_LLM_PROMPT = """这是一个对话记忆搜索系统。普通搜索用向量相似度就能回答（找到"像这个"的内容）。
但有些问题向量搜索答不了——需要时间排序、因果追溯或全局概览。判断下面的搜索属于哪种。

只输出一个词。绝大多数搜索是 none。

temporal_origin — 问"最初/一开始/怎么开始的/起源/从哪来的"，答案是最早的相关记录
  ✓ "我们刚开始的时候" "最初怎么聊起来的" "故事从哪里开始" "回到最初"
  ✗ "我们聊了什么"（不是问起源）"最近怎么样"（不是问最早）

temporal_evolution — 问"怎么一步步变成现在这样/发展过程/变化轨迹"，答案是时间线上的变化
  ✓ "这个功能怎么一步步做出来的" "从开始到现在经历了什么变化" "后来怎么样了"
  ✗ "现在的状态是什么"（只问当前，不问变化）

causal — 问"为什么/什么原因/是什么导致了"，答案是因果链
  ✓ "背后的原因" "是什么让她做出这个选择" "导致这个结果的"
  ✗ "她选了什么"（问事实，不问原因）

timeline — 问"总体回顾/大事记/梳理时间线/都经历了什么"，答案是跨月份的全局概览
  ✓ "回顾一下这一年" "梳理一下时间线" "总结聊过的话题"
  ✗ "最近聊了什么"（只问近期，不是全局回顾）

none — 普通搜索，向量相似度就够了。这是默认值，不确定就选 none。
  ✓ "攀岩" "吃什么" "生日" "拉屎" "照片"

搜索词：{query}
意图："""

_VALID_INTENTS = {"temporal_origin", "temporal_evolution", "causal", "timeline"}


def _detect_intent_llm(query: str) -> str | None:
    try:
        url = (f"https://api.cloudflare.com/client/v4/accounts/{_CF_ACCOUNT_ID}"
               f"/ai/run/{_CF_RERANK_MODEL}")
        payload = json.dumps({"messages": [
            {"role": "user", "content": _INTENT_LLM_PROMPT.format(query=query)},
        ], "max_tokens": 20}).encode()
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Authorization": f"Bearer {_CF_API_TOKEN}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        answer = (data.get("result", {}).get("response") or "").strip().lower()
        for intent in _VALID_INTENTS:
            if intent in answer:
                return intent
    except Exception:
        pass
    return None


def _scored_all_chunks(query_vec: list[float], db, top_n: int = 30) -> list[dict]:
    """Full-table vec scan of conversation_chunks via numpy batch matmul."""
    import numpy as _np
    rows = db.execute(
        "SELECT id, start_time, end_time, start_msg_id, end_msg_id, "
        "       msg_count, summary, keywords, embedding "
        "FROM conversation_chunks WHERE embedding IS NOT NULL"
    ).fetchall()
    if not rows:
        return []
    dim = len(query_vec)
    mat = _np.empty((len(rows), dim), dtype=_np.float32)
    valid = _np.ones(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        blob = r["embedding"]
        if not blob or len(blob) // 4 != dim:
            valid[i] = False
            continue
        mat[i] = _np.frombuffer(blob, dtype=_np.float32)
    q = _np.asarray(query_vec, dtype=_np.float32)
    row_norms = _np.linalg.norm(mat, axis=1)
    q_norm = float(_np.linalg.norm(q)) or 1.0
    sims = (mat @ q) / (row_norms * q_norm + 1e-8)
    sims[~valid] = -1.0
    sims[~_np.isfinite(sims)] = -1.0
    order = _np.argsort(-sims)[:top_n]
    return [
        {
            "id": rows[int(idx)]["id"],
            "pool": "chunk",
            "start_time": rows[int(idx)]["start_time"],
            "end_time": rows[int(idx)]["end_time"],
            "start_msg_id": rows[int(idx)]["start_msg_id"],
            "end_msg_id": rows[int(idx)]["end_msg_id"],
            "msg_count": rows[int(idx)]["msg_count"],
            "summary": rows[int(idx)]["summary"],
            "keywords": rows[int(idx)]["keywords"],
            "vec_similarity": float(sims[int(idx)]),
            "score": float(sims[int(idx)]),
        }
        for idx in order
        if sims[int(idx)] > 0
    ]


def _temporal_origin_section(query: str, rrf_results: list[dict], db, limit: int = 2) -> list[str]:
    """Earliest chunks topically related to query. Always full-scan —
    rrf_results are biased by time decay and miss early chunks."""
    qv = _embed(query)
    if not qv:
        return []
    candidates = _scored_all_chunks(qv, db, top_n=30)
    if not candidates:
        return []
    main_ids = {r.get("id") for r in rrf_results[:5] if r.get("pool") == "chunk"}
    candidates = [c for c in candidates if c["id"] not in main_ids]
    candidates.sort(key=lambda c: c.get("start_time") or "")
    lines = []
    for c in candidates[:limit]:
        ts = (c.get("start_time") or "")[:10]
        summary = " ".join((c.get("summary") or "").split())[:180]
        lines.append(f"[Origin|{ts}] {summary}")
    return lines


def _temporal_evolution_section(query: str, rrf_results: list[dict], db, limit: int = 5) -> list[str]:
    """Trace how a topic evolved over time via chunk_edges."""
    seed_chunks = []
    seen_ids: set[int] = set()
    for r in rrf_results:
        if r.get("pool") == "chunk" and r.get("id"):
            seed_chunks.append(r)
            seen_ids.add(r["id"])
    if not seed_chunks:
        return []

    all_events: list[tuple] = []
    for seed in seed_chunks[:3]:
        sid = seed["id"]
        st = seed.get("start_time") or ""
        sm = seed.get("summary") or seed.get("content", "")
        all_events.append((st, sid, sm, True))
        try:
            neighbors = db.execute(
                """SELECT ce.target_id, ce.similarity * ce.strength as edge_score,
                          c.summary, c.start_time
                   FROM chunk_edges ce
                   JOIN conversation_chunks c ON c.id = ce.target_id
                   WHERE ce.source_id = ? AND ce.similarity * ce.strength >= ?
                   ORDER BY edge_score DESC LIMIT 10""",
                (sid, _GRAPH_EDGE_FLOOR),
            ).fetchall()
            for n in neighbors:
                tid = n["target_id"]
                if tid in seen_ids:
                    continue
                seen_ids.add(tid)
                all_events.append((n["start_time"] or "", tid, n["summary"] or "", False))
        except Exception:
            pass

    if len(all_events) < 2:
        return []

    by_day: dict[str, tuple] = {}
    for ev in all_events:
        day = (ev[0] or "")[:10]
        if not day:
            continue
        if day not in by_day or ev[3]:
            by_day[day] = ev
    events = sorted(by_day.values(), key=lambda e: e[0])

    if len(events) < 2:
        return []

    try:
        first_ts = datetime.fromisoformat(events[0][0][:19])
        last_ts = datetime.fromisoformat(events[-1][0][:19])
        if (last_ts - first_ts).days < 14:
            return []
    except (ValueError, IndexError):
        pass

    seed_idx = next((i for i, e in enumerate(events) if e[3]), 0)
    lo = max(0, seed_idx - 2)
    hi = min(len(events), seed_idx + 3)
    window = events[lo:hi]

    main_ids = {r.get("id") for r in rrf_results[:5] if r.get("pool") == "chunk"}
    lines = []
    for ts_str, cid, summary, is_seed in window:
        if cid in main_ids:
            continue
        ts = ts_str[:10]
        sm = " ".join(summary.split())[:120]
        marker = " ★" if is_seed else ""
        lines.append(f"[Evo|{ts}] {sm}{marker}")
    return lines[:limit]


def _causal_section(query: str, rrf_results: list[dict], db, limit: int = 3) -> list[str]:
    """Light causal: topically related chunks sorted by time ASC.
    Always full-scan — rrf_results with small limit dedup everything."""
    qv = _embed(query)
    if not qv:
        return []
    candidates = _scored_all_chunks(qv, db, top_n=20)
    if not candidates:
        return []
    candidates.sort(key=lambda c: c.get("start_time") or "")
    main_ids = {r.get("id") for r in rrf_results[:5] if r.get("pool") == "chunk"}
    lines = []
    for c in candidates:
        if c.get("id") in main_ids:
            continue
        ts = (c.get("start_time") or "")[:10]
        summary = " ".join((c.get("summary") or "").split())[:180]
        lines.append(f"[Causal|{ts}] {summary}")
        if len(lines) >= limit:
            break
    return lines


def _timeline_section(query: str, rrf_results: list[dict], db, limit: int = 5) -> list[str]:
    """Month-sampled event overview. Full-scan to avoid small-limit dedup trap."""
    qv = _embed(query)
    if not qv:
        return []
    candidates = _scored_all_chunks(qv, db, top_n=50)
    if not candidates:
        return []
    by_month: dict[str, dict] = {}
    for c in candidates:
        ts = (c.get("start_time") or "")[:7]
        if not ts:
            continue
        if ts not in by_month or (c.get("score", 0) > by_month[ts].get("score", 0)):
            by_month[ts] = c
    picked = sorted(by_month.values(), key=lambda r: r.get("start_time") or "")
    main_ids = {r.get("id") for r in rrf_results[:5] if r.get("pool") == "chunk"}
    lines = []
    for c in picked:
        if c.get("id") in main_ids:
            continue
        ts = (c.get("start_time") or "")[:7]
        summary = " ".join((c.get("summary") or "").split())[:120]
        lines.append(f"[Timeline|{ts}] {summary}")
        if len(lines) >= limit:
            break
    return lines


_INTENT_BANNERS = {
    "temporal_origin": "━━ 最早的相关对话 ━━",
    "temporal_evolution": "━━ 演进时间线 ━━",
    "causal": "━━ 相关事件链 ━━",
    "timeline": "━━ 大事记 ━━",
}
_INTENT_BANNERS_SHORT = {
    "temporal_origin": "最早的相关：",
    "temporal_evolution": "演进：",
    "causal": "相关事件：",
    "timeline": "大事记：",
}


def _format_graph_lines(raw_lines: list[str]) -> list[str]:
    """Strip [Type|ts] prefix, return clean '  ts  summary' lines."""
    out = []
    for line in raw_lines:
        if line.startswith("[") and "] " in line:
            tag = line[1:line.index("]")]
            ts = tag.split("|", 1)[1] if "|" in tag else ""
            text = line[line.index("] ") + 2:]
            out.append(f"    {ts}  {text}")
        else:
            out.append(f"    {line}")
    return out


def _intent_graph_section(query: str, rrf_results: list[dict], limit: int = 2) -> list[str]:
    """Entry point replacing _graph_expansion_section. Routes by intent.
    Uses LLM fallback for ambiguous queries (regex miss → ask CF)."""
    intent = _detect_intent(query, use_llm=True)
    if not intent:
        return []
    db = _get_db()
    try:
        if intent == "temporal_origin":
            raw = _temporal_origin_section(query, rrf_results, db, limit=limit)
        elif intent == "temporal_evolution":
            raw = _temporal_evolution_section(query, rrf_results, db, limit=limit + 3)
        elif intent == "causal":
            raw = _causal_section(query, rrf_results, db, limit=limit + 1)
        elif intent == "timeline":
            raw = _timeline_section(query, rrf_results, db, limit=limit + 3)
        else:
            raw = []
        if not raw:
            return []
        lines = ["", _INTENT_BANNERS[intent]]
        lines.extend(_format_graph_lines(raw))
        return lines
    except Exception:
        return []
    finally:
        db.close()


def _intent_graph_surfacing(query: str, rrf_results: list[dict]) -> list[str]:
    """Compact intent-routed graph for surfacing (~400 char budget)."""
    intent = _detect_intent(query)
    if not intent:
        return []
    db = _get_db()
    try:
        if intent == "temporal_origin":
            raw = _temporal_origin_section(query, rrf_results, db, limit=2)
        elif intent == "temporal_evolution":
            raw = _temporal_evolution_section(query, rrf_results, db, limit=4)
        elif intent == "causal":
            raw = _causal_section(query, rrf_results, db, limit=3)
        elif intent == "timeline":
            raw = _timeline_section(query, rrf_results, db, limit=4)
        else:
            raw = []
        if not raw:
            return []
        lines = [_INTENT_BANNERS_SHORT[intent]]
        for line in _format_graph_lines(raw):
            lines.append(line.strip()[:80])
        return lines
    except Exception:
        return []
    finally:
        db.close()


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
