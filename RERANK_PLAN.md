# Memory Rerank Implementation Plan

## Background

**What we're keeping**: existing vector (bge-m3) + FTS5 retrieval.
**What we're changing**: rerank layer only — how search results are scored and sorted.

## Final Formula

```python
if pinned:
    final_score = rrf_score
else:
    final_score = rrf_score * time_factor * activation_factor * importance_factor
```

### RRF (Reciprocal Rank Fusion)

Replaces the current linear weighting (`vector*0.4 + fts*0.4 + recency*0.2`).

```python
RRF_K = 60
rrf_score = 1/(RRF_K + rank_fts) + 1/(RRF_K + rank_vec)
# If a memory wasn't found by one channel, that channel contributes 0 (no penalty)
```

Why: eliminates the normalization problem with FTS scores. No more `max_rank` hacks.

### time_factor (0.4 ~ 1.0)

Based on days since **last access** (falls back to created_at if never accessed).

```python
days = days_since_last_access  # use created_at if last_accessed_at is NULL
lam = 0.05 / (importance / 5)
time_factor = 0.4 + 0.6 * math.exp(-lam * days)
```

Effect of importance on decay rate:

| importance | lambda | 1 week | 1 month | 3 months | 6 months |
|-----------|--------|--------|---------|----------|----------|
| 1         | 0.250  | 0.51   | 0.40    | 0.40     | 0.40     |
| 3         | 0.083  | 0.83   | 0.49    | 0.40     | 0.40     |
| 5         | 0.050  | 0.88   | 0.61    | 0.41     | 0.40     |
| 7         | 0.036  | 0.91   | 0.70    | 0.46     | 0.40     |
| 10        | 0.025  | 0.94   | 0.79    | 0.55     | 0.43     |
| pinned    | 0      | 1.00   | 1.00    | 1.00     | 1.00     |

Floor of 0.4 means nothing truly disappears — a strong keyword/vector match can always rescue an old memory.

### activation_factor (0.8 ~ 1.0)

Memories that are frequently recalled decay slower.

```python
activation_factor = 0.8 + 0.2 * (math.log(recalled_count + 1) / math.log(51))
```

| recalled_count | factor |
|---------------|--------|
| 0             | 0.80   |
| 1             | 0.84   |
| 5             | 0.89   |
| 10            | 0.92   |
| 50            | 1.00   |

Narrow range (0.8~1.0) — a gentle boost, not a dominant signal.

### importance_factor (0.73 ~ 1.0)

```python
importance_factor = 0.7 + 0.3 * (importance / 10)
```

| importance | factor |
|-----------|--------|
| 1         | 0.73   |
| 5         | 0.85   |
| 7         | 0.91   |
| 10        | 1.00   |

### Combined examples

```
"Our first date at xxx" — intimate, importance=9, recalled=8, 3 months ago, last accessed 5 days ago
  time:       0.4 + 0.6 * e^(-0.028 * 5)  = 0.97
  activation: 0.8 + 0.2 * (log9/log51)     = 0.91
  importance: 0.7 + 0.3 * 0.9              = 0.97
  multiplier: 0.97 * 0.91 * 0.97 = 0.856

"Had ramen for lunch" — diary, importance=2, recalled=0, 3 months ago, never accessed
  time:       0.4 + 0.6 * e^(-0.125 * 90)  = 0.40 (hit floor)
  activation: 0.80
  importance: 0.76
  multiplier: 0.40 * 0.80 * 0.76 = 0.243

"Project uses Python 3.11 + SQLite" — factual, importance=5, recalled=3, 6 months ago, accessed 2 weeks ago
  time:       0.4 + 0.6 * e^(-0.05 * 14)   = 0.70
  activation: 0.8 + 0.2 * (log4/log51)      = 0.86
  importance: 0.85
  multiplier: 0.70 * 0.86 * 0.85 = 0.512

Pinned memory "relationship milestone" — any age, any recall count
  multiplier: 1.0 (always)
```

---

## Implementation Steps

### Step 1: DB migration (db.py)

Add to `_init_tables` CREATE TABLE statement:

```sql
last_accessed_at TEXT,
pinned INTEGER DEFAULT 0
```

Add migration block (same pattern as existing migrations):

```python
# Migration: add last_accessed_at column
if "last_accessed_at" not in mem_cols:
    try:
        db.execute("ALTER TABLE memories ADD COLUMN last_accessed_at TEXT")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise

# Migration: add pinned column
if "pinned" not in mem_cols:
    try:
        db.execute("ALTER TABLE memories ADD COLUMN pinned INTEGER DEFAULT 0")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise
```

### Step 2: Replace search scoring (memory_manager.py)

**Delete** these constants:

```python
WEIGHT_VECTOR = 0.4
WEIGHT_FTS = 0.4
WEIGHT_RECENCY = 0.2
```

**Delete** `_recency_score()` function.

**Add** new constants and rerank function:

```python
RRF_K = 60

def _rerank_score(rrf_score: float, row: dict) -> float:
    """Apply time decay, activation boost, and importance weighting to RRF score."""
    if row.get("pinned"):
        return rrf_score

    # Days since last access (fallback to created_at)
    ref_time = row.get("last_accessed_at") or row.get("created_at", "")
    try:
        t = datetime_strptime(ref_time)
        days = max(0, (now_local() - t).total_seconds() / 86400)
    except (ValueError, TypeError):
        days = 30  # safe default

    importance = row.get("importance", 5)
    recalled_count = row.get("recalled_count", 0)

    # time_factor: 0.4 ~ 1.0
    lam = 0.05 / (importance / 5) if importance > 0 else 0.25
    time_factor = 0.4 + 0.6 * math.exp(-lam * days)

    # activation_factor: 0.8 ~ 1.0
    activation_factor = 0.8 + 0.2 * (math.log(recalled_count + 1) / math.log(51))

    # importance_factor: 0.73 ~ 1.0
    importance_factor = 0.7 + 0.3 * (importance / 10)

    return rrf_score * time_factor * activation_factor * importance_factor
```

### Step 3: Rewrite search() scoring section (memory_manager.py)

Replace the current "Combined scoring" block (roughly lines 348-378) with:

```python
# --- RRF Fusion ---

# Rank FTS results
fts_ranked = sorted(results.values(), key=lambda x: x.get("fts_raw_rank", 0))
for rank, item in enumerate(fts_ranked):
    item["rrf_fts"] = 1.0 / (RRF_K + rank + 1) if item.get("fts_raw_rank") else 0.0

# Rank vector results
vec_ranked = sorted(results.values(), key=lambda x: x.get("vec_score", 0), reverse=True)
for rank, item in enumerate(vec_ranked):
    item["rrf_vec"] = 1.0 / (RRF_K + rank + 1) if item.get("vec_score", 0) > 0 else 0.0

# Combine
for mid, info in results.items():
    rrf_score = info.get("rrf_fts", 0) + info.get("rrf_vec", 0)
    info["final_score"] = _rerank_score(rrf_score, info)

ranked = sorted(results.values(), key=lambda x: x["final_score"], reverse=True)
ranked = ranked[:limit]
```

Note: store the raw FTS rank during FTS search (add `fts_raw_rank` field when building results dict from FTS rows). Also need to fetch `last_accessed_at` and `pinned` in the SQL queries.

### Step 4: Update last_accessed_at on search hit (memory_manager.py)

Change the existing recalled_count update:

```python
# Before:
db.execute(
    "UPDATE memories SET recalled_count = recalled_count + 1 WHERE id = ?",
    (r["id"],),
)

# After:
db.execute(
    "UPDATE memories SET recalled_count = recalled_count + 1, last_accessed_at = ? WHERE id = ?",
    (now_str(), r["id"]),
)
```

### Step 5: Add pinned support (server.py + memory_manager.py)

**memory_manager.py** — add pin/unpin functions:

```python
def pin_memory(memory_id: int) -> dict:
    db = _get_db()
    db.execute("UPDATE memories SET pinned = 1 WHERE id = ?", (memory_id,))
    db.commit()
    db.close()
    return {"ok": True, "pinned": memory_id}

def unpin_memory(memory_id: int) -> dict:
    db = _get_db()
    db.execute("UPDATE memories SET pinned = 0 WHERE id = ?", (memory_id,))
    db.commit()
    db.close()
    return {"ok": True, "unpinned": memory_id}
```

**server.py** — register as MCP tools with description:

> pinned is ONLY for identity-level core memories: relationship milestones, core promises, boundaries. Expected total: ~20 or fewer.

### Step 6: Simplify decay() (memory_manager.py)

Change `decay()` from "reduce importance by 1" to "archive cleanup":

```python
def decay(days: int = 90, dry_run: bool = True) -> dict:
    """Archive memories that have been inactive for too long.
    Criteria: no access in N days + importance < 3 + recalled_count < 2 + not pinned."""
    db = _get_db()
    cutoff = (now_local() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
    now = now_str()

    rows = db.execute("""
        SELECT id, content, category, importance, recalled_count, created_at
        FROM memories
        WHERE COALESCE(last_accessed_at, created_at) < ?
            AND importance < 3 AND recalled_count < 2
            AND pinned = 0 AND superseded_by IS NULL
        ORDER BY created_at ASC
    """, (cutoff,)).fetchall()

    archived = []
    for r in rows:
        archived.append({"id": r["id"], "content": r["content"][:100]})
        if not dry_run:
            db.execute(
                "UPDATE memories SET superseded_by = -1, updated_at = ? WHERE id = ?",
                (now, r["id"]),
            )

    if not dry_run:
        db.commit()
    db.close()
    return {"dry_run": dry_run, "archived": len(archived), "details": archived[:20]}
```

### Step 7: Periodic review task

Set up a weekly cc_execute prompt (via cron or manual):

```
Review all memories:
1. Check importance distribution — flag if >30% are importance >= 8
2. Check pinned count — flag if > 20
3. Find high importance + 0 recalled_count for 30+ days — may be over-scored
4. Find low importance + high recalled_count — may be under-scored
5. Write review summary to daily_log
```

---

## What NOT to change

- Vector generation (`_embed()`, `_embed_ollama()`, `_embed_openai()`)
- FTS5 index and triggers
- Bank file search (`_search_bank()`)
- Storage structure (SQLite + WAL)
- Dedup logic (supersede mechanism)
- Daily log, message bus, task queue, conversation log

## Estimated changes

| File | Changes |
|------|---------|
| db.py | +2 migrations, +2 columns to CREATE TABLE (~15 lines) |
| memory_manager.py | new `_rerank_score()` (~25 lines), rewrite search scoring (~30 lines), update `decay()` (~20 lines), add pin/unpin (~15 lines), delete old code (~20 lines) |
| server.py | add pin/unpin tools, update tool descriptions (~20 lines) |

Total: ~100 lines changed.
