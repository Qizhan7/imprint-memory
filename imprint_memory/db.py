"""
Shared database connection and table initialization.
All imprint-memory modules import _get_db() from here.
"""

import os
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --- Config -----------------------------------------------------------

TZ_OFFSET = int(os.environ.get("TZ_OFFSET", 0))
LOCAL_TZ = timezone(timedelta(hours=TZ_OFFSET))

# Data directory: env var > ~/.imprint/
DATA_DIR = Path(os.environ.get("IMPRINT_DATA_DIR", str(Path.home() / ".imprint"))).expanduser()

# DB path: env var > DATA_DIR/memory.db
DB_PATH = Path(os.environ.get("IMPRINT_DB", str(DATA_DIR / "memory.db"))).expanduser()

# Daily logs and bank files
DAILY_LOG_DIR = DATA_DIR / "memory"
BANK_DIR = DATA_DIR / "memory" / "bank"
MEMORY_INDEX = DATA_DIR / "MEMORY.md"



# --- CJK Segmentation (for FTS5 triggers) --------------------------------

_CJK_RE = re.compile(
    r'([\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df'
    r'\U0002a700-\U0002ebef\u3000-\u303f\uff00-\uffef])'
)

try:
    import jieba as _jieba
    _jieba.setLogLevel(20)
    _userdict = DATA_DIR / "userdict_emotional.txt"
    if not _userdict.exists():
        _userdict = Path(__file__).parent / "data" / "userdict_emotional.txt"
    if _userdict.exists():
        _jieba.load_userdict(str(_userdict))
    _JIEBA_OK = True
except ImportError:
    _JIEBA_OK = False


def sanitize_fts_query(query: str) -> str:
    """Sanitize a query string for FTS5 MATCH.
    Removes operators and special characters that could cause parse errors."""
    # Strip FTS5 operators and special characters
    cleaned = re.sub(r'["\(\)\*\:\^\{\}]', ' ', query)
    # Remove FTS5 boolean keywords when used as operators
    cleaned = re.sub(r'\b(AND|OR|NOT|NEAR)\b', ' ', cleaned)
    # Collapse whitespace
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()


def segment_cjk(text: str) -> str:
    """Segment CJK text for FTS5 indexing.
    With jieba: word-level ("喜欢攀岩" → "喜欢 攀岩")
    Without:    char-level fallback ("喜欢攀岩" → "喜 欢 攀 岩")
    """
    if not text:
        return text or ""
    if _JIEBA_OK:
        return " ".join(_jieba.cut_for_search(text))
    return re.sub(r'\s+', ' ', _CJK_RE.sub(r' \1 ', text)).strip()


# Assistant replies synced from chat platforms (e.g. the claude.ai browser
# extension) can carry their hidden reasoning inline as <think>...</think>
# blocks, prepended *before* the visible reply. Left in place they swamp the
# FTS index: the thinking monologue (often several times longer than the reply)
# dominates term frequency, so keyword search recalls messages that only mention
# a word inside their thinking and buries the real reply text. We keep the full
# content (incl. think) in conversation_log for archival, and strip it only at
# the index/display layer. Registered as a SQLite function so the FTS sync
# triggers can call segment_cjk(strip_think(content)).
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Drop <think>...</think> blocks. Safe on messages without any (returns
    them unchanged) — user turns never carry think blocks, so this is applied
    unconditionally regardless of direction."""
    if not text:
        return text or ""
    return _THINK_RE.sub("", text).strip()


def _get_db() -> sqlite3.Connection:
    """Get database connection, auto-create tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH), timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")
    db.row_factory = sqlite3.Row
    db.create_function("segment_cjk", 1, segment_cjk)
    db.create_function("strip_think", 1, strip_think)
    _init_tables(db)
    return db


def _init_tables(db: sqlite3.Connection):
    """Create all tables (idempotent)."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            source TEXT DEFAULT 'cc',
            tags TEXT DEFAULT '[]',
            importance INTEGER DEFAULT 5,
            recalled_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT,
            superseded_by INTEGER REFERENCES memories(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS memory_vectors (
            memory_id INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
            embedding BLOB NOT NULL,
            model TEXT DEFAULT 'bge-m3'
        );

        CREATE TABLE IF NOT EXISTS daily_logs (
            date TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            summary TEXT,
            embedding BLOB
        );

        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bank_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding BLOB,
            file_mtime REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cc_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            result TEXT,
            source TEXT DEFAULT 'chat',
            session_id TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS message_bus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            direction TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            direction TEXT NOT NULL,
            speaker TEXT DEFAULT '',
            content TEXT NOT NULL,
            external_id TEXT DEFAULT '',
            session_id TEXT DEFAULT '',
            entrypoint TEXT DEFAULT '',
            model TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory_tags (
            memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            tag TEXT NOT NULL,
            PRIMARY KEY (memory_id, tag)
        );
        CREATE INDEX IF NOT EXISTS idx_tags_tag ON memory_tags(tag);

        CREATE TABLE IF NOT EXISTS memory_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            target_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
            relation TEXT NOT NULL,
            context TEXT NOT NULL,
            surfaced_count INTEGER DEFAULT 0,
            used_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE(source_id, target_id)
        );
        CREATE INDEX IF NOT EXISTS idx_edges_source ON memory_edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_edges_target ON memory_edges(target_id);

        CREATE TABLE IF NOT EXISTS stopwords (
            word TEXT PRIMARY KEY,
            doc_freq REAL DEFAULT 0,
            source TEXT DEFAULT 'auto',
            active INTEGER DEFAULT 1,
            updated_at TEXT
        );
    """)

    # Migration: add superseded_by column if missing
    mem_cols = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in db.execute("PRAGMA table_info(memories)").fetchall()
    }
    if "superseded_by" not in mem_cols:
        try:
            db.execute("ALTER TABLE memories ADD COLUMN superseded_by INTEGER REFERENCES memories(id) ON DELETE SET NULL")
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

    # Migration: add last_accessed_at column
    if "last_accessed_at" not in mem_cols:
        try:
            db.execute("ALTER TABLE memories ADD COLUMN last_accessed_at TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise

    # Migration: add optional metadata columns to conversation_log
    convlog_cols = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in db.execute("PRAGMA table_info(conversation_log)").fetchall()
    }
    if "summary" not in convlog_cols:
        try:
            db.execute("ALTER TABLE conversation_log ADD COLUMN summary TEXT DEFAULT ''")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise
    if "model" not in convlog_cols:
        try:
            db.execute("ALTER TABLE conversation_log ADD COLUMN model TEXT DEFAULT ''")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise
    if "external_id" not in convlog_cols:
        try:
            db.execute("ALTER TABLE conversation_log ADD COLUMN external_id TEXT DEFAULT ''")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise

    # Messages whose user-side turn opens with "测试：" / "test:" get
    # is_test=1 so the chunker, surfacing, and search pools all skip them.
    if "is_test" not in convlog_cols:
        try:
            db.execute("ALTER TABLE conversation_log ADD COLUMN is_test INTEGER DEFAULT 0")
            db.execute("CREATE INDEX IF NOT EXISTS idx_convlog_is_test ON conversation_log(is_test)")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise

    db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_conversation_log_external_id
        ON conversation_log(platform, external_id)
        WHERE external_id IS NOT NULL AND external_id != ''
    """)

    # Migration: add session_id column to cc_tasks
    cctask_cols = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in db.execute("PRAGMA table_info(cc_tasks)").fetchall()
    }
    if "session_id" not in cctask_cols:
        try:
            db.execute("ALTER TABLE cc_tasks ADD COLUMN session_id TEXT DEFAULT ''")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise

    # Migration: add index_version column if missing
    bank_chunk_cols = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in db.execute("PRAGMA table_info(bank_chunks)").fetchall()
    }
    if "index_version" not in bank_chunk_cols:
        try:
            db.execute("ALTER TABLE bank_chunks ADD COLUMN index_version INTEGER DEFAULT 1")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise

    # FTS5 full-text index
    try:
        db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(content, category, tags, content=memories, content_rowid=id)
        """)
    except sqlite3.OperationalError:
        pass

    # FTS5 for conversation_log
    try:
        db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS conversation_log_fts
            USING fts5(content, platform, speaker, content=conversation_log, content_rowid=id)
        """)
    except sqlite3.OperationalError:
        pass

    # Drop old triggers so corrected versions (with segment_cjk on delete side) are created
    db.executescript("""
        DROP TRIGGER IF EXISTS memories_ai;
        DROP TRIGGER IF EXISTS memories_ad;
        DROP TRIGGER IF EXISTS memories_au;
        DROP TRIGGER IF EXISTS convlog_ai;
        DROP TRIGGER IF EXISTS convlog_ad;
        DROP TRIGGER IF EXISTS convlog_au;
    """)

    # FTS5 sync triggers (segment_cjk for CJK language support)
    db.executescript("""
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, category, tags)
            VALUES (new.id, segment_cjk(new.content), new.category, new.tags);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, category, tags)
            VALUES ('delete', old.id, segment_cjk(old.content), old.category, old.tags);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, category, tags)
            VALUES ('delete', old.id, segment_cjk(old.content), old.category, old.tags);
            INSERT INTO memories_fts(rowid, content, category, tags)
            VALUES (new.id, segment_cjk(new.content), new.category, new.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS convlog_ai AFTER INSERT ON conversation_log BEGIN
            INSERT INTO conversation_log_fts(rowid, content, platform, speaker)
            VALUES (new.id, segment_cjk(strip_think(new.content)), new.platform, new.speaker);
        END;
        CREATE TRIGGER IF NOT EXISTS convlog_ad AFTER DELETE ON conversation_log BEGIN
            INSERT INTO conversation_log_fts(conversation_log_fts, rowid, content, platform, speaker)
            VALUES ('delete', old.id, segment_cjk(strip_think(old.content)), old.platform, old.speaker);
        END;
        CREATE TRIGGER IF NOT EXISTS convlog_au AFTER UPDATE ON conversation_log BEGIN
            INSERT INTO conversation_log_fts(conversation_log_fts, rowid, content, platform, speaker)
            VALUES ('delete', old.id, segment_cjk(strip_think(old.content)), old.platform, old.speaker);
            INSERT INTO conversation_log_fts(rowid, content, platform, speaker)
            VALUES (new.id, segment_cjk(strip_think(new.content)), new.platform, new.speaker);
        END;
    """)
    db.commit()


# --- Time Utils -------------------------------------------------------

def now_local() -> datetime:
    """Return the current time using the configured local timezone."""
    return datetime.now(LOCAL_TZ)


def now_str() -> str:
    """Return a SQLite-friendly local timestamp."""
    return now_local().strftime("%Y-%m-%d %H:%M:%S")


# --- Init on import ---------------------------------------------------
_db = _get_db()
_db.close()
