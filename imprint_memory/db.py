"""
Shared database connection and table initialization.
All imprint-memory modules import _get_db() from here.
"""

import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --- Config -----------------------------------------------------------

TZ_OFFSET = int(os.environ.get("TZ_OFFSET", 0))
LOCAL_TZ = timezone(timedelta(hours=TZ_OFFSET))

# Data directory: env var > ~/.imprint/
DATA_DIR = Path(os.environ.get("IMPRINT_DATA_DIR", Path.home() / ".imprint"))

# DB path: env var > DATA_DIR/memory.db
DB_PATH = Path(os.environ.get("IMPRINT_DB", str(DATA_DIR / "memory.db")))

# Daily logs and bank files
DAILY_LOG_DIR = DATA_DIR / "memory"
BANK_DIR = DATA_DIR / "memory" / "bank"
MEMORY_INDEX = DATA_DIR / "MEMORY.md"


def _get_db() -> sqlite3.Connection:
    """Get database connection, auto-create tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH), timeout=10)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")
    db.row_factory = sqlite3.Row
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
            updated_at TEXT
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
    """)

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

    # FTS5 sync triggers
    db.executescript("""
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, content, category, tags)
            VALUES (new.id, new.content, new.category, new.tags);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, category, tags)
            VALUES ('delete', old.id, old.content, old.category, old.tags);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, content, category, tags)
            VALUES ('delete', old.id, old.content, old.category, old.tags);
            INSERT INTO memories_fts(rowid, content, category, tags)
            VALUES (new.id, new.content, new.category, new.tags);
        END;
    """)
    db.commit()


# --- Time Utils -------------------------------------------------------

def now_local() -> datetime:
    return datetime.now(LOCAL_TZ)


def now_str() -> str:
    return now_local().strftime("%Y-%m-%d %H:%M")


# --- Init on import ---------------------------------------------------
_db = _get_db()
_db.close()
