"""Rebuild conversation_log_fts so the keyword index excludes <think> blocks.

One-shot migration. Assistant replies imported from chat platforms (e.g. the
claude.ai browser extension) can carry hidden reasoning inline as
<think>...</think>, prepended before the visible reply. Builds before the
strip-think fix indexed that whole field into FTS5, so keyword search recalled
messages that only mention a term inside their thinking and buried the real
replies. The fix strips <think> at the index/display layer for new rows; rows
ingested earlier stay polluted until re-indexed.

This re-indexes them. It only touches conversation_log_fts; conversation_log
(the source of truth) is never modified, so it is safe to re-run.

Run after upgrading, then restart your memory server:
    imprint-rebuild-fts                 # uses the configured DB (IMPRINT_DB / IMPRINT_DATA_DIR)
    imprint-rebuild-fts /path/to/memory.db
"""

import sqlite3
import sys

from .db import DB_PATH, segment_cjk, strip_think


def rebuild(db_path: str | None = None) -> None:
    path = db_path or str(DB_PATH)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=10000")

    total = db.execute("SELECT COUNT(*) FROM conversation_log").fetchone()[0]
    with_think = db.execute(
        "SELECT COUNT(*) FROM conversation_log WHERE content LIKE '%<think>%'"
    ).fetchone()[0]
    print(f"DB: {path}")
    print(f"conversation_log: {total} rows, {with_think} carry <think>")
    if total == 0:
        print("nothing to do.")
        db.close()
        return

    # external-content FTS5: clear the whole index, then re-feed every row using
    # the same expression the corrected triggers use (strip_think then segment).
    # Direct inserts into the FTS table don't fire the conversation_log triggers,
    # so segmentation/stripping is done here in Python.
    print("clearing conversation_log_fts ('delete-all') ...")
    db.execute("INSERT INTO conversation_log_fts(conversation_log_fts) VALUES('delete-all')")

    print("re-indexing (strip_think + segment_cjk) ...")
    n = 0
    for row in db.execute(
        "SELECT id, content, platform, speaker FROM conversation_log ORDER BY id"
    ):
        db.execute(
            "INSERT INTO conversation_log_fts(rowid, content, platform, speaker) "
            "VALUES (?, ?, ?, ?)",
            (
                row["id"],
                segment_cjk(strip_think(row["content"])),
                row["platform"],
                row["speaker"],
            ),
        )
        n += 1
        if n % 5000 == 0:
            db.commit()
            print(f"  {n}/{total}")
    db.commit()
    print(f"rebuilt FTS index for {n} rows. Restart your memory server next.")
    db.close()


def main() -> None:
    rebuild(sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    main()
