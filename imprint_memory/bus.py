"""
Cross-channel message bus.
Shared "whiteboard" for logging messages sent/received across all channels.
"""

import os
from .db import _get_db, now_str

MESSAGE_BUS_LIMIT = int(os.environ.get("MESSAGE_BUS_LIMIT", 40))


def bus_post(source: str, direction: str, content: str) -> None:
    """Write a message to the bus. Auto-prunes old messages beyond limit.
    source: telegram/wechat/chat/cc_task/scheduled/heartbeat
    direction: in (user sent) / out (Claude sent)
    content: message content (auto-truncated to 200 chars)"""
    if len(content) > 200:
        content = content[:197] + "..."

    db = _get_db()
    db.execute(
        "INSERT INTO message_bus (source, direction, content, created_at) VALUES (?, ?, ?, ?)",
        (source, direction, content, now_str()),
    )
    db.execute(
        "DELETE FROM message_bus WHERE id NOT IN (SELECT id FROM message_bus ORDER BY id DESC LIMIT ?)",
        (MESSAGE_BUS_LIMIT,),
    )
    db.commit()
    db.close()


def bus_read(limit: int = 20) -> list[dict]:
    """Read recent bus messages."""
    db = _get_db()
    rows = db.execute(
        "SELECT source, direction, content, created_at FROM message_bus ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    db.close()
    return [dict(r) for r in reversed(rows)]


def bus_format(limit: int = 20) -> str:
    """Format bus messages for context injection."""
    messages = bus_read(limit)
    if not messages:
        return "(No recent cross-channel messages)"
    lines = ["# Recent Cross-Channel Messages\n"]
    for m in messages:
        arrow = "\u2192" if m["direction"] == "out" else "\u2190"
        lines.append(f"[{m['created_at']}] [{m['source']}] {arrow} {m['content']}")
    return "\n".join(lines)
