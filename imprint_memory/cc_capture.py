#!/usr/bin/env python3
"""Capture Claude Code conversations into conversation_log.

Claude Code writes every turn to a JSONL transcript. A `Stop` hook hands this
script the current transcript path; we incrementally read the new lines and
write each user/assistant message via log_message(), so automatic capture
works for the Claude Code channel with no "decide to remember" step.

Wire it up as a Stop hook (see README). stdin receives Claude Code's hook JSON
containing `transcript_path` and `session_id`.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .db import DATA_DIR, LOCAL_TZ, now_str
from .conversation import log_message

# Cap a single stored message. Normal turns are far shorter; the cap keeps an
# occasional giant context injection (a whole file pasted in, etc.) from bloating
# conversation_log.
MAX_CHARS = 8000
_OFFSET_DIR = DATA_DIR / ".cc-capture-offsets"


def _offset_path(session_id: str) -> Path:
    safe = (session_id or "default").replace("/", "_")
    return _OFFSET_DIR / safe


def _get_offset(session_id: str) -> int:
    try:
        return int(_offset_path(session_id).read_text().strip())
    except (OSError, ValueError):
        return 0


def _set_offset(session_id: str, offset: int) -> None:
    _OFFSET_DIR.mkdir(parents=True, exist_ok=True)
    _offset_path(session_id).write_text(str(offset))


def _extract_text(content: Any) -> str:
    """Pull readable text out of a transcript message's content."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = str(block.get("text") or "").strip()
                if t:
                    parts.append(t)
        return "\n".join(parts)
    return ""


def _is_tool_result_only(content: Any) -> bool:
    """A 'user' entry carrying only tool_result blocks is tool plumbing, not
    something the human typed — skip it."""
    if not isinstance(content, list):
        return False
    has_tool_result = any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )
    has_text = any(
        isinstance(b, dict) and b.get("type") == "text" and str(b.get("text") or "").strip()
        for b in content
    )
    return has_tool_result and not has_text


def _timestamp(entry: dict) -> str:
    value = entry.get("timestamp")
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.astimezone(LOCAL_TZ).isoformat(timespec="seconds")
        except ValueError:
            return value[:19]
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 1e12 else value
        return datetime.fromtimestamp(seconds, tz=LOCAL_TZ).isoformat(timespec="seconds")
    return now_str()


def export_transcript(transcript_path: str, session_id: str) -> int:
    """Read new lines from a Claude Code JSONL transcript and log each
    user/assistant message. Returns the number of messages written.

    Dedup is twofold: a per-session byte offset avoids re-reading old lines,
    and external_id (the transcript entry uuid) lets log_message drop anything
    that slips through twice."""
    try:
        file_size = os.path.getsize(transcript_path)
    except OSError:
        return 0

    offset = _get_offset(session_id)
    if offset >= file_size:
        return 0

    count = 0
    new_offset = offset
    with open(transcript_path, "rb") as f:
        f.seek(offset)
        while True:
            raw = f.readline()
            if not raw:
                break
            new_offset = f.tell()
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type")
            if entry_type not in ("user", "assistant"):
                continue
            msg = entry.get("message")
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")

            if entry_type == "user" and _is_tool_result_only(content):
                continue

            text = _extract_text(content)
            if not text or len(text) < 3:
                continue
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS] + "..."

            uuid = str(entry.get("uuid") or msg.get("id") or f"offset-{new_offset}")
            res = log_message(
                platform="cc",
                direction="in" if entry_type == "user" else "out",
                content=text,
                session_id=session_id,
                created_at=_timestamp(entry),
                external_id=f"{session_id}:{uuid}",
            )
            if res.get("id") and not res.get("skipped"):
                count += 1

    _set_offset(session_id, new_offset)
    return count


def main() -> None:
    """Stop-hook entry point. Reads Claude Code's hook JSON from stdin and
    captures the current transcript. Stays silent and never raises so a capture
    hiccup can't break the turn."""
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    transcript_path = payload.get("transcript_path") or ""
    session_id = payload.get("session_id") or payload.get("sessionId") or ""
    if not transcript_path or not os.path.isfile(transcript_path):
        return
    try:
        export_transcript(transcript_path, session_id)
    except Exception:
        pass


if __name__ == "__main__":
    main()
