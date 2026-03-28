"""
CC remote task queue.
Submit tasks for local Claude Code to execute asynchronously.
"""

import os
import shutil
import subprocess
import threading

from .db import _get_db, now_str
from .bus import bus_post


def submit_task(prompt: str, source: str = "chat") -> dict:
    """Submit a task for CC to execute (async)."""
    db = _get_db()
    db.execute(
        "INSERT INTO cc_tasks (prompt, status, source, created_at) VALUES (?, 'pending', ?, ?)",
        (prompt, source, now_str()),
    )
    task_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    db.commit()
    db.close()

    t = threading.Thread(target=_execute_task, args=(task_id, prompt), daemon=True)
    t.start()

    return {"task_id": task_id, "status": "pending", "message": f"Task submitted (ID: {task_id}), CC is running"}


def check_task(task_id: int) -> dict:
    """Check task status and result."""
    db = _get_db()
    row = db.execute(
        "SELECT id, prompt, status, result, created_at, started_at, completed_at FROM cc_tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    db.close()
    if not row:
        return {"error": f"Task {task_id} not found"}
    return {
        "task_id": row["id"],
        "prompt": row["prompt"][:100] + ("..." if len(row["prompt"]) > 100 else ""),
        "status": row["status"], "result": row["result"],
        "created_at": row["created_at"], "started_at": row["started_at"],
        "completed_at": row["completed_at"],
    }


def list_tasks(limit: int = 10) -> list[dict]:
    """List recent tasks."""
    db = _get_db()
    rows = db.execute(
        "SELECT id, prompt, status, created_at, completed_at FROM cc_tasks ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    db.close()
    return [{
        "task_id": r["id"],
        "prompt": r["prompt"][:80] + ("..." if len(r["prompt"]) > 80 else ""),
        "status": r["status"], "created_at": r["created_at"], "completed_at": r["completed_at"],
    } for r in rows]


def _execute_task(task_id: int, prompt: str):
    """Execute a CC task in background (subprocess)."""
    db = _get_db()
    db.execute("UPDATE cc_tasks SET status = 'running', started_at = ? WHERE id = ?", (now_str(), task_id))
    db.commit()
    db.close()

    claude_bin = shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
    env = {**os.environ}
    env.pop("CLAUDECODE", None)
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.path.expanduser("~/.bun/bin") + ":" + env.get("PATH", "")

    try:
        result = subprocess.run(
            [claude_bin, "-p", prompt, "--permission-mode", "auto",
             "--output-format", "text", "--max-budget-usd", "1.00"],
            capture_output=True, text=True, timeout=300, env=env,
        )
        if result.returncode != 0:
            stderr_msg = result.stderr.strip()
            output = f"Process exited with code {result.returncode}"
            if stderr_msg:
                output += f": {stderr_msg}"
            elif result.stdout.strip():
                output += f"\n{result.stdout.strip()}"
            status = "error"
        else:
            output = result.stdout.strip() or result.stderr.strip() or "(no output)"
            status = "completed"
    except subprocess.TimeoutExpired:
        output = "Task timed out (5 minutes)"
        status = "timeout"
    except Exception as e:
        output = f"Execution error: {str(e)}"
        status = "error"

    db = _get_db()
    db.execute(
        "UPDATE cc_tasks SET status = ?, result = ?, completed_at = ? WHERE id = ?",
        (status, output, now_str(), task_id),
    )
    db.commit()
    db.close()

    summary = output[:100] if len(output) <= 100 else output[:97] + "..."
    bus_post("cc_task", "out", f"[Task#{task_id} {status}] {summary}")
