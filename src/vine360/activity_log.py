"""The project's activity log (docs/adr/0037): what was run, in what
order, when, for how long, with exactly which parameters, and how it
ended -- persisted in the project's own index.sqlite (`activity_log`
table), so it survives restarts and travels with the project.

Every long-running operation the GUI or the queue starts goes through
`logged_operation` (vine360.gui.main_window wraps its worker functions
with it), so a run is logged identically whether it was started by hand
or by a queued job -- `origin` records which.

Stdlib-only, like vine360.project: nothing here needs numpy/pycolmap.
"""

from __future__ import annotations

import functools
import inspect
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"

ORIGIN_KWARG = "log_origin"  # passed by the queue; popped before the wrapped function sees it

_IGNORED_PARAMS = {"project_root", "progress_callback"}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class LogEntry:
    entry_id: int
    operation: str
    label: str
    target: str | None
    params: dict
    status: str
    started_at: str
    finished_at: str | None
    duration_seconds: float | None
    result: str | None
    error: str | None
    origin: str | None


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def start_entry(
    conn: sqlite3.Connection,
    operation: str,
    label: str,
    *,
    target: str | None = None,
    params: dict | None = None,
    origin: str = "manual",
) -> int:
    cursor = conn.execute(
        "INSERT INTO activity_log (operation, label, target, params, status, started_at, origin) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            operation,
            label,
            target,
            json.dumps(params or {}, default=_json_default, sort_keys=True),
            STATUS_RUNNING,
            _utcnow_iso(),
            origin,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def finish_entry(
    conn: sqlite3.Connection,
    entry_id: int,
    *,
    status: str,
    duration_seconds: float | None = None,
    result: str | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE activity_log SET status = ?, finished_at = ?, duration_seconds = ?, result = ?, error = ? "
        "WHERE entry_id = ?",
        (status, _utcnow_iso(), duration_seconds, result, error, entry_id),
    )
    conn.commit()


def list_entries(conn: sqlite3.Connection, *, limit: int | None = None) -> list[LogEntry]:
    """Oldest first -- the order things were run in."""
    sql = (
        "SELECT entry_id, operation, label, target, params, status, started_at, finished_at, "
        "duration_seconds, result, error, origin FROM activity_log ORDER BY entry_id"
    )
    rows = conn.execute(sql + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    return [
        LogEntry(r[0], r[1], r[2], r[3], json.loads(r[4] or "{}"), r[5], r[6], r[7], r[8], r[9], r[10], r[11])
        for r in rows
    ]


def mark_interrupted(conn: sqlite3.Connection) -> int:
    """Entries still "running" from a previous app session can't be
    running any more -- the process that ran them is gone (the queue makes
    the same call for its own RUNNING jobs, docs/adr/0023). Call once when
    a project is opened in the app. Returns how many were marked."""
    cursor = conn.execute(
        "UPDATE activity_log SET status = ?, error = COALESCE(error, 'interrupted (app closed or crashed)') "
        "WHERE status = ?",
        (STATUS_INTERRUPTED, STATUS_RUNNING),
    )
    conn.commit()
    return cursor.rowcount


def logged_operation(
    operation: str,
    label: str,
    *,
    target: Callable[[dict], str | None] | None = None,
    summarize: Callable[[object, sqlite3.Connection], str | None] | None = None,
    open_conn: Callable[[Path], sqlite3.Connection] | None = None,
):
    """Decorator for a worker function whose first argument is the project
    root. Records an entry before it runs (status "running", every other
    argument as its parameters) and completes it afterwards -- "done" with
    `summarize(result, conn)` as the result text, or "failed" with the
    exception's message (the exception is re-raised unchanged).

    Logging must never break the work: any failure to write the log itself
    is swallowed. Uses its own short-lived connection (workers run off the
    GUI thread; sqlite connections can't cross threads)."""

    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            origin = kwargs.pop(ORIGIN_KWARG, "manual")
            bound = signature.bind_partial(*args, **kwargs)
            project_root = Path(bound.arguments.get("project_root", args[0]))
            params = {k: v for k, v in bound.arguments.items() if k not in _IGNORED_PARAMS}
            opener = open_conn or _default_open
            entry_id = None
            try:
                conn = opener(project_root)
                try:
                    entry_id = start_entry(
                        conn, operation, label, target=target(params) if target else None, params=params, origin=origin
                    )
                finally:
                    conn.close()
            except Exception:
                entry_id = None

            started = time.monotonic()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                _finish_quietly(opener, project_root, entry_id, status=STATUS_FAILED,
                                duration=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}")
                raise
            _finish_quietly(opener, project_root, entry_id, status=STATUS_DONE,
                            duration=time.monotonic() - started, result_obj=result, summarize=summarize)
            return result

        return wrapper

    return decorate


def _default_open(project_root: Path) -> sqlite3.Connection:
    from vine360.project import open_index_db

    return open_index_db(project_root)


def _finish_quietly(opener, project_root, entry_id, *, status, duration, error=None, result_obj=None, summarize=None):
    if entry_id is None:
        return
    try:
        conn = opener(project_root)
        try:
            summary = None
            if summarize is not None and status == STATUS_DONE:
                try:
                    summary = summarize(result_obj, conn)
                except Exception:
                    summary = None
            finish_entry(conn, entry_id, status=status, duration_seconds=duration, result=summary, error=error)
        finally:
            conn.close()
    except Exception:
        pass
