"""vine360.activity_log: entries record order, timing, parameters and
outcome; logging wraps real worker functions without ever changing what
they return or raise; leftovers from a dead session are marked
interrupted (docs/adr/0037)."""

from pathlib import Path

import pytest

from vine360.activity_log import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    finish_entry,
    list_entries,
    logged_operation,
    mark_interrupted,
    start_entry,
)
from vine360.config import CaptureMode
from vine360.project import create_project, open_index_db


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Log", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    yield root, conn
    conn.close()


def test_entries_are_listed_in_the_order_they_were_run(project):
    _root, conn = project
    first = start_entry(conn, "frames.extract", "Frame extraction", target="s1~i1", params={"interval_seconds": 1.0})
    second = start_entry(conn, "projection.generate", "Projection", target="s1~i1", params={"face_size": 1024})
    finish_entry(conn, second, status=STATUS_DONE, duration_seconds=2.5, result="24 views")
    finish_entry(conn, first, status=STATUS_FAILED, duration_seconds=1.0, error="boom")

    entries = list_entries(conn)
    assert [e.entry_id for e in entries] == [first, second]
    assert entries[0].status == STATUS_FAILED and entries[0].error == "boom"
    assert entries[1].params == {"face_size": 1024} and entries[1].result == "24 views"
    assert entries[1].finished_at and entries[1].duration_seconds == 2.5


def test_logged_operation_records_parameters_result_and_origin(project):
    root, conn = project

    @logged_operation("demo.op", "Demo", target=lambda p: p["frame_set_id"], summarize=lambda r, _c: f"{r} things")
    def worker(project_root, frame_set_id, face_size, progress_callback=None):
        return 7

    assert worker(root, "s1~i1", 512, progress_callback=lambda *a: None, log_origin="queue: demo") == 7

    (entry,) = list_entries(conn)
    assert entry.operation == "demo.op" and entry.status == STATUS_DONE
    assert entry.params == {"frame_set_id": "s1~i1", "face_size": 512}  # no project_root / progress_callback
    assert entry.target == "s1~i1" and entry.result == "7 things" and entry.origin == "queue: demo"
    assert entry.duration_seconds is not None


def test_logged_operation_records_failures_and_reraises_unchanged(project):
    root, conn = project

    @logged_operation("demo.fail", "Demo failure")
    def worker(project_root, path: Path):
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        worker(root, Path("/some/where"))
    (entry,) = list_entries(conn)
    assert entry.status == STATUS_FAILED and entry.error == "ValueError: bad input"
    assert entry.params == {"path": "/some/where"}  # Paths stored as strings


def test_logging_problems_never_break_the_work(tmp_path):
    @logged_operation("demo.op", "Demo")
    def worker(project_root):
        return "still ran"

    assert worker(tmp_path / "not-a-project") == "still ran"


def test_mark_interrupted_only_touches_running_entries(project):
    _root, conn = project
    done = start_entry(conn, "a", "A")
    finish_entry(conn, done, status=STATUS_DONE)
    start_entry(conn, "b", "B")

    assert mark_interrupted(conn) == 1
    statuses = [e.status for e in list_entries(conn)]
    assert statuses == [STATUS_DONE, STATUS_INTERRUPTED]
    assert STATUS_RUNNING not in statuses


def test_real_gui_workers_are_logged(project):
    """The GUI's actual worker functions carry the logger -- e.g. a Data
    manager delete, run exactly as the panel runs it."""
    pytest.importorskip("PySide6")
    from vine360.gui.main_window import _data_manager_delete_worker

    root, conn = project
    (root / "exports" / "all" / "postshot").mkdir(parents=True)
    _data_manager_delete_worker(root, "export", "exports/all/postshot")

    (entry,) = list_entries(conn)
    assert entry.operation == "data.delete" and entry.status == STATUS_DONE
    assert entry.params == {"action": "export", "target": "exports/all/postshot"}
    assert entry.origin == "manual"
