"""The queue's Georeference stage (docs/adr/0047): detect_tags needs its
frame set's projection; solve needs an imported control-point set, a run,
and detect_tags on each of the run's frame sets -- auto-queued when
missing, BLOCKED (never guessed) when the control set or run is."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from test_gui_queue_manager import _insert_frames_and_views, _insert_frames_only, _state
from vine360.config import CaptureMode
from vine360.georef.control_points import import_control_points
from vine360.gui.queue_manager import (
    BLOCKED,
    GEOREF_DETECT,
    GEOREF_SOLVE,
    QUEUED,
    STAGE_EXPORT,
    STAGE_GEOREF,
    STAGE_PROJECTION,
    QueueManager,
)
from vine360.project import create_project, open_index_db

GCP = Path(__file__).resolve().parent.parent / "fixtures" / "gcp" / "ESTO1_GCPs_23Oct26.csv"


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    yield root, conn
    conn.close()


def _insert_run(conn, run_id, frame_set_ids):
    conn.execute(
        "INSERT INTO sfm_runs (run_id, image_set_hash, engine_version, config, model_stats, selected_model, created_at) "
        "VALUES (?, 'x', '{}', ?, '{}', 'sfm/sparse/x/0', '2026-10-25T00:00:00+00:00')",
        (run_id, json.dumps({"image_source": "projections", "frame_set_ids": frame_set_ids})),
    )
    conn.commit()


def test_detect_tags_auto_queues_projection(project, qapp):
    root, conn = project
    _insert_frames_only(conn, "s1", "s1:000000")
    manager = QueueManager(_state(root, conn))
    job = manager.add_job(STAGE_GEOREF, "s1", {"kind": GEOREF_DETECT}, "detect", target_frame_set_id="s1")
    assert job.status == QUEUED
    assert [j.stage for j in manager.jobs] == [STAGE_PROJECTION, STAGE_GEOREF]
    assert job.depends_on == [manager.jobs[0].job_id]


def test_solve_is_blocked_without_a_control_set(project, qapp):
    root, conn = project
    _insert_frames_and_views(conn, "s1", "s1:000000", "v1")
    _insert_run(conn, "sfm-1", ["s1"])
    manager = QueueManager(_state(root, conn))
    job = manager.add_job(STAGE_GEOREF, None, {"kind": GEOREF_SOLVE, "run_id": "sfm-1", "set_id": None}, "solve")
    assert job.status == BLOCKED and "control-point set" in job.error_message
    assert len(manager.jobs) == 1


def test_solve_auto_queues_detection_for_each_of_the_runs_frame_sets(project, qapp):
    root, conn = project
    _insert_frames_and_views(conn, "s1", "s1:000000", "v1")
    _insert_frames_and_views(conn, "s2", "s2:000000", "v2")
    _insert_run(conn, "sfm-1", ["s1", "s2"])
    set_id, _ = import_control_points(conn, GCP, epsg=7854)
    conn.execute("INSERT INTO tag_detection_runs VALUES ('v1', 'tag36h11', 0, 'now')")  # s1 already detected
    conn.commit()
    manager = QueueManager(_state(root, conn))
    job = manager.add_job(STAGE_GEOREF, None, {"kind": GEOREF_SOLVE, "run_id": "sfm-1", "set_id": set_id}, "solve")
    assert job.status == QUEUED
    detect = [j for j in manager.jobs if j.params.get("kind") == GEOREF_DETECT]
    assert [j.target_frame_set_id for j in detect] == ["s2"] and detect[0].auto_added
    assert job.depends_on == [detect[0].job_id]

    # A second solve reuses the queued detection instead of adding another.
    again = manager.add_job(STAGE_GEOREF, None, {"kind": GEOREF_SOLVE, "run_id": "sfm-1", "set_id": set_id}, "solve 2")
    assert again.depends_on == [detect[0].job_id]


def test_dispatch_passes_the_solution_to_export_and_georef_workers(project, qapp):
    root, conn = project
    manager = QueueManager(_state(root, conn))
    manager._owner = object()
    calls = []
    with patch("vine360.gui.main_window.run_in_background", side_effect=lambda owner, fn, *args, **kw: calls.append((fn.__name__, args))):
        export = manager.add_job(
            STAGE_EXPORT, None, {"mode": "poses", "output_dir": str(root / "out"), "run_id": "sfm-1", "georef_solution_id": "geo-1"},
            "export",
        )
        manager._start_job(export)
        manager._running_job_id = None
        solve = manager.add_job(STAGE_GEOREF, None, {"kind": GEOREF_SOLVE, "run_id": "sfm-1", "set_id": "gcp-1"}, "solve")
        manager._start_job(solve)
        manager._running_job_id = None
        detect = manager.add_job(STAGE_GEOREF, None, {"kind": GEOREF_DETECT, "frame_set_ids": ["a", "b"]}, "detect")
        manager._start_job(detect)
    assert calls[0] == ("_export_postshot_worker", (root, root / "out", "sfm-1", "geo-1"))
    assert calls[1] == ("_solve_georef_worker", (root, "sfm-1", "gcp-1", {}))
    assert calls[2] == ("_detect_tags_worker", (root, ["a", "b"], "tag36h11", {}))
