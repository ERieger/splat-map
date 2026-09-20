"""The first automated test for vine360.gui -- compute_stage_statuses is
pure logic (takes a sqlite3.Connection, returns a dict), so it doesn't
need a QApplication/display to test, unlike the rest of the GUI (which so
far has only been verified with manual offscreen smoke tests -- see
docs/status.md).
"""

import pytest

pytest.importorskip("PySide6")

from vine360.config import CaptureMode
from vine360.gui.main_window import (
    ACTIVE,
    DONE,
    PENDING,
    STAGE_FRAMES,
    STAGE_IMPORT,
    STAGE_MASKS,
    STAGE_POSE,
    STAGE_PROJECTION,
    compute_stage_statuses,
)
from vine360.project import create_project, open_index_db


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    yield conn
    conn.close()


def test_no_project_is_all_pending():
    statuses = compute_stage_statuses(None)
    assert set(statuses.values()) == {PENDING}


def test_empty_project_is_all_pending_except_import(project):
    statuses = compute_stage_statuses(project)
    assert statuses[STAGE_IMPORT] == ACTIVE
    assert statuses[STAGE_FRAMES] == PENDING
    assert statuses[STAGE_PROJECTION] == PENDING
    assert statuses[STAGE_MASKS] == PENDING
    assert statuses[STAGE_POSE] == PENDING


def _insert_source_and_frame(conn):
    conn.execute(
        "INSERT INTO sources (source_id, path, checksum, media_type, projection, width, height, timestamps, capture_group) "
        "VALUES ('s1', '/x.mp4', 'x', 'video', 'equirectangular', 640, 320, '{}', NULL)"
    )
    conn.execute(
        "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum) "
        "VALUES ('f1', 's1', 0.0, '{}', 'x', 'y')"
    )
    conn.commit()


def test_frames_active_once_video_source_exists(project):
    _insert_source_and_frame(project)
    assert compute_stage_statuses(project)[STAGE_FRAMES] == DONE


def test_pose_active_once_masks_exist_with_no_run_yet(project):
    _insert_source_and_frame(project)
    project.execute(
        "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, fixed_rotation, image_path, updated_at) "
        "VALUES ('v1', 'f1', 'six-face', 10, 10, '{}', '{}', 'x', '2026-01-01T00:00:00+00:00')"
    )
    project.execute(
        "INSERT INTO masks (view_id, model, model_version, prompts, thresholds, morphology, keep_fraction, edited, flagged_for_review, updated_at) "
        "VALUES ('v1', 'classical', '1', '[]', '{}', '{}', 0.8, 0, 0, '2026-01-01T00:00:00+00:00')"
    )
    project.commit()
    assert compute_stage_statuses(project)[STAGE_POSE] == ACTIVE


def test_pose_done_when_run_is_newer_than_all_inputs(project):
    _insert_source_and_frame(project)
    project.execute(
        "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, fixed_rotation, image_path, updated_at) "
        "VALUES ('v1', 'f1', 'six-face', 10, 10, '{}', '{}', 'x', '2026-01-01T00:00:00+00:00')"
    )
    project.execute(
        "INSERT INTO masks (view_id, model, model_version, prompts, thresholds, morphology, keep_fraction, edited, flagged_for_review, updated_at) "
        "VALUES ('v1', 'classical', '1', '[]', '{}', '{}', 0.8, 0, 0, '2026-01-01T00:00:00+00:00')"
    )
    project.execute(
        "INSERT INTO sfm_runs (run_id, image_set_hash, engine_version, config, model_stats, selected_model, created_at) "
        "VALUES ('run1', 'h', '{}', '{}', '{}', 'run1', '2026-01-02T00:00:00+00:00')"
    )
    project.commit()
    assert compute_stage_statuses(project)[STAGE_POSE] == DONE


def test_pose_reverts_to_active_when_masks_rebuilt_after_the_run(project):
    """The real scenario this feature was requested for: re-running an
    earlier stage (here, rebuilding a mask) after a successful SfM run
    should show Pose as needing attention again, not still DONE."""
    _insert_source_and_frame(project)
    project.execute(
        "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, fixed_rotation, image_path, updated_at) "
        "VALUES ('v1', 'f1', 'six-face', 10, 10, '{}', '{}', 'x', '2026-01-01T00:00:00+00:00')"
    )
    project.execute(
        "INSERT INTO masks (view_id, model, model_version, prompts, thresholds, morphology, keep_fraction, edited, flagged_for_review, updated_at) "
        "VALUES ('v1', 'classical', '1', '[]', '{}', '{}', 0.8, 0, 0, '2026-01-01T00:00:00+00:00')"
    )
    project.execute(
        "INSERT INTO sfm_runs (run_id, image_set_hash, engine_version, config, model_stats, selected_model, created_at) "
        "VALUES ('run1', 'h', '{}', '{}', '{}', 'run1', '2026-01-02T00:00:00+00:00')"
    )
    project.commit()
    assert compute_stage_statuses(project)[STAGE_POSE] == DONE

    # rebuild the mask (as if the user reran masking) -- newer than the run
    project.execute("UPDATE masks SET updated_at = '2026-01-03T00:00:00+00:00' WHERE view_id = 'v1'")
    project.commit()

    assert compute_stage_statuses(project)[STAGE_POSE] == ACTIVE
