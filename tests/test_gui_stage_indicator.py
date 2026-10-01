"""First automated (non-manual-smoke-test) coverage of an actual QWidget
panel's behavior -- everything else in vine360.gui so far is either pure
logic (test_gui_stage_statuses.py) or a manual offscreen smoke test (see
docs/status.md's noted gap: no pytest-qt yet). Runs under the offscreen
Qt platform so it needs no real display, same as every manual smoke test
run during development.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from vine360.config import CaptureMode
from vine360.gui.main_window import (
    ACTIVE,
    DONE,
    ENGINE_COLMAP_EQUIRECTANGULAR,
    PENDING,
    AppState,
    ExportPanel,
    MasksPanel,
    PoseEstimationPanel,
    ProjectionPanel,
    StageIndicator,
)
from vine360.masking.build import build_mask_for_view, set_view_flagged
from vine360.project import create_project, open_index_db

import numpy as np
from PIL import Image


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    yield root, conn
    conn.close()


def _insert_view(conn, root, view_id, frame_id):
    image_path = root / "projections" / frame_id / "front.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[0:20, :] = [135, 206, 235]
    image[20:64, :] = [34, 90, 34]
    Image.fromarray(image, "RGB").save(image_path)
    conn.execute(
        "INSERT OR IGNORE INTO sources (source_id, path, checksum, media_type, projection, width, height, "
        "timestamps, capture_group) VALUES ('s1', '/x.mp4', 'deadbeef', 'video', 'equirectangular', 64, 32, "
        "'{}', NULL)"
    )
    conn.execute(
        "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum) "
        "VALUES (?, 's1', 0.0, '{}', 'x', 'y')",
        (frame_id,),
    )
    conn.execute(
        "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, "
        "fixed_rotation, image_path) VALUES (?, ?, 'p', 64, 64, '{}', '{}', ?)",
        (view_id, frame_id, str(image_path.relative_to(root))),
    )
    conn.commit()


def _insert_source_only(conn, source_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sources (source_id, path, checksum, media_type, projection, width, height, "
        "timestamps, capture_group) VALUES (?, ?, 'deadbeef', 'video', 'equirectangular', 64, 32, '{}', NULL)",
        (source_id, f"/{source_id}.mp4"),
    )
    conn.commit()


def _insert_frames_only(conn, source_id: str, frame_id: str) -> None:
    _insert_source_only(conn, source_id)
    conn.execute(
        "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum) "
        "VALUES (?, ?, 0.0, '{}', 'x', 'y')",
        (frame_id, source_id),
    )
    conn.commit()


def test_stage_indicator_set_stages_records_statuses(qapp):
    indicator = StageIndicator(["one", "two"])
    assert indicator.statuses == [PENDING, PENDING]
    indicator.set_stages([DONE, ACTIVE])
    assert indicator.statuses == [DONE, ACTIVE]


def test_masks_panel_stage_indicator_before_and_after_build(qapp, project):
    root, conn = project
    _insert_view(conn, root, "frame-1:front", "frame-1")

    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    panel = MasksPanel(state)
    panel.on_shown()

    assert panel.stage_indicator.statuses == [ACTIVE, PENDING], "no masks built yet -- Build should be current"

    build_mask_for_view(conn, root, "frame-1:front")
    panel.refresh_sources()

    assert panel.stage_indicator.statuses == [DONE, DONE], "built, nothing flagged -- both stages done"


def test_masks_panel_stage_indicator_reflects_flagged_review(qapp, project):
    root, conn = project
    _insert_view(conn, root, "frame-1:front", "frame-1")
    build_mask_for_view(conn, root, "frame-1:front")
    set_view_flagged(conn, "frame-1:front", True)

    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    panel = MasksPanel(state)
    panel.on_shown()

    assert panel.stage_indicator.statuses == [DONE, ACTIVE], "built but a flag remains -- Review is current"


def test_export_panel_stage_indicator_progression(qapp, project):
    root, conn = project
    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    panel = ExportPanel(state)
    panel.mode_combo.setCurrentIndex(1)  # "frames_masks" -- needs no SfM run, simplest to configure
    panel.on_shown()

    assert panel.stage_indicator.statuses[0] == DONE, "output_dir defaults on_shown -- already configured"
    assert panel.stage_indicator.statuses[1] == ACTIVE, "not exported yet"

    panel._exported = True
    panel._refresh_enabled()
    assert panel.stage_indicator.statuses[1] == DONE

    # _on_mode_changed/_on_choose_output_dir both reset _exported before calling
    # _refresh_enabled -- exercised directly here since triggering the real file
    # dialog isn't feasible in a test.
    panel._exported = False
    panel._refresh_enabled()
    assert panel.stage_indicator.statuses[1] == ACTIVE, "changing configuration should un-mark Export as done"


def test_export_panel_button_label_follows_the_selected_mode(qapp, project):
    """Real bug: export_btn's text was set once at construction
    ("Export for Postshot") and never updated -- switching to RealityScan
    mode still showed "Export for Postshot"."""
    root, conn = project
    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    panel = ExportPanel(state)
    panel.on_shown()

    assert panel.export_btn.text() == "Export for Postshot"  # poses mode, index 0
    panel.mode_combo.setCurrentIndex(1)  # frames_masks
    assert panel.export_btn.text() == "Export for Postshot"
    panel.mode_combo.setCurrentIndex(2)  # realityscan
    assert panel.export_btn.text() == "Export for RealityScan"


def test_projection_panel_lists_a_source_with_no_frames_and_disables_run_now(qapp, project):
    """Real bug: refresh_sources used an INNER JOIN on frames, so a
    source with zero frames never appeared in the combo at all -- making
    it impossible to select it and use "Add to Queue" to auto-queue
    frame extraction ahead of a projection job."""
    root, conn = project
    _insert_source_only(conn, "s1")
    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    panel = ProjectionPanel(state)
    panel.on_shown()

    assert panel.source_combo.count() == 1
    assert panel.generate_btn.isEnabled() is False
    assert panel.queue_btn.isEnabled() is True


def test_projection_panel_enables_run_now_once_the_selected_source_has_frames(qapp, project):
    root, conn = project
    _insert_frames_only(conn, "s1", "frame-1")
    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    panel = ProjectionPanel(state)
    panel.on_shown()

    assert panel.source_combo.count() == 1
    assert panel.generate_btn.isEnabled() is True


def test_masks_panel_lists_a_source_with_frames_but_no_views_and_disables_build(qapp, project):
    """Same bug as Projection's, one stage later: MasksPanel's combo used
    a double INNER JOIN (frames, views), hiding a source that has frames
    but no projected views yet."""
    root, conn = project
    _insert_frames_only(conn, "s1", "frame-1")
    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    panel = MasksPanel(state)
    panel.on_shown()

    assert panel.source_combo.count() == 1
    assert panel.build_btn.isEnabled() is False
    assert panel.queue_btn.isEnabled() is True


def test_pose_estimation_panel_disables_run_for_a_selected_source_with_no_frames(qapp, project):
    root, conn = project
    _insert_source_only(conn, "s1")
    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    panel = PoseEstimationPanel(state)
    panel.on_shown()
    index = panel.engine_combo.findData(ENGINE_COLMAP_EQUIRECTANGULAR)
    panel.engine_combo.setCurrentIndex(index)

    assert panel.source_combo.count() == 1
    assert panel.run_btn.isEnabled() is False


def test_export_panel_frames_masks_mode_disables_export_for_a_selected_source_with_no_views(qapp, project):
    root, conn = project
    _insert_frames_only(conn, "s1", "frame-1")
    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    panel = ExportPanel(state)
    panel.mode_combo.setCurrentIndex(1)  # frames_masks
    panel.on_shown()

    assert panel.export_btn.isEnabled() is True, "default 'All sources' selection stays allowed"

    index = panel.source_combo.findData("s1")
    assert index >= 0, "the zero-view source must still be selectable"
    panel.source_combo.setCurrentIndex(index)

    assert panel.export_btn.isEnabled() is False
    assert panel.queue_btn.isEnabled() is True


