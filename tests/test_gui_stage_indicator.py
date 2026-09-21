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
from vine360.gui.main_window import ACTIVE, DONE, PENDING, AppState, ExportPanel, MasksPanel, StageIndicator
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
