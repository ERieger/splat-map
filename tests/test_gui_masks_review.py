"""MasksPanel review behaviour (docs/adr/0038), offscreen."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PIL import Image
from PySide6.QtWidgets import QApplication

from vine360.config import CaptureMode
from vine360.gui.main_window import AppState, MasksPanel
from vine360.masking.layers import LAYER_SKY, build_layers_for_frame_set
from vine360.project import create_project, open_index_db


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qapp, tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    conn.execute(
        "INSERT INTO sources (source_id, path, checksum, media_type, projection, width, height, timestamps, "
        "capture_group) VALUES ('s1', '/x.mp4', 'd', 'video', 'equirectangular', 64, 32, '{}', NULL)"
    )
    conn.execute(
        "INSERT INTO frame_sets (frame_set_id, source_id, label, extraction_settings, created_at) "
        "VALUES ('s1', 's1', 'every 1s', '{}', '2026-01-01T00:00:00+00:00')"
    )
    for i in range(3):
        frame_id = f"s1:{i:06d}"
        conn.execute(
            "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
            "VALUES (?, 's1', ?, '{}', 'x', 'y', 's1')",
            (frame_id, float(i)),
        )
        for face in ("back", "front", "left"):
            rel = f"projections/{frame_id}/{face}.png"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            image = np.full((64, 64, 3), [34, 90, 34], dtype=np.uint8)
            image[:20] = [135, 206, 235]
            Image.fromarray(image, "RGB").save(root / rel)
            conn.execute(
                "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, fixed_rotation, "
                "image_path) VALUES (?, ?, 'p', 64, 64, '{}', '{}', ?)",
                (f"{frame_id}:{face}", frame_id, rel),
            )
    conn.commit()
    build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None})
    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    widget = MasksPanel(state)
    widget.on_shown()
    yield widget
    conn.close()


def test_face_stays_selected_when_changing_frame(panel):
    """Regression: changing frame used to reset the View combo to 'back'."""
    panel.face_combo.setCurrentIndex(panel.face_combo.findData("s1:000000:front"))
    panel.frame_selector._step(1)
    assert panel.face_combo.currentData() == "s1:000001:front"
    panel.frame_selector._step(1)
    assert panel.face_combo.currentData() == "s1:000002:front"


def test_exclude_this_view_zeroes_its_keep_fraction(panel):
    panel.face_combo.setCurrentIndex(panel.face_combo.findData("s1:000000:left"))
    panel.exclude_view_btn.setChecked(True)
    kf = panel.state.conn.execute("SELECT keep_fraction FROM masks WHERE view_id = 's1:000000:left'").fetchone()[0]
    assert kf == 0.0
    assert "excluded" in panel.face_combo.currentText()
    panel.exclude_view_btn.setChecked(False)
    kf = panel.state.conn.execute("SELECT keep_fraction FROM masks WHERE view_id = 's1:000000:left'").fetchone()[0]
    assert 0.0 < kf < 1.0


def test_build_options_are_explicit_worker_arguments(panel):
    panel.layer_rows["overexposed"].include.setChecked(True)
    panel.clip_spin.setValue(245)
    options = panel._build_options()
    assert options == {
        "use_sam3_person": False,
        "use_sam3_sky": False,
        "mask_sky": True,
        "mask_overexposure": True,
        "overexposure_clip": 245,
        "overexposure_bloom_radius": 40,
    }
