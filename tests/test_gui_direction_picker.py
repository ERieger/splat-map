"""The Projection panel's graphical view-direction picker (docs/adr/0041),
under the offscreen Qt platform like every other GUI widget test."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from vine360.config import CaptureMode
from vine360.gui.direction_picker import DirectionPicker
from vine360.gui.main_window import AppState, ProjectionPanel
from vine360.project import create_project, open_index_db
from vine360.projection.cubemap import CARDINAL_NAMES, DIRECTION_PRESETS, DIRECTIONS_BY_NAME, direction_preset


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def picker(qapp):
    widget = DirectionPicker()
    widget.resize(900, 520)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()


def _click_marker(picker, name):
    point = picker.canvas.marker_position(DIRECTIONS_BY_NAME[name])
    QTest.mouseClick(picker.canvas, Qt.LeftButton, Qt.NoModifier, QPoint(round(point.x()), round(point.y())))


def test_defaults_to_the_four_cardinal_views(picker):
    assert picker.selected_names() == CARDINAL_NAMES
    assert picker.ring_tilt() == 45.0


def test_clicking_markers_toggles_them_and_emits(picker):
    emitted = []
    picker.selection_changed.connect(emitted.append)

    _click_marker(picker, "front-left-down")
    assert "front-left-down" in picker.selected_names()
    _click_marker(picker, "front")
    assert "front" not in picker.selected_names()
    _click_marker(picker, "down")
    assert "down" in picker.selected_names()

    assert emitted[-1] == picker.selected_names()
    assert len(emitted) == 3


def test_every_marker_is_individually_clickable(picker):
    """No two markers overlap closely enough for a click to hit the wrong one."""
    picker.set_selected([])
    for name in DIRECTIONS_BY_NAME:
        point = picker.canvas.marker_position(DIRECTIONS_BY_NAME[name])
        assert picker.canvas.name_at(QPointF(point)) == name


def test_presets_replace_the_selection(picker):
    picker.preset_buttons["Vine-row tunnel"].click()
    assert set(picker.selected_names()) == set(DIRECTION_PRESETS["Vine-row tunnel"])
    picker.set_selected([])
    assert picker.selected_names() == []
    with pytest.raises(ValueError):
        picker.set_selected(["sideways"])


def test_summary_counts_images_across_frames(picker):
    picker.set_selected(DIRECTION_PRESETS["Horizon + lower ring (16)"])
    picker.set_frame_count(100)
    assert "16 views per frame × 100 frames = 1,600 images" in picker.summary_label.text()


def test_tilt_moves_lower_ring_markers(picker):
    before = picker.canvas.marker_position(DIRECTIONS_BY_NAME["front-down"]).y()
    picker.set_ring_tilt(60.0)
    after = picker.canvas.marker_position(DIRECTIONS_BY_NAME["front-down"]).y()
    assert after > before  # further down the panorama


def test_background_and_hover_preview_render_from_the_panorama(picker, tmp_path):
    pano = np.zeros((180, 360, 3), dtype=np.uint8)
    pano[90:, :] = [200, 50, 50]  # everything below the horizon is red
    path = tmp_path / "pano.jpg"
    Image.fromarray(pano).save(path)

    picker.set_background(path)
    assert picker.background_qimage() is not None
    picker.grab()  # paints without raising, footprints included

    picker._on_hovered("front-down")
    pixmap = picker.hover_preview.pixmap()
    assert not pixmap.isNull()
    center = pixmap.toImage().pixelColor(80, 120)
    assert center.red() > 150  # a downward view sees the red floor

    picker.set_background(tmp_path / "missing.jpg")  # never raises
    assert picker.background_qimage() is None


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    yield root, conn
    conn.close()


def _insert_frame_with_views(conn, root, names, tilt):
    conn.execute(
        "INSERT INTO sources (source_id, path, checksum, media_type, projection, width, height, timestamps, capture_group) "
        "VALUES ('s1', '/x.mp4', 'deadbeef', 'video', 'equirectangular', 64, 32, '{}', NULL)"
    )
    conn.execute(
        "INSERT INTO frame_sets (frame_set_id, source_id, label, extraction_settings, created_at) "
        "VALUES ('s1', 's1', 'every 1s', '{\"mode\": \"interval\", \"interval_seconds\": 1.0}', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
        "VALUES ('f1', 's1', 0.0, '{}', 'frames/s1/frame_000001.png', 'y', 's1')"
    )
    for face in direction_preset(16, 90.0, names, ring_tilt_degrees=tilt):
        conn.execute(
            "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, fixed_rotation, image_path) "
            "VALUES (?, 'f1', 'directions', 16, 16, '{}', ?, ?)",
            (f"f1:{face.name}", json.dumps(face.fixed_rotation_to_dict()), f"projections/f1/{face.name}.png"),
        )
    conn.commit()


def test_projection_panel_preselects_the_directions_and_tilt_already_generated(qapp, project):
    root, conn = project
    _insert_frame_with_views(conn, root, ["front", "left-down", "back-right-down"], tilt=30.0)
    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    panel = ProjectionPanel(state)
    panel.on_shown()

    assert panel.direction_picker.selected_names() == ["front", "back-right-down", "left-down"]
    assert panel.direction_picker.ring_tilt() == 30.0
    assert panel._selected_face_names() == panel.direction_picker.selected_names()


def test_projection_panel_queues_the_picker_selection_and_tilt(qapp, project):
    root, conn = project
    _insert_frame_with_views(conn, root, ["front"], tilt=45.0)
    added = []

    class FakeQueue:
        is_running = False
        jobs: list = []

        class _Sig:
            def connect(self, _fn):
                pass

        queue_changed = _Sig()

        def add_job(self, *args, **kwargs):
            added.append((args, kwargs))

    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    state.queue_manager = FakeQueue()
    panel = ProjectionPanel(state)
    panel.on_shown()
    panel.direction_picker.set_selected(["front-right-down", "front-left-down"])
    panel.direction_picker.set_ring_tilt(40.0)
    panel._on_add_to_queue()

    (_stage, _source, params, label), _kwargs = added[0]
    assert params["face_names"] == ["front-right-down", "front-left-down"]
    assert params["ring_tilt_degrees"] == 40.0
    assert "front-right-down" in label
