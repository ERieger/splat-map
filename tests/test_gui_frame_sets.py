"""GUI behavior around frame sets (docs/adr/0034), the Data manager tab
(docs/adr/0035), the Frames interval box and the Export folder default.
Offscreen Qt, same as test_gui_stage_indicator.py."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from vine360.config import FRAME_PRESET_INTERVALS, CaptureMode, FramePreset
from vine360.gui.main_window import (
    ENGINE_COLMAP_EQUIRECTANGULAR,
    AppState,
    DataManagerPanel,
    ExportPanel,
    FramesPanel,
    PoseEstimationPanel,
    ProjectionPanel,
)
from vine360.gui.queue_manager import QueueManager
from vine360.project import create_project, open_index_db


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


def _state(root, conn, *, with_queue: bool = False) -> AppState:
    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    if with_queue:
        state.queue_manager = QueueManager(state)
    return state


def _insert_source(conn, source_id: str, projection: str = "equirectangular") -> None:
    conn.execute(
        "INSERT INTO sources (source_id, path, checksum, media_type, projection, width, height, timestamps, "
        "capture_group) VALUES (?, ?, 'x', 'video', ?, 64, 32, '{\"duration_seconds\": 10.0}', NULL)",
        (source_id, f"/media/{source_id}.mp4", projection),
    )
    conn.commit()


def _insert_frame_set(conn, source_id: str, frame_set_id: str, label: str, n_frames: int = 1) -> None:
    conn.execute(
        "INSERT INTO frame_sets (frame_set_id, source_id, label, extraction_settings, created_at) "
        "VALUES (?, ?, ?, '{}', '2026-01-01')",
        (frame_set_id, source_id, label),
    )
    for i in range(n_frames):
        conn.execute(
            "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
            "VALUES (?, ?, ?, '{}', 'x', 'y', ?)",
            (f"{frame_set_id}:{i:06d}", source_id, float(i), frame_set_id),
        )
    conn.commit()


def test_interval_box_shows_each_presets_interval_and_restores_the_custom_value(qapp, project):
    root, conn = project
    panel = FramesPanel(_state(root, conn))

    for preset in (FramePreset.PREVIEW, FramePreset.QUALITY, FramePreset.BALANCED):
        panel.preset_combo.setCurrentText(preset.value)
        assert panel.custom_interval_spin.value() == FRAME_PRESET_INTERVALS[preset]
        assert not panel.custom_interval_spin.isEnabled()

    panel.preset_combo.setCurrentText(FramePreset.CUSTOM.value)
    assert panel.custom_interval_spin.isEnabled()
    panel.custom_interval_spin.setValue(0.25)
    panel.preset_combo.setCurrentText(FramePreset.PREVIEW.value)
    assert panel.custom_interval_spin.value() == FRAME_PRESET_INTERVALS[FramePreset.PREVIEW]
    panel.preset_combo.setCurrentText(FramePreset.CUSTOM.value)
    assert panel.custom_interval_spin.value() == 0.25
    assert panel._current_interval() == 0.25


def test_frames_panel_lists_the_selected_sources_frame_sets(qapp, project):
    root, conn = project
    _insert_source(conn, "s1")
    _insert_frame_set(conn, "s1", "s1~i0.5", "every 0.5s", n_frames=3)
    _insert_frame_set(conn, "s1", "s1~i1", "every 1s", n_frames=2)
    panel = FramesPanel(_state(root, conn))
    panel.on_project_changed()

    assert "2 frame sets" in panel.source_combo.itemText(0)
    texts = [panel.frame_sets_list.item(i).text() for i in range(panel.frame_sets_list.count())]
    assert texts == ["every 0.5s — 3 frames, 0 views, 0 masks", "every 1s — 2 frames, 0 views, 0 masks"]


def test_projection_panel_offers_each_frame_set_plus_queued_and_unextracted_ones(qapp, project):
    root, conn = project
    _insert_source(conn, "s1")
    _insert_source(conn, "s2")
    _insert_frame_set(conn, "s1", "s1~i0.5", "every 0.5s")
    _insert_frame_set(conn, "s1", "s1~i1", "every 1s")
    state = _state(root, conn, with_queue=True)
    state.queue_manager.add_job(
        "frames", "s1", {"interval_seconds": 2.0, "start_time": None, "end_time": None}, "frames s1 2s"
    )
    panel = ProjectionPanel(state)
    panel.on_shown()

    ids = [panel.source_combo.itemData(i) for i in range(panel.source_combo.count())]
    assert ids == ["s1~i0.5", "s1~i1", "s1~i2", "s2~i1"]  # extracted, extracted, queued, s2's balanced default
    assert "queued" in panel.source_combo.itemText(2)

    panel.source_combo.setCurrentIndex(1)
    assert panel.generate_btn.isEnabled()
    panel.refresh_sources()
    assert panel.source_combo.currentData() == "s1~i1", "a refresh keeps the chosen frame set selected"

    panel.source_combo.setCurrentIndex(2)
    assert not panel.generate_btn.isEnabled()  # queued, no frames yet -- only "Add to Queue"
    panel._on_add_to_queue()
    job = state.queue_manager.jobs[-1]
    assert job.target_frame_set_id == "s1~i2"
    assert job.depends_on == [state.queue_manager.jobs[0].job_id]


def test_pose_panel_scope_choices_follow_the_engine(qapp, project):
    root, conn = project
    _insert_source(conn, "s1")
    _insert_source(conn, "flat", projection="perspective")
    _insert_frame_set(conn, "s1", "s1~i1", "every 1s")
    _insert_frame_set(conn, "flat", "flat~i1", "every 1s")
    panel = PoseEstimationPanel(_state(root, conn))
    panel.on_shown()

    six_face = [panel.source_combo.itemData(i) for i in range(panel.source_combo.count())]
    assert six_face == [None, "s1~i1", "flat~i1"]  # "All frame sets" first

    panel.engine_combo.setCurrentIndex(panel.engine_combo.findData(ENGINE_COLMAP_EQUIRECTANGULAR))
    equirect = [panel.source_combo.itemData(i) for i in range(panel.source_combo.count())]
    assert equirect == ["s1~i1"]  # no "All", equirectangular sources only
    assert panel._current_sfm_args() == ("frames", "s1~i1", "EQUIRECTANGULAR")


def test_export_panel_defaults_into_the_new_projects_exports_after_switching_project(qapp, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    create_project(first, "First", CaptureMode.THREE_SIXTY)
    create_project(second, "Second", CaptureMode.THREE_SIXTY)
    conn = open_index_db(first)
    state = _state(first, conn)
    panel = ExportPanel(state)
    panel.on_project_changed()
    assert panel.output_dir == first / "exports" / "all" / "colmap"
    # Qt's folder dialog falls back to the working directory for a folder
    # that doesn't exist yet -- start it at the nearest existing one.
    assert panel._dialog_start_dir() == first / "exports"

    panel.output_dir = tmp_path / "somewhere-else"  # what "Choose output folder…" does
    panel._output_dir_user_chosen = True

    conn.close()
    conn = open_index_db(second)
    state.project_root, state.conn = second, conn
    panel.on_project_changed()
    assert panel.output_dir == second / "exports" / "all" / "colmap"
    conn.close()


def test_export_panel_per_frame_set_default_folder_names(qapp, project):
    root, conn = project
    _insert_source(conn, "s1")
    _insert_frame_set(conn, "s1", "s1", "every 1s")  # legacy (migrated) frame set
    _insert_frame_set(conn, "s1", "s1~i0.5", "every 0.5s")
    panel = ExportPanel(_state(root, conn))
    panel.mode_combo.setCurrentIndex(1)  # frames_masks
    panel.on_shown()

    assert panel._frame_set_folder_name("s1") == "s1"  # legacy exports keep landing where they always did
    assert panel._frame_set_folder_name("s1~i0.5") == "s1_i0.5"
    panel.source_combo.setCurrentIndex(panel.source_combo.findData("s1~i0.5"))
    assert panel.output_dir == root / "exports" / "s1_i0.5" / "postshot"


def test_data_manager_panel_lists_everything_and_gates_deletes_on_selection(qapp, project):
    root, conn = project
    _insert_source(conn, "s1")
    _insert_frame_set(conn, "s1", "s1~i0.5", "every 0.5s", n_frames=2)
    (root / "exports" / "all" / "postshot").mkdir(parents=True)
    panel = DataManagerPanel(_state(root, conn))
    panel.on_shown()

    assert panel.sources_table.rowCount() == 1
    assert panel.sets_tree.topLevelItemCount() == 1
    source_item = panel.sets_tree.topLevelItem(0)
    assert source_item.text(0) == "s1.mp4"
    set_item = source_item.child(0)
    assert (set_item.text(0), set_item.text(1)) == ("every 0.5s", "2")
    assert panel.exports_table.rowCount() == 1
    assert panel.summary_label.text().startswith("1 source(s), 1 frame set(s), 0 SfM run(s), 1 export folder(s)")

    assert not panel.delete_set_btn.isEnabled()
    source_item.setSelected(True)
    assert not panel.delete_set_btn.isEnabled(), "a source row can't be deleted from here"
    source_item.setSelected(False)
    set_item.setSelected(True)
    assert panel.delete_set_btn.isEnabled()
    assert not panel.delete_views_btn.isEnabled()  # nothing projected yet
    assert not panel.delete_masks_btn.isEnabled()
