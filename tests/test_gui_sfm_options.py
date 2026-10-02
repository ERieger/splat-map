"""The Pose estimation panel's "Advanced options" (docs/adr/0040): the
editor built from OPTION_SPECS, how the panel hands options to a run and
to the queue, and the worker recording them. Offscreen Qt, same as
test_gui_frame_sets.py; the worker test is a real (synthetic) pycolmap
run, nothing mocked."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from vine360.activity_log import list_entries
from vine360.config import CaptureMode
from vine360.gui.main_window import (
    ENGINE_COLMAP_EQUIRECTANGULAR,
    ENGINE_COLMAP_PROJECTIONS,
    ENGINE_SPHERESFM,
    AppState,
    PoseEstimationPanel,
    _run_sfm_worker,
)
from vine360.gui.queue_manager import QueueManager
from vine360.gui.sfm_options import SfmOptionsEditor
from vine360.project import create_project, open_index_db
from vine360.sfm.options import VARIANT_EQUIRECT, VARIANT_PROJECTIONS, VARIANT_SPHERESFM, SfmConfig


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


def _state(root, conn) -> AppState:
    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    state.queue_manager = QueueManager(state)
    return state


def _equirect_frame_set(conn, frame_set_id="s1~i1", n_frames=3) -> None:
    conn.execute(
        "INSERT INTO sources (source_id, path, checksum, media_type, projection, width, height, timestamps, "
        "capture_group) VALUES ('s1', '/media/s1.mp4', 'x', 'video', 'equirectangular', 64, 32, "
        "'{\"duration_seconds\": 10.0}', NULL)"
    )
    conn.execute(
        "INSERT INTO frame_sets (frame_set_id, source_id, label, extraction_settings, created_at) "
        "VALUES (?, 's1', 'every 1s', '{}', '2026-01-01')",
        (frame_set_id,),
    )
    for i in range(n_frames):
        conn.execute(
            "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
            "VALUES (?, 's1', ?, '{}', 'x', 'y', ?)",
            (f"{frame_set_id}:{i:06d}", float(i), frame_set_id),
        )
    conn.commit()


def _field(editor: SfmOptionsEditor, key: str):
    return editor._fields[key][0]


def _set(editor: SfmOptionsEditor, key: str, value) -> None:
    editor._fields[key][2](value)


def _row_visible(editor: SfmOptionsEditor, key: str) -> bool:
    widget, _get, _set, _label, form = editor._fields[key]
    return form.isRowVisible(widget)


def test_editor_shows_only_what_the_engine_uses(qapp):
    editor = SfmOptionsEditor()
    assert editor.config() == SfmConfig()  # untouched fields read back as exact defaults
    assert editor.summary_label.text() == "All defaults"

    editor.set_variant(VARIANT_PROJECTIONS)
    assert _row_visible(editor, "camera_model") and _row_visible(editor, "mapper")
    editor.set_variant(VARIANT_EQUIRECT)
    assert not _row_visible(editor, "camera_model") and _row_visible(editor, "refine_focal_length")
    editor.set_variant(VARIANT_SPHERESFM)
    assert not _row_visible(editor, "mapper") and not _row_visible(editor, "refine_focal_length")
    assert _row_visible(editor, "max_num_features") and _row_visible(editor, "matcher")


def test_editor_enables_dependent_rows_validates_and_applies_presets(qapp):
    editor = SfmOptionsEditor()
    editor.set_variant(VARIANT_EQUIRECT)
    assert _field(editor, "sequential_overlap").isEnabled()
    assert not _field(editor, "exhaustive_block_size").isEnabled()
    _set(editor, "matcher", "exhaustive")
    assert not _field(editor, "sequential_overlap").isEnabled()
    assert _field(editor, "exhaustive_block_size").isEnabled()

    _set(editor, "matcher", "vocab_tree")
    assert editor.problems() and "vocabulary tree" in editor.problems_label.text()

    editor.set_config(SfmConfig())
    editor.preset_combo.setCurrentIndex(editor.preset_combo.findData("Fast preview"))
    editor._on_preset()
    assert editor.config().max_image_size == 1600
    assert editor.preset_combo.currentText() == "Fast preview"
    assert "max image size (px) 1600" in editor.summary_label.text()
    _set(editor, "max_num_features", 1024)
    assert editor.preset_combo.currentText() == "Custom"


def test_panel_hands_non_default_options_to_runs_and_queued_jobs(qapp, project):
    root, conn = project
    _equirect_frame_set(conn)
    state = _state(root, conn)
    panel = PoseEstimationPanel(state)
    panel.on_project_changed()

    panel.engine_combo.setCurrentIndex(panel.engine_combo.findData(ENGINE_COLMAP_EQUIRECTANGULAR))
    editor = panel.options_editor
    _set(editor, "matcher", "exhaustive")
    _set(editor, "max_num_features", 4096)
    _set(editor, "camera_model", "PINHOLE")  # projections-only: not passed on for raw frames
    image_source, frame_set_id, camera_model, engine, options = panel._current_sfm_args()
    assert (image_source, frame_set_id, camera_model, engine) == ("frames", "s1~i1", "EQUIRECTANGULAR", "pycolmap")
    assert options == {"matcher": "exhaustive", "max_num_features": 4096}
    assert panel.run_btn.isEnabled()

    panel._on_add_to_queue()
    job = state.queue_manager.jobs[-1]
    assert job.params["options"] == {"matcher": "exhaustive", "max_num_features": 4096}
    assert "max features per image 4096" in job.label

    panel.engine_combo.setCurrentIndex(panel.engine_combo.findData(ENGINE_COLMAP_PROJECTIONS))
    assert panel._current_sfm_args()[2] == "PINHOLE"

    _set(editor, "matcher", "vocab_tree")  # no tree file -> can't run or queue
    assert not panel.run_btn.isEnabled() and not panel.queue_btn.isEnabled()
    assert "vocabulary tree" in panel.run_btn.toolTip()


def test_panel_starts_from_the_latest_runs_options(qapp, project):
    root, conn = project
    stored = SfmConfig(matcher="exhaustive", upright=True).for_variant(VARIANT_SPHERESFM)
    conn.execute(
        "INSERT INTO sfm_runs (run_id, image_set_hash, engine_version, config, model_stats, selected_model, created_at) "
        "VALUES ('sfm-1', 'h', 'v', ?, ?, 'sfm/sparse/sfm-1/0', '2026-01-01T00:00:00+00:00')",
        (
            json.dumps({"engine": "spheresfm", "camera_model": "SPHERE", "image_source": "frames", "options": stored.to_dict()}),
            json.dumps({"registered_images": 3, "total_images": 3, "registered_ratio": 1.0, "num_points3d": 10,
                        "mean_reprojection_error": 0.5, "num_connected_models": 1}),
        ),
    )
    conn.commit()
    panel = PoseEstimationPanel(_state(root, conn))
    panel.on_project_changed()
    config = panel.options_editor.config()
    assert config.matcher == "exhaustive" and config.upright
    assert config.camera_model == SfmConfig().camera_model  # SPHERE was the engine's, not a choice
    assert "Options:" in panel.stats_label.text() and "upright features on" in panel.stats_label.text()


def test_spheresfm_engine_hides_the_mapper_choice(qapp, project, monkeypatch):
    monkeypatch.setattr(
        PoseEstimationPanel, "_spheresfm_status",
        lambda self: {"found": True, "path": "/x/colmap", "version": "SphereSfM", "cuda": False, "searched": [], "env_var": "X"},
    )
    root, conn = project
    _equirect_frame_set(conn)
    panel = PoseEstimationPanel(_state(root, conn))
    panel.on_project_changed()
    _set(panel.options_editor, "mapper", "global")
    panel.engine_combo.setCurrentIndex(panel.engine_combo.findData(ENGINE_SPHERESFM))
    assert not _row_visible(panel.options_editor, "mapper")
    assert "mapper" not in panel._current_sfm_args()[4]  # dropped for SphereSfM, not passed on
    assert panel.run_btn.isEnabled()


def test_worker_runs_with_the_options_and_logs_them(project):
    pytest.importorskip("pycolmap")
    from synthetic_360 import add_synthetic_frame_set

    root, conn = project
    frame_set_id = add_synthetic_frame_set(conn, root, n=6, width=1024)
    diagnostics, _ = _run_sfm_worker(
        root, "frames", frame_set_id, "EQUIRECTANGULAR", "pycolmap", {"matcher": "exhaustive", "random_seed": 5}
    )
    assert diagnostics.registered_images >= 5
    stored = json.loads(conn.execute("SELECT config FROM sfm_runs").fetchone()[0])
    assert stored["camera_model"] == "EQUIRECTANGULAR"
    assert stored["options"]["matcher"] == "exhaustive" and stored["options"]["random_seed"] == 5
    entry = list_entries(conn)[0]
    assert entry.params["options"] == {"matcher": "exhaustive", "random_seed": 5}
