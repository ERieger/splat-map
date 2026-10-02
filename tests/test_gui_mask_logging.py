"""Every mask change made from the Masks panel lands in the Activity log
(docs/adr/0037): per-view layer toggles and Exclude-view clicks as instant
entries, the set-wide "merged" toggle as a logged background run. Real
project, real layer files; offscreen Qt like test_gui_stage_indicator.py."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from test_masking_layers import _add_view
from vine360.activity_log import STATUS_DONE, list_entries
from vine360.config import CaptureMode
from vine360.gui.main_window import AppState, MasksPanel
from vine360.masking.layers import LAYER_SKY, build_layers_for_frame_set, view_layers
from vine360.project import create_project, open_index_db


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qapp, tmp_path):
    root = tmp_path / "proj"
    create_project(root, "T", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    view_ids = [_add_view(conn, root, i) for i in range(3)]
    build_layers_for_frame_set(conn, root, "s1", {LAYER_SKY: None})
    panel = MasksPanel(AppState(project_root=root, conn=conn))
    panel.source_combo.blockSignals(True)
    panel.source_combo.addItem("s1", "s1")
    panel.source_combo.blockSignals(False)
    # Isolate the handlers' own work + logging from review-UI refreshing.
    panel._current_view_id = lambda: view_ids[0]
    panel._after_view_change = lambda _view_id: None
    yield panel, conn, view_ids
    conn.close()


def _mask_entries(conn):
    return [e for e in list_entries(conn) if e.operation.startswith("masks.")]


def test_per_view_layer_toggle_is_logged(panel):
    panel, conn, view_ids = panel
    panel._on_view_layer_toggled(LAYER_SKY, False)

    (entry,) = _mask_entries(conn)
    assert entry.operation == "masks.view_layer"
    assert entry.status == STATUS_DONE
    assert entry.params == {"view_id": view_ids[0], "layer": LAYER_SKY, "enabled": False}
    assert "removed from" in entry.result


def test_exclude_and_restore_view_are_logged(panel):
    panel, conn, view_ids = panel
    panel._on_exclude_view_toggled(True)
    panel._on_exclude_view_toggled(False)

    entries = _mask_entries(conn)
    assert [e.label for e in entries] == ["Exclude view", "Restore excluded view"]
    assert [e.params["excluded"] for e in entries] == [True, False]
    assert all(e.target == view_ids[0] for e in entries)


def test_set_wide_merge_toggle_is_logged_with_direction(panel, qapp):
    panel, conn, view_ids = panel
    panel.layer_rows[LAYER_SKY].merged.setCheckState(Qt.Unchecked)  # what the click leaves behind
    panel._on_layer_merged_clicked(LAYER_SKY)
    deadline = time.time() + 20
    while panel._job_running and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert {view_layers(conn, v)[0]["enabled"] for v in view_ids} == {False}
    (entry,) = _mask_entries(conn)
    assert entry.operation == "masks.merge"
    assert entry.status == STATUS_DONE
    assert entry.params == {"frame_set_id": "s1", "layer": LAYER_SKY, "enabled": False}
    assert entry.target == "sky off @ s1"
    assert entry.result == "sky layer removed from the keep-mask of 3 view(s)"
