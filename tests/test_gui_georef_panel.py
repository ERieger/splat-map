"""Offscreen smoke test of the Georeference panel, the Export panel's
georeference choice and the Data manager's georeference rows
(docs/adr/0047), on the synthetic scene of tests/georef_project.py."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pycolmap")

from PySide6.QtWidgets import QApplication

from georef_project import FRAME_SET, RUN_ID, build_scene
from vine360.gui.main_window import (
    DONE,
    STAGE_GEOREF,
    AppState,
    DataManagerPanel,
    ExportPanel,
    GeorefPanel,
    _solve_georef_worker,
    compute_stage_statuses,
)
from vine360.georef.control_points import load_control_points
from vine360.georef.solve import active_solution
from vine360.project import load_project


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def state(tmp_path, qapp):
    scene = build_scene(tmp_path)
    st = AppState(project_root=scene.root, project=load_project(scene.root), conn=scene.conn)
    yield st, scene
    scene.conn.close()


def test_panels_show_control_points_solutions_and_choices(state):
    st, scene = state
    panel = GeorefPanel(st)
    panel.on_shown()
    assert panel.points_table.rowCount() == 15
    assert panel.points_table.item(0, 0).text() == "PRS778941614462"
    assert panel.points_table.cellWidget(0, 5).currentData() == "ignore"
    assert "10.4 km" in panel.points_table.item(0, 0).toolTip()
    assert [panel.origin_spins[i].value() for i in range(3)] == [298900.0, 6126100.0, 0.0]
    assert panel.detect_combo.findData(FRAME_SET) >= 0
    assert panel.run_combo.currentData() == RUN_ID
    assert panel.solve_btn.isEnabled()

    # Editing the table writes through to the index.
    row = next(r for r in range(15) if panel.points_table.item(r, 0).text() == "021")
    panel.points_table.item(row, 6).setText("0.25")
    assert {p.name: p.tag_size_m for p in load_control_points(scene.conn, scene.set_id)}["021"] == 0.25

    solution_id = _solve_georef_worker(scene.root, RUN_ID, scene.set_id, {})
    panel._on_run_changed(select=solution_id)
    assert panel.solution_combo.currentData() == solution_id
    assert panel.report_table.rowCount() == 14 and "RMSE 3D" in panel.report_label.text()
    assert compute_stage_statuses(scene.conn)[STAGE_GEOREF] == DONE

    export = ExportPanel(st)
    export.on_shown()
    assert export.georef_combo.itemData(0) is None and export.georef_combo.itemData(1) == solution_id
    assert export.georef_combo.currentData() is None  # no active choice yet -> None
    export.georef_combo.setCurrentIndex(1)
    assert active_solution(scene.root, RUN_ID) == solution_id
    export.on_shown()
    assert export.georef_combo.currentData() == solution_id  # remembered per run

    manager = DataManagerPanel(st)
    manager.on_shown()
    texts = []
    for i in range(manager.georef_tree.topLevelItemCount()):
        top = manager.georef_tree.topLevelItem(i)
        texts.append(top.text(0))
        for k in range(top.childCount()):
            child = top.child(k)
            texts += [child.text(0)] + [child.child(j).text(0) for j in range(child.childCount())]
    assert "ESTO1_GCPs_23Oct26.csv" in texts and any(solution_id in t for t in texts)
    _wait_for_background(manager)


def _wait_for_background(widget, timeout_s: float = 10.0) -> None:
    """Let the Data manager's disk-usage scan finish before the widget goes
    away (Qt aborts on a QThread destroyed mid-run)."""
    import time

    deadline = time.monotonic() + timeout_s
    while getattr(widget, "_bg_pairs", None) and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.01)


def test_tag_size_table_controls(state, tmp_path, monkeypatch):
    from vine360.georef.tag_sizes import BUILTIN_TABLE

    monkeypatch.setenv("VINE360_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr("vine360.gui.main_window.QMessageBox.information", lambda *a, **k: None)
    st, scene = state
    panel = GeorefPanel(st)
    panel.on_shown()
    assert panel.tag_table_label.text().startswith("Built-in: vine360_apriltag_id_sizes.csv — 48 tags")
    assert "20 × 250 mm" not in panel.tag_table_label.text() and "40 × 250 mm" in panel.tag_table_label.text()
    assert not panel.tag_table_reset_btn.isEnabled()
    panel._on_apply_tag_sizes()  # the scene's 020/021 were 0.25, the kit sheet says 400 mm
    sizes = {p.name: p.tag_size_m for p in load_control_points(scene.conn, scene.set_id)}
    assert sizes["020"] == 0.4 and sizes["021"] == 0.4 and sizes["005"] == 0.25
    row = next(r for r in range(15) if panel.points_table.item(r, 0).text() == "020")
    assert panel.points_table.item(row, 6).text() == "0.4"
    assert BUILTIN_TABLE.is_file()


def test_splat_section_defaults_to_a_new_file_and_follows_the_solution(state, tmp_path):
    import numpy as np

    from test_georef_apply import FIELDS, _write_ply
    from vine360.gui.main_window import _georef_splat_worker
    from vine360.georef.similarity import Similarity
    from vine360.georef.solve import load_solution
    from vine360.georef.splat import reference_points

    st, scene = state
    solution_id = _solve_georef_worker(scene.root, RUN_ID, scene.set_id, {})
    panel = GeorefPanel(st)
    panel.on_shown()
    panel._on_run_changed(select=solution_id)
    assert panel.splat_solution_combo.currentData() == solution_id  # follows section 3
    assert not panel.splat_btn.isEnabled()  # no input yet

    solution = load_solution(scene.conn, solution_id)
    sim = Similarity.from_dict(solution["transform"])
    local = reference_points(scene.root, solution)
    data = np.zeros((len(local), len(FIELDS)))
    data[:, :3] = (local - sim.translation) @ sim.rotation / sim.scale
    src = tmp_path / "trained.ply"
    _write_ply(src, data)
    panel.splat_in_edit.setText(str(src))
    panel._on_splat_inputs_changed()
    expected = tmp_path / f"trained.georef-{solution_id[:8]}.ply"
    assert panel.splat_out_edit.text() == str(expected) and panel.splat_btn.isEnabled()

    result = _georef_splat_worker(scene.root, src, expected, solution_id)
    panel._on_splat_done(result)
    assert expected.exists() and "Wrote" in panel.splat_result.text()
    # The next default never replaces the one just written.
    assert panel.splat_out_edit.text() == str(tmp_path / f"trained.georef-{solution_id[:8]}-2.ply")

    export = ExportPanel(st)
    export.on_shown()
    assert not hasattr(export, "splat_btn")  # Export is for the sparse model / image sets only
