"""The Georeference panel's control-point map (docs/adr/0047): hull and
scale-bar helpers, point placement (north up, equal scale, ignored points
out of the zoom), click-to-switch control/check, and the panel wiring
(the map's request reaches the index; residuals overlay after a solve).
Offscreen Qt like every other GUI widget test."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from vine360.georef.control_points import ControlPoint
from vine360.gui.control_point_map import ControlPointMap, convex_hull, nice_length


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _point(name, east, north, use="control", tag_id=1):
    return ControlPoint(name, east, north, 10.0, tag_id, use)


@pytest.fixture
def cp_map(qapp):
    widget = ControlPointMap()
    widget.resize(500, 400)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()


def test_convex_hull_drops_interior_and_collinear_points():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hull = convex_hull(square + [(5, 5), (5, 0), (2, 3)])
    assert sorted(hull) == sorted(square)
    assert convex_hull([(0, 0), (1, 1)]) == [(0, 0), (1, 1)]


def test_nice_length_picks_1_2_5_steps():
    assert nice_length(37.0) == 20.0
    assert nice_length(0.8) == 0.5
    assert nice_length(1000.0) == 1000.0


def test_points_are_north_up_and_ignored_far_points_stay_out_of_the_zoom(cp_map):
    origin = (298900.0, 6126100.0)
    points = [
        _point("a", 298900.0, 6126100.0),
        _point("b", 299000.0, 6126100.0, use="check"),
        _point("c", 298900.0, 6126200.0),
        _point("base", 309000.0, 6130000.0, use="ignore", tag_id=None),  # ~10 km away
    ]
    cp_map.set_points(points, origin)
    a, b, c = (cp_map.to_pixel(p) for p in points[:3])
    assert b.x() > a.x() and abs(b.y() - a.y()) < 1e-6  # east is right
    assert c.y() < a.y() and abs(c.x() - a.x()) < 1e-6  # north is up
    assert abs((b.x() - a.x()) - (a.y() - c.y())) < 1e-6  # equal scale on both axes
    assert b.x() - a.x() > 200  # the base station didn't shrink the site to a dot
    far = cp_map.to_pixel(points[3])
    assert far.x() > cp_map.width()  # it's off the map, and counted in the legend
    cp_map.grab()  # paints without raising


def test_click_requests_control_check_switch_and_selects(cp_map, qapp):
    points = [_point("a", 0.0, 0.0), _point("b", 50.0, 0.0, use="check"), _point("x", 25.0, 40.0, use="ignore")]
    cp_map.set_points(points, (0.0, 0.0))
    requests, selected = [], []
    cp_map.use_change_requested.connect(lambda name, use: requests.append((name, use)))
    cp_map.point_selected.connect(selected.append)
    for p in points:
        QTest.mouseClick(cp_map, Qt.LeftButton, pos=cp_map.to_pixel(p).toPoint())
    assert requests == [("a", "check"), ("b", "control")]  # an ignored point needs the right-click menu
    assert selected == ["a", "b", "x"] and cp_map.selected() == "x"
    QTest.mouseClick(cp_map, Qt.LeftButton, pos=QPoint(3, 3))
    assert len(requests) == 2  # empty space does nothing


def test_residual_overlay_paints_and_clears(cp_map):
    cp_map.set_points([_point("a", 0.0, 0.0), _point("b", 50.0, 20.0)], (0.0, 0.0))
    cp_map.set_residuals([{"name": "a", "dE": 0.02, "dN": -0.01}, {"name": "b", "dE": 0.0, "dN": 0.0}], "geo-1234")
    cp_map.grab()
    assert cp_map._residuals["a"] == (0.02, -0.01)
    cp_map.set_residuals(None)
    assert not cp_map._residuals


def test_panel_map_switches_use_and_shows_residuals(tmp_path, qapp):
    pytest.importorskip("pycolmap")
    from georef_project import RUN_ID, build_scene
    from vine360.georef.control_points import load_control_points
    from vine360.gui.main_window import AppState, GeorefPanel, _solve_georef_worker
    from vine360.project import load_project

    scene = build_scene(tmp_path)
    try:
        st = AppState(project_root=scene.root, project=load_project(scene.root), conn=scene.conn)
        panel = GeorefPanel(st)
        panel.resize(1400, 900)
        panel.on_shown()
        assert len(panel.cp_map._points) == 15
        assert not panel.show_origin_check.isChecked() and not panel.cp_map.show_origin()
        panel.show_origin_check.setChecked(True)
        assert panel.cp_map.show_origin()

        panel.cp_map.use_change_requested.emit("021", "check")
        uses = {p.name: p.use for p in load_control_points(scene.conn, scene.set_id)}
        assert uses["021"] == "check"
        row = next(r for r in range(15) if panel.points_table.item(r, 0).text() == "021")
        assert panel.points_table.cellWidget(row, 5).currentData() == "check"  # table follows the map

        panel.cp_map.point_selected.emit("021")
        assert {i.row() for i in panel.points_table.selectedIndexes()} == {row}
        panel.points_table.selectRow(0)
        assert panel.cp_map.selected() == panel.points_table.item(0, 0).text()

        solution_id = _solve_georef_worker(scene.root, RUN_ID, scene.set_id, {})
        panel._on_run_changed(select=solution_id)
        assert panel.cp_map._residuals and panel.cp_map._residual_label == solution_id
        panel.cp_map.grab()
    finally:
        scene.conn.close()


def test_origin_is_zoomed_to_when_near_and_pointed_to_when_far(cp_map):
    points = [_point("a", 298950.0, 6126150.0), _point("b", 299010.0, 6126220.0)]
    cp_map.set_points(points, (298900.0, 6126100.0, 0.0))
    assert not cp_map.show_origin() and not cp_map.origin_in_zoom()  # hidden by default: zoom is on the points
    zoomed_span = cp_map.to_pixel(points[1]).x() - cp_map.to_pixel(points[0]).x()
    cp_map.set_show_origin(True)
    assert cp_map.origin_in_zoom()
    assert cp_map.to_pixel(points[1]).x() - cp_map.to_pixel(points[0]).x() < zoomed_span
    origin, a, b = cp_map.origin_pixel(), cp_map.to_pixel(points[0]), cp_map.to_pixel(points[1])
    assert origin.x() < a.x() < b.x() and origin.y() > a.y() > b.y()  # south-west of the points, on the map
    assert 0 <= origin.x() <= cp_map.width() and 0 <= origin.y() <= cp_map.height()
    cp_map.grab()

    cp_map.set_points(points, (290000.0, 6120000.0, 0.0))  # ~10 km away
    assert not cp_map.origin_in_zoom()
    assert cp_map.to_pixel(points[1]).x() - cp_map.to_pixel(points[0]).x() > 100  # zoom stays on the points
    assert cp_map.origin_pixel().x() < 0  # off the map; an edge arrow points to it
    cp_map.grab()


def test_main_window_opens_at_a_fraction_of_the_screen():
    from PySide6.QtCore import QRect

    from vine360.gui.main_window import MIN_WINDOW_SIZE, default_window_geometry

    big = default_window_geometry(QRect(0, 0, 2560, 1400))
    assert (big.width(), big.height()) == (2048, 1120) and big.center() == QRect(0, 0, 2560, 1400).center()
    small = default_window_geometry(QRect(0, 0, 1000, 700))
    assert (small.width(), small.height()) == MIN_WINDOW_SIZE  # never below the old fixed size...
    tiny = default_window_geometry(QRect(100, 50, 800, 600))
    assert tiny == QRect(100, 50, 800, 600)  # ...nor above the screen
