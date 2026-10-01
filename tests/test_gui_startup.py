"""The app's real startup sequence: every panel is constructed once,
up front, before any project is open (Vine360MainWindow.__init__ builds
ProjectPanel/ImportPanel/.../ExportPanel/QueuePanel immediately, and a
project is only opened afterward via the Project panel). Nothing else in
this test suite exercises that exact sequence -- every other GUI test
builds an AppState with project_root/conn already set. Real bug this
would have caught: ExportPanel's __init__ unconditionally calls
_on_mode_changed(), which (after the export source-selection feature)
started computing a default output_dir via `state.project_root / ...`
before state.project_root was ever set, crashing every real launch with
TypeError: unsupported operand type(s) for /: 'NoneType' and 'str'.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from vine360.gui.main_window import Vine360MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_constructs_with_no_project_open(qapp):
    window = Vine360MainWindow()
    assert window.state.conn is None
    assert window.state.project_root is None
    # every stage panel + the queue dock should exist and show the
    # "no project" placeholder rather than having crashed mid-construction
    assert len(window._panels) == 7
    assert window.queue_manager is not None
