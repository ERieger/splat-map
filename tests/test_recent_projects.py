import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from vine360.config import CaptureMode
from vine360.project import create_project
from vine360.recent_projects import (
    MAX_RECENT_PROJECTS,
    config_dir,
    load_recent_projects,
    record_recent_project,
    recent_projects_path,
    remove_recent_project,
)


def test_config_dir_honours_override_then_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("VINE360_CONFIG_DIR", str(tmp_path / "override"))
    assert config_dir() == tmp_path / "override"
    assert config_dir("win32") == tmp_path / "override"  # override wins on every OS
    monkeypatch.delenv("VINE360_CONFIG_DIR")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert config_dir("linux") == tmp_path / "xdg" / "vine360"


def test_config_dir_linux_default(monkeypatch, tmp_path):
    monkeypatch.delenv("VINE360_CONFIG_DIR")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config_dir("linux") == tmp_path / ".config" / "vine360"


def test_config_dir_windows(monkeypatch, tmp_path):
    monkeypatch.delenv("VINE360_CONFIG_DIR")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert config_dir("win32") == tmp_path / "Roaming" / "vine360"
    monkeypatch.delenv("APPDATA")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config_dir("win32") == tmp_path / "AppData" / "Roaming" / "vine360"


def test_config_dir_macos(monkeypatch, tmp_path):
    monkeypatch.delenv("VINE360_CONFIG_DIR")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "ignored"))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert config_dir("darwin") == tmp_path / "Library" / "Application Support" / "vine360"


def test_empty_when_no_file():
    assert load_recent_projects() == []


def test_record_moves_to_top_without_duplicates(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    record_recent_project(a, "A")
    record_recent_project(b, "B")
    record_recent_project(a, "A renamed")
    entries = load_recent_projects()
    assert [e.name for e in entries] == ["A renamed", "B"]
    assert entries[0].path == str(a.resolve())


def test_list_is_capped(tmp_path):
    for i in range(MAX_RECENT_PROJECTS + 3):
        record_recent_project(tmp_path / f"p{i}", f"P{i}")
    entries = load_recent_projects()
    assert len(entries) == MAX_RECENT_PROJECTS
    assert entries[0].name == f"P{MAX_RECENT_PROJECTS + 2}"


def test_remove(tmp_path):
    record_recent_project(tmp_path / "a", "A")
    record_recent_project(tmp_path / "b", "B")
    remove_recent_project(tmp_path / "a")
    assert [e.name for e in load_recent_projects()] == ["B"]


def test_corrupt_file_reads_as_empty():
    path = recent_projects_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert load_recent_projects() == []
    record_recent_project("/somewhere", "X")
    assert [e.name for e in load_recent_projects()] == ["X"]


def test_exists_checks_for_project_yaml(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Real", CaptureMode.THREE_SIXTY)
    record_recent_project(root, "Real")
    record_recent_project(tmp_path / "gone", "Gone")
    by_name = {e.name: e.exists() for e in load_recent_projects()}
    assert by_name == {"Real": True, "Gone": False}


def test_project_panel_records_and_reopens(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from vine360.gui.main_window import AppState, ProjectPanel

    app = QApplication.instance() or QApplication([])
    root = tmp_path / "proj"
    create_project(root, "Block 7", CaptureMode.THREE_SIXTY)

    panel = ProjectPanel(AppState())
    assert panel.recent_list.count() == 1  # the "No recent projects yet." placeholder
    assert not panel.recent_open_btn.isEnabled()

    assert panel._open_path(root)
    assert [e.name for e in load_recent_projects()] == ["Block 7"]

    # A fresh panel (i.e. the next app launch) lists it and can reopen it.
    panel2 = ProjectPanel(AppState())
    assert panel2.recent_list.count() == 1
    panel2.recent_list.setCurrentRow(0)
    assert panel2.recent_open_btn.isEnabled()
    panel2._on_open_recent()
    assert panel2.state.project_root == root
    assert panel2.state.project.name == "Block 7"

    panel2.recent_list.setCurrentRow(0)  # opening refreshes the list, clearing the selection
    panel2._on_remove_recent()
    assert load_recent_projects() == []
    for p in (panel, panel2):
        if p.state.conn is not None:
            p.state.conn.close()
    assert app is not None
