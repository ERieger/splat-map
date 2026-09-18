import sqlite3

import pytest

from vine360.config import CaptureMode
from vine360.project import (
    LAYOUT_DIRS,
    ProjectAlreadyExistsError,
    ProjectNotFoundError,
    create_project,
    load_project,
    open_index_db,
)


def test_create_project_writes_expected_layout(tmp_path):
    root = tmp_path / "myproject"
    project = create_project(root, "My Vineyard", CaptureMode.THREE_SIXTY)

    assert (root / "project.yaml").exists()
    assert (root / "index.sqlite").exists()
    for rel in LAYOUT_DIRS:
        assert (root / rel).is_dir(), f"missing layout dir: {rel}"

    assert project.name == "My Vineyard"
    assert project.capture_mode == CaptureMode.THREE_SIXTY
    assert project.schema_version == 1


def test_create_project_twice_raises(tmp_path):
    root = tmp_path / "myproject"
    create_project(root, "My Vineyard", CaptureMode.THREE_SIXTY)
    with pytest.raises(ProjectAlreadyExistsError):
        create_project(root, "My Vineyard", CaptureMode.THREE_SIXTY)


def test_load_project_round_trips(tmp_path):
    root = tmp_path / "myproject"
    created = create_project(root, "My Vineyard", CaptureMode.MIXED)
    loaded = load_project(root)
    assert loaded == created


def test_load_project_missing_raises(tmp_path):
    with pytest.raises(ProjectNotFoundError):
        load_project(tmp_path / "does-not-exist")


def test_index_db_has_sources_and_frames_tables(tmp_path):
    root = tmp_path / "myproject"
    create_project(root, "My Vineyard", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"sources", "frames"} <= tables
    finally:
        conn.close()


def test_open_index_db_missing_raises(tmp_path):
    with pytest.raises(ProjectNotFoundError):
        open_index_db(tmp_path / "does-not-exist")
