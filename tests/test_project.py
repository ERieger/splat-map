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


def test_index_db_has_all_tables(tmp_path):
    root = tmp_path / "myproject"
    create_project(root, "My Vineyard", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"sources", "frames", "views", "masks", "sfm_runs"} <= tables
    finally:
        conn.close()


def test_open_index_db_missing_raises(tmp_path):
    with pytest.raises(ProjectNotFoundError):
        open_index_db(tmp_path / "does-not-exist")


def test_opening_an_older_project_migrates_missing_tables(tmp_path):
    """Regression test: a project created before views/masks/sfm_runs
    existed (e.g. from an earlier version of this app) must not raise
    "no such table" when opened -- open_index_db upgrades the schema on
    every open, not just at creation."""
    root = tmp_path / "myproject"
    create_project(root, "My Vineyard", CaptureMode.THREE_SIXTY)

    # simulate a pre-upgrade project: drop the newer tables directly
    conn = sqlite3.connect(root / "index.sqlite")
    conn.executescript("DROP TABLE views; DROP TABLE masks; DROP TABLE sfm_runs;")
    conn.commit()
    conn.close()

    reopened = open_index_db(root)
    try:
        tables = {row[0] for row in reopened.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"sources", "frames", "views", "masks", "sfm_runs"} <= tables
        # and it's actually usable, not just present
        reopened.execute("SELECT COUNT(*) FROM views").fetchone()
    finally:
        reopened.close()
