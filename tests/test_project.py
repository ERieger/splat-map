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


def test_opening_an_older_project_migrates_missing_layout_dirs(tmp_path):
    """Regression test: a project created before a LAYOUT_DIRS entry
    (e.g. models/) existed must get it retroactively on open, not only at
    creation -- open_index_db recreates any missing layout directory
    every time, mirroring the schema migration above (docs/adr/0024)."""
    import shutil

    root = tmp_path / "myproject"
    create_project(root, "My Vineyard", CaptureMode.THREE_SIXTY)

    # simulate a pre-upgrade project: remove a layout dir directly
    shutil.rmtree(root / "models")
    assert not (root / "models").exists()

    reopened = open_index_db(root)
    reopened.close()
    assert (root / "models").is_dir()


def test_opening_an_older_project_migrates_missing_columns(tmp_path):
    """Regression test: CREATE TABLE IF NOT EXISTS does nothing for a
    table that already exists without a newer column (e.g.
    masks.flagged_for_review, added after some projects' masks tables
    already existed) -- open_index_db must add it via ALTER TABLE."""
    root = tmp_path / "myproject"
    create_project(root, "My Vineyard", CaptureMode.THREE_SIXTY)

    conn = sqlite3.connect(root / "index.sqlite")
    conn.execute("ALTER TABLE masks RENAME TO masks_old")
    conn.execute(
        "CREATE TABLE masks (view_id TEXT PRIMARY KEY, model TEXT NOT NULL, model_version TEXT NOT NULL, "
        "prompts TEXT NOT NULL, thresholds TEXT NOT NULL, morphology TEXT NOT NULL, keep_fraction REAL, "
        "edited INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute("DROP TABLE masks_old")
    conn.commit()
    conn.close()

    reopened = open_index_db(root)
    try:
        columns = {row[1] for row in reopened.execute("PRAGMA table_info(masks)")}
        assert "flagged_for_review" in columns
        reopened.execute("SELECT flagged_for_review FROM masks").fetchall()
    finally:
        reopened.close()


def test_opening_an_older_project_migrates_frames_into_legacy_frame_sets(tmp_path):
    """docs/adr/0034: frames extracted before frame sets existed have no
    frames.frame_set_id. On open, each source's frames become one frame
    set whose id is the source_id itself -- so the existing frames/
    <source_id>/ paths and "<source_id>:NNNNNN" frame_ids stay valid with
    nothing moved."""
    root = tmp_path / "myproject"
    create_project(root, "My Vineyard", CaptureMode.THREE_SIXTY)

    conn = sqlite3.connect(root / "index.sqlite")
    conn.execute("DROP TABLE frame_sets")
    conn.execute("ALTER TABLE frames DROP COLUMN frame_set_id")
    conn.execute(
        "INSERT INTO sources (source_id, path, checksum, media_type, projection, timestamps) "
        "VALUES ('src-a', '/a.mp4', 'x', 'video', 'equirectangular', '{}')"
    )
    for i in range(3):
        conn.execute(
            "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum) "
            "VALUES (?, 'src-a', ?, ?, ?, 'c')",
            (
                f"src-a:{i:06d}",
                float(i),
                '{"mode": "interval", "interval_seconds": 0.5, "requested_count": null}',
                f"frames/src-a/frame_{i + 1:06d}.png",
            ),
        )
    conn.commit()
    conn.close()

    reopened = open_index_db(root)
    try:
        assert reopened.execute("SELECT frame_set_id, source_id, label FROM frame_sets").fetchall() == [
            ("src-a", "src-a", "every 0.5s")
        ]
        assert {r[0] for r in reopened.execute("SELECT frame_set_id FROM frames")} == {"src-a"}
        assert reopened.execute("SELECT frame_id FROM frames ORDER BY frame_id").fetchall()[0] == ("src-a:000000",)
    finally:
        reopened.close()

    again = open_index_db(root)  # idempotent
    try:
        assert again.execute("SELECT COUNT(*) FROM frame_sets").fetchone()[0] == 1
    finally:
        again.close()
