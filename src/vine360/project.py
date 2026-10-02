"""Project creation and the on-disk layout from the handover doc, section 5.

See docs/adr/0003-project-layout-and-index.md for the directory layout and
the decision to record sources by reference (checksum + path) rather than
copying media into the project on ingest.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from vine360 import __version__
from vine360.config import CaptureMode
from vine360.models import SCHEMA_VERSION, Project

PROJECT_FILE = "project.yaml"
INDEX_FILE = "index.sqlite"

# Directories created for every new project. "sfm/database.db" is created
# lazily by the COLMAP adapter in M4, not here. "sources", "datasets" and
# "cache" were dropped (docs/adr/0029) -- sources are recorded by
# reference and never copied in (ADR 0003), datasets was reserved for
# Training, which is out of scope (ADR 0025), and cache had no code
# ever reading or writing it.
LAYOUT_DIRS = [
    "frames",
    "projections",
    "masks/classes",
    "masks/keep",
    "sfm/sparse",
    "runs",
    "exports",
    "models",
]


def _ensure_layout_dirs(root: Path) -> None:
    """Idempotent (mkdir(exist_ok=True) throughout) -- called both when a
    project is created AND every time an existing one is opened, so a
    project made before a LAYOUT_DIRS entry was added (e.g. before
    "models" existed) gets it too, the same migrate-on-open philosophy
    _ensure_schema/_ensure_column already use for the database. "models"
    isn't written to or read from by any vine360 code -- it's where a
    user drops the trained model output (e.g. .ply/.splat) that comes
    back from Postshot or another external trainer after training on an
    exports/ bundle (see docs/adr/0024)."""
    for rel in LAYOUT_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)


class ProjectError(Exception):
    pass


class ProjectAlreadyExistsError(ProjectError):
    pass


class ProjectNotFoundError(ProjectError):
    pass


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Idempotent (CREATE TABLE IF NOT EXISTS throughout) -- called both
    when a project is created AND every time an existing one is opened, so
    a project made before a schema change (e.g. before views/masks/
    sfm_runs existed) gets upgraded rather than silently missing tables.
    Real bug this fixes: opening an older project and switching to the
    Projection panel raised "no such table: views", which its refresh
    handler didn't surface as an error -- it just looked like the source
    picker silently refused to populate."""
    conn.executescript(
        """
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                checksum TEXT NOT NULL,
                media_type TEXT NOT NULL,
                projection TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                timestamps TEXT NOT NULL,
                capture_group TEXT
            );
            CREATE TABLE IF NOT EXISTS frames (
                frame_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(source_id),
                source_time REAL NOT NULL,
                extraction_settings TEXT NOT NULL,
                path TEXT NOT NULL,
                checksum TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS frame_sets (
                frame_set_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(source_id),
                label TEXT NOT NULL,
                extraction_settings TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS views (
                view_id TEXT PRIMARY KEY,
                frame_id TEXT NOT NULL REFERENCES frames(frame_id),
                projection_id TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                intrinsics TEXT NOT NULL,
                fixed_rotation TEXT NOT NULL,
                image_path TEXT NOT NULL,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS masks (
                view_id TEXT PRIMARY KEY REFERENCES views(view_id),
                model TEXT NOT NULL,
                model_version TEXT NOT NULL,
                prompts TEXT NOT NULL,
                thresholds TEXT NOT NULL,
                morphology TEXT NOT NULL,
                keep_fraction REAL,
                edited INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS sfm_runs (
                run_id TEXT PRIMARY KEY,
                image_set_hash TEXT NOT NULL,
                engine_version TEXT NOT NULL,
                config TEXT NOT NULL,
                model_stats TEXT NOT NULL,
                selected_model TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS activity_log (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                label TEXT NOT NULL,
                target TEXT,
                params TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                duration_seconds REAL,
                result TEXT,
                error TEXT,
                origin TEXT
            );
            CREATE TABLE IF NOT EXISTS queue_jobs (
                job_id TEXT PRIMARY KEY,
                order_index INTEGER NOT NULL,
                stage TEXT NOT NULL,
                label TEXT NOT NULL,
                target_source_id TEXT,
                params TEXT NOT NULL,
                depends_on TEXT NOT NULL,
                status TEXT NOT NULL,
                auto_added INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT
            );
        """
    )
    conn.commit()
    _ensure_column(conn, "masks", "flagged_for_review", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "views", "updated_at", "TEXT")
    _ensure_column(conn, "masks", "updated_at", "TEXT")
    _ensure_column(conn, "frames", "frame_set_id", "TEXT")
    _ensure_column(conn, "queue_jobs", "target_frame_set_id", "TEXT")
    _migrate_legacy_frame_sets(conn)


def _migrate_legacy_frame_sets(conn: sqlite3.Connection) -> None:
    """Frames extracted before frame sets existed (docs/adr/0034) have no
    frame_set_id. Each source's existing frames become one frame set whose
    id is the source_id itself -- deliberately, so every path and id built
    from it (frames/<source_id>/, frame_id "<source_id>:NNNNNN",
    projections/<frame_id>/, masks/keep/<frame_id>/, a "frames"-engine
    sfm run's recorded source_id) is already correct as-is, with no file
    moved or row rewritten beyond setting frames.frame_set_id. Imports
    frame_set_label lazily: vine360.ingest.frames imports this module."""
    rows = conn.execute(
        "SELECT source_id, MIN(extraction_settings) FROM frames WHERE frame_set_id IS NULL GROUP BY source_id"
    ).fetchall()
    if not rows:
        return
    from vine360.ingest.frames import frame_set_label

    for source_id, settings_json in rows:
        settings = json.loads(settings_json)
        conn.execute(
            "INSERT OR IGNORE INTO frame_sets (frame_set_id, source_id, label, extraction_settings, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_id, source_id, frame_set_label(settings), settings_json, _utcnow_iso()),
        )
        conn.execute("UPDATE frames SET frame_set_id = ? WHERE source_id = ? AND frame_set_id IS NULL", (source_id, source_id))
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """CREATE TABLE IF NOT EXISTS handles new tables, but not new columns
    on a table that already exists -- an older project's `masks` table
    predates `flagged_for_review` and needs it added explicitly. Table/
    column names here are always our own hardcoded schema, never user
    input, so this f-string is not a SQL-injection concern."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        conn.commit()


def create_project(root: Path, name: str, capture_mode: CaptureMode) -> Project:
    root = Path(root)
    if (root / PROJECT_FILE).exists():
        raise ProjectAlreadyExistsError(f"{root} already contains a project")

    root.mkdir(parents=True, exist_ok=True)
    _ensure_layout_dirs(root)

    project = Project(
        schema_version=SCHEMA_VERSION,
        project_id=str(uuid.uuid4()),
        name=name,
        created_at=_utcnow_iso(),
        working_crs=None,
        software_versions={"vine360": __version__},
        capture_mode=capture_mode,
    )
    _write_project_yaml(root, project)
    conn = sqlite3.connect(root / INDEX_FILE)
    try:
        _ensure_schema(conn)
    finally:
        conn.close()
    return project


def _write_project_yaml(root: Path, project: Project) -> None:
    with open(root / PROJECT_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(project.to_dict(), f, sort_keys=False)


def load_project(root: Path) -> Project:
    root = Path(root)
    project_file = root / PROJECT_FILE
    if not project_file.exists():
        raise ProjectNotFoundError(f"no {PROJECT_FILE} found under {root}")
    with open(project_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Project.from_dict(data)


def index_db_path(root: Path) -> Path:
    return Path(root) / INDEX_FILE


def open_index_db(root: Path) -> sqlite3.Connection:
    path = index_db_path(root)
    if not path.exists():
        raise ProjectNotFoundError(f"no {INDEX_FILE} found under {root}")
    _ensure_layout_dirs(Path(root))
    conn = sqlite3.connect(path)
    _ensure_schema(conn)
    return conn
