"""Project creation and the on-disk layout from the handover doc, section 5.

See docs/adr/0003-project-layout-and-index.md for the directory layout and
the decision to record sources by reference (checksum + path) rather than
copying media into the project on ingest.
"""

from __future__ import annotations

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
# lazily by the COLMAP adapter in M4, not here.
LAYOUT_DIRS = [
    "sources",
    "frames",
    "projections",
    "masks/classes",
    "masks/keep",
    "sfm/sparse",
    "datasets",
    "runs",
    "exports",
    "cache",
]


class ProjectError(Exception):
    pass


class ProjectAlreadyExistsError(ProjectError):
    pass


class ProjectNotFoundError(ProjectError):
    pass


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _init_index_db(root: Path) -> None:
    conn = sqlite3.connect(root / INDEX_FILE)
    try:
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
            CREATE TABLE IF NOT EXISTS views (
                view_id TEXT PRIMARY KEY,
                frame_id TEXT NOT NULL REFERENCES frames(frame_id),
                projection_id TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                intrinsics TEXT NOT NULL,
                fixed_rotation TEXT NOT NULL,
                image_path TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS masks (
                view_id TEXT PRIMARY KEY REFERENCES views(view_id),
                model TEXT NOT NULL,
                model_version TEXT NOT NULL,
                prompts TEXT NOT NULL,
                thresholds TEXT NOT NULL,
                morphology TEXT NOT NULL,
                keep_fraction REAL,
                edited INTEGER NOT NULL DEFAULT 0
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
            """
        )
        conn.commit()
    finally:
        conn.close()


def create_project(root: Path, name: str, capture_mode: CaptureMode) -> Project:
    root = Path(root)
    if (root / PROJECT_FILE).exists():
        raise ProjectAlreadyExistsError(f"{root} already contains a project")

    root.mkdir(parents=True, exist_ok=True)
    for rel in LAYOUT_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)

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
    _init_index_db(root)
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
    return sqlite3.connect(path)
