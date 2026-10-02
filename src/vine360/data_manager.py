"""Listing and cleaning up a project's derived data -- the library half of
the GUI's Data manager tab (docs/adr/0035). Qt-free so every delete is
testable directly.

Sources are listed only, never modified here (removing a source stays on
Import, where its cascade already lives). Every delete preserves the
cascade-delete invariant (docs/adr/0017): removing something also removes
everything derived from it, rows and files both, and nothing upstream or
in a sibling frame set.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from vine360.cleanup import remove_mask_files
from vine360.ingest.frames import clear_frame_set
from vine360.projection.generate import clear_views_for_frame

EXPORT_FORMAT_DIRNAMES = ("colmap", "postshot", "realityscan")
# What an export function writes into its output folder (vine360.export.
# postshot's managed subdirectories plus RealityScan's priors CSV) -- the
# only things an export written straight into exports/ owns there.
EXPORT_CONTENT_NAMES = ("images", "masks", "sparse", "CameraPriors.csv")


class DataManagerError(Exception):
    pass


@dataclass
class SourceInfo:
    source_id: str
    path: str
    media_type: str
    projection: str
    capture_group: str | None
    duration_seconds: float | None
    frame_set_count: int

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass
class FrameSetInfo:
    frame_set_id: str
    source_id: str
    source_path: str
    source_projection: str
    label: str
    created_at: str
    frame_count: int
    view_count: int
    mask_count: int

    @property
    def display_name(self) -> str:
        return f"{Path(self.source_path).name} — {self.label}"


@dataclass
class SfmRunInfo:
    run_id: str
    created_at: str
    image_source: str
    frame_set_id: str | None
    registered_images: int
    total_images: int
    engine: str = "pycolmap"  # or "spheresfm" (docs/adr/0036)

    @property
    def engine_label(self) -> str:
        if self.engine == "spheresfm":
            return "SphereSfM (raw 360 frames)"
        return "COLMAP equirectangular (raw frames)" if self.image_source == "frames" else "COLMAP six-face projections"


@dataclass
class ExportDirInfo:
    relative_path: str  # relative to the project root, e.g. "exports/all/postshot"
    legacy_layout: bool  # a pre-ADR-0028 flat exports/<capture>/{images,masks,sparse}/ folder
    # An export written with exports/ itself chosen as the output folder:
    # its images/masks/sparse sit directly in exports/, next to the
    # <capture>/ folders, so only EXPORT_CONTENT_NAMES belong to it.
    in_exports_root: bool = False


def list_sources(conn: sqlite3.Connection) -> list[SourceInfo]:
    rows = conn.execute(
        "SELECT s.source_id, s.path, s.media_type, s.projection, s.capture_group, s.timestamps, "
        "(SELECT COUNT(*) FROM frame_sets fs WHERE fs.source_id = s.source_id) "
        "FROM sources s ORDER BY s.rowid"
    ).fetchall()
    result = []
    for source_id, path, media_type, projection, capture_group, timestamps_json, n_sets in rows:
        timestamps = json.loads(timestamps_json) if timestamps_json else {}
        result.append(
            SourceInfo(
                source_id=source_id,
                path=path,
                media_type=media_type,
                projection=projection,
                capture_group=capture_group,
                duration_seconds=timestamps.get("duration_seconds"),
                frame_set_count=n_sets,
            )
        )
    return result


def list_frame_sets(conn: sqlite3.Connection, *, source_id: str | None = None) -> list[FrameSetInfo]:
    """Every frame set (optionally just one source's), ordered by source
    then creation time, with frame/view/mask counts."""
    where, args = ("WHERE fs.source_id = ?", (source_id,)) if source_id else ("", ())
    rows = conn.execute(
        "SELECT fs.frame_set_id, fs.source_id, s.path, s.projection, fs.label, fs.created_at, "
        "(SELECT COUNT(*) FROM frames f WHERE f.frame_set_id = fs.frame_set_id), "
        "(SELECT COUNT(*) FROM views v JOIN frames f ON f.frame_id = v.frame_id "
        " WHERE f.frame_set_id = fs.frame_set_id), "
        "(SELECT COUNT(*) FROM masks m JOIN views v ON v.view_id = m.view_id "
        " JOIN frames f ON f.frame_id = v.frame_id WHERE f.frame_set_id = fs.frame_set_id) "
        f"FROM frame_sets fs JOIN sources s ON s.source_id = fs.source_id {where} "
        "ORDER BY s.rowid, fs.created_at, fs.frame_set_id",
        args,
    ).fetchall()
    return [FrameSetInfo(*row) for row in rows]


def directory_size(path: Path) -> int:
    """Total bytes of every file under path (0 if missing). Never raises on
    a file vanishing mid-walk."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def frame_set_disk_usage(conn: sqlite3.Connection, project_root: Path, frame_set_id: str) -> dict[str, int]:
    """Bytes on disk per artifact kind for one frame set. Walks the
    filesystem -- slow on a big project, so the GUI calls it off the GUI
    thread."""
    project_root = Path(project_root)
    frame_ids = [r[0] for r in conn.execute("SELECT frame_id FROM frames WHERE frame_set_id = ?", (frame_set_id,))]
    view_ids = [
        r[0]
        for r in conn.execute(
            "SELECT v.view_id FROM views v JOIN frames f ON f.frame_id = v.frame_id WHERE f.frame_set_id = ?",
            (frame_set_id,),
        )
    ]
    return {
        "frames": directory_size(project_root / "frames" / frame_set_id),
        "projections": sum(directory_size(project_root / "projections" / fid) for fid in frame_ids),
        "masks": sum(directory_size(project_root / "masks" / "keep" / fid) for fid in frame_ids)
        + sum(directory_size(project_root / "masks" / "classes" / vid) for vid in view_ids),
    }


def delete_frame_set(conn: sqlite3.Connection, project_root: Path, frame_set_id: str) -> None:
    if conn.execute("SELECT 1 FROM frame_sets WHERE frame_set_id = ?", (frame_set_id,)).fetchone() is None:
        raise DataManagerError(f"unknown frame set: {frame_set_id}")
    clear_frame_set(conn, project_root, frame_set_id)


def delete_views_for_frame_set(conn: sqlite3.Connection, project_root: Path, frame_set_id: str) -> int:
    """Removes a frame set's projected views and the masks built from them;
    keeps the frames. Returns the number of frames cleared."""
    frame_ids = [r[0] for r in conn.execute("SELECT frame_id FROM frames WHERE frame_set_id = ?", (frame_set_id,))]
    for frame_id in frame_ids:
        clear_views_for_frame(conn, project_root, frame_id)
    return len(frame_ids)


def delete_masks_for_frame_set(conn: sqlite3.Connection, project_root: Path, frame_set_id: str) -> int:
    """Removes a frame set's masks only; keeps frames and views. Returns
    the number of mask rows deleted."""
    by_frame: dict[str, list[str]] = {}
    for view_id, frame_id in conn.execute(
        "SELECT v.view_id, v.frame_id FROM views v JOIN frames f ON f.frame_id = v.frame_id "
        "WHERE f.frame_set_id = ?",
        (frame_set_id,),
    ):
        by_frame.setdefault(frame_id, []).append(view_id)
    conn.execute(
        "DELETE FROM mask_layers WHERE view_id IN (SELECT v.view_id FROM views v JOIN frames f "
        "ON f.frame_id = v.frame_id WHERE f.frame_set_id = ?)",
        (frame_set_id,),
    )
    cursor = conn.execute(
        "DELETE FROM masks WHERE view_id IN (SELECT v.view_id FROM views v JOIN frames f "
        "ON f.frame_id = v.frame_id WHERE f.frame_set_id = ?)",
        (frame_set_id,),
    )
    conn.commit()
    for frame_id, view_ids in by_frame.items():
        remove_mask_files(project_root, frame_id, view_ids)
    return cursor.rowcount


def list_sfm_runs(conn: sqlite3.Connection) -> list[SfmRunInfo]:
    rows = conn.execute("SELECT run_id, created_at, config, model_stats FROM sfm_runs ORDER BY created_at DESC").fetchall()
    result = []
    for run_id, created_at, config_json, stats_json in rows:
        config = json.loads(config_json)
        stats = json.loads(stats_json)
        image_source = config.get("image_source", "projections")
        # Pre-frame-set "frames"-engine runs only recorded source_id, which
        # is also their (migrated, legacy) frame_set_id -- docs/adr/0034.
        frame_set_id = config.get("frame_set_id") or (config.get("source_id") if image_source == "frames" else None)
        result.append(
            SfmRunInfo(
                run_id=run_id,
                created_at=created_at,
                image_source=image_source,
                frame_set_id=frame_set_id,
                registered_images=stats.get("registered_images", 0),
                total_images=stats.get("total_images", 0),
                engine=config.get("engine", "pycolmap"),
            )
        )
    return result


def delete_sfm_run(conn: sqlite3.Connection, project_root: Path, run_id: str) -> None:
    """Removes the run's row and its own sfm/sparse/<run_id>/ directory
    (models plus, for runs since docs/adr/0034, its COLMAP database).
    Runs recorded before per-run directories existed (docs/adr/0019)
    point into the shared sfm/sparse/<key>/ layout -- those files can't
    be safely attributed to one run, so only the row is removed."""
    if conn.execute("SELECT 1 FROM sfm_runs WHERE run_id = ?", (run_id,)).fetchone() is None:
        raise DataManagerError(f"unknown sfm run: {run_id}")
    conn.execute("DELETE FROM sfm_runs WHERE run_id = ?", (run_id,))
    conn.commit()
    run_dir = Path(project_root) / "sfm" / "sparse" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)


def list_export_dirs(project_root: Path) -> list[ExportDirInfo]:
    """Every exports/<capture>/<format>/ folder (docs/adr/0028), any
    pre-0028 flat exports/<capture>/ folder still holding images/masks/
    sparse directly, and an export written straight into exports/ itself
    (listed as "exports", in_exports_root=True)."""
    exports_root = Path(project_root) / "exports"
    if not exports_root.is_dir():
        return []
    result = []
    if any((exports_root / name).exists() for name in EXPORT_CONTENT_NAMES):
        result.append(ExportDirInfo("exports", legacy_layout=False, in_exports_root=True))
    for capture in sorted(p for p in exports_root.iterdir() if p.is_dir() and p.name not in EXPORT_CONTENT_NAMES):
        children = [c for c in capture.iterdir() if c.is_dir()]
        for child in sorted(children):
            if child.name in EXPORT_FORMAT_DIRNAMES:
                result.append(ExportDirInfo(child.relative_to(project_root).as_posix(), legacy_layout=False))
        if any(c.name not in EXPORT_FORMAT_DIRNAMES for c in children):
            result.append(ExportDirInfo(capture.relative_to(project_root).as_posix(), legacy_layout=True))
    return result


def export_dir_size(project_root: Path, info: ExportDirInfo) -> int:
    """Bytes on disk for one listed export -- for an export straight into
    exports/, just its own contents, not the <capture>/ folders beside it."""
    root = Path(project_root) / info.relative_path
    if not info.in_exports_root:
        return directory_size(root)
    total = 0
    for name in EXPORT_CONTENT_NAMES:
        path = root / name
        if path.is_file():
            total += path.stat().st_size
        else:
            total += directory_size(path)
    return total


def delete_export_dir(project_root: Path, relative_path: str) -> None:
    """Deletes one folder under exports/ -- refuses anything that doesn't
    resolve strictly inside it. Removes the now-empty <capture>/ parent
    too. relative_path "exports" itself means an export written straight
    into exports/: only its EXPORT_CONTENT_NAMES are removed, never the
    <capture>/ folders beside them."""
    exports_root = (Path(project_root) / "exports").resolve()
    target = (Path(project_root) / relative_path).resolve()
    if target == exports_root:
        present = [exports_root / name for name in EXPORT_CONTENT_NAMES if (exports_root / name).exists()]
        if not present:
            raise DataManagerError("no export written directly into exports/")
        for path in present:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        return
    if target == exports_root or exports_root not in target.parents:
        raise DataManagerError(f"refusing to delete {relative_path!r}: not a folder inside exports/")
    if not target.is_dir():
        raise DataManagerError(f"no such export folder: {relative_path}")
    shutil.rmtree(target)
    parent = target.parent
    if parent != exports_root and not any(parent.iterdir()):
        parent.rmdir()
