"""Persists View records + rendered face images for extracted frames
(handover doc, section 4, step 3 "Project"). Bridges the pure projection
library (geometry.py, cubemap.py, render.py -- which know nothing about
sqlite or the project layout) to the project's index database and
`project/projections/<frame_id>/<direction>.png` filesystem layout.
"""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from PIL import Image

from vine360.cleanup import remove_view_files
from vine360.models import View
from vine360.projection.cubemap import DEFAULT_RING_TILT_DEGREES, FaceSpec, six_face_preset
from vine360.projection.render import project_equirect_to_face


class ProjectionGenerationError(Exception):
    pass


def clear_views_for_frame(conn: sqlite3.Connection, project_root: Path, frame_id: str) -> None:
    """Removes a frame's views (and, per section 5's invalidation
    principle, any masks built from them) before regenerating -- the same
    clear-before-regenerate pattern used for frames, needed for the same
    reason: view_id is deterministic per frame+face, so a second
    generation with a different preset would otherwise collide or leave
    stale faces behind. Mask *files* (masks/keep/<frame_id>/,
    masks/classes/<view_id>/) are removed along with the rows -- they
    used to be left behind as orphans (docs/adr/0035)."""
    view_ids = [r[0] for r in conn.execute("SELECT view_id FROM views WHERE frame_id = ?", (frame_id,))]
    conn.execute(
        "DELETE FROM masks WHERE view_id IN (SELECT view_id FROM views WHERE frame_id = ?)", (frame_id,)
    )
    conn.execute(
        "DELETE FROM mask_layers WHERE view_id IN (SELECT view_id FROM views WHERE frame_id = ?)", (frame_id,)
    )
    conn.execute("DELETE FROM views WHERE frame_id = ?", (frame_id,))
    conn.commit()
    remove_view_files(project_root, frame_id, view_ids)


def _render_and_save_frame_views(
    project_root: Path, frame_id: str, equirect_relpath: str, faces: list[FaceSpec]
) -> list[View]:
    """The CPU-bound half of view generation for one frame -- opens the
    equirect image, renders + saves every face, builds the View objects --
    with no sqlite3.Connection involved at all. Deliberately a plain
    module-level function taking only picklable arguments (Path, str,
    FaceSpec dataclasses) so it can run unmodified inside a
    ProcessPoolExecutor worker (see generate_views_for_frame_set's parallel
    path) as well as in-process for the sequential path and single-frame
    generate_views_for_frame."""
    project_root = Path(project_root)
    with Image.open(project_root / equirect_relpath) as img:
        equirect = np.asarray(img.convert("RGB"))

    output_dir = project_root / "projections" / frame_id
    output_dir.mkdir(parents=True, exist_ok=True)

    views: list[View] = []
    for face in faces:
        rendered = project_equirect_to_face(equirect, face)
        image_path = output_dir / f"{face.name}.png"
        Image.fromarray(rendered, "RGB").save(image_path)

        views.append(
            View(
                view_id=f"{frame_id}:{face.name}",
                frame_id=frame_id,
                projection_id="directions",
                width=face.intrinsics.width,
                height=face.intrinsics.height,
                intrinsics=face.intrinsics.to_dict(),
                fixed_rotation=face.fixed_rotation_to_dict(),
                image_path=str(image_path.relative_to(project_root)),
            )
        )
    return views


def _insert_view_rows(conn: sqlite3.Connection, views: list[View]) -> None:
    for view in views:
        conn.execute(
            """
            INSERT INTO views
                (view_id, frame_id, projection_id, width, height, intrinsics, fixed_rotation, image_path, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                view.view_id,
                view.frame_id,
                view.projection_id,
                view.width,
                view.height,
                json.dumps(view.intrinsics),
                json.dumps(view.fixed_rotation),
                view.image_path,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
    conn.commit()


def generate_views_for_frame(
    conn: sqlite3.Connection,
    project_root: Path,
    frame_id: str,
    *,
    face_size: int = 1024,
    fov_degrees: float = 90.0,
    include_polar_faces: bool = False,
    face_names: list[str] | None = None,
    ring_tilt_degrees: float = DEFAULT_RING_TILT_DEGREES,
) -> list[View]:
    """face_names, if given, selects an explicit set of view directions
    to generate (any catalog name, overrides include_polar_faces);
    ring_tilt_degrees is the pitch of the -down/-up rings -- see
    vine360.projection.cubemap."""
    project_root = Path(project_root)
    row = conn.execute("SELECT path FROM frames WHERE frame_id = ?", (frame_id,)).fetchone()
    if row is None:
        raise ProjectionGenerationError(f"unknown frame_id: {frame_id}")

    clear_views_for_frame(conn, project_root, frame_id)

    faces = six_face_preset(
        face_size=face_size,
        fov_degrees=fov_degrees,
        include_polar_faces=include_polar_faces,
        face_names=face_names,
        ring_tilt_degrees=ring_tilt_degrees,
    )
    views = _render_and_save_frame_views(project_root, frame_id, row[0], faces)
    _insert_view_rows(conn, views)
    return views


def generate_views_for_frame_set(
    conn: sqlite3.Connection,
    project_root: Path,
    frame_set_id: str,
    *,
    face_size: int = 1024,
    fov_degrees: float = 90.0,
    include_polar_faces: bool = False,
    face_names: list[str] | None = None,
    ring_tilt_degrees: float = DEFAULT_RING_TILT_DEGREES,
    progress_callback=None,
    max_workers: int | None = None,
) -> list[View]:
    """progress_callback(message, current, total), current/total counted in
    frames processed (not individual face images). face_names /
    ring_tilt_degrees: as generate_views_for_frame.

    max_workers, when >= 2 and the frame set has more than one frame, renders
    frames concurrently across that many worker processes
    (ProcessPoolExecutor with a "spawn" context -- see docs/adr/0032).
    Defaults to None (sequential, in-process, today's exact behavior and
    progress-message shape) so every existing caller is unaffected. Each
    frame's views are still inserted into conn (and committed) on the main
    process as its result arrives -- worker processes never touch conn.

    Calling this with max_workers >= 2 from a plain script (not the GUI,
    which already guards its entry point -- see main_window.py's
    `if __name__ == "__main__":`) requires the caller's own top-level
    code to be guarded the same way -- a bare Python `spawn` requirement
    verified for real while building this: an unguarded caller raises
    RuntimeError("An attempt has been made to start a new process
    before the current process has finished its bootstrapping phase...")."""
    project_root = Path(project_root)
    notify = progress_callback or (lambda *a: None)
    frame_rows = conn.execute(
        "SELECT frame_id, path FROM frames WHERE frame_set_id = ? ORDER BY source_time", (frame_set_id,)
    ).fetchall()
    if not frame_rows:
        raise ProjectionGenerationError(f"no frames found for frame set {frame_set_id}; extract frames first")

    faces = six_face_preset(
        face_size=face_size,
        fov_degrees=fov_degrees,
        include_polar_faces=include_polar_faces,
        face_names=face_names,
        ring_tilt_degrees=ring_tilt_degrees,
    )

    # Clear every frame's stale views/masks up front, before any frame's
    # images are (re)written -- must fully precede rendering (parallel or
    # not) so a worker never writes a frame's new files while that
    # frame's old directory hasn't been cleared yet.
    for frame_id, _path in frame_rows:
        clear_views_for_frame(conn, project_root, frame_id)

    total = len(frame_rows)
    all_views: list[View] = []

    if not max_workers or max_workers <= 1 or total <= 1:
        for index, (frame_id, path) in enumerate(frame_rows):
            notify(f"Projecting frame {index + 1}/{total}…", index, total)
            views = _render_and_save_frame_views(project_root, frame_id, path, faces)
            _insert_view_rows(conn, views)
            all_views.extend(views)
    else:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn")) as pool:
            futures = {
                pool.submit(_render_and_save_frame_views, project_root, frame_id, path, faces): frame_id
                for frame_id, path in frame_rows
            }
            completed = 0
            for future in as_completed(futures):
                views = future.result()
                _insert_view_rows(conn, views)
                all_views.extend(views)
                completed += 1
                notify(f"Projecting frame {completed}/{total}…", completed, total)

    notify(f"Projected {total} frames ({len(all_views)} views, {len(faces)} per frame).", total, total)
    return all_views
