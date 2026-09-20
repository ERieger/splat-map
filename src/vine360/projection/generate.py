"""Persists View records + rendered face images for extracted frames
(handover doc, section 4, step 3 "Project"). Bridges the pure projection
library (geometry.py, cubemap.py, render.py -- which know nothing about
sqlite or the project layout) to the project's index database and
`project/projections/<frame_id>/<face>.png` filesystem layout.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image

from vine360.models import View
from vine360.projection.cubemap import six_face_preset
from vine360.projection.render import project_equirect_to_face


class ProjectionGenerationError(Exception):
    pass


def clear_views_for_frame(conn: sqlite3.Connection, project_root: Path, frame_id: str) -> None:
    """Removes a frame's views (and, per section 5's invalidation
    principle, any masks built from them) before regenerating -- the same
    clear-before-regenerate pattern used for frames, needed for the same
    reason: view_id is deterministic per frame+face, so a second
    generation with a different preset would otherwise collide or leave
    stale faces behind."""
    conn.execute(
        "DELETE FROM masks WHERE view_id IN (SELECT view_id FROM views WHERE frame_id = ?)", (frame_id,)
    )
    conn.execute("DELETE FROM views WHERE frame_id = ?", (frame_id,))
    conn.commit()
    output_dir = Path(project_root) / "projections" / frame_id
    if output_dir.exists():
        shutil.rmtree(output_dir)


def generate_views_for_frame(
    conn: sqlite3.Connection,
    project_root: Path,
    frame_id: str,
    *,
    face_size: int = 1024,
    fov_degrees: float = 90.0,
    include_polar_faces: bool = False,
    face_names: list[str] | None = None,
) -> list[View]:
    """face_names, if given, selects an explicit subset of faces to
    generate (overrides include_polar_faces) -- see
    vine360.projection.cubemap.six_face_preset."""
    project_root = Path(project_root)
    row = conn.execute("SELECT path FROM frames WHERE frame_id = ?", (frame_id,)).fetchone()
    if row is None:
        raise ProjectionGenerationError(f"unknown frame_id: {frame_id}")

    clear_views_for_frame(conn, project_root, frame_id)

    with Image.open(project_root / row[0]) as img:
        equirect = np.asarray(img.convert("RGB"))

    faces = six_face_preset(
        face_size=face_size, fov_degrees=fov_degrees, include_polar_faces=include_polar_faces, face_names=face_names
    )
    output_dir = project_root / "projections" / frame_id
    output_dir.mkdir(parents=True, exist_ok=True)

    views: list[View] = []
    for face in faces:
        rendered = project_equirect_to_face(equirect, face)
        image_path = output_dir / f"{face.name}.png"
        Image.fromarray(rendered, "RGB").save(image_path)

        view = View(
            view_id=f"{frame_id}:{face.name}",
            frame_id=frame_id,
            projection_id="six-face",
            width=face.intrinsics.width,
            height=face.intrinsics.height,
            intrinsics=face.intrinsics.to_dict(),
            fixed_rotation=face.fixed_rotation_to_dict(),
            image_path=str(image_path.relative_to(project_root)),
        )
        views.append(view)
        conn.execute(
            """
            INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, fixed_rotation, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
    conn.commit()
    return views


def generate_views_for_source(
    conn: sqlite3.Connection,
    project_root: Path,
    source_id: str,
    *,
    face_size: int = 1024,
    fov_degrees: float = 90.0,
    include_polar_faces: bool = False,
    face_names: list[str] | None = None,
    progress_callback=None,
) -> list[View]:
    """progress_callback(message, current, total), current/total counted in
    frames processed (not individual face images)."""
    notify = progress_callback or (lambda *a: None)
    frame_rows = conn.execute(
        "SELECT frame_id FROM frames WHERE source_id = ? ORDER BY source_time", (source_id,)
    ).fetchall()
    if not frame_rows:
        raise ProjectionGenerationError(f"no frames found for source {source_id}; extract frames first")

    total = len(frame_rows)
    all_views: list[View] = []
    for index, (frame_id,) in enumerate(frame_rows):
        notify(f"Projecting frame {index + 1}/{total}…", index, total)
        all_views.extend(
            generate_views_for_frame(
                conn,
                project_root,
                frame_id,
                face_size=face_size,
                fov_degrees=fov_degrees,
                include_polar_faces=include_polar_faces,
                face_names=face_names,
            )
        )
    notify(f"Projected {total} frames ({len(all_views)} views).", total, total)
    return all_views
