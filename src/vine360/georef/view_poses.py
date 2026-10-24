"""Every registered projected view's COLMAP-convention camera, for either
kind of SfM run (docs/adr/0047) -- the only place in vine360.georef that
knows the two kinds apart:

- a raw-frame (360) run, SphereSfM or pycolmap EQUIRECTANGULAR: each
  registered frame's pose turned into its views' poses by
  vine360.sfm.frame_poses (load_frame_model + view_cam_from_world, the
  conversion pinned in docs/adr/0036);
- a projections run: the model's images *are* the views, named by
  image_path relative to projections/ (vine360.sfm.project_run), read with
  pycolmap and matched back through the views table.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vine360.sfm.frame_poses import (
    FramePoseError,
    ViewRecord,
    is_frame_run,
    load_frame_model,
    view_cam_from_world,
    views_for_frame_set,
)
from vine360.sfm.project_run import run_frame_set_ids


@dataclass
class ViewPose:
    R: np.ndarray  # cam_from_world, COLMAP convention (y down)
    t: np.ndarray
    view: ViewRecord

    @property
    def K(self) -> np.ndarray:
        v = self.view
        return np.array([[v.fx, 0.0, v.cx], [0.0, v.fy, v.cy], [0.0, 0.0, 1.0]])


def run_config(conn: sqlite3.Connection, run_id: str) -> dict:
    row = conn.execute("SELECT config FROM sfm_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise FramePoseError(f"unknown sfm run: {run_id}")
    return json.loads(row[0])


def run_scope_frame_set_ids(conn: sqlite3.Connection, run_id: str) -> list[str]:
    """The frame sets a run's views come from: its recorded scope, or -- for
    a project-wide projections run -- every frame set with projected views."""
    ids = run_frame_set_ids(run_config(conn, run_id))
    if ids:
        return ids
    return [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT f.frame_set_id FROM views v JOIN frames f ON f.frame_id = v.frame_id "
            "WHERE f.frame_set_id IS NOT NULL ORDER BY f.frame_set_id"
        )
    ]


def view_poses_for_run(conn: sqlite3.Connection, project_root: Path, run_id: str) -> dict[str, ViewPose]:
    """{view_id: ViewPose} for every projected view the run registered.
    Raises FramePoseError for an unknown run or a missing model."""
    project_root = Path(project_root)
    config = run_config(conn, run_id)
    if is_frame_run(config):
        model = load_frame_model(conn, project_root, run_id)
        out = {}
        for view in views_for_frame_set(conn, model.frame_set_ids):
            pose = model.poses.get(view.frame_id)
            if pose is None:
                continue
            R, t = view_cam_from_world(pose, view.fixed_rotation, model.engine)
            out[view.view_id] = ViewPose(R, t, view)
        return out

    import pycolmap

    row = conn.execute("SELECT selected_model FROM sfm_runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row or not row[0]:
        raise FramePoseError(f"sfm run {run_id!r} has no recorded model -- re-run Pose estimation")
    model_dir = project_root / row[0]
    if not model_dir.exists():
        raise FramePoseError(f"sfm run {run_id!r}'s model is missing on disk at {model_dir}")
    views_by_name = {
        Path(v.image_path).relative_to("projections").as_posix(): v
        for v in views_for_frame_set(conn, run_scope_frame_set_ids(conn, run_id))
    }
    reconstruction = pycolmap.Reconstruction(model_dir)
    out = {}
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        view = views_by_name.get(image.name)
        if view is None:
            continue
        cam_from_world = image.cam_from_world()
        out[view.view_id] = ViewPose(
            np.asarray(cam_from_world.rotation.matrix()), np.asarray(cam_from_world.translation), view
        )
    return out
