"""Poses of raw 360 frames -> poses of vine360's own projected views
(docs/adr/0036).

Both frame-based SfM engines -- SphereSfM and pycolmap's native
EQUIRECTANGULAR model -- reconstruct whole equirectangular frames. Postshot
and other pinhole/COLMAP trainers need ordinary perspective cameras, and
vine360's masks are built on its own projected cube-face views, not on raw
frames. Every view of a frame shares the frame's optical centre (zero-
translation fixed_rotation, vine360.projection.cubemap), so a view's pose
is just the frame's pose rotated by that view's fixed_rotation -- no
re-estimation needed.

Conventions (measured, not assumed -- tests/test_frame_poses.py pins them):
- vine360's panorama/face frames are x right, **y up**, z forward
  (vine360.projection.geometry); COLMAP cameras are x right, **y down**,
  z forward. F = diag(1, -1, 1) converts between the two.
- Each engine's 360 camera frame relates to vine360's panorama frame by a
  fixed matrix M (direction_vine360 = M @ bearing_engine). Fitted on real
  reconstructions of a synthetic 360 sequence (tests/synthetic_360.py) for
  both engines: M = F, to within 0.04 degrees median angular error.
- vine360 face pixel index p maps to COLMAP pixel coordinate p + 0.5, so a
  view's COLMAP PINHOLE params are exactly its stored (fx, fy, cx, cy).

So for a frame with COLMAP-convention cam_from_world rotation R_frame and
centre C, and a view with R_panorama_face = R_face:

    R_view = F @ R_face.T @ M.T @ R_frame,   t_view = -R_view @ C

**pycolmap never loads a SphereSfM model** -- its SPHERE camera model id
(11) is a different model in pycolmap (vine360.sfm.spheresfm_adapter).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vine360.sfm.project_run import ENGINE_PYCOLMAP, ENGINE_SPHERESFM, run_engine

F = np.diag([1.0, -1.0, 1.0])
BEARING_CONVENTION = {ENGINE_SPHERESFM: F, ENGINE_PYCOLMAP: F}  # measured -- see module docstring


class FramePoseError(Exception):
    pass


@dataclass
class FramePose:
    frame_id: str
    rotation: np.ndarray  # cam_from_world, the engine's own camera convention
    center: np.ndarray  # world


@dataclass
class SparsePoint:
    xyz: np.ndarray
    rgb: tuple[int, int, int]
    frame_ids: list[str] = field(default_factory=list)  # frames that observed it


@dataclass
class FrameModel:
    run_id: str
    engine: str
    frame_set_id: str
    poses: dict[str, FramePose]
    points: list[SparsePoint]


def quaternion_to_matrix(qvec) -> np.ndarray:
    w, x, y, z = (float(v) for v in qvec)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _run_row(conn: sqlite3.Connection, run_id: str) -> tuple[str, dict]:
    row = conn.execute("SELECT selected_model, config FROM sfm_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise FramePoseError(f"unknown sfm run: {run_id}")
    selected_model, config_json = row
    if not selected_model:
        raise FramePoseError(f"sfm run {run_id!r} has no recorded model -- re-run Pose estimation")
    return selected_model, json.loads(config_json)


def is_frame_run(config: dict) -> bool:
    return config.get("image_source", "projections") == "frames"


def load_frame_model(conn: sqlite3.Connection, project_root: Path, run_id: str) -> FrameModel:
    """A frame-based run's registered frames (keyed by frame_id, matched
    through frames.path) and its sparse points, from either engine."""
    project_root = Path(project_root)
    selected_model, config = _run_row(conn, run_id)
    if not is_frame_run(config):
        raise FramePoseError(f"sfm run {run_id!r} isn't a raw-frame (360) run")
    frame_set_id = config.get("frame_set_id") or config.get("source_id")
    if not frame_set_id:
        raise FramePoseError(f"sfm run {run_id!r} has no frame_set_id")
    model_dir = project_root / selected_model
    if not model_dir.exists():
        raise FramePoseError(f"sfm run {run_id!r}'s model is missing on disk at {model_dir}")

    frame_id_by_name = {
        Path(path).name: frame_id
        for frame_id, path in conn.execute("SELECT frame_id, path FROM frames WHERE frame_set_id = ?", (frame_set_id,))
    }
    engine = run_engine(config)
    poses: dict[str, FramePose] = {}
    points: list[SparsePoint] = []

    if engine == ENGINE_SPHERESFM:
        from vine360.sfm.spheresfm_adapter import read_text_model

        txt_dir = model_dir / "txt"
        if not (txt_dir / "images.txt").exists():
            raise FramePoseError(f"SphereSfM run {run_id!r} has no TXT model at {txt_dir}")
        model = read_text_model(txt_dir)
        frame_by_image_id = {}
        for image in model.images.values():
            frame_id = frame_id_by_name.get(Path(image.name).name)
            if frame_id is None:
                continue
            rotation = quaternion_to_matrix(image.qvec)
            poses[frame_id] = FramePose(frame_id, rotation, -rotation.T @ np.asarray(image.tvec))
            frame_by_image_id[image.image_id] = frame_id
        for point in model.points.values():
            frames = sorted({frame_by_image_id[i] for i, _ in point.track if i in frame_by_image_id})
            if frames:
                points.append(SparsePoint(np.asarray(point.xyz), point.rgb, frames))
    elif engine == ENGINE_PYCOLMAP:
        import pycolmap

        reconstruction = pycolmap.Reconstruction(model_dir)
        frame_by_image_id = {}
        for image_id, image in reconstruction.images.items():
            if not image.has_pose:
                continue
            frame_id = frame_id_by_name.get(Path(image.name).name)
            if frame_id is None:
                continue
            cam_from_world = image.cam_from_world()
            rotation = np.asarray(cam_from_world.rotation.matrix())
            poses[frame_id] = FramePose(frame_id, rotation, np.asarray(image.projection_center()))
            frame_by_image_id[image_id] = frame_id
        for point in reconstruction.points3D.values():
            frames = sorted(
                {frame_by_image_id[e.image_id] for e in point.track.elements if e.image_id in frame_by_image_id}
            )
            if frames:
                points.append(SparsePoint(np.asarray(point.xyz), tuple(int(c) for c in point.color), frames))
    else:
        raise FramePoseError(f"unknown SfM engine {engine!r} for run {run_id!r}")

    if not poses:
        raise FramePoseError(f"sfm run {run_id!r} registered none of frame set {frame_set_id!r}'s frames")
    return FrameModel(run_id, engine, frame_set_id, poses, points)


def view_cam_from_world(pose: FramePose, fixed_rotation: np.ndarray, engine: str) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of a projected view as a COLMAP-convention pinhole camera."""
    convention = BEARING_CONVENTION[engine]
    rotation = F @ np.asarray(fixed_rotation).T @ convention.T @ pose.rotation
    return rotation, -rotation @ pose.center


@dataclass
class ViewRecord:
    view_id: str
    frame_id: str
    image_path: str  # relative to the project, e.g. "projections/<frame_id>/front.png"
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    fixed_rotation: np.ndarray


def views_for_frame_set(conn: sqlite3.Connection, frame_set_id: str) -> list[ViewRecord]:
    rows = conn.execute(
        "SELECT v.view_id, v.frame_id, v.image_path, v.width, v.height, v.intrinsics, v.fixed_rotation "
        "FROM views v JOIN frames f ON f.frame_id = v.frame_id WHERE f.frame_set_id = ? ORDER BY v.view_id",
        (frame_set_id,),
    ).fetchall()
    records = []
    for view_id, frame_id, image_path, width, height, intrinsics_json, rotation_json in rows:
        intrinsics = json.loads(intrinsics_json)
        records.append(
            ViewRecord(
                view_id, frame_id, image_path, width, height,
                intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"],
                np.asarray(json.loads(rotation_json)["matrix"]),
            )
        )
    return records


def build_view_reconstruction(conn: sqlite3.Connection, project_root: Path, run_id: str):
    """A standard PINHOLE pycolmap.Reconstruction over the projected views
    of the run's registered frames: one image per view (named by its path
    relative to projections/, exactly like a six-face run's images), and
    the run's sparse points, each observed in every view of an observing
    frame it projects into (in front of the camera, inside the image) --
    the same visibility rule as SphereSfM's own ExportPerspectiveCubic.

    Returns (reconstruction, model) -- raises FramePoseError if the frame
    set has no projected views."""
    import pycolmap

    model = load_frame_model(conn, project_root, run_id)
    views = [v for v in views_for_frame_set(conn, model.frame_set_id) if v.frame_id in model.poses]
    if not views:
        raise FramePoseError(
            f"no projected views for frame set {model.frame_set_id!r} -- generate projections for this frame "
            "set first (the export uses vine360's own projected views, so masks can come along)"
        )

    reconstruction = pycolmap.Reconstruction()
    camera_ids: dict[tuple, int] = {}
    view_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    views_by_frame: dict[str, list[ViewRecord]] = {}
    for view in views:
        key = (view.width, view.height, view.fx, view.fy, view.cx, view.cy)
        if key not in camera_ids:
            camera_ids[key] = len(camera_ids) + 1
            camera = pycolmap.Camera(
                model="PINHOLE", width=view.width, height=view.height,
                params=[view.fx, view.fy, view.cx, view.cy], camera_id=camera_ids[key],
            )
            reconstruction.add_camera_with_trivial_rig(camera)
        view_poses[view.view_id] = view_cam_from_world(model.poses[view.frame_id], view.fixed_rotation, model.engine)
        views_by_frame.setdefault(view.frame_id, []).append(view)

    # Project every sparse point into the views of the frames that saw it.
    keypoints: dict[str, list[np.ndarray]] = {v.view_id: [] for v in views}
    tracks: list[tuple[SparsePoint, list[tuple[str, int]]]] = []
    for point in model.points:
        elements = []
        for frame_id in point.frame_ids:
            for view in views_by_frame.get(frame_id, []):
                rotation, translation = view_poses[view.view_id]
                cam = rotation @ point.xyz + translation
                if cam[2] <= 1e-9:
                    continue
                x = view.fx * cam[0] / cam[2] + view.cx
                y = view.fy * cam[1] / cam[2] + view.cy
                if 0 <= x < view.width and 0 <= y < view.height:
                    keypoints[view.view_id].append(np.array([x, y]))
                    elements.append((view.view_id, len(keypoints[view.view_id]) - 1))
        if elements:
            tracks.append((point, elements))

    image_ids = {}
    for index, view in enumerate(views, start=1):
        rotation, translation = view_poses[view.view_id]
        points2d = np.array(keypoints[view.view_id]) if keypoints[view.view_id] else np.zeros((0, 2))
        image = pycolmap.Image(
            name=Path(view.image_path).relative_to("projections").as_posix(),
            keypoints=points2d,
            camera_id=camera_ids[(view.width, view.height, view.fx, view.fy, view.cx, view.cy)],
            image_id=index,
        )
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), translation)
        reconstruction.add_image_with_trivial_frame(image, cam_from_world)
        image_ids[view.view_id] = index

    for point, elements in tracks:
        track = pycolmap.Track()
        for view_id, point2d_idx in elements:
            track.add_element(image_ids[view_id], point2d_idx)
        reconstruction.add_point3D(point.xyz, track, np.asarray(point.rgb, dtype=np.uint8))
    return reconstruction, model
