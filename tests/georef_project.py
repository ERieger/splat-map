"""A synthetic, fully known georeferencing scene in a real vine360 project
(docs/adr/0047): the EstoWines Capture 2 control points, a known
model -> local similarity, pinhole views around the site as a *projections*
SfM run whose model is a real pycolmap reconstruction on disk, and tag
detections projected from ground truth (with optional pixel noise).

The run's model is written directly rather than reconstructed: SfM itself
is covered by tests/test_sfm_integration.py, and what's under test here is
georef's own wiring around a run (the same reason test_sfm_project_run.py
mocks run_sfm)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from georef_synthetic import intrinsics, look_at, project, random_rotation
from vine360.config import CaptureMode
from vine360.georef.control_points import USE_CHECK, import_control_points, load_control_points, update_control_point
from vine360.georef.similarity import Similarity
from vine360.project import create_project, open_index_db

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "gcp" / "ESTO1_GCPs_23Oct26.csv"
FRAME_SET = "src1~i1"
RUN_ID = "sfm-synthetic"
WIDTH = HEIGHT = 1024
FOCAL = 500.0


@dataclass
class Scene:
    root: Path
    conn: object
    set_id: str
    truth: Similarity  # model -> local
    run_id: str = RUN_ID


def build_scene(tmp_path: Path, *, noise_px: float = 0.3, seed: int = 0, check_points=("005",), small_tags=("020", "021")) -> Scene:
    import pycolmap

    rng = np.random.default_rng(seed)
    root = tmp_path / "project"
    create_project(root, "georef", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    set_id, _ = import_control_points(conn, FIXTURE, epsg=7854, tag_size_m=0.4)
    for name in check_points:
        update_control_point(conn, set_id, name, use=USE_CHECK)
    for name in small_tags:
        update_control_point(conn, set_id, name, tag_size_m=0.25)
    points = [p for p in load_control_points(conn, set_id) if p.tag_id is not None]
    origin = np.array([298900.0, 6126100.0, 0.0])

    truth = Similarity(0.42, random_rotation(rng), rng.normal(scale=50.0, size=3))  # model -> local
    inverse_R = truth.rotation.T

    def to_model(local: np.ndarray) -> np.ndarray:
        return (np.asarray(local) - truth.translation) @ inverse_R.T / truth.scale

    local_points = {p.tag_id: np.array([p.east, p.north, p.height]) - origin for p in points}
    sizes = {p.tag_id: p.tag_size_m for p in points}
    centroid = np.mean(list(local_points.values()), axis=0)

    # 24 cameras on two rings above the site, looking at its centre.
    K = intrinsics(WIDTH, HEIGHT, FOCAL)
    cameras_local = []
    for k in range(24):
        a = 2 * np.pi * k / 12
        radius, height = (55.0, 25.0) if k < 12 else (30.0, 40.0)
        center = centroid + np.array([radius * np.cos(a), radius * np.sin(a), height])
        cameras_local.append(center)

    conn.execute(
        "INSERT INTO sources (source_id, path, checksum, media_type, projection, timestamps) VALUES (?, ?, ?, ?, ?, ?)",
        ("src1", "/nowhere/clip.mp4", "x", "video", "equirectangular", "{}"),
    )
    conn.execute(
        "INSERT INTO frame_sets (frame_set_id, source_id, label, extraction_settings, created_at) VALUES (?, ?, ?, ?, ?)",
        (FRAME_SET, "src1", "every 1s", "{}", "2026-10-25T00:00:00+00:00"),
    )
    reconstruction = pycolmap.Reconstruction()
    camera = pycolmap.Camera(model="PINHOLE", width=WIDTH, height=HEIGHT, params=[FOCAL, FOCAL, WIDTH / 2, HEIGHT / 2], camera_id=1)
    reconstruction.add_camera_with_trivial_rig(camera)
    tracks: dict[int, list] = {}
    for k, center_local in enumerate(cameras_local):
        frame_id = f"{FRAME_SET}:{k:06d}"
        view_id = f"{frame_id}/front"
        image_path = f"projections/{frame_id}/front.png"
        conn.execute(
            "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (frame_id, "src1", float(k), "{}", f"frames/{FRAME_SET}/frame_{k:06d}.png", "x", FRAME_SET),
        )
        conn.execute(
            "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, fixed_rotation, image_path, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                view_id, frame_id, "front", WIDTH, HEIGHT,
                json.dumps({"width": WIDTH, "height": HEIGHT, "fx": FOCAL, "fy": FOCAL, "cx": WIDTH / 2, "cy": HEIGHT / 2}),
                json.dumps({"matrix": np.eye(3).tolist()}), image_path, "2026-10-25T00:00:00+00:00",
            ),
        )
        R, t = look_at(to_model(center_local), to_model(centroid), up=truth.rotation.T @ np.array([0.0, 0.0, 1.0]))
        keypoints = []
        for tag_id, local in local_points.items():
            X = to_model(local)
            cam = R @ X + t
            px = project(K, R, t, X)
            if cam[2] <= 0 or not (0 <= px[0] < WIDTH and 0 <= px[1] < HEIGHT):
                continue
            half = sizes[tag_id] / 2
            corners_local = [local + np.array(d) for d in ([-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0])]
            corners = np.array([project(K, R, t, to_model(c)) for c in corners_local]) + rng.normal(scale=noise_px, size=(4, 2))
            center_px = px + rng.normal(scale=noise_px, size=2)
            conn.execute(
                "INSERT INTO tag_observations (view_id, tag_family, tag_id, corners, center, decision_margin, hamming, "
                "detector, detector_version, params, detected_at) VALUES (?, 'tag36h11', ?, ?, ?, 80, 0, 'synthetic', "
                "'0', '{}', '2026-10-25T00:00:00+00:00')",
                (view_id, tag_id, json.dumps(corners.tolist()), json.dumps(center_px.tolist())),
            )
            tracks.setdefault(tag_id, []).append((k + 1, len(keypoints)))
            keypoints.append(px)
        conn.execute(
            "INSERT INTO tag_detection_runs (view_id, tag_family, n_tags, detected_at) VALUES (?, 'tag36h11', ?, ?)",
            (view_id, len(keypoints), "2026-10-25T00:00:00+00:00"),
        )
        image = pycolmap.Image(
            name=f"{frame_id}/front.png", keypoints=np.array(keypoints) if keypoints else np.zeros((0, 2)),
            camera_id=1, image_id=k + 1,
        )
        reconstruction.add_image_with_trivial_frame(image, pycolmap.Rigid3d(pycolmap.Rotation3d(R), t))
        image_file = root / image_path
        image_file.parent.mkdir(parents=True, exist_ok=True)
        image_file.write_bytes(f"image {k}".encode())  # export only copies bytes
    for tag_id, elements in tracks.items():
        track = pycolmap.Track()
        for image_id, idx in elements:
            track.add_element(image_id, idx)
        reconstruction.add_point3D(to_model(local_points[tag_id]), track, np.array([200, 200, 200], dtype=np.uint8))

    model_dir = root / "sfm" / "sparse" / RUN_ID / "0"
    model_dir.mkdir(parents=True)
    reconstruction.write(model_dir)
    conn.execute(
        "INSERT INTO sfm_runs (run_id, image_set_hash, engine_version, config, model_stats, selected_model, created_at) "
        "VALUES (?, 'x', '{}', ?, ?, ?, '2026-10-25T00:00:00+00:00')",
        (
            RUN_ID,
            json.dumps({"engine": "pycolmap", "image_source": "projections", "frame_set_id": FRAME_SET,
                        "frame_set_ids": [FRAME_SET], "source_id": "src1"}),
            json.dumps({"registered_images": 24, "total_images": 24, "registered_ratio": 1.0, "num_points3d": len(tracks),
                        "mean_reprojection_error": 0.0, "num_connected_models": 1}),
            f"sfm/sparse/{RUN_ID}/0",
        ),
    )
    conn.commit()
    return Scene(root, conn, set_id, truth)
