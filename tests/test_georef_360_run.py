"""Georeferencing a *real* raw-frame (360) reconstruction -- pycolmap
EQUIRECTANGULAR (and SphereSfM when built) on the synthetic ray-cast room
(tests/synthetic_360.py) -- end to end through view_poses_for_run,
triangulation, the solve and the materialised pinhole model
(docs/adr/0047).

The room has no tags, so a handful of the run's own sparse points stand in
for them: each point's *observed* equirect pixel is carried into the
projected views purely through vine360's projection geometry (exactly as
tests/test_frame_poses.py checks view poses), and becomes a "detection" in
every view it lands in. Their surveyed coordinates are the points mapped
through a known similarity plus an origin."""

import json

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")

from conftest import HAVE_SPHERESFM
from georef_synthetic import random_rotation
from synthetic_360 import add_synthetic_frame_set
from vine360.config import CaptureMode
from vine360.georef.control_points import import_control_points
from vine360.georef.similarity import Similarity
from vine360.georef.solve import load_solution, solve_georef
from vine360.georef.view_poses import view_poses_for_run
from vine360.project import create_project, open_index_db
from vine360.projection.generate import generate_views_for_frame_set
from vine360.projection.geometry import Intrinsics, direction_to_camera_pixel, equirect_pixel_to_direction
from vine360.sfm.colmap_adapter import SfmConfig
from vine360.sfm.frame_poses import views_for_frame_set
from vine360.sfm.project_run import ENGINE_PYCOLMAP, ENGINE_SPHERESFM, run_sfm_for_project

WIDTH = 1024
ORIGIN = np.array([298900.0, 6126100.0, 0.0])


@pytest.fixture(scope="module", params=[ENGINE_PYCOLMAP] + ([ENGINE_SPHERESFM] if HAVE_SPHERESFM else []))
def run(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("georef360") / "proj"
    create_project(root, "Synthetic 360", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    frame_set_id = add_synthetic_frame_set(conn, root, width=WIDTH)
    generate_views_for_frame_set(conn, root, frame_set_id, face_size=256, face_names=["front", "right", "back", "left", "up", "down"])
    if request.param == ENGINE_PYCOLMAP:
        run_sfm_for_project(conn, root, config=SfmConfig(camera_model="EQUIRECTANGULAR"), image_source="frames", frame_set_id=frame_set_id)
    else:
        run_sfm_for_project(conn, root, image_source="frames", frame_set_id=frame_set_id, engine=ENGINE_SPHERESFM)
    run_id = conn.execute("SELECT run_id FROM sfm_runs").fetchone()[0]
    yield root, conn, frame_set_id, run_id, request.param
    conn.close()


def _equirect_observations(conn, root, run_id, engine):
    """{point key: (xyz, [(frame_id, x, y)])} from the engine's own model."""
    selected_model = conn.execute("SELECT selected_model FROM sfm_runs WHERE run_id = ?", (run_id,)).fetchone()[0]
    frame_by_name = {p.split("/")[-1]: f for f, p in conn.execute("SELECT frame_id, path FROM frames")}
    points: dict = {}
    if engine == ENGINE_SPHERESFM:
        from vine360.sfm import spheresfm_adapter as sph

        model = sph.read_text_model(root / selected_model / "txt")
        for image in model.images.values():
            for x, y, point_id in image.points2d:
                if point_id >= 0:
                    entry = points.setdefault(point_id, (np.array(model.points[point_id].xyz), []))
                    entry[1].append((frame_by_name[image.name], x, y))
    else:
        reconstruction = pycolmap.Reconstruction(root / selected_model)
        for image in reconstruction.images.values():
            for p2d in image.points2D:
                if p2d.has_point3D():
                    entry = points.setdefault(p2d.point3D_id, (np.array(reconstruction.points3D[p2d.point3D_id].xyz), []))
                    entry[1].append((frame_by_name[image.name], p2d.xy[0], p2d.xy[1]))
    return points


def test_solve_on_a_real_360_run(run, tmp_path):
    root, conn, frame_set_id, run_id, engine = run
    poses = view_poses_for_run(conn, root, run_id)
    assert poses and all(vp.view.view_id == vid for vid, vp in poses.items())
    views_by_frame = {}
    for view in views_for_frame_set(conn, frame_set_id):
        views_by_frame.setdefault(view.frame_id, []).append(view)

    # Stand-in tags: the best-observed, well-spread sparse points.
    candidates = sorted(_equirect_observations(conn, root, run_id, engine).items(), key=lambda kv: -len(kv[1][1]))
    extent = np.ptp(np.array([xyz for _key, (xyz, _obs) in candidates]), axis=0).max()
    chosen: list = []
    for key, (xyz, obs) in candidates:
        well_seen = len({f for f, *_ in obs}) >= 4
        if well_seen and all(np.linalg.norm(xyz - c[0]) > extent / 8 for c in chosen):
            chosen.append((xyz, obs))
        if len(chosen) == 6:
            break
    assert len(chosen) == 6

    rng = np.random.default_rng(4)
    truth = Similarity(2.3, random_rotation(rng), np.array([80.0, 120.0, 360.0]))
    lines = []
    for tag_id, (xyz, obs) in enumerate(chosen):
        e, n, h = truth.apply(xyz[None])[0] + ORIGIN
        lines.append(f"{tag_id:03d},{e:.4f},{n:.4f},{h:.4f},")
        for frame_id, x, y in obs:
            direction = equirect_pixel_to_direction(x - 0.5, y - 0.5, WIDTH, WIDTH // 2)
            for view in views_by_frame[frame_id]:
                if view.view_id not in poses:
                    continue
                intrinsics = Intrinsics(view.width, view.height, view.fx, view.fy, view.cx, view.cy)
                pixel = direction_to_camera_pixel(view.fixed_rotation.T @ direction, intrinsics)
                if pixel is None or not (0 <= pixel[0] < view.width and 0 <= pixel[1] < view.height):
                    continue
                center = [pixel[0] + 0.5, pixel[1] + 0.5]
                conn.execute(
                    "INSERT OR REPLACE INTO tag_observations VALUES (?, 'tag36h11', ?, ?, ?, 80, 0, 'synthetic', '0', '{}', 'now')",
                    (view.view_id, tag_id, json.dumps([center] * 4), json.dumps(center)),
                )
    for view_id in poses:
        conn.execute("INSERT OR REPLACE INTO tag_detection_runs VALUES (?, 'tag36h11', 0, 'now')", (view_id,))
    conn.commit()
    gcp = tmp_path / "gcp.csv"
    gcp.write_text("\n".join(lines) + "\n")
    set_id, _ = import_control_points(conn, gcp, epsg=7854)

    solution = load_solution(conn, solve_georef(conn, root, run_id, set_id))
    fit = Similarity.from_dict(solution["transform"])
    assert solution["report"]["n_control"] == 6
    # The stand-in "tags" are SfM's own sparse points from 6 frames on 256 px
    # faces, so they carry real triangulation noise; a convention or
    # composition error would be off by the whole scene, not by a few percent.
    local = np.array([truth.apply(xyz[None])[0] for xyz, _obs in chosen])
    spread = np.ptp(local, axis=0).max()
    assert fit.scale == pytest.approx(truth.scale, rel=0.05)
    np.testing.assert_allclose(fit.rotation, truth.rotation, atol=0.05)
    assert solution["report"]["rmse"]["d3"] < 0.1 * spread

    # The materialised model is the pinhole view model, moved into local metres.
    moved = pycolmap.Reconstruction(root / solution["model_dir"])
    assert len(moved.images) == len(poses)
    for image in moved.images.values():
        view_id = next(v for v, vp in poses.items() if vp.view.image_path == f"projections/{image.name}")
        R, t = poses[view_id].R, poses[view_id].t
        np.testing.assert_allclose(image.projection_center(), fit.apply((-R.T @ t)[None])[0], atol=1e-6)
