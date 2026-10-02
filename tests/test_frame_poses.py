"""vine360.sfm.frame_poses against *real* reconstructions of a synthetic 360
sequence (tests/synthetic_360.py) from both raw-frame engines: SphereSfM
(when its build is present) and pycolmap's EQUIRECTANGULAR model. This is
what pins the engine bearing conventions and the view-pose composition
(docs/adr/0036) -- nothing here is mocked.

The core check: a sparse point's *observed* equirect pixel, carried into a
projected view purely through vine360's own projection geometry
(equirect pixel -> panorama ray -> face via fixed_rotation -> face pixel),
must land where the derived pinhole view pose projects the point's 3D
position. If the convention or composition order were wrong this is off by
tens or hundreds of pixels, not a fraction of one."""

import json

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")

from conftest import HAVE_SPHERESFM, requires_spheresfm
from synthetic_360 import add_synthetic_frame_set, ground_truth_trajectory
from vine360.config import CaptureMode
from vine360.projection.generate import generate_views_for_frame_set
from vine360.projection.geometry import Intrinsics, direction_to_camera_pixel, equirect_pixel_to_direction
from vine360.project import create_project, open_index_db
from vine360.sfm import spheresfm_adapter as sph
from vine360.sfm.colmap_adapter import SfmConfig
from vine360.sfm.frame_poses import (
    build_view_reconstruction,
    load_frame_model,
    quaternion_to_matrix,
    view_cam_from_world,
    views_for_frame_set,
)
from vine360.sfm.project_run import ENGINE_PYCOLMAP, ENGINE_SPHERESFM, run_sfm_for_project

FACE_SIZE = 256
WIDTH = 1024


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """One project, one synthetic frame set, projected to all six faces, plus
    a real run from each available engine. Read-only for the tests below."""
    root = tmp_path_factory.mktemp("frame_poses") / "proj"
    create_project(root, "Synthetic 360", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    frame_set_id = add_synthetic_frame_set(conn, root, width=WIDTH)
    generate_views_for_frame_set(
        conn, root, frame_set_id, face_size=FACE_SIZE, face_names=["front", "right", "back", "left", "up", "down"]
    )
    runs = {}
    run_sfm_for_project(
        conn, root, config=SfmConfig(camera_model="EQUIRECTANGULAR"), image_source="frames", frame_set_id=frame_set_id
    )
    runs[ENGINE_PYCOLMAP] = conn.execute("SELECT run_id FROM sfm_runs ORDER BY created_at DESC LIMIT 1").fetchone()[0]
    if HAVE_SPHERESFM:
        run_sfm_for_project(conn, root, image_source="frames", frame_set_id=frame_set_id, engine=ENGINE_SPHERESFM)
        runs[ENGINE_SPHERESFM] = conn.execute(
            "SELECT run_id FROM sfm_runs WHERE config LIKE '%spheresfm%'"
        ).fetchone()[0]
    yield root, conn, frame_set_id, runs
    conn.close()


def _observations(conn, root, run_id, engine):
    """(frame_id, xyz, x, y) for every sparse-point observation, straight
    from the engine's own model -- COLMAP pixel coordinates."""
    selected_model, config = conn.execute("SELECT selected_model, config FROM sfm_runs WHERE run_id = ?", (run_id,)).fetchone()
    frame_by_name = {p.split("/")[-1]: f for f, p in conn.execute("SELECT frame_id, path FROM frames")}
    observations = []
    if engine == ENGINE_SPHERESFM:
        model = sph.read_text_model(root / selected_model / "txt")
        for image in model.images.values():
            for x, y, point_id in image.points2d:
                if point_id >= 0:
                    observations.append((frame_by_name[image.name], np.array(model.points[point_id].xyz), x, y))
    else:
        reconstruction = pycolmap.Reconstruction(root / selected_model)
        for image in reconstruction.images.values():
            for point2d in image.points2D:
                if point2d.has_point3D():
                    xyz = reconstruction.points3D[point2d.point3D_id].xyz
                    observations.append((frame_by_name[image.name], np.array(xyz), point2d.xy[0], point2d.xy[1]))
    return observations


def _engines():
    return [ENGINE_PYCOLMAP, pytest.param(ENGINE_SPHERESFM, marks=requires_spheresfm)]


@pytest.mark.parametrize("engine", _engines())
def test_both_engines_register_the_whole_synthetic_sequence(scene, engine):
    root, conn, frame_set_id, runs = scene
    model = load_frame_model(conn, root, runs[engine])
    assert model.engine == engine and model.frame_set_id == frame_set_id
    assert len(model.poses) == 6
    assert len(model.points) > 1000


@pytest.mark.parametrize("engine", _engines())
def test_view_poses_agree_with_vine360s_own_projection_geometry(scene, engine):
    root, conn, frame_set_id, runs = scene
    model = load_frame_model(conn, root, runs[engine])
    views_by_frame = {}
    for view in views_for_frame_set(conn, frame_set_id):
        views_by_frame.setdefault(view.frame_id, []).append(view)

    errors = []
    for frame_id, xyz, x, y in _observations(conn, root, runs[engine], engine):
        direction = equirect_pixel_to_direction(x - 0.5, y - 0.5, WIDTH, WIDTH // 2)  # vine360 panorama ray
        for view in views_by_frame[frame_id]:
            intrinsics = Intrinsics(view.width, view.height, view.fx, view.fy, view.cx, view.cy)
            pixel = direction_to_camera_pixel(view.fixed_rotation.T @ direction, intrinsics)
            if pixel is None or not (0 <= pixel[0] < view.width and 0 <= pixel[1] < view.height):
                continue
            expected = np.array(pixel) + 0.5  # vine360 pixel index -> COLMAP pixel coordinate
            rotation, translation = view_cam_from_world(model.poses[frame_id], view.fixed_rotation, engine)
            cam = rotation @ xyz + translation
            assert cam[2] > 0, "a point the face sees must be in front of the derived camera"
            projected = np.array([view.fx * cam[0] / cam[2] + view.cx, view.fy * cam[1] / cam[2] + view.cy])
            errors.append(np.linalg.norm(projected - expected))

    errors = np.array(errors)
    assert len(errors) > 1000
    assert np.median(errors) < 0.5, f"median {np.median(errors):.3f}px"
    assert np.percentile(errors, 95) < 2.0, f"95th percentile {np.percentile(errors, 95):.3f}px"


@pytest.mark.parametrize("engine", _engines())
def test_recovered_camera_centres_match_ground_truth_up_to_similarity(scene, engine):
    root, conn, _frame_set_id, runs = scene
    model = load_frame_model(conn, root, runs[engine])
    truth = {f"synth~i1:{i:06d}": center for i, (center, _rotation) in enumerate(ground_truth_trajectory())}
    ids = sorted(model.poses)
    estimated = np.array([model.poses[i].center for i in ids])
    expected = np.array([truth[i] for i in ids])

    # Umeyama similarity alignment estimated -> expected.
    mu_e, mu_x = estimated.mean(0), expected.mean(0)
    e, x = estimated - mu_e, expected - mu_x
    u, s, vt = np.linalg.svd(x.T @ e)
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ d @ vt
    scale = np.trace(np.diag(s) @ d) / (e**2).sum()
    aligned = scale * e @ rotation.T + mu_x
    residual = np.linalg.norm(aligned - expected, axis=1)
    assert residual.max() < 0.02 * np.linalg.norm(x, axis=1).max(), residual


@requires_spheresfm
def test_front_and_side_views_match_spheresfms_own_cube_face_exporter(scene, tmp_path):
    """Independent cross-check: SphereSfM's own sphere_cubic_reprojecer
    composes face poses its own way (reconstruction.cc
    ExportPerspectiveCubic). Its faces 0-3 are front/right/back/left --
    the same yaws as vine360's -- and their rotations must match ours."""
    root, conn, frame_set_id, runs = scene
    run_id = runs[ENGINE_SPHERESFM]
    selected_model = conn.execute("SELECT selected_model FROM sfm_runs WHERE run_id = ?", (run_id,)).fetchone()[0]
    binary = sph.find_binary()
    out = tmp_path / "cubic"
    sph.run_command(
        sph.LocalRunner(),
        sph.build_cubic_reprojection_command(
            binary, root / selected_model, root / "frames" / frame_set_id, out, image_size=FACE_SIZE
        ),
        "cube-face export",
    )
    (out / "txt").mkdir()
    sph.run_command(sph.LocalRunner(), sph.build_model_converter_command(binary, out / "sparse", out / "txt"), "conversion")
    theirs = {image.name: quaternion_to_matrix(image.qvec) for image in sph.read_text_model(out / "txt").images.values()}

    model = load_frame_model(conn, root, run_id)
    views = {(v.frame_id, v.image_path.rsplit("/", 1)[-1]): v for v in views_for_frame_set(conn, frame_set_id)}
    face_index = {"front.png": 0, "right.png": 1, "back.png": 2, "left.png": 3}
    compared = 0
    for frame_id, pose in model.poses.items():
        stem = conn.execute("SELECT path FROM frames WHERE frame_id = ?", (frame_id,)).fetchone()[0].rsplit("/", 1)[-1][:-4]
        for face_name, index in face_index.items():
            ours, _ = view_cam_from_world(pose, views[(frame_id, face_name)].fixed_rotation, ENGINE_SPHERESFM)
            their_rotation = theirs[f"{stem}_perspective_{index:08d}.png"]
            angle = np.degrees(np.arccos(np.clip((np.trace(ours.T @ their_rotation) - 1) / 2, -1, 1)))
            assert angle < 0.1, f"{frame_id} {face_name}: {angle:.3f} degrees apart"
            compared += 1
    assert compared == 24


@pytest.mark.parametrize("engine", _engines())
def test_build_view_reconstruction_is_a_standard_pinhole_model(scene, engine, tmp_path):
    root, conn, frame_set_id, runs = scene
    reconstruction, _model = build_view_reconstruction(conn, root, runs[engine])

    reconstruction.write(tmp_path)
    reread = pycolmap.Reconstruction(tmp_path)
    assert {camera.model.name for camera in reread.cameras.values()} == {"PINHOLE"}
    assert reread.num_reg_images() == 36  # 6 frames x 6 faces
    names = {image.name for image in reread.images.values()}
    assert f"{frame_set_id}:000000/front.png" in names
    assert reread.num_points3D() > 1000
    # The model's own reprojection error is ~0 by construction; the real
    # check is that every observation refers to an existing image.
    for point in reread.points3D.values():
        for element in point.track.elements:
            assert element.image_id in reread.images
    assert reread.compute_mean_reprojection_error() < 1e-6


def test_spheresfm_run_is_recorded_like_any_other_run(scene):
    root, conn, frame_set_id, runs = scene
    if ENGINE_SPHERESFM not in runs:
        pytest.skip("no SphereSfM build")
    run_id = runs[ENGINE_SPHERESFM]
    selected_model, config_json, stats_json = conn.execute(
        "SELECT selected_model, config, model_stats FROM sfm_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    config = json.loads(config_json)
    assert config["engine"] == "spheresfm" and config["camera_model"] == "SPHERE"
    assert config["frame_set_id"] == frame_set_id and config["image_source"] == "frames"
    stats = json.loads(stats_json)
    assert stats["registered_images"] == stats["total_images"] == 6
    run_dir = root / "sfm" / "sparse" / run_id
    assert selected_model.startswith(f"sfm/sparse/{run_id}/")
    assert (run_dir / "database.db").exists() and (run_dir / "image_list.txt").exists()
    assert (root / selected_model / "txt" / "images.txt").exists()
    assert "thumbs" not in (run_dir / "image_list.txt").read_text()


@pytest.mark.parametrize("engine", _engines())
def test_postshot_export_of_a_360_run_writes_masked_pinhole_views(scene, engine, tmp_path):
    """The export pipeline end to end: a raw-frame run exports vine360's
    projected views (not raw equirect frames) as a PINHOLE model, with each
    view's keep-mask under the same flattened name."""
    from vine360.export.postshot import export_for_postshot
    from vine360.masking.build import build_mask_for_view

    root, conn, frame_set_id, runs = scene
    masked = [f"{frame_set_id}:000000:front", f"{frame_set_id}:000001:right"]
    for view_id in masked:
        build_mask_for_view(conn, root, view_id)

    result = export_for_postshot(conn, root, tmp_path / "out", run_id=runs[engine])

    assert result.num_images == 36
    exported = pycolmap.Reconstruction(result.sparse_dir)
    assert {camera.model.name for camera in exported.cameras.values()} == {"PINHOLE"}
    image_names = {image.name for image in exported.images.values()}
    assert image_names == {p.name for p in result.images_dir.iterdir()}
    assert result.num_masks == len(masked)
    assert {p.name for p in result.masks_dir.iterdir()} <= image_names
    assert not any(name.endswith(".png") and "frame_" in name for name in image_names), "no raw equirect frames"


@pytest.mark.parametrize("engine", _engines())
def test_realityscan_priors_come_from_a_360_run_of_either_engine(scene, engine, tmp_path):
    from vine360.export.postshot import export_for_realityscan

    root, conn, frame_set_id, runs = scene
    # Make this engine's run the most recent, so priors selection picks it.
    conn.execute("UPDATE sfm_runs SET created_at = '2000-01-01' WHERE run_id != ?", (runs[engine],))
    conn.execute("UPDATE sfm_runs SET created_at = '2099-01-01' WHERE run_id = ?", (runs[engine],))
    conn.commit()

    result = export_for_realityscan(
        conn, root, tmp_path / "rs", frame_set_id=frame_set_id, include_camera_priors=True
    )
    rows = (tmp_path / "rs" / "CameraPriors.csv").read_text().splitlines()
    assert rows[0] == "#name,x,y,alt"
    assert len(rows) - 1 == result.num_images == 36  # every view gets its frame's centre
