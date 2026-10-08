"""Multi-frame-set SfM (docs/adr/0046): one reconstruction over several
frame sets -- e.g. an aerial and a ground 360 capture of one block.

Real runs, nothing mocked except where marked: two synthetic 360 frame
sets of the same room (tests/synthetic_360.py), the second walked higher
up and rendered at a *different resolution*, so a per-set camera actually
matters. Both raw-frame engines run on them together (SphereSfM when its
build is present); the projections engine's scoping is checked with the
heavy COLMAP call mocked, as in test_sfm_project_run.py."""

import json
from unittest.mock import patch

import pytest

pycolmap = pytest.importorskip("pycolmap")

from conftest import HAVE_SPHERESFM, requires_spheresfm
from synthetic_360 import add_synthetic_frame_set
from vine360.config import CaptureMode
from vine360.export.postshot import _select_priors_run, export_for_postshot
from vine360.projection.generate import generate_views_for_frame_set
from vine360.project import create_project, open_index_db
from vine360.sfm import spheresfm_adapter as sph
from vine360.sfm.colmap_adapter import SfmConfig, SfmDiagnostics
from vine360.sfm.frame_poses import build_view_reconstruction, load_frame_model
from vine360.sfm.project_run import (
    ENGINE_PYCOLMAP,
    ENGINE_SPHERESFM,
    multi_set_matcher_warning,
    run_frame_set_ids,
    run_sfm_for_project,
)

FACES = ["front", "right", "back", "left"]
GROUND_WIDTH, AIR_WIDTH = 1024, 768
# Out of 12 frames; a tiny scene occasionally loses a frame (see test_frame_poses.py).
MIN_REGISTERED = 10


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    root = tmp_path_factory.mktemp("multi_set") / "proj"
    create_project(root, "Two captures", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    ground = add_synthetic_frame_set(conn, root, width=GROUND_WIDTH, source_id="ground")
    air = add_synthetic_frame_set(conn, root, width=AIR_WIDTH, source_id="air", offset=(0.3, 0.8, 0.4))
    for frame_set_id in (ground, air):
        generate_views_for_frame_set(conn, root, frame_set_id, face_size=128, face_names=FACES)

    runs = {}
    config = SfmConfig(matcher="exhaustive")
    run_sfm_for_project(
        conn, root, config=config.replace(camera_model="EQUIRECTANGULAR"), image_source="frames",
        frame_set_ids=[ground, air],
    )
    runs[ENGINE_PYCOLMAP] = _latest_run(conn)
    if HAVE_SPHERESFM:
        run_sfm_for_project(
            conn, root, config=config, image_source="frames", frame_set_ids=[ground, air], engine=ENGINE_SPHERESFM
        )
        runs[ENGINE_SPHERESFM] = _latest_run(conn)
    yield root, conn, ground, air, runs
    conn.close()


def _latest_run(conn) -> str:
    return conn.execute("SELECT run_id FROM sfm_runs ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()[0]


def _engines():
    return [ENGINE_PYCOLMAP, pytest.param(ENGINE_SPHERESFM, marks=requires_spheresfm)]


@pytest.mark.parametrize("engine", _engines())
def test_both_frame_sets_register_into_one_model(scene, engine):
    root, conn, ground, air, runs = scene
    model = load_frame_model(conn, root, runs[engine])
    assert model.frame_set_ids == [ground, air] and model.frame_set_id is None
    registered_sets = {frame_id.split(":")[0] for frame_id in model.poses}
    assert registered_sets == {ground, air}, "frames from both captures in the one selected model"
    assert len(model.poses) >= MIN_REGISTERED
    stats = json.loads(conn.execute("SELECT model_stats FROM sfm_runs WHERE run_id = ?", (runs[engine],)).fetchone()[0])
    assert stats["total_images"] == 12


@pytest.mark.parametrize("engine", _engines())
def test_the_run_records_every_frame_set(scene, engine):
    root, conn, ground, air, runs = scene
    config = json.loads(conn.execute("SELECT config FROM sfm_runs WHERE run_id = ?", (runs[engine],)).fetchone()[0])
    assert config["frame_set_ids"] == [ground, air]
    assert config["source_ids"] == ["ground", "air"]
    # The single-set keys stay unscoped, so nothing older reads this as a run of one set.
    assert config["frame_set_id"] is None and config["source_id"] is None
    assert run_frame_set_ids(config) == [ground, air]


def test_pycolmap_gives_each_frame_set_its_own_camera(scene):
    root, conn, ground, air, runs = scene
    selected_model, config_json = conn.execute(
        "SELECT selected_model, config FROM sfm_runs WHERE run_id = ?", (runs[ENGINE_PYCOLMAP],)
    ).fetchone()
    assert json.loads(config_json)["options"]["camera_mode"] == "per_folder"
    reconstruction = pycolmap.Reconstruction(root / selected_model)
    widths = {reconstruction.cameras[image.camera_id].width for image in reconstruction.images.values()}
    assert widths == {GROUND_WIDTH, AIR_WIDTH}
    assert all(image.name.split("/")[0] in (ground, air) for image in reconstruction.images.values())


@requires_spheresfm
def test_spheresfm_extracts_each_frame_set_with_its_own_sphere_camera(scene):
    root, conn, ground, air, runs = scene
    run_id = runs[ENGINE_SPHERESFM]
    selected_model = conn.execute("SELECT selected_model FROM sfm_runs WHERE run_id = ?", (run_id,)).fetchone()[0]
    model = sph.read_text_model(root / selected_model / "txt")
    cameras = {(c.width, c.height): c.params for c in model.cameras.values()}
    # SPHERE params are f, cx, cy -- the image centre of *that* set's frames.
    assert cameras[(GROUND_WIDTH, GROUND_WIDTH // 2)][1:] == [GROUND_WIDTH / 2, GROUND_WIDTH / 4]
    assert cameras[(AIR_WIDTH, AIR_WIDTH // 2)][1:] == [AIR_WIDTH / 2, AIR_WIDTH / 4]
    run_dir = root / "sfm" / "sparse" / run_id
    assert (run_dir / "1_feature_extractor_1.log").exists() and (run_dir / "1_feature_extractor_2.log").exists()
    assert (run_dir / "image_list_2.txt").read_text().split() == [f"{air}/frame_{i:06d}.png" for i in range(1, 7)]


@pytest.mark.parametrize("engine", _engines())
def test_export_covers_the_views_of_every_frame_set(scene, engine, tmp_path):
    root, conn, ground, air, runs = scene
    reconstruction, model = build_view_reconstruction(conn, root, runs[engine])
    view_sets = {image.name.split(":")[0] for image in reconstruction.images.values()}
    assert view_sets == {ground, air}
    assert reconstruction.num_reg_images() == len(FACES) * len(model.poses)

    result = export_for_postshot(conn, root, tmp_path / "out", run_id=runs[engine])
    assert result.num_images == len(FACES) * len(model.poses)


def test_camera_priors_use_a_multi_set_run_for_each_of_its_sets(scene):
    root, conn, ground, air, runs = scene
    for frame_set_id in (ground, air):
        selected = _select_priors_run(conn, frame_set_id=frame_set_id)
        assert selected is not None and selected[0] in runs.values()
    assert _select_priors_run(conn, frame_set_id="someone~else") is None


def test_sequential_matching_on_several_sets_is_warned_about():
    config = SfmConfig()
    assert multi_set_matcher_warning(["a", "b"], config)
    assert multi_set_matcher_warning(["a"], config) is None
    assert multi_set_matcher_warning(["a", "b"], config.replace(matcher="exhaustive")) is None
    assert multi_set_matcher_warning(["a", "b"], config.replace(matcher="vocab_tree")) is None


def test_projections_run_uses_only_the_chosen_sets_views(scene, tmp_path):
    """Mocks only the heavy COLMAP call (covered for real in
    test_sfm_integration.py) to check which images it would be given."""
    root, conn, ground, air, _runs = scene
    fake_model_dir = root / "sfm" / "sparse" / "fake"
    fake_model_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = SfmDiagnostics(8, 8, 1.0, 1, 10, 0.1, 2.0, 2.0)
    with patch("vine360.sfm.project_run.run_sfm", return_value=(object(), diagnostics, fake_model_dir)) as mock_run:
        _diagnostics, warnings = run_sfm_for_project(conn, root, frame_set_ids=[ground, air])
        names = mock_run.call_args.kwargs["image_names"]
        assert {name.split(":")[0] for name in names} == {ground, air}
        assert len(names) == 12 * len(FACES)
        assert any("sequential matching" in w for w in warnings)

        run_sfm_for_project(conn, root, frame_set_ids=[air])
        assert {n.split(":")[0] for n in mock_run.call_args.kwargs["image_names"]} == {air}
    conn.execute("DELETE FROM sfm_runs WHERE selected_model = 'sfm/sparse/fake'")
    conn.commit()


def test_frame_set_id_and_frame_set_ids_cant_disagree(scene):
    root, conn, ground, air, _runs = scene
    with pytest.raises(ValueError, match="not both"):
        run_sfm_for_project(conn, root, frame_set_id=ground, frame_set_ids=[ground, air])
