"""User-tunable SfM options on the real engines (docs/adr/0040), nothing
mocked:
- the pycolmap mapping: every option lands on the installed pycolmap's
  real option objects, and SfmConfig() reproduces pycolmap's own defaults
  exactly (so a run with default options behaves as it did before options
  existed);
- real runs on the synthetic 360 sequence with non-default options --
  exhaustive matching, the global (GLOMAP) mapper, tuned features -- for
  pycolmap and (when a build is present) SphereSfM.
The SphereSfM command-line side is in test_spheresfm_adapter.py."""

import json

import pytest

pycolmap = pytest.importorskip("pycolmap")

from conftest import requires_spheresfm
from synthetic_360 import add_synthetic_frame_set
from vine360.config import CaptureMode
from vine360.project import create_project, open_index_db
from vine360.sfm import colmap_adapter as ca
from vine360.sfm.options import VARIANT_EQUIRECT, VARIANT_SPHERESFM, SfmConfig
from vine360.sfm.project_run import ENGINE_SPHERESFM, run_sfm_for_project


# -- pycolmap mapping -----------------------------------------------------------


def test_default_options_reproduce_pycolmaps_own_defaults():
    """SfmConfig() must change nothing about how pycolmap runs."""
    config = SfmConfig()
    assert ca.extraction_options(config).sift.todict() == pycolmap.FeatureExtractionOptions().sift.todict()
    assert ca.extraction_options(config).max_image_size == pycolmap.FeatureExtractionOptions().max_image_size
    matching = ca.matching_options(config).todict()
    default_matching = pycolmap.FeatureMatchingOptions().todict()
    matching.pop("use_gpu"), default_matching.pop("use_gpu")  # host-dependent (CUDA build or not)
    assert matching == default_matching
    assert ca.verification_options(config).todict() == pycolmap.TwoViewGeometryOptions().todict()
    assert ca.incremental_options(config).todict() == pycolmap.IncrementalPipelineOptions().todict()
    _name, match_fn, pairing = ca.pairing_options(config)
    assert match_fn is pycolmap.match_sequential
    assert pairing.todict() == pycolmap.SequentialPairingOptions(overlap=10).todict()


def test_every_option_lands_on_the_pycolmap_objects(tmp_path):
    tree = tmp_path / "tree.bin"
    tree.write_bytes(b"\0")
    config = SfmConfig(
        camera_model="PINHOLE", max_image_size=1600, max_num_features=2048, peak_threshold=0.004, edge_threshold=12.0,
        upright=True, max_ratio=0.7, max_distance=0.6, cross_check=False, max_num_matches=4096, guided_matching=True,
        min_num_inliers=20, max_error=3.0, min_inlier_ratio=0.3, sequential_overlap=7, quadratic_overlap=False,
        loop_detection=True, loop_detection_period=5, loop_detection_num_images=20, vocab_tree_path=str(tree),
        min_num_matches=20, multiple_models=False, min_model_size=4, init_min_num_inliers=50, init_min_tri_angle=8.0,
        init_max_forward_motion=0.99, abs_pose_min_num_inliers=20, abs_pose_min_inlier_ratio=0.2,
        filter_max_reproj_error=3.0, filter_min_tri_angle=1.0, refine_focal_length=False, refine_principal_point=True,
        refine_extra_params=False, random_seed=7,
    )
    assert ca.reader_options(config).camera_model == "PINHOLE"
    extraction = ca.extraction_options(config)
    assert extraction.max_image_size == 1600
    assert (extraction.sift.max_num_features, extraction.sift.peak_threshold, extraction.sift.edge_threshold) == (2048, 0.004, 12.0)
    assert extraction.sift.upright
    matching = ca.matching_options(config)
    assert (matching.sift.max_ratio, matching.sift.max_distance, matching.sift.cross_check) == (0.7, 0.6, False)
    assert matching.max_num_matches == 4096 and matching.guided_matching
    verification = ca.verification_options(config)
    assert (verification.min_num_inliers, verification.ransac.max_error, verification.ransac.min_inlier_ratio) == (20, 3.0, 0.3)
    assert verification.ransac.random_seed == 7
    _name, _fn, pairing = ca.pairing_options(config)
    assert (pairing.overlap, pairing.quadratic_overlap, pairing.loop_detection) == (7, False, True)
    assert (pairing.loop_detection_period, pairing.loop_detection_num_images) == (5, 20)
    assert pairing.vocab_tree_path == tree
    incremental = ca.incremental_options(config)
    assert (incremental.min_num_matches, incremental.multiple_models, incremental.min_model_size) == (20, False, 4)
    assert (incremental.ba_refine_focal_length, incremental.ba_refine_principal_point, incremental.ba_refine_extra_params) == (
        False, True, False,
    )
    mapper = incremental.mapper
    assert (mapper.init_min_num_inliers, mapper.init_min_tri_angle, mapper.init_max_forward_motion) == (50, 8.0, 0.99)
    assert (mapper.abs_pose_min_num_inliers, mapper.abs_pose_min_inlier_ratio) == (20, 0.2)
    assert (mapper.filter_max_reproj_error, mapper.filter_min_tri_angle) == (3.0, 1.0)
    assert incremental.random_seed == mapper.random_seed == 7
    glomap = ca.global_options(config)
    assert (glomap.min_num_matches, glomap.multiple_models, glomap.min_model_size, glomap.random_seed) == (20, False, 4, 7)
    assert not glomap.mapper.bundle_adjustment.refine_focal_length

    for matcher, fn in (("exhaustive", pycolmap.match_exhaustive), ("vocab_tree", pycolmap.match_vocabtree)):
        _name, match_fn, _pairing = ca.pairing_options(config.replace(matcher=matcher))
        assert match_fn is fn


def test_cpu_only_sift_variants_are_reported_as_cpu():
    assert ca.devices(SfmConfig(use_gpu=False)) == ("CPU", "CPU")
    extract, match = ca.devices(SfmConfig(estimate_affine_shape=True))
    assert extract == "CPU"
    assert match == ("GPU" if pycolmap.has_cuda else "CPU")


# -- real runs ------------------------------------------------------------------


@pytest.fixture
def frame_set(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Options", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    frame_set_id = add_synthetic_frame_set(conn, root, n=6, width=1024)
    yield root, conn, frame_set_id
    conn.close()


def _stored_config(conn) -> dict:
    return json.loads(conn.execute("SELECT config FROM sfm_runs ORDER BY rowid DESC LIMIT 1").fetchone()[0])


def test_pycolmap_global_mapper_with_exhaustive_matching(frame_set):
    root, conn, frame_set_id = frame_set
    config = SfmConfig(matcher="exhaustive", mapper="global", camera_mode="single", random_seed=1).for_variant(VARIANT_EQUIRECT)
    events = []
    diagnostics, _ = run_sfm_for_project(
        conn, root, config=config, image_source="frames", frame_set_id=frame_set_id,
        progress_callback=lambda m, c, t: events.append((m, c, t)),
    )
    assert diagnostics.registered_images >= 5
    assert any("exhaustive" in m for m, _, _ in events)
    assert any("block 1/1" in m for m, _, _ in events)
    stages = [(c, t) for m, c, t in events if "(global mapping): stage" in m]
    assert stages[-1] == (5, 5) and len(stages) >= 5
    stored = _stored_config(conn)
    assert stored["options"]["mapper"] == "global" and stored["options"]["matcher"] == "exhaustive"
    assert SfmConfig.from_dict(stored["options"]) == config


def test_pycolmap_tuned_incremental_run_and_quality_thresholds(frame_set):
    root, conn, frame_set_id = frame_set
    config = SfmConfig(
        max_image_size=800, max_num_features=3000, upright=True, guided_matching=True, init_min_tri_angle=8.0,
        random_seed=2, max_mean_reprojection_error=0.0001,
    ).for_variant(VARIANT_EQUIRECT)
    diagnostics, warnings = run_sfm_for_project(conn, root, config=config, image_source="frames", frame_set_id=frame_set_id)
    assert diagnostics.registered_images >= 5
    assert any("poor reprojection error" in w and "0.00px" in w for w in warnings)  # the configured threshold
    assert _stored_config(conn)["options"]["max_image_size"] == 800


def test_invalid_options_are_rejected_before_any_work(frame_set):
    root, conn, frame_set_id = frame_set
    with pytest.raises(ValueError, match="vocabulary tree"):
        run_sfm_for_project(conn, root, config=SfmConfig(matcher="vocab_tree"), image_source="frames", frame_set_id=frame_set_id)
    with pytest.raises(ValueError, match="global mapper"):
        run_sfm_for_project(
            conn, root, config=SfmConfig(mapper="global"), image_source="frames", frame_set_id=frame_set_id,
            engine=ENGINE_SPHERESFM,
        )
    assert not (root / "sfm" / "sparse").exists() or not any((root / "sfm" / "sparse").iterdir())
    assert conn.execute("SELECT COUNT(*) FROM sfm_runs").fetchone()[0] == 0


@requires_spheresfm
def test_spheresfm_exhaustive_run_with_options(frame_set):
    root, conn, frame_set_id = frame_set
    config = SfmConfig(matcher="exhaustive", max_num_features=4000, init_min_tri_angle=8.0, random_seed=3)
    events = []
    diagnostics, _ = run_sfm_for_project(
        conn, root, config=config.for_variant(VARIANT_SPHERESFM), image_source="frames", frame_set_id=frame_set_id,
        engine=ENGINE_SPHERESFM, progress_callback=lambda m, c, t: events.append((m, c, t)),
    )
    assert diagnostics.registered_images >= 5
    assert any("step 2/4" in m and "block 1/1" in m for m, _, _ in events)
    run_id = conn.execute("SELECT run_id FROM sfm_runs").fetchone()[0]
    assert (root / "sfm" / "sparse" / run_id / "2_exhaustive_matcher.log").stat().st_size > 0
    assert _stored_config(conn)["options"]["matcher"] == "exhaustive"
