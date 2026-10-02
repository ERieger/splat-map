"""vine360.sfm.spheresfm_adapter. The command builders and TXT reader are
checked against the real SphereSfM build when one is present -- every
flag the builders use must appear in that binary's own `<command> -h`
(docs/adr/0036); the pure builder tests run regardless."""

import re
import subprocess
from pathlib import Path

import pytest

from conftest import requires_spheresfm
from vine360.sfm import spheresfm_adapter as sph
from vine360.sfm.options import SfmConfig

B = Path("/opt/spheresfm/colmap")


def test_validate_installation_reports_what_it_searched():
    status = sph.validate_installation()
    assert status["found"] in (True, False)
    assert status["env_var"] == "VINE360_SPHERESFM_COLMAP"
    assert status["searched"]


def test_env_var_is_searched_first(monkeypatch, tmp_path):
    monkeypatch.setenv("VINE360_SPHERESFM_COLMAP", str(tmp_path / "colmap"))
    assert sph.candidate_binaries()[0] == tmp_path / "colmap"


def test_a_binary_that_isnt_spheresfm_is_rejected(monkeypatch, tmp_path):
    fake = tmp_path / "colmap"
    fake.write_text("#!/bin/sh\necho 'COLMAP 3.9 stock'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("VINE360_SPHERESFM_COLMAP", str(fake))
    monkeypatch.setattr(sph, "candidate_binaries", lambda: [fake])
    assert sph.find_binary() is None


def test_feature_extraction_uses_the_readme_sphere_params():
    cmd = sph.build_feature_extraction_command(B, Path("/p/db.db"), Path("/p/img"), Path("/p/list.txt"), width=7040, height=3520)
    assert cmd[:2] == [str(B), "feature_extractor"]
    assert cmd[cmd.index("--ImageReader.camera_model") + 1] == "SPHERE"
    assert cmd[cmd.index("--ImageReader.camera_params") + 1] == "1,3520,1760"  # README example
    assert cmd[cmd.index("--SiftExtraction.use_gpu") + 1] == "0"


def test_mapper_keeps_intrinsics_fixed():
    cmd = sph.build_mapper_command(B, Path("/p/db.db"), Path("/p/img"), Path("/p/out"))
    for flag in ("--Mapper.ba_refine_focal_length", "--Mapper.ba_refine_principal_point", "--Mapper.ba_refine_extra_params"):
        assert cmd[cmd.index(flag) + 1] == "0"
    assert cmd[cmd.index("--Mapper.sphere_camera") + 1] == "1"


def test_cubic_reprojection_uses_the_real_subcommand_spelling():
    cmd = sph.build_cubic_reprojection_command(B, Path("/p/m"), Path("/p/img"), Path("/p/out"))
    assert cmd[1] == "sphere_cubic_reprojecer"


# Every option set away from its default, so every flag a builder can emit
# is checked against the real binary (docs/adr/0040).
_ALL_OPTIONS = SfmConfig(
    camera_mode="per_folder", max_image_size=1600, max_num_features=2048, peak_threshold=0.004, edge_threshold=12.0,
    upright=True, estimate_affine_shape=True, domain_size_pooling=True, max_ratio=0.7, max_distance=0.6,
    cross_check=False, max_num_matches=4096, guided_matching=True, min_num_inliers=20, max_error=3.0,
    min_inlier_ratio=0.3, sequential_overlap=7, quadratic_overlap=False, loop_detection=True, loop_detection_period=5,
    loop_detection_num_images=20, vocab_tree_path="/p/tree.bin", min_num_matches=20, multiple_models=False,
    min_model_size=4, init_min_num_inliers=50, init_min_tri_angle=8.0, init_max_forward_motion=0.99,
    abs_pose_min_num_inliers=20, abs_pose_min_inlier_ratio=0.2, filter_max_reproj_error=3.0, filter_min_tri_angle=1.0,
    random_seed=7,
)


def _builders():
    p = Path("/p")
    return [
        sph.build_feature_extraction_command(B, p / "db", p / "img", p / "list", width=64, height=32),
        sph.build_matcher_command(B, p / "db", overlap=10),
        sph.build_mapper_command(B, p / "db", p / "img", p / "out"),
        sph.build_model_converter_command(B, p / "in", p / "out"),
        sph.build_cubic_reprojection_command(B, p / "in", p / "img", p / "out"),
        sph.build_feature_extraction_command(B, p / "db", p / "img", p / "list", width=64, height=32, config=_ALL_OPTIONS),
        sph.build_feature_extraction_command(
            B, p / "db", p / "img", p / "list", width=64, height=32, config=_ALL_OPTIONS.replace(camera_mode="per_image")
        ),
        sph.build_matcher_command(B, p / "db", config=_ALL_OPTIONS),
        sph.build_matcher_command(B, p / "db", config=_ALL_OPTIONS.replace(matcher="exhaustive")),
        sph.build_matcher_command(B, p / "db", config=_ALL_OPTIONS.replace(matcher="vocab_tree")),
        sph.build_mapper_command(B, p / "db", p / "img", p / "out", config=_ALL_OPTIONS),
    ]


def _help_text(subcommand: str) -> str:
    binary = sph.find_binary()
    return subprocess.run([str(binary), subcommand, "-h"], capture_output=True, text=True, env=sph._env()).stdout


@requires_spheresfm
@pytest.mark.parametrize("command", _builders(), ids=lambda c: c[1])
def test_every_flag_exists_in_the_real_binary(command):
    binary = sph.find_binary()
    help_text = subprocess.run([str(binary), command[1], "-h"], capture_output=True, text=True, env=sph._env()).stdout
    flags = [part for part in command[2:] if part.startswith("--")]
    missing = [flag for flag in flags if flag not in help_text]
    assert not missing, f"{command[1]} -h doesn't list {missing}"


def test_read_text_model_parses_colmaps_txt_format(tmp_path):
    (tmp_path / "cameras.txt").write_text("# c\n1 SPHERE 1024 512 1 512 256\n")
    (tmp_path / "images.txt").write_text(
        "# i\n"
        "1 1 0 0 0 0.5 0 -1 1 frame with space.png\n"
        "10.5 20.5 7 30 40 -1\n"
        "2 0.7071 0 0.7071 0 1 2 3 1 frame_000002.png\n"
        "\n"
    )
    (tmp_path / "points3D.txt").write_text("# p\n7 1 2 3 10 20 30 0.25 1 0\n")
    model = sph.read_text_model(tmp_path)
    assert model.cameras[1].model == "SPHERE" and model.cameras[1].params == [1.0, 512.0, 256.0]
    assert model.images[1].name == "frame with space.png"
    assert model.images[1].points2d == [(10.5, 20.5, 7), (30.0, 40.0, -1)]
    assert model.images[2].points2d == []
    assert model.points[7].track == [(1, 0)] and model.points[7].rgb == (10, 20, 30)


def test_gpu_sift_only_for_a_cuda_build():
    assert sph.has_cuda("COLMAP 3.8 (Commit 6b40b2d on 2026-02-24 with CUDA)")
    assert not sph.has_cuda("COLMAP 3.8 (Commit 6b40b2d on 2026-02-24 without CUDA)")
    assert not sph.has_cuda(None)
    p = Path("/p")
    cpu = sph.build_feature_extraction_command(B, p / "db", p / "img", p / "l", width=64, height=32)
    gpu = sph.build_feature_extraction_command(B, p / "db", p / "img", p / "l", width=64, height=32, use_gpu=True)
    assert cpu[cpu.index("--SiftExtraction.use_gpu") + 1] == "0"
    assert gpu[gpu.index("--SiftExtraction.use_gpu") + 1] == "1"
    matcher = sph.build_matcher_command(B, p / "db", overlap=5, use_gpu=True)
    assert matcher[matcher.index("--SiftMatching.use_gpu") + 1] == "1"


def test_options_map_onto_the_flags():
    p = Path("/p")
    extract = sph.build_feature_extraction_command(
        B, p / "db", p / "img", p / "l", width=64, height=32, use_gpu=True, config=_ALL_OPTIONS
    )
    assert extract[extract.index("--ImageReader.single_camera") + 1] == "0"
    assert extract[extract.index("--ImageReader.single_camera_per_folder") + 1] == "1"
    assert extract[extract.index("--SiftExtraction.max_image_size") + 1] == "1600"
    assert extract[extract.index("--SiftExtraction.peak_threshold") + 1] == "0.004"
    assert extract[extract.index("--SiftExtraction.use_gpu") + 1] == "0"  # affine/DSP SIFT are CPU-only
    assert extract[extract.index("--random_seed") + 1] == "7"
    assert "--random_seed" not in sph.build_matcher_command(B, p / "db")  # -1 = not fixed: left to the binary

    sequential = sph.build_matcher_command(B, p / "db", use_gpu=True, config=_ALL_OPTIONS)
    assert sequential[1] == "sequential_matcher"
    assert sequential[sequential.index("--SiftMatching.use_gpu") + 1] == "1"
    assert sequential[sequential.index("--SequentialMatching.overlap") + 1] == "7"
    assert sequential[sequential.index("--SequentialMatching.vocab_tree_path") + 1] == "/p/tree.bin"
    exhaustive = sph.build_matcher_command(B, p / "db", config=_ALL_OPTIONS.replace(matcher="exhaustive"))
    assert exhaustive[1] == "exhaustive_matcher" and "--SequentialMatching.overlap" not in exhaustive
    vocab = sph.build_matcher_command(B, p / "db", config=_ALL_OPTIONS.replace(matcher="vocab_tree"))
    assert vocab[1] == "vocab_tree_matcher"

    mapper = sph.build_mapper_command(B, p / "db", p / "img", p / "out", config=_ALL_OPTIONS)
    assert mapper[mapper.index("--Mapper.init_min_tri_angle") + 1] == "8"
    assert mapper[mapper.index("--Mapper.min_model_size") + 1] == "4"
    assert "--Mapper.min_model_size" not in sph.build_mapper_command(B, p / "db", p / "img", p / "out")
    for flag in ("--Mapper.ba_refine_focal_length", "--Mapper.ba_refine_extra_params"):
        assert mapper[mapper.index(flag) + 1] == "0"  # SPHERE intrinsics stay fixed whatever the options say


_HELP_DEFAULT = re.compile(r"(--[\w.]+) arg \(=([^)]*)\)")


@requires_spheresfm
@pytest.mark.parametrize("command", _builders()[:3], ids=lambda c: c[1])
def test_default_options_pass_the_binarys_own_defaults(command):
    """With SfmConfig() every tunable flag vine360 now passes must equal the
    binary's own default -- so a default-options run behaves exactly as it
    did before options existed. (The flags vine360 always set on purpose --
    SPHERE camera, sphere_camera, fixed intrinsics, GPU -- are excluded.)"""
    defaults = dict(_HELP_DEFAULT.findall(_help_text(command[1])))
    deliberate = {
        "--ImageReader.camera_model", "--ImageReader.camera_params", "--SiftExtraction.use_gpu", "--SiftMatching.use_gpu",
        "--Mapper.sphere_camera", "--Mapper.ba_refine_focal_length", "--Mapper.ba_refine_extra_params",
    }
    checked = 0
    for flag, value in zip(command[2:], command[3:]):
        if flag in defaults and flag not in deliberate:
            assert float(value) == pytest.approx(float(defaults[flag])), f"{flag}: {value} != binary default {defaults[flag]}"
            checked += 1
    assert checked >= 5
