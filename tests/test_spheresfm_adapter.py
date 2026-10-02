"""vine360.sfm.spheresfm_adapter. The command builders and TXT reader are
checked against the real SphereSfM build when one is present -- every
flag the builders use must appear in that binary's own `<command> -h`
(docs/adr/0036); the pure builder tests run regardless."""

import subprocess
from pathlib import Path

import pytest

from conftest import requires_spheresfm
from vine360.sfm import spheresfm_adapter as sph

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


def _builders():
    p = Path("/p")
    return [
        sph.build_feature_extraction_command(B, p / "db", p / "img", p / "list", width=64, height=32),
        sph.build_matcher_command(B, p / "db", overlap=10),
        sph.build_mapper_command(B, p / "db", p / "img", p / "out"),
        sph.build_model_converter_command(B, p / "in", p / "out"),
        sph.build_cubic_reprojection_command(B, p / "in", p / "img", p / "out"),
    ]


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
