"""Tests for the command-construction functions only -- SphereSfM itself
is unverified in this environment (no build available), so these only
confirm the commands are built as documented, not that they actually run.
See vine360.sfm.spheresfm_adapter's module docstring and docs/adr/0017.
"""

from pathlib import Path

from vine360.sfm.spheresfm_adapter import (
    build_cubic_reprojection_command,
    build_feature_extraction_command,
    build_mapper_command,
    build_matcher_command,
    validate_installation,
)


def test_validate_installation_never_raises():
    status = validate_installation()
    assert status["colmap_binary_found"] in (True, False)
    assert "note" in status


def test_build_feature_extraction_command():
    cmd = build_feature_extraction_command(Path("/p/db.db"), Path("/p/images"), width=2048, height=1024)
    assert cmd[:2] == ["colmap", "feature_extractor"]
    assert "--ImageReader.camera_model" in cmd
    assert "SPHERE" in cmd
    assert "1024.0,1024.0" in " ".join(cmd) or "2048,1024.0,512.0" in " ".join(cmd)
    params_index = cmd.index("--ImageReader.camera_params") + 1
    assert cmd[params_index] == "2048,1024.0,512.0"


def test_build_matcher_command():
    cmd = build_matcher_command(Path("/p/db.db"))
    assert cmd == ["colmap", "spatial_matcher", "--database_path", "/p/db.db"]


def test_build_mapper_command():
    cmd = build_mapper_command(Path("/p/db.db"), Path("/p/images"), Path("/p/sparse"))
    assert cmd[:2] == ["colmap", "mapper"]
    assert "--Mapper.sphere_camera" in cmd
    assert cmd[cmd.index("--Mapper.sphere_camera") + 1] == "1"


def test_build_cubic_reprojection_command():
    cmd = build_cubic_reprojection_command(Path("/p/sparse/0"), Path("/p/cubic"))
    assert cmd == [
        "colmap",
        "sphere_cubic_reprojecter",
        "--input_path",
        "/p/sparse/0",
        "--output_path",
        "/p/cubic",
    ]
