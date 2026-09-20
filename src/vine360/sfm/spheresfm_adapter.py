"""SphereSfM adapter.

**UNVERIFIED, like the Nerfstudio training adapter (docs/adr/0009) --
unlike this project's other adapters, which were each confirmed against
real installed source, a real live endpoint, or real registration
results.** SphereSfM (https://github.com/json87/SphereSfM) is a genuine,
separate fork of COLMAP with its own `SPHERE` camera model, a
`--Mapper.sphere_camera` mapper flag, and a `sphere_cubic_reprojecter`
tool for converting a spherical reconstruction to cubic format for dense
matching. It is **not** available as a pip package or prebuilt binary --
it must be compiled from C++ source, which this environment cannot do
(confirmed: no prebuilt binary, no wheel; its own README points at
COLMAP's standard from-source build instructions). It is also a
different thing from `pycolmap`'s own native `EQUIRECTANGULAR` camera
model (see `vine360.sfm.project_run`'s `image_source="frames"` option,
which *is* real and verified) -- SphereSfM's `SPHERE` model and its
sphere-aware mapper/matching are specific to this fork and have no
equivalent in stock COLMAP or pycolmap. See docs/adr/0017.

The command shapes below reflect SphereSfM's own README usage examples,
not source-verified against an installed build. **Run
`colmap feature_extractor --help` and `colmap mapper --help` against a
real SphereSfM build and correct this file before depending on it.**
"""

from __future__ import annotations

import shutil
from pathlib import Path

SPHERESFM_REPO_URL = "https://github.com/json87/SphereSfM"


def validate_installation() -> dict:
    """Reports whether *some* `colmap` binary is on PATH -- this cannot
    distinguish a SphereSfM-flavored build from stock COLMAP (both are
    named `colmap`; the only way to tell is to actually run
    `colmap feature_extractor --help` and check for a `SPHERE` camera
    model, which this function doesn't do to avoid a slow subprocess call
    on every check)."""
    path = shutil.which("colmap")
    return {
        "colmap_binary_found": path is not None,
        "path": path,
        "note": "cannot confirm this is a SphereSfM build rather than stock COLMAP without running it",
    }


def build_feature_extraction_command(
    database_path: Path, image_path: Path, *, width: int, height: int
) -> list[str]:
    """SphereSfM's SPHERE camera model params, per its README, are
    "f,cx,cy" for a full equirectangular image -- unlike stock COLMAP's
    EQUIRECTANGULAR model (confirmed via pycolmap: just "w,h", no focal
    length). Using width as a stand-in focal length and the image center
    matches the README's own example (`--ImageReader.camera_params
    "1,3520,1760"` for a presumably ~7040x3520 image)."""
    camera_params = f"{width},{width / 2},{height / 2}"
    return [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--ImageReader.camera_model",
        "SPHERE",
        "--ImageReader.camera_params",
        camera_params,
    ]


def build_matcher_command(database_path: Path) -> list[str]:
    return ["colmap", "spatial_matcher", "--database_path", str(database_path)]


def build_mapper_command(database_path: Path, image_path: Path, output_path: Path) -> list[str]:
    return [
        "colmap",
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_path),
        "--Mapper.sphere_camera",
        "1",
    ]


def build_cubic_reprojection_command(input_path: Path, output_path: Path) -> list[str]:
    """The README's custom tool for converting a spherical reconstruction
    to cubic format for dense matching -- only meaningful with an actual
    SphereSfM build, since `sphere_cubic_reprojecter` isn't a stock COLMAP
    subcommand at all."""
    return [
        "colmap",
        "sphere_cubic_reprojecter",
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
    ]
