"""Real end-to-end SfM tests: nothing here mocks pycolmap.

Two genuinely different things need testing, and neither substitutes for
the other:

1. Feature extraction + masking + matching against real rendered images
   (uses vine360's own M2 projection code to generate overlapping views).
   This does NOT test 3D reconstruction, because every view here is
   rendered from a single equirectangular panorama -- i.e. one optical
   center, zero baseline. COLMAP's incremental mapper requires real
   parallax (`init_min_tri_angle`) to triangulate anything; a rotation-only
   image set correctly fails to register, no matter how good the 2D
   matches are. That failure is asserted explicitly below, not treated as
   a bug -- see docs/adr/0007.

2. Incremental mapping + diagnostics, tested with `pycolmap`'s own
   synthetic-dataset generator, which does produce real multi-position 3D
   ground truth (an actual baseline between camera frames).
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

pycolmap = pytest.importorskip("pycolmap")

from vine360.masking.semantics import colmap_mask_path, save_mask
from vine360.projection.cubemap import six_face_preset
from vine360.projection.render import project_equirect_to_face
from vine360.sfm.colmap_adapter import (
    SfmRegistrationError,
    evaluate_registration_quality,
    extract_and_match,
    map_and_diagnose,
)


def _make_textured_equirect(width: int, height: int, seed: int = 0) -> np.ndarray:
    """A noisy background plus many small randomly placed/colored/sized
    shapes -- gives SIFT plenty of distinct, matchable local features."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(noise, "RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(400):
        x, y = rng.integers(0, width), rng.integers(0, height)
        r = rng.integers(4, 14)
        color = tuple(int(c) for c in rng.integers(80, 255, size=3))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    return np.asarray(img)


@pytest.fixture(scope="module")
def rendered_scene(tmp_path_factory):
    """Four overlapping (100 deg FOV) cubemap-style faces of one panorama,
    each with a masked-out corner block, then extracted+matched for real."""
    tmp_path = tmp_path_factory.mktemp("sfm_scene")
    equirect = _make_textured_equirect(2048, 1024, seed=42)

    image_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    image_dir.mkdir()

    faces = six_face_preset(face_size=480, fov_degrees=100.0, include_polar_faces=False)
    names = []
    for face in faces:
        rendered = project_equirect_to_face(equirect, face)
        name = f"{face.name}.png"
        Image.fromarray(rendered, "RGB").save(image_dir / name)
        names.append(name)

        keep_mask = np.full((rendered.shape[0], rendered.shape[1]), 255, dtype=np.uint8)
        keep_mask[0:60, 0:60] = 0  # a fixed masked-out corner, as if person/sky
        save_mask(keep_mask, colmap_mask_path(name, masks_dir))

    database_path = tmp_path / "sfm" / "database.db"
    extract_and_match(image_dir, database_path, mask_dir=masks_dir)

    return {
        "tmp_path": tmp_path,
        "image_dir": image_dir,
        "masks_dir": masks_dir,
        "database_path": database_path,
        "names": names,
    }


def test_a04_no_keypoints_inside_masked_region(rendered_scene):
    """Acceptance test A04: "No extracted keypoints lie inside zero-valued
    mask regions on a controlled fixture." Real feature extraction, real
    mask files, real database -- confirmed against actual COLMAP output."""
    db = pycolmap.Database.open(rendered_scene["database_path"])
    try:
        images = db.read_all_images()
        assert len(images) == 4
        for image in images:
            keypoints = db.read_keypoints(image.image_id)
            assert len(keypoints) > 0, f"expected keypoints for {image.name}"
            mask = np.asarray(Image.open(colmap_mask_path(image.name, rendered_scene["masks_dir"])))
            for x, y in keypoints[:, :2]:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
                    assert mask[yi, xi] != 0, (
                        f"keypoint at ({xi},{yi}) in {image.name} falls inside the masked-out region"
                    )
    finally:
        db.close()


def test_zero_parallax_single_panorama_views_correctly_fail_to_register(rendered_scene, tmp_path):
    """Documents real, correct COLMAP behavior: views that all share one
    optical center (any single-frame cubemap/projection preset) have zero
    baseline and cannot be triangulated, regardless of 2D match quality."""
    sparse_dir = tmp_path / "sparse"
    with pytest.raises(SfmRegistrationError):
        map_and_diagnose(rendered_scene["database_path"], rendered_scene["image_dir"], sparse_dir)


@pytest.fixture(scope="module")
def synthetic_multiview_database(tmp_path_factory):
    """A dataset with genuine 3D ground truth and real camera-position
    parallax, via pycolmap's own synthetic-dataset generator -- exercises
    map_and_diagnose the way an actual multi-position vineyard capture
    would (distinct camera translations, not a single rotated viewpoint)."""
    tmp_path = tmp_path_factory.mktemp("sfm_synthetic")
    database_path = tmp_path / "database.db"
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    db = pycolmap.Database.open(database_path)
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 1
    options.num_cameras_per_rig = 1
    options.num_frames_per_rig = 12
    options.num_points3D = 300
    pycolmap.synthesize_dataset(options, db)
    db.close()

    return {"database_path": database_path, "image_dir": image_dir, "num_frames": 12, "num_points": 300}


def test_map_and_diagnose_with_real_parallax(synthetic_multiview_database, tmp_path):
    sparse_dir = tmp_path / "sparse"
    reconstruction, diagnostics = map_and_diagnose(
        synthetic_multiview_database["database_path"],
        synthetic_multiview_database["image_dir"],
        sparse_dir,
        total_images=synthetic_multiview_database["num_frames"],
    )

    assert diagnostics.total_images == synthetic_multiview_database["num_frames"]
    assert diagnostics.registered_images == synthetic_multiview_database["num_frames"]
    assert diagnostics.registered_ratio == 1.0
    assert diagnostics.num_connected_models == 1
    assert diagnostics.num_points3d == synthetic_multiview_database["num_points"]
    assert diagnostics.mean_reprojection_error < 0.1  # exact synthetic ground truth

    assert evaluate_registration_quality(diagnostics) == []

    model_dirs = list(sparse_dir.iterdir())
    assert len(model_dirs) == 1
    assert (model_dirs[0] / "cameras.bin").exists()
    assert (model_dirs[0] / "images.bin").exists()
    assert (model_dirs[0] / "points3D.bin").exists()


def test_evaluate_registration_quality_flags_problems():
    from vine360.sfm.colmap_adapter import SfmDiagnostics

    bad = SfmDiagnostics(
        total_images=10,
        registered_images=3,
        registered_ratio=0.3,
        num_connected_models=2,
        num_points3d=0,
        mean_reprojection_error=5.0,
        mean_track_length=1.0,
        mean_observations_per_reg_image=1.0,
    )
    warnings = evaluate_registration_quality(bad)
    assert len(warnings) == 4
