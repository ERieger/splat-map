"""Real pycolmap reconstructions throughout (via pycolmap.synthesize_dataset
+ colmap_adapter.map_and_diagnose, the same real pipeline test_sfm_
integration.py exercises) -- this module's own job is just the file
wiring around an already-written model, so nothing about SfM itself is
mocked."""

import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("pycolmap")

import pycolmap

from vine360.config import CaptureMode
from vine360.export.postshot import PostshotExportError, export_for_postshot
from vine360.masking.build import build_mask_for_view
from vine360.masking.semantics import colmap_mask_path
from vine360.project import create_project, open_index_db
from vine360.sfm.colmap_adapter import map_and_diagnose


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    yield root, conn
    conn.close()


_model_counter = 0


def _build_real_model(root, *, num_frames=6, num_points=50):
    """Writes a genuine registered COLMAP model under root/sfm/sparse-N/
    using real camera-position parallax (see docs/adr/0007) -- returns
    the sorted list of image names pycolmap assigned."""
    global _model_counter
    _model_counter += 1
    n = _model_counter

    database_path = root / "sfm" / f"database-{n}.db"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir = root / f"sfm_synth_images-{n}"  # not used for pixel content, only for total_images count
    image_dir.mkdir()

    db = pycolmap.Database.open(database_path)
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 1
    options.num_cameras_per_rig = 1
    options.num_frames_per_rig = num_frames
    options.num_points3D = num_points
    pycolmap.synthesize_dataset(options, db)
    db.close()

    sparse_dir = root / "sfm" / f"sparse-{n}"
    _reconstruction, diagnostics, model_dir = map_and_diagnose(
        database_path, image_dir, sparse_dir, total_images=num_frames
    )
    assert diagnostics.registered_images == num_frames  # sanity: real full registration

    reconstruction = pycolmap.Reconstruction(model_dir)
    image_names = sorted(image.name for image in reconstruction.images.values())
    return model_dir, image_names


def _write_dummy_image(path, size=32):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8), "RGB").save(path)


def _insert_sfm_run(conn, root, model_dir, *, config=None, run_id="sfm-test1"):
    conn.execute(
        "INSERT INTO sfm_runs (run_id, image_set_hash, engine_version, config, model_stats, selected_model, "
        "created_at) VALUES (?, 'hash', 'v', ?, '{}', ?, datetime('now'))",
        (run_id, json.dumps(config or {"image_source": "projections"}), model_dir.relative_to(root).as_posix()),
    )
    conn.commit()


def test_export_for_postshot_no_run_raises(project):
    root, conn = project
    with pytest.raises(PostshotExportError, match="no SfM run"):
        export_for_postshot(conn, root, root / "exports" / "postshot")


def test_export_for_postshot_missing_selected_model_raises(project):
    root, conn = project
    conn.execute(
        "INSERT INTO sfm_runs (run_id, image_set_hash, engine_version, config, model_stats, selected_model, "
        "created_at) VALUES ('sfm-old', 'h', 'v', '{}', '{}', NULL, datetime('now'))"
    )
    conn.commit()
    with pytest.raises(PostshotExportError, match="re-run Pose estimation"):
        export_for_postshot(conn, root, root / "exports" / "postshot")


def test_export_for_postshot_stale_selected_model_raises_actionable_error(project):
    """Real regression: every sfm_runs row created before the selected_
    model bug fix (docs/adr/0018) has selected_model == its own run_id,
    which never resolves to a real directory -- confirmed against a real
    project on disk with exactly this old data. Must not surface as a
    bare 'missing files' error."""
    root, conn = project
    conn.execute(
        "INSERT INTO sfm_runs (run_id, image_set_hash, engine_version, config, model_stats, selected_model, "
        "created_at) VALUES ('sfm-old1', 'h', 'v', '{}', '{}', 'sfm-old1', datetime('now'))"
    )
    conn.commit()
    with pytest.raises(PostshotExportError, match="predates a bug fix"):
        export_for_postshot(conn, root, root / "exports" / "postshot")


def test_export_for_postshot_copies_images_sparse_and_masks(project):
    root, conn = project
    model_dir, image_names = _build_real_model(root)
    for name in image_names:
        _write_dummy_image(root / "projections" / name)

    # Build a real keep-mask for every exported image, via a real frames/views row each.
    for i, name in enumerate(image_names):
        frame_id = f"frame-{i}"
        conn.execute(
            "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum) "
            "VALUES (?, 's1', 0.0, '{}', 'x', 'y')",
            (frame_id,),
        )
        view_id = f"{frame_id}:v"
        conn.execute(
            "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, "
            "fixed_rotation, image_path) VALUES (?, ?, 'p', 32, 32, '{}', '{}', ?)",
            (view_id, frame_id, f"projections/{name}"),
        )
        conn.commit()
        build_mask_for_view(conn, root, view_id)

    _insert_sfm_run(conn, root, model_dir)

    result = export_for_postshot(conn, root, root / "exports" / "postshot")

    assert result.num_images == len(image_names)
    assert result.num_masks == len(image_names)
    assert result.warnings == []
    for name in image_names:
        assert (result.images_dir / name).exists()
        assert (result.masks_dir / name).exists()  # same basename as the color image, not "<name>.png"
    for fname in ("cameras.bin", "images.bin", "points3D.bin"):
        assert (result.sparse_dir / fname).exists()
    reloaded = pycolmap.Reconstruction(result.sparse_dir)
    assert len(reloaded.images) == len(image_names)


def test_export_for_postshot_no_masks_built_warns(project):
    root, conn = project
    model_dir, image_names = _build_real_model(root)
    for name in image_names:
        _write_dummy_image(root / "projections" / name)
    _insert_sfm_run(conn, root, model_dir)

    result = export_for_postshot(conn, root, root / "exports" / "postshot")

    assert result.num_images == len(image_names)
    assert result.masks_dir is None
    assert result.num_masks == 0
    assert any("no masks" in w for w in result.warnings)


def test_export_for_postshot_missing_source_image_warns_and_skips(project):
    root, conn = project
    model_dir, image_names = _build_real_model(root)
    for name in image_names[:-1]:  # leave the last one missing on purpose
        _write_dummy_image(root / "projections" / name)
    _insert_sfm_run(conn, root, model_dir)

    result = export_for_postshot(conn, root, root / "exports" / "postshot")

    assert result.num_images == len(image_names) - 1
    assert any("missing, skipped" in w for w in result.warnings)


def test_export_for_postshot_defaults_to_latest_run(project):
    root, conn = project
    model_dir_1, names_1 = _build_real_model(root, num_frames=4)
    for name in names_1:
        _write_dummy_image(root / "projections" / name)
    _insert_sfm_run(conn, root, model_dir_1, run_id="sfm-first")

    model_dir_2, names_2 = _build_real_model(root, num_frames=5)
    for name in names_2:
        _write_dummy_image(root / "projections" / name)
    conn.execute("UPDATE sfm_runs SET created_at = '2000-01-01T00:00:00' WHERE run_id = 'sfm-first'")
    _insert_sfm_run(conn, root, model_dir_2, run_id="sfm-second")

    result = export_for_postshot(conn, root, root / "exports" / "postshot")
    assert result.num_images == len(names_2)


def test_export_for_postshot_frames_engine_skips_masks(project):
    root, conn = project
    model_dir, image_names = _build_real_model(root)
    for name in image_names:
        _write_dummy_image(root / "frames" / "s1" / name)
    _insert_sfm_run(conn, root, model_dir, config={"image_source": "frames", "source_id": "s1"})

    result = export_for_postshot(conn, root, root / "exports" / "postshot")

    assert result.num_images == len(image_names)
    assert result.masks_dir is None
    assert result.num_masks == 0
