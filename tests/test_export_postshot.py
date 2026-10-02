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
from vine360.export.postshot import (
    PostshotExportError,
    _migrate_legacy_export_layout,
    _select_priors_run,
    export_for_postshot,
    export_for_realityscan,
    export_frames_and_masks_for_postshot,
    realityscan_priors_available,
)
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
            "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
            "VALUES (?, 's1', 0.0, '{}', 'x', 'y', 's1')",
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

    progress_calls = []
    result = export_for_postshot(
        conn, root, root / "exports" / "postshot", progress_callback=lambda m, c, t: progress_calls.append((m, c, t))
    )

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

    # Real progress reporting, not just an indeterminate spinner: one call per
    # image and per mask, with an accurate running (current, total).
    image_progress_calls = [c for c in progress_calls if c[0].startswith("Copying images")]
    mask_progress_calls = [c for c in progress_calls if c[0].startswith("Copying masks")]
    assert len(image_progress_calls) == len(image_names)
    assert len(mask_progress_calls) == len(image_names)
    assert image_progress_calls[-1][1:] == (len(image_names), len(image_names))


def test_export_for_postshot_disambiguates_same_face_name_across_frames(project):
    """Same real bug as the frames-and-masks-only mode, but here the
    filenames also have to stay in sync with images.bin, or Postshot's
    COLMAP import can't find them at all."""
    root, conn = project
    model_dir, image_names = _build_real_model(root, num_frames=4)

    # Rename this model's images to vine360's real naming shape --
    # "<source_id>:<index>/<face>.png" -- so every image shares the same
    # leaf filename ("front.png"), exactly the real-project collision.
    reconstruction = pycolmap.Reconstruction(model_dir)
    renamed = {}
    for i, image in enumerate(reconstruction.images.values()):
        new_name = f"source-a:{i:06d}/front.png"
        renamed[image.name] = new_name
        image.name = new_name
    reconstruction.write(model_dir)
    for original_name, new_name in renamed.items():
        _write_dummy_image(root / "projections" / new_name)

    _insert_sfm_run(conn, root, model_dir)
    result = export_for_postshot(conn, root, root / "exports" / "postshot")

    assert result.num_images == len(image_names)
    image_files = sorted(p.name for p in result.images_dir.iterdir())
    assert len(image_files) == len(image_names)  # not collapsed into one "front.png"
    assert len(set(image_files)) == len(image_names)
    for name in image_files:
        assert ":" not in name

    # images.bin must reference the same flattened filenames the files were written under.
    reloaded = pycolmap.Reconstruction(result.sparse_dir)
    reloaded_names = sorted(image.name for image in reloaded.images.values())
    assert reloaded_names == image_files


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


def _insert_view_with_image(conn, root, frame_id, face_name, *, with_mask=False, source_id="s1"):
    image_path = root / "projections" / frame_id / f"{face_name}.png"
    _write_dummy_image(image_path)
    conn.execute(
        "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
        "VALUES (?, ?, 0.0, '{}', 'x', 'y', ?) ON CONFLICT(frame_id) DO NOTHING",
        (frame_id, source_id, source_id),
    )
    view_id = f"{frame_id}:{face_name}"
    conn.execute(
        "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, "
        "fixed_rotation, image_path) VALUES (?, ?, 'p', 32, 32, '{}', '{}', ?)",
        (view_id, frame_id, f"projections/{frame_id}/{face_name}.png"),
    )
    conn.commit()
    if with_mask:
        build_mask_for_view(conn, root, view_id)
    return view_id


def test_export_frames_and_masks_no_views_raises(project):
    root, conn = project
    with pytest.raises(PostshotExportError, match="generate projections"):
        export_frames_and_masks_for_postshot(conn, root, root / "exports" / "frames")


def test_export_frames_and_masks_copies_images_and_masks(project):
    root, conn = project
    _insert_view_with_image(conn, root, "frame-1", "front", with_mask=True)
    _insert_view_with_image(conn, root, "frame-1", "back", with_mask=True)

    progress_calls = []
    result = export_frames_and_masks_for_postshot(
        conn, root, root / "exports" / "frames", progress_callback=lambda m, c, t: progress_calls.append((m, c, t))
    )

    assert result.sparse_dir is None  # no pose data at all -- Postshot solves its own
    assert result.num_images == 2
    assert result.num_masks == 2
    assert result.warnings == []
    assert (result.images_dir / "frame-1__front.png").exists()  # flattened, not nested
    assert (result.masks_dir / "frame-1__front.png").exists()  # not "frame-1/front.png.png"

    determinate_calls = [c for c in progress_calls if c[1] is not None]
    assert len(determinate_calls) == 2
    assert determinate_calls[-1][1:] == (2, 2)


def test_export_frames_and_masks_disambiguates_same_face_name_across_frames(project):
    """Real bug: every frame's projections share the same five leaf
    filenames (front.png, back.png, ...) -- fine for our own on-disk
    layout since the frame_id folder disambiguates them, but Postshot
    associates a dropped mask to a color image by filename alone (its own
    docs: "the same base filename"), so two different frames' front.png
    files must not become two files of the same name in the export."""
    root, conn = project
    _insert_view_with_image(conn, root, "source-a:000000", "front", with_mask=True)
    _insert_view_with_image(conn, root, "source-a:000001", "front", with_mask=True)

    result = export_frames_and_masks_for_postshot(conn, root, root / "exports" / "frames")

    assert result.num_images == 2
    assert result.num_masks == 2
    image_files = sorted(p.name for p in result.images_dir.iterdir())
    mask_files = sorted(p.name for p in result.masks_dir.iterdir())
    assert len(image_files) == 2  # not silently overwritten into one "front.png"
    assert len(set(image_files)) == 2
    assert image_files == mask_files  # each mask shares its color image's exact flattened name
    for name in image_files:
        assert ":" not in name  # frame_ids can contain ":" -- not valid in a bare Windows filename


def test_export_frames_and_masks_without_masks_warns(project):
    root, conn = project
    _insert_view_with_image(conn, root, "frame-1", "front", with_mask=False)

    result = export_frames_and_masks_for_postshot(conn, root, root / "exports" / "frames")

    assert result.num_images == 1
    assert result.masks_dir is None
    assert any("no masks" in w for w in result.warnings)


def test_export_frames_and_masks_filters_by_source_id(project):
    root, conn = project
    _insert_view_with_image(conn, root, "frame-a", "front", with_mask=True, source_id="source-a")
    _insert_view_with_image(conn, root, "frame-b", "front", with_mask=True, source_id="source-b")

    result = export_frames_and_masks_for_postshot(
        conn, root, root / "exports" / "source-a-only", frame_set_id="source-a"
    )

    assert result.num_images == 1
    assert (result.images_dir / "frame-a__front.png").exists()
    assert not (result.images_dir / "frame-b__front.png").exists()


def test_export_frames_and_masks_no_filter_includes_every_source(project):
    root, conn = project
    _insert_view_with_image(conn, root, "frame-a", "front", with_mask=False, source_id="source-a")
    _insert_view_with_image(conn, root, "frame-b", "front", with_mask=False, source_id="source-b")

    result = export_frames_and_masks_for_postshot(conn, root, root / "exports" / "all")

    assert result.num_images == 2  # source_id=None (the default) -- unfiltered, matches prior behavior


def test_export_frames_and_masks_source_filter_with_no_views_raises_actionable_error(project):
    root, conn = project
    _insert_view_with_image(conn, root, "frame-a", "front", source_id="source-a")

    with pytest.raises(PostshotExportError, match="source-b.*generate projections"):
        export_frames_and_masks_for_postshot(
            conn, root, root / "exports" / "source-b-only", frame_set_id="source-b"
        )


def test_export_for_realityscan_no_views_raises(project):
    root, conn = project
    with pytest.raises(PostshotExportError, match="generate projections"):
        export_for_realityscan(conn, root, root / "exports" / "cap" / "realityscan")


def test_export_for_realityscan_writes_mask_in_dedicated_subfolder(project):
    """RealityScan's other documented mask convention (used here instead
    of the flat-adjacent one, so a plain images/ listing stays mask-free
    -- see docs/adr/0028): a mask sits in images/layers/.mask/, using the
    *same* filename as its color image, no extra suffix."""
    root, conn = project
    _insert_view_with_image(conn, root, "frame-1", "front", with_mask=True)

    result = export_for_realityscan(conn, root, root / "exports" / "cap" / "realityscan")

    assert result.num_images == 1
    assert result.num_masks == 1
    assert (result.images_dir / "frame-1__front.png").exists()
    mask_path = result.images_dir / "layers" / ".mask" / "frame-1__front.png"
    assert mask_path.exists()
    assert result.masks_dir == mask_path.parent
    # images/ itself stays mask-free -- the whole point of the subfolder
    assert not (result.images_dir / "frame-1__front.png.mask.png").exists()
    top_level_names = {p.name for p in result.images_dir.iterdir()}
    assert "frame-1__front.png.mask.png" not in top_level_names


def test_export_for_realityscan_filters_by_source_id(project):
    root, conn = project
    _insert_view_with_image(conn, root, "frame-a", "front", with_mask=True, source_id="source-a")
    _insert_view_with_image(conn, root, "frame-b", "front", with_mask=True, source_id="source-b")

    result = export_for_realityscan(
        conn, root, root / "exports" / "source-a" / "realityscan", frame_set_id="source-a"
    )

    assert result.num_images == 1
    assert (result.images_dir / "frame-a__front.png").exists()
    assert not (result.images_dir / "frame-b__front.png").exists()


def test_export_for_realityscan_without_masks_warns(project):
    root, conn = project
    _insert_view_with_image(conn, root, "frame-1", "front", with_mask=False)

    result = export_for_realityscan(conn, root, root / "exports" / "cap" / "realityscan")

    assert result.num_images == 1
    assert result.num_masks == 0
    assert any("no masks" in w for w in result.warnings)


# -- camera-priors CSV (RealityScan only) --------------------------------


def _insert_view_for_projections_model(conn, root, frame_id, image_name):
    """Mirrors test_export_for_postshot_copies_images_sparse_and_masks's
    setup -- a real frames/views row whose image_path matches a real
    COLMAP-registered image's own name exactly, for the "projections"
    engine priors path."""
    _write_dummy_image(root / "projections" / image_name)
    conn.execute(
        "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
        "VALUES (?, 's1', 0.0, '{}', 'x', 'y', 's1')",
        (frame_id,),
    )
    conn.execute(
        "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, "
        "fixed_rotation, image_path) VALUES (?, ?, 'p', 32, 32, '{}', '{}', ?)",
        (f"{frame_id}:v", frame_id, f"projections/{image_name}"),
    )


def test_export_for_realityscan_camera_priors_default_off(project):
    root, conn = project
    _insert_view_with_image(conn, root, "frame-1", "front", with_mask=False)

    result = export_for_realityscan(conn, root, root / "exports" / "cap" / "realityscan")

    assert result.priors_path is None
    assert result.num_priors == 0
    assert not (result.output_dir / "CameraPriors.csv").exists()


def test_export_for_realityscan_camera_priors_projections_engine(project):
    root, conn = project
    model_dir, image_names = _build_real_model(root)
    for i, name in enumerate(image_names):
        _insert_view_for_projections_model(conn, root, f"frame-{i}", name)
    conn.commit()
    _insert_sfm_run(conn, root, model_dir)

    result = export_for_realityscan(
        conn, root, root / "exports" / "cap" / "realityscan", include_camera_priors=True
    )

    assert result.priors_path == result.output_dir / "CameraPriors.csv"
    assert result.num_priors == len(image_names)
    lines = result.priors_path.read_text().strip().splitlines()
    assert lines[0] == "#name,x,y,alt"
    assert len(lines) - 1 == len(image_names)
    exported_names = {p.name for p in result.images_dir.iterdir() if p.is_file()}
    csv_names = {line.split(",")[0] for line in lines[1:]}
    assert csv_names == exported_names


def test_export_for_realityscan_camera_priors_frames_engine_shares_position_across_faces(project):
    """The key regression: every cube face rendered from the same frame
    shares the panorama's own optical center (fixed_rotation is
    rotation-only), so two views under the same frame_id must get
    byte-identical positions in the priors CSV -- no pose composition
    should be needed or attempted."""
    root, conn = project
    model_dir, image_names = _build_real_model(root, num_frames=3)

    reconstruction = pycolmap.Reconstruction(model_dir)
    for i, image in enumerate(reconstruction.images.values()):
        image.name = f"frame_{i:06d}.png"
    reconstruction.write(model_dir)

    num_frames = len(image_names)
    for i in range(num_frames):
        frame_id = f"s1:{i:06d}"
        conn.execute(
            "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
            "VALUES (?, 's1', 0.0, '{}', ?, 'y', 's1')",
            (frame_id, f"frames/s1/frame_{i:06d}.png"),
        )
        for face in ("front", "back"):
            image_path = root / "projections" / frame_id / f"{face}.png"
            _write_dummy_image(image_path)
            conn.execute(
                "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, "
                "fixed_rotation, image_path) VALUES (?, ?, 'p', 32, 32, '{}', '{}', ?)",
                (f"{frame_id}:{face}", frame_id, f"projections/{frame_id}/{face}.png"),
            )
    conn.commit()
    _insert_sfm_run(conn, root, model_dir, config={"image_source": "frames", "source_id": "s1"})

    result = export_for_realityscan(
        conn, root, root / "exports" / "cap" / "realityscan", include_camera_priors=True
    )

    assert result.num_priors == 2 * num_frames
    priors_by_name = {}
    for line in result.priors_path.read_text().strip().splitlines()[1:]:
        name, x, y, alt = line.split(",")
        priors_by_name[name] = (x, y, alt)
    for i in range(num_frames):
        front = priors_by_name[f"s1_{i:06d}__front.png"]
        back = priors_by_name[f"s1_{i:06d}__back.png"]
        assert front == back  # same frame -> same camera center regardless of face


def test_export_for_realityscan_camera_priors_partial_overlap_warns(project):
    root, conn = project
    model_dir, image_names = _build_real_model(root, num_frames=3)
    for i, name in enumerate(image_names):
        _insert_view_for_projections_model(conn, root, f"frame-{i}", name)
    _insert_view_with_image(conn, root, "frame-extra", "front", source_id="s1")  # no matching registered pose
    conn.commit()
    _insert_sfm_run(conn, root, model_dir)

    result = export_for_realityscan(
        conn, root, root / "exports" / "cap" / "realityscan", include_camera_priors=True
    )

    assert result.num_images == len(image_names) + 1
    assert result.num_priors == len(image_names)
    assert any("camera priors" in w and "wrote" in w for w in result.warnings)


def test_export_for_realityscan_camera_priors_no_sfm_run_is_non_fatal(project):
    root, conn = project
    _insert_view_with_image(conn, root, "frame-1", "front", with_mask=False)

    result = export_for_realityscan(
        conn, root, root / "exports" / "cap" / "realityscan", include_camera_priors=True
    )

    assert result.num_images == 1
    assert result.priors_path is None
    assert any("no usable SfM run" in w for w in result.warnings)


def test_export_for_realityscan_camera_priors_stale_model_missing_on_disk_warns(project):
    root, conn = project
    _insert_view_with_image(conn, root, "frame-1", "front", with_mask=False)
    conn.execute(
        "INSERT INTO sfm_runs (run_id, image_set_hash, engine_version, config, model_stats, selected_model, "
        "created_at) VALUES ('sfm-missing', 'h', 'v', '{\"image_source\": \"projections\"}', '{}', "
        "'sfm/sparse-missing', datetime('now'))"
    )
    conn.commit()

    result = export_for_realityscan(
        conn, root, root / "exports" / "cap" / "realityscan", include_camera_priors=True
    )

    assert result.num_images == 1
    assert result.priors_path is None
    assert any("missing on disk" in w for w in result.warnings)


def test_select_priors_run_prefers_matching_frames_over_projections(project):
    root, conn = project
    _insert_sfm_run(conn, root, root / "sfm" / "m1", config={"image_source": "projections"}, run_id="proj-run")
    conn.execute("UPDATE sfm_runs SET created_at = '2000-01-01T00:00:00' WHERE run_id = 'proj-run'")
    _insert_sfm_run(
        conn, root, root / "sfm" / "m2", config={"image_source": "frames", "source_id": "s1"}, run_id="frames-run"
    )

    selection = _select_priors_run(conn, frame_set_id="s1")
    assert selection[0] == "frames-run"


def test_select_priors_run_falls_back_to_projections_when_no_matching_frames_run(project):
    root, conn = project
    _insert_sfm_run(conn, root, root / "sfm" / "m1", config={"image_source": "projections"}, run_id="proj-run")
    _insert_sfm_run(
        conn,
        root,
        root / "sfm" / "m2",
        config={"image_source": "frames", "source_id": "other-source"},
        run_id="frames-run",
    )

    selection = _select_priors_run(conn, frame_set_id="s1")
    assert selection[0] == "proj-run"


def test_select_priors_run_returns_none_with_no_runs(project):
    root, conn = project
    assert _select_priors_run(conn, frame_set_id=None) is None
    assert _select_priors_run(conn, frame_set_id="s1") is None


def test_select_priors_run_all_sources_prefers_projections_but_falls_back(project):
    root, conn = project
    _insert_sfm_run(
        conn, root, root / "sfm" / "m1", config={"image_source": "frames", "source_id": "s1"}, run_id="frames-run"
    )
    assert _select_priors_run(conn, frame_set_id=None)[0] == "frames-run"  # only run available -- falls back to it

    _insert_sfm_run(conn, root, root / "sfm" / "m2", config={"image_source": "projections"}, run_id="proj-run")
    assert _select_priors_run(conn, frame_set_id=None)[0] == "proj-run"  # now prefers projections over frames


def test_select_priors_run_respects_a_projections_run_scoped_to_a_frame_set(project):
    """docs/adr/0034: a six-face run can now be scoped to one frame set.
    It's the best match for that frame set, but has zero overlap with any
    other -- it must never be picked for a different frame set, nor as
    the "project-wide" fallback."""
    root, conn = project
    _insert_sfm_run(
        conn,
        root,
        root / "sfm" / "m1",
        config={"image_source": "projections", "frame_set_id": "s1~i0.5", "source_id": "s1"},
        run_id="scoped-run",
    )
    assert _select_priors_run(conn, frame_set_id="s1~i0.5")[0] == "scoped-run"
    assert _select_priors_run(conn, frame_set_id="s1~i1") is None

    _insert_sfm_run(conn, root, root / "sfm" / "m2", config={"image_source": "projections"}, run_id="wide-run")
    conn.execute("UPDATE sfm_runs SET created_at = '2000-01-01T00:00:00' WHERE run_id = 'wide-run'")
    assert _select_priors_run(conn, frame_set_id="s1~i1")[0] == "wide-run"
    assert _select_priors_run(conn, frame_set_id=None)[0] == "wide-run"


def test_realityscan_priors_available_reflects_selection_logic(project):
    root, conn = project
    assert realityscan_priors_available(conn, frame_set_id="s1") is False
    _insert_sfm_run(conn, root, root / "sfm" / "m1", config={"image_source": "projections"}, run_id="proj-run")
    assert realityscan_priors_available(conn, frame_set_id="s1") is True


# -- legacy export-layout migration -------------------------------------


def test_migrate_legacy_export_layout_infers_colmap_from_sparse_dir(tmp_path):
    capture_dir = tmp_path / "cap"
    (capture_dir / "sparse").mkdir(parents=True)
    (capture_dir / "sparse" / "cameras.bin").write_bytes(b"x")
    (capture_dir / "images").mkdir()
    (capture_dir / "images" / "a.png").write_bytes(b"x")

    _migrate_legacy_export_layout(capture_dir)

    assert not (capture_dir / "sparse").exists()
    assert not (capture_dir / "images").exists()
    assert (capture_dir / "colmap" / "sparse" / "cameras.bin").exists()
    assert (capture_dir / "colmap" / "images" / "a.png").exists()


def test_migrate_legacy_export_layout_infers_postshot_when_no_sparse_dir(tmp_path):
    capture_dir = tmp_path / "cap"
    (capture_dir / "images").mkdir(parents=True)
    (capture_dir / "images" / "a.png").write_bytes(b"x")
    (capture_dir / "masks").mkdir()
    (capture_dir / "masks" / "a.png").write_bytes(b"x")

    _migrate_legacy_export_layout(capture_dir)

    assert not (capture_dir / "images").exists()
    assert not (capture_dir / "masks").exists()
    assert (capture_dir / "postshot" / "images" / "a.png").exists()
    assert (capture_dir / "postshot" / "masks" / "a.png").exists()


def test_migrate_legacy_export_layout_is_a_noop_with_no_legacy_folder(tmp_path):
    capture_dir = tmp_path / "cap"
    capture_dir.mkdir()

    _migrate_legacy_export_layout(capture_dir)  # must not raise

    assert list(capture_dir.iterdir()) == []


def test_migrate_legacy_export_layout_does_not_clobber_an_existing_target(tmp_path):
    """Conservative, matching the project's "never guess under ambiguity"
    precedent: if colmap/ already exists, leave the old flat folder
    alone rather than merge or overwrite into it."""
    capture_dir = tmp_path / "cap"
    (capture_dir / "sparse").mkdir(parents=True)
    (capture_dir / "sparse" / "cameras.bin").write_bytes(b"old")
    (capture_dir / "colmap").mkdir()
    (capture_dir / "colmap" / "marker.txt").write_bytes(b"already migrated")

    _migrate_legacy_export_layout(capture_dir)

    assert (capture_dir / "sparse" / "cameras.bin").exists()  # untouched
    assert (capture_dir / "colmap" / "marker.txt").exists()  # untouched


def test_migrate_legacy_export_layout_renames_old_all_sources_default(tmp_path):
    """The old all-sources default was literally named "postshot" at the
    exports/ root (every mode shared it, since per-source capture
    folders didn't exist before ADR 0026). The new all-sources default
    is "all" -- a legacy flat export bound for exports/all/<format>/ is
    actually sitting at the sibling exports/postshot/ instead."""
    exports_dir = tmp_path / "exports"
    legacy = exports_dir / "postshot"
    (legacy / "images").mkdir(parents=True)
    (legacy / "images" / "a.png").write_bytes(b"x")

    _migrate_legacy_export_layout(exports_dir / "all")

    assert not legacy.exists()  # old exports/postshot/ folder cleaned up entirely
    assert (exports_dir / "all" / "postshot" / "images" / "a.png").exists()


def test_export_for_postshot_migrates_legacy_layout_before_writing(project):
    """End-to-end: a real export function actually calls the migration
    helper on the parent of its own output_dir before writing, not just
    the helper in isolation."""
    root, conn = project
    model_dir, image_names = _build_real_model(root, num_frames=4)
    for name in image_names:
        _write_dummy_image(root / "projections" / name)
    _insert_sfm_run(conn, root, model_dir)

    capture_dir = root / "exports" / "cap"
    legacy_sparse = capture_dir / "sparse"
    legacy_sparse.mkdir(parents=True)
    (legacy_sparse / "old.bin").write_bytes(b"legacy")

    export_for_postshot(conn, root, capture_dir / "colmap")

    # The legacy flat folder is gone from its old location -- migrated into
    # capture_dir/colmap/, then immediately superseded there by the real
    # export's own _reset_managed_subdirs + fresh write (correct: migration
    # only promises "not left at the wrong path", not "old files preserved
    # forever" -- same "never leaves stale files" invariant every export
    # already guarantees).
    assert not legacy_sparse.exists()
    assert not (capture_dir / "sparse").exists()
    assert (capture_dir / "colmap" / "sparse" / "cameras.bin").exists()


def test_frames_and_masks_export_removes_stale_sparse_dir_from_a_prior_poses_export(project):
    """The real scenario this guards against: export poses+images+masks to
    a folder, then export images+masks-only to that *same* folder -- the
    old sparse/ (real poses) must not linger where Postshot, or a person,
    could mistake it for part of the second, pose-free export."""
    root, conn = project
    model_dir, image_names = _build_real_model(root, num_frames=4)
    for name in image_names:
        _write_dummy_image(root / "projections" / name)
    _insert_sfm_run(conn, root, model_dir)

    output_dir = root / "exports" / "shared"
    first = export_for_postshot(conn, root, output_dir)
    assert first.sparse_dir.exists()
    assert (output_dir / "sparse" / "cameras.bin").exists()

    _insert_view_with_image(conn, root, "frame-x", "front", with_mask=False)
    second = export_frames_and_masks_for_postshot(conn, root, output_dir)

    assert second.sparse_dir is None
    assert not (output_dir / "sparse").exists()  # not left over from the first export
    assert (output_dir / "images" / "frame-x__front.png").exists()


def test_export_for_postshot_does_not_leave_images_from_a_previous_larger_run(project):
    """A second export to the same folder with a smaller image set must
    not leave the first export's now-irrelevant extra images behind."""
    root, conn = project
    model_dir_1, names_1 = _build_real_model(root, num_frames=6)
    for name in names_1:
        _write_dummy_image(root / "projections" / name)
    output_dir = root / "exports" / "shrinking"
    _insert_sfm_run(conn, root, model_dir_1, run_id="sfm-big")
    export_for_postshot(conn, root, output_dir, run_id="sfm-big")
    assert len(list((output_dir / "images").iterdir())) == len(names_1)

    model_dir_2, names_2 = _build_real_model(root, num_frames=4)
    for name in names_2:
        _write_dummy_image(root / "projections" / name)
    _insert_sfm_run(conn, root, model_dir_2, run_id="sfm-small")
    export_for_postshot(conn, root, output_dir, run_id="sfm-small")

    assert len(names_2) < len(names_1)
    assert sorted(p.name for p in (output_dir / "images").iterdir()) == sorted(names_2)


def test_export_into_exports_root_leaves_the_projects_own_masks_alone(project):
    """Real bug: with exports/ itself chosen as output_dir, the legacy
    migration ran on its parent -- the project root -- took the project's
    own masks/ for a flat legacy export and moved it to <root>/postshot/,
    orphaning every mask row."""
    root, conn = project
    _insert_view_with_image(conn, root, "frame-1", "front", with_mask=True)
    keep_mask = root / "masks" / "keep" / "frame-1" / "front.png.png"
    assert keep_mask.exists()

    result = export_frames_and_masks_for_postshot(conn, root, root / "exports")

    assert keep_mask.exists()
    assert not (root / "postshot").exists()
    assert result.num_masks == 1
    assert (root / "exports" / "masks" / "frame-1__front.png").exists()


def test_migrate_legacy_export_layout_never_touches_a_project_root(project):
    root, _conn = project
    (root / "masks" / "keep" / "f.png").write_bytes(b"x")

    _migrate_legacy_export_layout(root)

    assert (root / "masks" / "keep" / "f.png").exists()
    assert not (root / "postshot").exists()


def test_export_refuses_a_project_root_as_output_dir(project):
    """_reset_managed_subdirs would otherwise delete the project's masks/."""
    root, conn = project
    _insert_view_with_image(conn, root, "frame-1", "front", with_mask=True)

    with pytest.raises(PostshotExportError, match="project folder"):
        export_frames_and_masks_for_postshot(conn, root, root)
    assert (root / "masks" / "keep" / "frame-1" / "front.png.png").exists()
