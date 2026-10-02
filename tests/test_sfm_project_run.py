import json
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image, ImageDraw

pytest.importorskip("pycolmap")

from vine360.config import CaptureMode
from vine360.project import create_project, open_index_db
from vine360.projection.generate import generate_views_for_frame
from vine360.sfm.colmap_adapter import SfmDiagnostics
from vine360.sfm.project_run import SfmRegistrationError, run_sfm_for_project


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    yield root, conn
    conn.close()


def _insert_fake_source_and_textured_frame(conn, project_root, frame_id="frame-1", source_id="s1"):
    conn.execute(
        "INSERT INTO sources (source_id, path, checksum, media_type, projection, width, height, timestamps, capture_group) "
        "VALUES (?, '/x.mp4', 'deadbeef', 'video', 'equirectangular', 1024, 512, '{}', NULL)",
        (source_id,),
    )
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 60, size=(512, 1024, 3), dtype=np.uint8)
    img = Image.fromarray(noise, "RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(200):
        x, y = rng.integers(0, 1024), rng.integers(0, 512)
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=tuple(int(c) for c in rng.integers(80, 255, size=3)))
    equirect_path = project_root / "frames" / source_id / "frame_000001.png"
    equirect_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(equirect_path)
    conn.execute(
        "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
        "VALUES (?, ?, 0.0, '{}', ?, 'cafef00d', ?)",
        (frame_id, source_id, str(equirect_path.relative_to(project_root)), source_id),
    )
    conn.commit()


def test_run_sfm_for_project_no_views_raises(project):
    root, conn = project
    with pytest.raises(SfmRegistrationError):
        run_sfm_for_project(conn, root)


def test_run_sfm_for_project_zero_parallax_real_views_raises_registration_error(project):
    """Real pipeline (no mocking): views generated from a single frame's
    projection share one optical center and cannot be 3D-reconstructed --
    correct COLMAP behavior, not a bug (see docs/adr/0007)."""
    root, conn = project
    _insert_fake_source_and_textured_frame(conn, root)
    generate_views_for_frame(conn, root, "frame-1", face_size=256, fov_degrees=100.0)

    with pytest.raises(SfmRegistrationError):
        run_sfm_for_project(conn, root)


def test_run_sfm_for_project_success_records_sfm_run_row(project):
    """Mocks only the heavy COLMAP call (colmap_adapter.run_sfm, already
    tested for real in tests/test_sfm_integration.py) to exercise this
    module's own job: the database/file wiring around it."""
    root, conn = project
    _insert_fake_source_and_textured_frame(conn, root)
    generate_views_for_frame(conn, root, "frame-1", face_size=64)

    fake_diagnostics = SfmDiagnostics(
        total_images=4,
        registered_images=4,
        registered_ratio=1.0,
        num_connected_models=1,
        num_points3d=150,
        mean_reprojection_error=0.4,
        mean_track_length=3.0,
        mean_observations_per_reg_image=20.0,
    )
    fake_model_dir = root / "sfm" / "sparse" / "0"
    fake_model_dir.mkdir(parents=True)
    messages = []
    with patch(
        "vine360.sfm.project_run.run_sfm", return_value=(object(), fake_diagnostics, fake_model_dir)
    ):
        diagnostics, warnings = run_sfm_for_project(
            conn, root, progress_callback=lambda m, c, t: messages.append(m)
        )

    assert diagnostics is fake_diagnostics
    assert warnings == []
    assert any(m.startswith("COLMAP done: 4/4 images registered") for m in messages)

    row = conn.execute("SELECT run_id, model_stats, selected_model FROM sfm_runs").fetchone()
    assert row is not None
    stats = json.loads(row[1])
    assert stats["num_points3d"] == 150
    assert row[2] == "sfm/sparse/0"  # the actual model directory, not the run_id


def test_run_sfm_for_project_uses_mask_dir_only_when_masks_exist(project):
    root, conn = project
    _insert_fake_source_and_textured_frame(conn, root)
    generate_views_for_frame(conn, root, "frame-1", face_size=64)

    fake_model_dir = root / "sfm" / "sparse" / "0"
    fake_model_dir.mkdir(parents=True)
    fake_diagnostics = SfmDiagnostics(4, 4, 1.0, 1, 1, 0.1, 1.0, 1.0)
    with patch(
        "vine360.sfm.project_run.run_sfm", return_value=(object(), fake_diagnostics, fake_model_dir)
    ) as mock_run:
        run_sfm_for_project(conn, root)
    _, kwargs = mock_run.call_args
    assert kwargs["mask_dir"] is None  # no masks/ files were ever written in this test


def test_run_sfm_for_project_uses_a_distinct_sparse_dir_per_run(project):
    """Real regression: sparse_dir used to be the same project/sfm/sparse/
    for every run, so a later run's pycolmap.incremental_mapping (which
    numbers its own candidate reconstructions starting from 0 every time)
    could silently overwrite an earlier run's model files on disk, even
    though the sfm_runs row was kept as history (see docs/adr/0019).
    Confirmed for real on a project with 4 historical runs where only the
    most recent one's files still existed."""
    root, conn = project
    _insert_fake_source_and_textured_frame(conn, root)
    generate_views_for_frame(conn, root, "frame-1", face_size=64)

    fake_diagnostics = SfmDiagnostics(4, 4, 1.0, 1, 1, 0.1, 1.0, 1.0)
    sparse_dirs_used = []

    def fake_run_sfm(image_dir, database_path, sparse_dir, *, mask_dir=None, config=None, image_names=None, progress_callback=None):
        sparse_dirs_used.append(sparse_dir)
        model_dir = sparse_dir / "0"
        model_dir.mkdir(parents=True)
        return object(), fake_diagnostics, model_dir

    with patch("vine360.sfm.project_run.run_sfm", side_effect=fake_run_sfm):
        run_sfm_for_project(conn, root)
        run_sfm_for_project(conn, root)

    assert len(sparse_dirs_used) == 2
    assert sparse_dirs_used[0] != sparse_dirs_used[1]

    selected_models = [row[0] for row in conn.execute("SELECT selected_model FROM sfm_runs").fetchall()]
    assert len(set(selected_models)) == 2  # each run's recorded model path is distinct


def test_run_sfm_for_project_frames_requires_frame_set_id(project):
    root, conn = project
    with pytest.raises(ValueError):
        run_sfm_for_project(conn, root, image_source="frames")


def test_run_sfm_for_project_unknown_image_source_raises(project):
    root, conn = project
    with pytest.raises(ValueError):
        run_sfm_for_project(conn, root, image_source="bogus")


def test_run_sfm_for_project_frames_missing_raises_registration_error(project):
    root, conn = project
    with pytest.raises(SfmRegistrationError):
        run_sfm_for_project(conn, root, image_source="frames", frame_set_id="no-such-frame-set")


def test_run_sfm_for_project_frames_real_equirectangular_camera_model(project):
    """Real pipeline (no mocking) against raw equirectangular frames using
    COLMAP's native EQUIRECTANGULAR camera model -- confirmed for real via
    pycolmap.synthesize_dataset that this camera model registers correctly
    given real parallax (see docs/adr/0017). This single-frame source has
    none (same limitation as the "projections" path -- ADR 0007), so it
    correctly raises rather than fabricating a reconstruction; the point
    of this test is that extraction/matching runs cleanly against a real
    equirectangular image with this camera model, not that it registers."""
    root, conn = project
    _insert_fake_source_and_textured_frame(conn, root, frame_id="frame-1", source_id="s1")

    from vine360.sfm.colmap_adapter import SfmConfig

    with pytest.raises(SfmRegistrationError):
        run_sfm_for_project(
            conn, root, config=SfmConfig(camera_model="EQUIRECTANGULAR"), image_source="frames", frame_set_id="s1"
        )

    row = conn.execute("SELECT COUNT(*) FROM sfm_runs").fetchone()
    assert row[0] == 0  # no run recorded on failure


def _fake_run_sfm_recording(calls):
    fake_diagnostics = SfmDiagnostics(4, 4, 1.0, 1, 1, 0.1, 1.0, 1.0)

    def fake_run_sfm(image_dir, database_path, sparse_dir, *, mask_dir=None, config=None, image_names=None, progress_callback=None):
        calls.append({"image_dir": image_dir, "database_path": database_path, "sparse_dir": sparse_dir,
                      "image_names": image_names})
        model_dir = sparse_dir / "0"
        model_dir.mkdir(parents=True)
        return object(), fake_diagnostics, model_dir

    return fake_run_sfm


def test_run_sfm_for_project_scoped_to_one_frame_set_passes_only_its_views(project):
    """docs/adr/0034: with two frame sets in the project, a projections run
    scoped to one must hand COLMAP only that frame set's views (by name),
    record the scope, and keep its COLMAP database inside its own run
    directory rather than the shared sfm/database.db."""
    root, conn = project
    _insert_fake_source_and_textured_frame(conn, root, frame_id="s1:000000", source_id="s1")
    _insert_fake_source_and_textured_frame(conn, root, frame_id="s2:000000", source_id="s2")
    generate_views_for_frame(conn, root, "s1:000000", face_size=32)
    generate_views_for_frame(conn, root, "s2:000000", face_size=32)

    calls = []
    with patch("vine360.sfm.project_run.run_sfm", side_effect=_fake_run_sfm_recording(calls)):
        run_sfm_for_project(conn, root, frame_set_id="s2")

    (call,) = calls
    assert call["image_names"] and all(name.startswith("s2:000000/") for name in call["image_names"])
    assert call["database_path"] == call["sparse_dir"] / "database.db"
    config = json.loads(conn.execute("SELECT config FROM sfm_runs").fetchone()[0])
    assert config["frame_set_id"] == "s2"

    calls.clear()
    with patch("vine360.sfm.project_run.run_sfm", side_effect=_fake_run_sfm_recording(calls)):
        run_sfm_for_project(conn, root)  # project-wide: both frame sets
    assert {name.split("/")[0] for name in calls[0]["image_names"]} == {"s1:000000", "s2:000000"}


def test_run_sfm_for_project_frames_engine_lists_frames_not_thumbnails(project):
    root, conn = project
    _insert_fake_source_and_textured_frame(conn, root, frame_id="s1:000000", source_id="s1")
    thumbs = root / "frames" / "s1" / "thumbs"
    thumbs.mkdir()
    (thumbs / "frame_000001.jpg").write_bytes(b"not a frame")

    calls = []
    with patch("vine360.sfm.project_run.run_sfm", side_effect=_fake_run_sfm_recording(calls)):
        run_sfm_for_project(conn, root, image_source="frames", frame_set_id="s1")
    assert calls[0]["image_names"] == ["frame_000001.png"]


def test_extract_and_match_image_names_restricts_what_colmap_reads(tmp_path):
    """Real pycolmap (no mocking): extract_features' image_names really does
    limit extraction to the named subset -- what frame-set-scoped SfM runs
    rely on."""
    import pycolmap

    from vine360.sfm.colmap_adapter import extract_and_match

    rng = np.random.default_rng(3)
    image_dir = tmp_path / "images"
    (image_dir / "a").mkdir(parents=True)
    (image_dir / "b").mkdir(parents=True)
    for sub in ("a", "b"):
        for i in range(2):
            img = Image.fromarray(rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8), "RGB")
            img.save(image_dir / sub / f"{i}.png")

    database_path = tmp_path / "database.db"
    extract_and_match(image_dir, database_path, image_names=["b/0.png", "b/1.png"])

    db = pycolmap.Database.open(database_path)
    try:
        names = sorted(image.name for image in db.read_all_images())
    finally:
        db.close()
    assert names == ["b/0.png", "b/1.png"]
