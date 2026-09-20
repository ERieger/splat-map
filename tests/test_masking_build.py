import json

import numpy as np
import pytest
from PIL import Image

from vine360.config import CaptureMode
from vine360.masking.build import MaskBuildError, build_mask_for_view, build_masks_for_source
from vine360.project import create_project, open_index_db


def _insert_fake_view(conn, project_root, view_id="frame-1:front", frame_id="frame-1"):
    image_path = project_root / "projections" / frame_id / "front.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[0:40, :] = [135, 206, 235]  # sky-blue top strip
    image[40:100, :] = [34, 90, 34]  # green ground
    Image.fromarray(image, "RGB").save(image_path)

    conn.execute(
        "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum) "
        "VALUES (?, 's1', 0.0, '{}', 'x', 'y') ",
        (frame_id,),
    )
    conn.execute(
        "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, fixed_rotation, image_path) "
        "VALUES (?, ?, 'six-face', 100, 100, '{}', '{}', ?)",
        (view_id, frame_id, str(image_path.relative_to(project_root))),
    )
    conn.commit()
    return view_id


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    yield root, conn
    conn.close()


def test_build_mask_for_view_classical_writes_files_and_row(project):
    root, conn = project
    view_id = _insert_fake_view(conn, root)

    mask = build_mask_for_view(conn, root, view_id)

    assert mask.model == "classical-sky"
    assert 0.0 < mask.keep_fraction < 1.0
    keep_path = root / "masks" / "keep" / "frame-1" / "front.png.png"
    assert keep_path.exists()
    assert (root / "masks" / "classes" / view_id / "exclude.png").exists()

    row = conn.execute("SELECT model, keep_fraction FROM masks WHERE view_id = ?", (view_id,)).fetchone()
    assert row[0] == "classical-sky"
    assert row[1] == pytest.approx(mask.keep_fraction)


def test_build_mask_for_view_unknown_view_raises(project):
    root, conn = project
    with pytest.raises(MaskBuildError):
        build_mask_for_view(conn, root, "no-such-view")


def test_keep_mask_excludes_sky_region(project):
    root, conn = project
    view_id = _insert_fake_view(conn, root)
    build_mask_for_view(conn, root, view_id)

    keep = np.asarray(Image.open(root / "masks" / "keep" / "frame-1" / "front.png.png"))
    assert keep[10, 50] == 0  # sky region excluded
    assert keep[90, 50] == 255  # ground region kept


def test_build_masks_for_source_processes_all_views_and_reports_progress(project):
    root, conn = project
    _insert_fake_view(conn, root, view_id="frame-1:front", frame_id="frame-1")
    # a second view sharing the same source via the frames.source_id join
    image_path = root / "projections" / "frame-1" / "right.png"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), "RGB").save(image_path)
    conn.execute(
        "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, fixed_rotation, image_path) "
        "VALUES ('frame-1:right', 'frame-1', 'six-face', 100, 100, '{}', '{}', ?)",
        (str(image_path.relative_to(root)),),
    )
    conn.commit()

    messages = []
    masks = build_masks_for_source(conn, root, "s1", progress_callback=lambda m, c, t: messages.append((m, c, t)))

    assert len(masks) == 2
    assert messages[0] == ("Masking view 1/2…", 0, 2)
    assert messages[-1][1:] == (2, 2)


def test_build_masks_for_source_no_views_raises(project):
    root, conn = project
    with pytest.raises(MaskBuildError):
        build_masks_for_source(conn, root, "no-such-source")


@pytest.mark.network
def test_build_mask_for_view_with_real_sam3_sky(project):
    """Real end-to-end: real SAM 3 inference, through the full persistence
    path (files on disk + a real masks row), not just the adapter in
    isolation (see tests/test_masking_sam3_adapter.py)."""
    root, conn = project
    view_id = _insert_fake_view(conn, root)

    mask = build_mask_for_view(conn, root, view_id, use_sam3_sky=True)

    assert mask.model == "sam3"
    keep = np.asarray(Image.open(root / "masks" / "keep" / "frame-1" / "front.png.png"))
    assert keep[10, 50] == 0, "sky region should be excluded by real SAM3 inference"
