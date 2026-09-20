import json

import numpy as np
import pytest
from PIL import Image

from vine360.config import CaptureMode
from vine360.project import create_project, open_index_db
from vine360.projection.generate import (
    ProjectionGenerationError,
    generate_views_for_frame,
    generate_views_for_source,
)


def _insert_fake_source_and_frame(conn, project_root, frame_id="frame-1", source_id="source-1"):
    conn.execute(
        "INSERT INTO sources (source_id, path, checksum, media_type, projection, width, height, timestamps, capture_group) "
        "VALUES (?, '/x.mp4', 'deadbeef', 'video', 'equirectangular', 640, 320, '{}', NULL)",
        (source_id,),
    )
    equirect_path = project_root / "frames" / source_id / "frame_000001.png"
    equirect_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((320, 640, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    Image.fromarray(image, "RGB").save(equirect_path)
    conn.execute(
        "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum) "
        "VALUES (?, ?, 0.0, '{}', ?, 'cafef00d')",
        (frame_id, source_id, str(equirect_path.relative_to(project_root))),
    )
    conn.commit()
    return source_id, frame_id


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    yield root, conn
    conn.close()


def test_generate_views_for_frame_writes_files_and_rows(project):
    root, conn = project
    _source_id, frame_id = _insert_fake_source_and_frame(conn, root)

    views = generate_views_for_frame(conn, root, frame_id, face_size=64)

    assert {v.projection_id for v in views} == {"six-face"}
    assert {v.view_id for v in views} == {f"{frame_id}:{name}" for name in ("front", "right", "back", "left")}
    for view in views:
        assert (root / view.image_path).exists()

    rows = conn.execute("SELECT view_id, intrinsics, fixed_rotation FROM views WHERE frame_id = ?", (frame_id,)).fetchall()
    assert len(rows) == 4
    intrinsics = json.loads(rows[0][1])
    assert intrinsics["width"] == 64
    fixed_rotation = json.loads(rows[0][2])
    assert "matrix" in fixed_rotation


def test_generate_views_for_frame_with_explicit_face_names(project):
    root, conn = project
    _source_id, frame_id = _insert_fake_source_and_frame(conn, root)

    views = generate_views_for_frame(conn, root, frame_id, face_size=64, face_names=["front", "up"])

    assert {v.view_id for v in views} == {f"{frame_id}:front", f"{frame_id}:up"}


def test_generate_views_for_frame_unknown_frame_raises(project):
    root, conn = project
    with pytest.raises(ProjectionGenerationError):
        generate_views_for_frame(conn, root, "no-such-frame")


def test_regenerating_views_replaces_rather_than_collides(project):
    root, conn = project
    _source_id, frame_id = _insert_fake_source_and_frame(conn, root)

    generate_views_for_frame(conn, root, frame_id, face_size=64, include_polar_faces=True)
    first_count = conn.execute("SELECT COUNT(*) FROM views WHERE frame_id = ?", (frame_id,)).fetchone()[0]
    assert first_count == 6

    generate_views_for_frame(conn, root, frame_id, face_size=64, include_polar_faces=False)
    second_count = conn.execute("SELECT COUNT(*) FROM views WHERE frame_id = ?", (frame_id,)).fetchone()[0]
    assert second_count == 4  # polar faces from the first run are gone, not left stale


def test_regenerating_views_clears_dependent_masks(project):
    root, conn = project
    _source_id, frame_id = _insert_fake_source_and_frame(conn, root)
    views = generate_views_for_frame(conn, root, frame_id, face_size=64)
    conn.execute(
        "INSERT INTO masks (view_id, model, model_version, prompts, thresholds, morphology, keep_fraction, edited) "
        "VALUES (?, 'classical', '1', '[]', '{}', '{}', 0.9, 0)",
        (views[0].view_id,),
    )
    conn.commit()

    generate_views_for_frame(conn, root, frame_id, face_size=64)

    assert conn.execute("SELECT COUNT(*) FROM masks").fetchone()[0] == 0


def test_generate_views_for_source_processes_all_frames_and_reports_progress(project):
    root, conn = project
    source_id, frame_id_1 = _insert_fake_source_and_frame(conn, root, frame_id="frame-1", source_id="source-1")
    conn.execute(
        "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum) "
        "VALUES ('frame-2', ?, 1.0, '{}', ?, 'aa11bb22')",
        (source_id, f"frames/{source_id}/frame_000001.png"),
    )
    conn.commit()

    messages = []
    views = generate_views_for_source(conn, root, source_id, face_size=64, progress_callback=lambda m, c, t: messages.append((m, c, t)))

    assert len(views) == 8  # 2 frames x 4 faces
    assert messages[0] == ("Projecting frame 1/2…", 0, 2)
    assert messages[-1][1:] == (2, 2)


def test_generate_views_for_source_no_frames_raises(project):
    root, conn = project
    conn.execute(
        "INSERT INTO sources (source_id, path, checksum, media_type, projection, width, height, timestamps, capture_group) "
        "VALUES ('s1', '/x.mp4', 'deadbeef', 'video', 'equirectangular', 640, 320, '{}', NULL)"
    )
    conn.commit()
    with pytest.raises(ProjectionGenerationError):
        generate_views_for_source(conn, root, "s1")
