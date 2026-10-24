"""vine360.georef.observe: real detection on real view images stored in a
project, re-runs replacing rows, and the cascade-delete invariant
(re-projecting a frame drops its detections) -- docs/adr/0047."""

import json

import numpy as np
import pytest

pytest.importorskip("pupil_apriltags")

from PIL import Image

from test_georef_apriltag import _area_sampled, _tag_cells
from vine360.config import CaptureMode
from vine360.georef.observe import (
    delete_tag_observations,
    detect_tags_for_frame_sets,
    detection_status,
    tag_counts_by_frame_set,
)
from vine360.project import create_project, open_index_db
from vine360.projection.generate import clear_views_for_frame

FS = "src1~i1"


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "p"
    create_project(root, "p", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    conn.execute("INSERT INTO sources VALUES ('src1', '/x.mp4', 'x', 'video', 'equirectangular', 0, 0, '{}', NULL)")
    conn.execute("INSERT INTO frame_sets VALUES (?, 'src1', 'every 1s', '{}', 'now')", (FS,))
    intrinsics = json.dumps({"width": 700, "height": 600, "fx": 350.0, "fy": 350.0, "cx": 350.0, "cy": 300.0})
    for k, tags in enumerate([[20], [20, 22], []]):
        frame_id = f"{FS}:{k:06d}"
        conn.execute(
            "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
            "VALUES (?, 'src1', ?, '{}', 'f.png', 'x', ?)",
            (frame_id, float(k), FS),
        )
        image = np.full((600, 700), 255, np.uint8)
        for n, tag_id in enumerate(tags):
            image = np.minimum(image, _area_sampled(_tag_cells(tag_id), 40.0 + 330 * n, 150.0, 28.0))
        path = root / "projections" / frame_id / "front.png"
        path.parent.mkdir(parents=True)
        Image.fromarray(image).convert("RGB").save(path)
        conn.execute(
            "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, fixed_rotation, image_path) "
            "VALUES (?, ?, 'front', 700, 600, ?, ?, ?)",
            (f"{frame_id}/front", frame_id, intrinsics, json.dumps({"matrix": np.eye(3).tolist()}), f"projections/{frame_id}/front.png"),
        )
    conn.commit()
    yield root, conn
    conn.close()


def test_detects_counts_and_replaces(project):
    root, conn = project
    progress = []
    summary = detect_tags_for_frame_sets(conn, root, [FS], progress_callback=lambda *a: progress.append(a))
    assert summary.views == 3 and summary.views_with_tags == 2 and summary.detections == 3
    assert summary.views_per_tag == {20: 2, 22: 1}
    assert progress[-1][0].startswith("Tag detection done") and progress[1][1:] == (1, 3)
    assert detection_status(conn, FS) == (3, 3)
    assert tag_counts_by_frame_set(conn)[FS] == {"views_detected": 3, "detections": 3, "tags": [20, 22]}
    row = conn.execute("SELECT detector, center FROM tag_observations WHERE tag_id = 22").fetchone()
    assert row[0] == "pupil-apriltags" and len(json.loads(row[1])) == 2

    detect_tags_for_frame_sets(conn, root, [FS])  # a re-run replaces, never duplicates
    assert conn.execute("SELECT COUNT(*) FROM tag_observations").fetchone()[0] == 3


def test_reprojecting_a_frame_drops_its_detections(project):
    root, conn = project
    detect_tags_for_frame_sets(conn, root, [FS])
    clear_views_for_frame(conn, root, f"{FS}:000001")
    assert conn.execute("SELECT COUNT(*) FROM tag_observations").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM tag_detection_runs").fetchone()[0] == 2
    assert delete_tag_observations(conn, FS) == 1
    assert detection_status(conn, FS) == (0, 2)
