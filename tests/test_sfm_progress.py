"""Live progress from real SfM runs (both engines) on the synthetic 360
sequence: every step reports counted progress, and the counts are right.
Nothing mocked -- the SphereSfM numbers come from parsing the real
binary's own log lines, the pycolmap ones from batched extraction and the
mapper's registration callbacks."""

import pytest

pytest.importorskip("pycolmap")

from conftest import requires_spheresfm
from synthetic_360 import add_synthetic_frame_set
from vine360.config import CaptureMode
from vine360.project import create_project, open_index_db
from vine360.sfm.colmap_adapter import SfmConfig
from vine360.sfm.project_run import ENGINE_SPHERESFM, run_sfm_for_project
from vine360.sfm.spheresfm_adapter import ProgressParser


@pytest.fixture
def frame_set(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Progress", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    frame_set_id = add_synthetic_frame_set(conn, root, n=6, width=1024)
    yield root, conn, frame_set_id
    conn.close()


def _counted(events, needle):
    return [(c, t) for m, c, t in events if needle in m and c is not None]


def test_pycolmap_equirect_run_reports_each_step(frame_set):
    root, conn, frame_set_id = frame_set
    events = []
    diagnostics, _ = run_sfm_for_project(
        conn,
        root,
        config=SfmConfig(camera_model="EQUIRECTANGULAR"),
        image_source="frames",
        frame_set_id=frame_set_id,
        progress_callback=lambda m, c, t: events.append((m, c, t)),
    )
    features = _counted(events, "step 1/3 (features")
    assert features[0] == (0, 6) and features[-1] == (6, 6)
    assert any("step 2/3 (matching" in m for m, _, _ in events)
    mapping = _counted(events, "step 3/3 (mapping)")
    # Every registration is reported, so the count ends where the model
    # does (a tiny scene can leave one frame unregistered -- see
    # test_frame_poses.MIN_REGISTERED).
    registered = diagnostics.registered_images
    assert registered >= 5
    assert mapping[-1] == (registered, 6)
    assert [c for c, _ in mapping] == sorted(c for c, _ in mapping)
    assert events[-1][0].startswith(f"COLMAP done: {registered}/6 images registered")


@requires_spheresfm
def test_spheresfm_run_reports_each_step_and_keeps_its_logs(frame_set):
    root, conn, frame_set_id = frame_set
    events = []
    diagnostics, _ = run_sfm_for_project(
        conn,
        root,
        image_source="frames",
        frame_set_id=frame_set_id,
        engine=ENGINE_SPHERESFM,
        progress_callback=lambda m, c, t: events.append((m, c, t)),
    )
    assert _counted(events, "step 1/4")[-1] == (6, 6)
    assert _counted(events, "step 2/4 (matching")[-1] == (6, 6)
    registered = diagnostics.registered_images
    assert registered >= 5
    assert _counted(events, "frames registered")[-1] == (registered, 6)
    assert events[-1][0].startswith(f"SphereSfM done: {registered}/6 frames registered")

    run_id = conn.execute("SELECT run_id FROM sfm_runs").fetchone()[0]
    run_dir = root / "sfm" / "sparse" / run_id
    for log in ("1_feature_extractor.log", "2_sequential_matcher.log", "3_mapper.log"):
        assert (run_dir / log).stat().st_size > 0


def test_progress_parser_reads_colmap_38_log_lines():
    events = []
    parser = ProgressParser("X: ", 10, lambda m, c, t: events.append((m, c, t)))
    for line in [
        "Processed file [3/10]",
        "Matching image [4/10] in 0.1s",
        "==============================================================================",
        "Finding good initial image pair",
        "Initializing with image pair #4 and #5",
        "Registering image #6 (3)",
        "Global bundle adjustment",
        "unrelated line",
    ]:
        parser.feed(line)
    assert events[0] == ("X: frame 3/10", 3, 10)
    assert events[1] == ("X: frame 4/10", 4, 10)
    assert [c for _, c, _ in events[2:]] == [0, 2, 3, 3]
    assert events[-1][0] == "X: 3/10 frames registered (global bundle adjustment)"
