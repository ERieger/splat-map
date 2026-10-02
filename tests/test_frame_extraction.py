import threading
import time
from pathlib import Path

import pytest

from vine360.ingest.frames import (
    _poll_output_frame_count,
    build_frame_extraction_command,
    build_thumbnail_command,
    resolve_interval_seconds,
    resolve_time_range,
)


def test_build_frame_extraction_command():
    cmd = build_frame_extraction_command(Path("/media/clip.mp4"), Path("/proj/frames/s1/frame_%06d.png"), 2.0)
    assert cmd == [
        "ffmpeg",
        "-y",
        "-i",
        "/media/clip.mp4",
        "-vf",
        "fps=1/2.0",
        "-vsync",
        "0",
        "/proj/frames/s1/frame_%06d.png",
    ]


def test_build_thumbnail_command():
    cmd = build_thumbnail_command(Path("/proj/frame_000001.png"), Path("/proj/thumbs/frame_000001.jpg"))
    assert cmd == [
        "ffmpeg",
        "-y",
        "-i",
        "/proj/frame_000001.png",
        "-vf",
        "scale=320:-1",
        "/proj/thumbs/frame_000001.jpg",
    ]


def test_resolve_interval_seconds_explicit_interval():
    interval, settings = resolve_interval_seconds(30.0, interval_seconds=1.5, target_count=None)
    assert interval == 1.5
    assert settings.mode == "interval"
    assert settings.requested_count is None


def test_resolve_interval_seconds_from_target_count():
    interval, settings = resolve_interval_seconds(30.0, interval_seconds=None, target_count=10)
    assert interval == 3.0
    assert settings.mode == "count"
    assert settings.requested_count == 10


def test_resolve_interval_seconds_rejects_both():
    with pytest.raises(ValueError):
        resolve_interval_seconds(30.0, interval_seconds=1.0, target_count=10)


def test_resolve_interval_seconds_rejects_neither():
    with pytest.raises(ValueError):
        resolve_interval_seconds(30.0, interval_seconds=None, target_count=None)


def test_resolve_interval_seconds_rejects_zero_duration_with_count():
    with pytest.raises(ValueError):
        resolve_interval_seconds(0.0, interval_seconds=None, target_count=10)


def test_build_frame_extraction_command_with_range():
    cmd = build_frame_extraction_command(
        Path("/media/clip.mp4"),
        Path("/proj/frames/s1/frame_%06d.png"),
        2.0,
        start_time=1.5,
        duration_limit=4.0,
    )
    assert cmd == [
        "ffmpeg",
        "-y",
        "-ss",
        "1.5",
        "-i",
        "/media/clip.mp4",
        "-t",
        "4.0",
        "-vf",
        "fps=1/2.0",
        "-vsync",
        "0",
        "/proj/frames/s1/frame_%06d.png",
    ]


def test_build_frame_extraction_command_zero_start_matches_no_range():
    with_zero_start = build_frame_extraction_command(
        Path("/media/clip.mp4"), Path("/proj/frames/s1/frame_%06d.png"), 2.0, start_time=0.0, duration_limit=None
    )
    without_range = build_frame_extraction_command(
        Path("/media/clip.mp4"), Path("/proj/frames/s1/frame_%06d.png"), 2.0
    )
    assert with_zero_start == without_range


def test_resolve_time_range_explicit_range():
    assert resolve_time_range(10.0, start_time=2.0, end_time=6.0) == (2.0, 4.0)


def test_resolve_time_range_only_start():
    assert resolve_time_range(10.0, start_time=3.0, end_time=None) == (3.0, None)


def test_resolve_time_range_only_end():
    assert resolve_time_range(10.0, start_time=None, end_time=6.0) == (0.0, 6.0)


def test_resolve_time_range_no_range():
    assert resolve_time_range(10.0, start_time=None, end_time=None) == (0.0, None)


def test_resolve_time_range_clamps_end_beyond_duration():
    assert resolve_time_range(10.0, start_time=1.0, end_time=100.0) == (1.0, 9.0)


def test_resolve_time_range_rejects_negative_start():
    with pytest.raises(ValueError):
        resolve_time_range(10.0, start_time=-1.0, end_time=None)


def test_resolve_time_range_rejects_start_at_or_past_duration():
    with pytest.raises(ValueError):
        resolve_time_range(10.0, start_time=10.0, end_time=None)
    with pytest.raises(ValueError):
        resolve_time_range(10.0, start_time=11.0, end_time=None)


def test_resolve_time_range_rejects_end_before_or_equal_start():
    with pytest.raises(ValueError):
        resolve_time_range(10.0, start_time=5.0, end_time=5.0)
    with pytest.raises(ValueError):
        resolve_time_range(10.0, start_time=5.0, end_time=4.0)


def test_poll_output_frame_count_reports_files_as_they_appear(tmp_path):
    """Real regression: the raw ffmpeg extraction call previously reported
    zero progress at all -- current/total always None -- for however long
    it took, indistinguishable from a hang for a large real source. This
    poller runs concurrently with that blocking call and counts files
    ffmpeg has already written, the same way the thumbnail loop already
    reported real per-frame progress."""
    events: list[tuple[int, int]] = []
    stop_event = threading.Event()
    poller = threading.Thread(
        target=_poll_output_frame_count,
        args=(tmp_path, 2, lambda m, c, t: events.append((c, t)), stop_event, 0.01),
    )
    poller.start()
    try:
        time.sleep(0.05)
        assert events, "should have polled at least once before any files existed"
        assert events[-1] == (0, 2)

        (tmp_path / "frame_000000.png").touch()
        time.sleep(0.05)
        assert events[-1] == (1, 2)

        (tmp_path / "frame_000001.png").touch()
        time.sleep(0.05)
        assert events[-1] == (2, 2)
    finally:
        stop_event.set()
        poller.join(timeout=1.0)
    assert not poller.is_alive()


def test_frame_set_tag_and_label():
    from vine360.ingest.frames import frame_set_label, frame_set_tag

    plain = {"mode": "interval", "interval_seconds": 0.5}
    assert frame_set_tag(plain) == "i0.5"
    assert frame_set_label(plain) == "every 0.5s"
    ranged = {"mode": "interval", "interval_seconds": 1.0, "start_time_seconds": 10.0, "end_time_seconds": 60.0}
    assert frame_set_tag(ranged) == "i1_r10-60"
    assert frame_set_label(ranged) == "every 1s, 10.0s–60.0s"
    open_ended = {"mode": "interval", "interval_seconds": 2.0, "start_time_seconds": 5.0, "end_time_seconds": None}
    assert frame_set_tag(open_ended) == "i2_r5-end"
    counted = {"mode": "count", "interval_seconds": 0.37, "requested_count": 200}
    assert frame_set_tag(counted) == "n200"
    assert frame_set_label(counted) == "200 frames total"


def test_frame_set_id_for_is_deterministic_and_reuses_a_matching_set(tmp_path):
    from vine360.config import CaptureMode
    from vine360.ingest.frames import frame_set_id_for
    from vine360.project import create_project, open_index_db

    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    try:
        assert frame_set_id_for(conn, "s1", interval_seconds=0.5) == "s1~i0.5"
        # start_time 0 is the same config as no start time
        assert frame_set_id_for(conn, "s1", interval_seconds=0.5, start_time=0.0) == "s1~i0.5"
        assert frame_set_id_for(conn, "s1", interval_seconds=0.5, end_time=30.0) == "s1~i0.5_r0-30"

        # A legacy frame set (id == source_id, docs/adr/0034) with the same
        # config is reused rather than duplicated.
        conn.execute(
            "INSERT INTO sources (source_id, path, checksum, media_type, projection, timestamps) "
            "VALUES ('s1', '/a.mp4', 'x', 'video', 'equirectangular', '{}')"
        )
        conn.execute(
            "INSERT INTO frame_sets (frame_set_id, source_id, label, extraction_settings, created_at) "
            "VALUES ('s1', 's1', 'every 1s', ?, '2026-01-01')",
            ('{"mode": "interval", "interval_seconds": 1.0, "start_time_seconds": null, "end_time_seconds": null}',),
        )
        assert frame_set_id_for(conn, "s1", interval_seconds=1.0) == "s1"
        assert frame_set_id_for(conn, "s1", interval_seconds=0.5) == "s1~i0.5"
    finally:
        conn.close()
