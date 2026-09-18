from pathlib import Path

import pytest

from vine360.ingest.frames import (
    build_frame_extraction_command,
    build_thumbnail_command,
    resolve_interval_seconds,
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
