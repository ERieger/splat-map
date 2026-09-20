"""End-to-end ingest test against real ffmpeg/ffprobe.

Skipped when ffmpeg/ffprobe are not installed (this sandbox has neither --
see docs/status.md). Uses a synthetically generated clip rather than a real
Insta360 X6 export; open question in the handover doc, section 12, asks
whether a small redistributable reference dataset can be committed instead.
"""

import subprocess

from conftest import requires_ffmpeg

from vine360.config import CaptureMode, MediaType
from vine360.ingest.frames import extract_frames
from vine360.ingest.manifest import build_manifest
from vine360.ingest.sources import add_source
from vine360.project import create_project, open_index_db
from vine360.runners.local import LocalRunner


def _make_synthetic_clip(path, duration=4, width=640, height=320):
    # width:height = 2:1 to exercise the equirectangular-detection path,
    # mirroring an Insta360 X6 equirectangular export's aspect ratio.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size={width}x{height}:rate=10",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@requires_ffmpeg
def test_ingest_and_extract_frames_end_to_end(tmp_path):
    clip_path = tmp_path / "source_clip.mp4"
    _make_synthetic_clip(clip_path)

    project_root = tmp_path / "project"
    create_project(project_root, "Integration Test Vineyard", CaptureMode.THREE_SIXTY)
    conn = open_index_db(project_root)
    runner = LocalRunner()

    source = add_source(
        conn,
        clip_path,
        runner,
        media_type_override=MediaType.VIDEO,
        added_at="2026-01-01T00:00:00+00:00",
    )
    assert source.timestamps["duration_seconds"] > 0

    frames = extract_frames(conn, project_root, source.source_id, runner, interval_seconds=1.0)
    assert len(frames) >= 3
    for i, frame in enumerate(frames):
        assert (project_root / frame.path).exists()
        assert frame.source_time == i * 1.0

    manifest = build_manifest(conn)
    assert manifest["source_frame_map"][source.source_id] == [f.frame_id for f in frames]
    conn.close()


@requires_ffmpeg
def test_re_extracting_frames_with_a_different_interval_does_not_collide(tmp_path):
    """Regression test: extracting frames twice for the same source (e.g.
    the user tries one interval, doesn't like it, picks another) used to
    fail with a sqlite UNIQUE constraint error on frame_id, and could leave
    stale files behind from the first, larger extraction."""
    clip_path = tmp_path / "source_clip.mp4"
    _make_synthetic_clip(clip_path, duration=4)

    project_root = tmp_path / "project"
    create_project(project_root, "Integration Test Vineyard", CaptureMode.THREE_SIXTY)
    conn = open_index_db(project_root)
    runner = LocalRunner()

    source = add_source(
        conn, clip_path, runner, media_type_override=MediaType.VIDEO, added_at="2026-01-01T00:00:00+00:00"
    )

    first = extract_frames(conn, project_root, source.source_id, runner, interval_seconds=0.5)
    second = extract_frames(conn, project_root, source.source_id, runner, interval_seconds=2.0)

    assert len(first) > len(second)  # smaller interval -> more frames
    frames_dir = project_root / "frames" / source.source_id
    remaining_frame_files = sorted(frames_dir.glob("frame_*.png"))
    assert len(remaining_frame_files) == len(second), "stale files from the first extraction should be gone"

    rows = conn.execute("SELECT frame_id FROM frames WHERE source_id = ?", (source.source_id,)).fetchall()
    assert len(rows) == len(second)
    conn.close()


@requires_ffmpeg
def test_extract_frames_reports_progress(tmp_path):
    clip_path = tmp_path / "source_clip.mp4"
    _make_synthetic_clip(clip_path, duration=3)

    project_root = tmp_path / "project"
    create_project(project_root, "Integration Test Vineyard", CaptureMode.THREE_SIXTY)
    conn = open_index_db(project_root)
    runner = LocalRunner()
    source = add_source(
        conn, clip_path, runner, media_type_override=MediaType.VIDEO, added_at="2026-01-01T00:00:00+00:00"
    )

    events = []
    frames = extract_frames(
        conn,
        project_root,
        source.source_id,
        runner,
        interval_seconds=1.0,
        progress_callback=lambda m, c, t: events.append((m, c, t)),
    )
    messages = [m for m, _, _ in events]
    assert any("Extracting raw frames" in m for m in messages)
    assert any("Generating thumbnails" in m for m in messages)
    assert any(f"({len(frames)}/{len(frames)})" in m for m in messages), "should report reaching the last frame"
    thumbnail_events = [(c, t) for m, c, t in events if "Generating thumbnails" in m]
    assert thumbnail_events[-1] == (len(frames), len(frames))
    conn.close()
