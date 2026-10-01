"""Deterministic frame extraction (handover doc, section 4, step 2 "Extract").

Command construction (`build_frame_extraction_command`,
`build_thumbnail_command`) is separated from execution so the exact ffmpeg
invocation is unit-testable without ffmpeg installed; `extract_frames` is
the integration path that actually runs it through a Runner.

Near-duplicate frame filtering (called for in the handover) is not
implemented yet -- see docs/status.md.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

from vine360.config import ExtractionSettings
from vine360.ingest.sources import get_source, sha256_file
from vine360.models import Frame
from vine360.runners.base import Runner

THUMBNAIL_WIDTH = 320


class FrameExtractionError(Exception):
    pass


def build_frame_extraction_command(
    source: Path,
    output_pattern: Path,
    interval_seconds: float,
    *,
    start_time: float | None = None,
    duration_limit: float | None = None,
) -> list[str]:
    command = ["ffmpeg", "-y"]
    if start_time:
        command += ["-ss", str(start_time)]
    command += ["-i", str(source)]
    if duration_limit is not None:
        command += ["-t", str(duration_limit)]
    command += ["-vf", f"fps=1/{interval_seconds}", "-vsync", "0", str(output_pattern)]
    return command


def build_thumbnail_command(frame: Path, thumbnail: Path, width: int = THUMBNAIL_WIDTH) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(frame),
        "-vf",
        f"scale={width}:-1",
        str(thumbnail),
    ]


def resolve_interval_seconds(
    duration_seconds: float, *, interval_seconds: float | None, target_count: int | None
) -> tuple[float, ExtractionSettings]:
    if interval_seconds is not None and target_count is not None:
        raise ValueError("specify interval_seconds or target_count, not both")
    if interval_seconds is not None:
        return interval_seconds, ExtractionSettings(mode="interval", interval_seconds=interval_seconds)
    if target_count is not None:
        if target_count < 1:
            raise ValueError("target_count must be >= 1")
        if duration_seconds <= 0:
            raise ValueError("cannot derive an interval from a non-positive duration")
        interval = duration_seconds / target_count
        return interval, ExtractionSettings(
            mode="count", interval_seconds=interval, requested_count=target_count
        )
    raise ValueError("one of interval_seconds or target_count is required")


def resolve_time_range(
    duration_seconds: float, *, start_time: float | None, end_time: float | None
) -> tuple[float, float | None]:
    """Validates user-facing absolute start/end (seconds from the start of
    the source file, either/both may be None meaning "from the start"/"to
    the end") and converts them into ffmpeg-shaped (effective_start,
    duration_limit) -- duration_limit is None when end_time is None
    (extract to EOF). Maps onto ffmpeg's own -ss <start> (input option) +
    -t <duration> (output option); -to is deliberately avoided since its
    meaning becomes range-relative, not absolute, once -ss has already
    been given as an input option (the output timeline resets to 0) --
    confusing to reason about and to test.

    end_time beyond the real duration is clamped, not rejected -- ffmpeg
    would just run to EOF anyway, and erroring on "asked for slightly
    more than exists" would be unfriendly UX for a value a caller derived
    from a slightly-stale duration estimate.
    """
    start = start_time if start_time is not None else 0.0
    if start < 0:
        raise ValueError("start_time must be >= 0")
    if start >= duration_seconds:
        raise ValueError(f"start_time ({start}) must be less than the source duration ({duration_seconds})")

    if end_time is None:
        return start, None

    end = min(end_time, duration_seconds)
    if end <= start:
        raise ValueError(f"end_time ({end_time}) must be greater than start_time ({start})")

    return start, end - start


def clear_frames_for_source(conn: sqlite3.Connection, project_root: Path, source_id: str) -> None:
    """Removes any previously extracted frames for this source -- rows and
    files -- so extract_frames can be safely re-run with different
    settings (e.g. a different interval). Without this, a second
    extraction hits the frames table's frame_id PRIMARY KEY (frame_id is
    deterministic per source+index) and can also leave stale files behind
    from a larger prior extraction that a smaller new one wouldn't
    overwrite.

    Real data-integrity gap this also fixes: re-extracting frames used to
    leave any downstream views/masks built from the *old* frames as
    orphans -- their frame_id no longer existed, but the rows, files and
    project/projections/<frame_id>/ directories stuck around. Regenerating
    frames now cascades exactly like regenerating a single frame's views
    already does (vine360.projection.generate.clear_views_for_frame):
    delete masks for the affected views, delete the views, remove their
    projected-image directories, then remove the frames themselves.
    """
    frame_rows = conn.execute("SELECT frame_id FROM frames WHERE source_id = ?", (source_id,)).fetchall()
    frame_ids = [row[0] for row in frame_rows]

    if frame_ids:
        placeholders = ",".join("?" * len(frame_ids))
        conn.execute(
            f"DELETE FROM masks WHERE view_id IN "
            f"(SELECT view_id FROM views WHERE frame_id IN ({placeholders}))",
            frame_ids,
        )
        conn.execute(f"DELETE FROM views WHERE frame_id IN ({placeholders})", frame_ids)
    conn.execute("DELETE FROM frames WHERE source_id = ?", (source_id,))
    conn.commit()

    output_dir = Path(project_root) / "frames" / source_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    projections_root = Path(project_root) / "projections"
    for frame_id in frame_ids:
        frame_projections_dir = projections_root / frame_id
        if frame_projections_dir.exists():
            shutil.rmtree(frame_projections_dir)


def _poll_output_frame_count(
    output_dir: Path, expected_total: int, notify, stop_event: threading.Event, poll_interval_seconds: float
) -> None:
    """Runs on a side thread for the duration of the blocking ffmpeg
    extraction call, which otherwise gives no feedback at all until it
    fully completes -- for a real high-resolution/360 source this can run
    long enough to look identical to a hang (the same problem the
    thumbnail loop below already had, and was fixed for -- see docs/adr
    for this fix). ffmpeg's image2 muxer (the `frame_%06d.png` pattern)
    writes each numbered frame to disk as soon as it's decoded, so
    counting files on disk is a real, if approximate, progress signal
    (the last file may be mid-write for one tick, undercounting by at
    most one)."""
    while not stop_event.wait(poll_interval_seconds):
        count = len(list(output_dir.glob("frame_*.png")))
        notify(f"Extracting raw frames via ffmpeg… ({count}/{expected_total})", count, expected_total)


def extract_frames(
    conn: sqlite3.Connection,
    project_root: Path,
    source_id: str,
    runner: Runner,
    *,
    interval_seconds: float | None = None,
    target_count: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    generate_thumbnails: bool = True,
    progress_callback=None,
    poll_interval_seconds: float = 0.5,
) -> list[Frame]:
    """progress_callback(message, current, total), if given, is called for
    each phase. The raw ffmpeg extraction's progress is polled from a side
    thread (see _poll_output_frame_count) since the extraction itself is a
    single blocking command with no built-in sub-progress of its own; the
    thumbnail loop that follows reports real per-frame counts directly,
    and is the slow part for a high-resolution source (e.g. ~0.8s/frame at
    8K -- 172 frames is ~2.5 minutes). Frame rows are also committed
    incrementally (every 20 frames) rather than only once at the very end,
    so an interrupted run leaves a partially-usable record instead of
    none -- full crash-resume (re-using already-done work) is still not
    implemented; a re-run still clears and starts over via
    clear_frames_for_source. poll_interval_seconds is a constructor knob
    mainly so tests can shrink it well below a real ffmpeg call's
    duration.

    start_time/end_time optionally restrict extraction to a sub-range of
    the source (seconds from the start of the file; see
    resolve_time_range). source_time on returned Frames is always
    absolute -- offset by start_time -- never range-relative, since every
    downstream consumer relies on it meaning "seconds from the start of
    the original source file".

    generate_thumbnails, when False, skips the per-frame thumbnail pass
    entirely -- it's currently the slow part of this function (see above)
    for no benefit right now, since nothing in the app reads
    frames/<source_id>/thumbs/ back; PreviewGallery loads full-resolution
    view/mask images directly and lets Qt scale them in-memory instead.
    Defaults to True to keep existing behavior unchanged for any caller
    that doesn't pass it."""
    notify = progress_callback or (lambda *a: None)

    source = get_source(conn, source_id)
    if source.media_type.value != "video":
        raise FrameExtractionError(
            f"source {source_id} is media_type={source.media_type.value}; "
            "frame extraction only applies to video sources"
        )
    duration = source.timestamps.get("duration_seconds")
    if not duration:
        raise FrameExtractionError(f"source {source_id} has no known duration; cannot extract frames")

    range_start, duration_limit = resolve_time_range(duration, start_time=start_time, end_time=end_time)
    range_duration = duration_limit if duration_limit is not None else duration - range_start

    interval, extraction_settings = resolve_interval_seconds(
        range_duration, interval_seconds=interval_seconds, target_count=target_count
    )
    extraction_settings = replace(extraction_settings, start_time_seconds=start_time, end_time_seconds=end_time)

    clear_frames_for_source(conn, project_root, source_id)
    output_dir = Path(project_root) / "frames" / source_id
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = output_dir / "thumbs"
    if generate_thumbnails:
        thumbs_dir.mkdir(parents=True, exist_ok=True)

    notify("Extracting raw frames via ffmpeg…", None, None)
    pattern = output_dir / "frame_%06d.png"
    command = build_frame_extraction_command(
        Path(source.path), pattern, interval, start_time=range_start, duration_limit=duration_limit
    )

    expected_frame_count = max(1, round(range_duration / interval))
    stop_polling = threading.Event()
    poller = threading.Thread(
        target=_poll_output_frame_count,
        args=(output_dir, expected_frame_count, notify, stop_polling, poll_interval_seconds),
        daemon=True,
    )
    poller.start()
    try:
        result = runner.run(command)
    finally:
        stop_polling.set()
        poller.join(timeout=2.0)

    if not result.ok:
        raise FrameExtractionError(f"ffmpeg failed extracting {source_id}: {result.stderr.strip()}")

    frame_files = sorted(output_dir.glob("frame_*.png"))
    if not frame_files:
        raise FrameExtractionError(f"ffmpeg reported success but produced no frames for {source_id}")

    frames: list[Frame] = []
    for index, frame_path in enumerate(frame_files):
        if generate_thumbnails:
            notify(f"Generating thumbnails ({index + 1}/{len(frame_files)})…", index + 1, len(frame_files))
            thumb_path = thumbs_dir / f"{frame_path.stem}.jpg"
            thumb_result = runner.run(build_thumbnail_command(frame_path, thumb_path))
            if not thumb_result.ok:
                raise FrameExtractionError(
                    f"ffmpeg failed generating thumbnail for {frame_path}: {thumb_result.stderr.strip()}"
                )
        else:
            notify(f"Recording frames ({index + 1}/{len(frame_files)})…", index + 1, len(frame_files))

        frame = Frame(
            frame_id=f"{source_id}:{index:06d}",
            source_id=source_id,
            source_time=range_start + index * interval,
            extraction_settings=extraction_settings.to_dict(),
            path=str(frame_path.relative_to(project_root)),
            checksum=sha256_file(frame_path),
        )
        frames.append(frame)
        conn.execute(
            """
            INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                frame.frame_id,
                frame.source_id,
                frame.source_time,
                json.dumps(frame.extraction_settings),
                frame.path,
                frame.checksum,
            ),
        )
        if (index + 1) % 20 == 0:
            conn.commit()  # incremental, so an interrupted run keeps partial progress
    conn.commit()
    return frames
