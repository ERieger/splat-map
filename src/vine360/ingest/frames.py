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
import sqlite3
from pathlib import Path

from vine360.config import ExtractionSettings
from vine360.ingest.sources import get_source, sha256_file
from vine360.models import Frame
from vine360.runners.base import Runner

THUMBNAIL_WIDTH = 320


class FrameExtractionError(Exception):
    pass


def build_frame_extraction_command(source: Path, output_pattern: Path, interval_seconds: float) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"fps=1/{interval_seconds}",
        "-vsync",
        "0",
        str(output_pattern),
    ]


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


def extract_frames(
    conn: sqlite3.Connection,
    project_root: Path,
    source_id: str,
    runner: Runner,
    *,
    interval_seconds: float | None = None,
    target_count: int | None = None,
) -> list[Frame]:
    source = get_source(conn, source_id)
    if source.media_type.value != "video":
        raise FrameExtractionError(
            f"source {source_id} is media_type={source.media_type.value}; "
            "frame extraction only applies to video sources"
        )
    duration = source.timestamps.get("duration_seconds")
    if not duration:
        raise FrameExtractionError(f"source {source_id} has no known duration; cannot extract frames")

    interval, extraction_settings = resolve_interval_seconds(
        duration, interval_seconds=interval_seconds, target_count=target_count
    )

    output_dir = Path(project_root) / "frames" / source_id
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = output_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    pattern = output_dir / "frame_%06d.png"
    command = build_frame_extraction_command(Path(source.path), pattern, interval)
    result = runner.run(command)
    if not result.ok:
        raise FrameExtractionError(f"ffmpeg failed extracting {source_id}: {result.stderr.strip()}")

    frame_files = sorted(output_dir.glob("frame_*.png"))
    if not frame_files:
        raise FrameExtractionError(f"ffmpeg reported success but produced no frames for {source_id}")

    frames: list[Frame] = []
    for index, frame_path in enumerate(frame_files):
        thumb_path = thumbs_dir / f"{frame_path.stem}.jpg"
        thumb_result = runner.run(build_thumbnail_command(frame_path, thumb_path))
        if not thumb_result.ok:
            raise FrameExtractionError(
                f"ffmpeg failed generating thumbnail for {frame_path}: {thumb_result.stderr.strip()}"
            )

        frame = Frame(
            frame_id=f"{source_id}:{index:06d}",
            source_id=source_id,
            source_time=index * interval,
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
    conn.commit()
    return frames
