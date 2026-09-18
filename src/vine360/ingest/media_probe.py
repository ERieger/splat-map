"""ffprobe adapter (handover doc, section 4, step 1 "Inspect").

Command construction is a pure function (`build_ffprobe_command`) so it can
be unit-tested without ffprobe installed; only `probe_media` actually shells
out, through a Runner so no other module calls ffprobe directly.

GPS/EXIF extraction for still photos (e.g. drone imagery) is not implemented
yet -- ffprobe surfaces container-level tags (which covers most video GPS
metadata, including Insta360 X6 exports) but not JPEG EXIF GPS. This is a
documented gap; see docs/status.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vine360.config import (
    EQUIRECTANGULAR_ASPECT_RATIO,
    EQUIRECTANGULAR_ASPECT_TOLERANCE,
)
from vine360.runners.base import Runner


class ProbeError(Exception):
    pass


@dataclass
class MediaMetadata:
    width: int | None
    height: int | None
    duration_seconds: float | None
    codec: str | None
    frame_rate: float | None
    format_tags: dict
    is_2to1_aspect: bool

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "codec": self.codec,
            "frame_rate": self.frame_rate,
            "format_tags": self.format_tags,
            "is_2to1_aspect": self.is_2to1_aspect,
        }


def build_ffprobe_command(path: Path) -> list[str]:
    return [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]


def _parse_frame_rate(rate_str: str | None) -> float | None:
    if not rate_str or rate_str in ("0/0",):
        return None
    if "/" in rate_str:
        num, _, den = rate_str.partition("/")
        try:
            num_f, den_f = float(num), float(den)
            return num_f / den_f if den_f else None
        except ValueError:
            return None
    try:
        return float(rate_str)
    except ValueError:
        return None


def parse_ffprobe_json(raw: str) -> MediaMetadata:
    data = json.loads(raw)
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

    width = video_stream.get("width") if video_stream else None
    height = video_stream.get("height") if video_stream else None
    codec = video_stream.get("codec_name") if video_stream else None
    frame_rate = _parse_frame_rate(video_stream.get("avg_frame_rate")) if video_stream else None

    duration_raw = fmt.get("duration") or (video_stream or {}).get("duration")
    duration = float(duration_raw) if duration_raw is not None else None

    is_2to1 = False
    if width and height:
        ratio = width / height
        is_2to1 = abs(ratio - EQUIRECTANGULAR_ASPECT_RATIO) <= EQUIRECTANGULAR_ASPECT_TOLERANCE

    return MediaMetadata(
        width=width,
        height=height,
        duration_seconds=duration,
        codec=codec,
        frame_rate=frame_rate,
        format_tags=fmt.get("tags", {}),
        is_2to1_aspect=is_2to1,
    )


def probe_media(path: Path, runner: Runner) -> MediaMetadata:
    command = build_ffprobe_command(path)
    result = runner.run(command)
    if not result.ok:
        raise ProbeError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    return parse_ffprobe_json(result.stdout)
