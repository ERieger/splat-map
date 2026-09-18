"""Typed configuration values shared across the processing core.

Dataclasses stand in for pydantic here; see docs/adr/0001-stdlib-first-cli.md
for why. They still give every config value a declared type and a single
place to change defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CaptureMode(str, Enum):
    THREE_SIXTY = "360"
    CONVENTIONAL = "conventional"
    MIXED = "mixed"


class MediaType(str, Enum):
    VIDEO = "video"
    EQUIRECT_STILL = "equirect_still"
    PHOTO = "photo"


class Projection(str, Enum):
    EQUIRECTANGULAR = "equirectangular"
    PERSPECTIVE = "perspective"
    UNKNOWN = "unknown"


class FramePreset(str, Enum):
    PREVIEW = "preview"
    BALANCED = "balanced"
    QUALITY = "quality"
    CUSTOM = "custom"


# Frame-extraction interval, in seconds, for each preset (section 6:
# "Preset strategy"). Custom presets carry no default and must be supplied
# explicitly by the caller.
FRAME_PRESET_INTERVALS: dict[FramePreset, float] = {
    FramePreset.PREVIEW: 2.0,
    FramePreset.BALANCED: 1.0,
    FramePreset.QUALITY: 0.5,
}

# Aspect ratios within this tolerance of 2:1 are treated as *candidate*
# equirectangular stills, but section 4 requires explicit user confirmation
# rather than silent inference, so this only ever gates a prompt/flag.
EQUIRECTANGULAR_ASPECT_RATIO = 2.0
EQUIRECTANGULAR_ASPECT_TOLERANCE = 0.02


@dataclass(frozen=True)
class ExtractionSettings:
    """Recorded verbatim on every Frame so extraction is reproducible."""

    mode: str  # "interval" or "count"
    interval_seconds: float
    requested_count: int | None = None

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "interval_seconds": self.interval_seconds,
            "requested_count": self.requested_count,
        }
