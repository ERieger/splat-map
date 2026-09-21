"""Data contracts from the handover doc, section 5 ("Project layout and
data contracts").

Every dataclass here carries at least the "Required fields" listed for its
entity in the handover. Project, Source and Frame are wired up to real
persistence in M0/M1. View, Mask, SfmRun and TrainingRun are declared now so
the schema is stable and reviewable, but nothing constructs them yet -- that
is M2 through M5 work; do not add processing logic against them early.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from vine360.config import CaptureMode, MediaType, Projection

SCHEMA_VERSION = 1


@dataclass
class Project:
    schema_version: int
    project_id: str
    name: str
    created_at: str  # ISO-8601 UTC
    working_crs: str | None
    software_versions: dict[str, str]
    capture_mode: CaptureMode

    def to_dict(self) -> dict:
        d = asdict(self)
        d["capture_mode"] = self.capture_mode.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "Project":
        d = dict(d)
        d["capture_mode"] = CaptureMode(d["capture_mode"])
        return Project(**d)


@dataclass
class Source:
    source_id: str
    path: str  # path to the original media, recorded not copied (ADR 0003)
    checksum: str  # sha256
    media_type: MediaType
    projection: Projection
    dimensions: tuple[int, int] | None
    timestamps: dict  # e.g. {"duration_seconds": ..., "added_at": ...}
    capture_group: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["media_type"] = self.media_type.value
        d["projection"] = self.projection.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "Source":
        d = dict(d)
        d["media_type"] = MediaType(d["media_type"])
        d["projection"] = Projection(d["projection"])
        if d.get("dimensions") is not None:
            d["dimensions"] = tuple(d["dimensions"])
        return Source(**d)


@dataclass
class Frame:
    frame_id: str
    source_id: str
    source_time: float  # seconds into the source
    extraction_settings: dict
    path: str
    checksum: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Frame":
        return Frame(**d)


# --- Reserved for later milestones (M2-M5). Fields mirror the handover's
# data-contract table exactly; do not add behaviour around these yet. ---


@dataclass
class View:
    view_id: str
    frame_id: str
    projection_id: str
    width: int
    height: int
    intrinsics: dict
    fixed_rotation: dict
    image_path: str


@dataclass
class Mask:
    view_id: str
    model: str
    model_version: str
    prompts: list[str] = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    morphology: dict = field(default_factory=dict)
    keep_fraction: float | None = None
    edited: bool = False
    # Not in the handover doc's original data contract -- added for the
    # GUI's review workflow (auto-set for keep-fraction anomalies, freely
    # toggleable by the user afterward). Distinct from `edited`: flagged
    # means "needs a look", edited means "a human already corrected it".
    flagged_for_review: bool = False


@dataclass
class SfmRun:
    run_id: str
    image_set_hash: str
    engine_version: str
    config: dict
    model_stats: dict
    # Path to the written COLMAP model directory, relative to project_root
    # (e.g. "sfm/sparse/0") -- pycolmap.incremental_mapping writes every
    # candidate reconstruction it finds under sfm/sparse/<key>/, not just
    # the best one, so this field is what actually identifies which
    # subdirectory holds the reconstruction this run selected.
    selected_model: str | None


@dataclass
class TrainingRun:
    run_id: str
    backend_version: str
    dataset_hash: str
    config: dict
    checkpoint: str | None
    metrics: dict
    output_path: str | None
