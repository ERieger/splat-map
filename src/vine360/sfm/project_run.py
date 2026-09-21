"""Persists an SfmRun record for a project-level SfM run (handover doc,
section 4, step 5 "Estimate poses"). Bridges colmap_adapter.py (which
knows nothing about sqlite or the project layout) to the project's index
database.

Two real, verified image sources:
- "projections" (default): `project/projections/` (the six-face cubemap
  renders) + `project/masks/keep/` if built -- the doc's originally
  recommended pipeline.
- "frames": raw equirectangular frames directly
  (`project/frames/<source_id>/`), using COLMAP's own native
  EQUIRECTANGULAR camera model. Confirmed for real via
  `pycolmap.synthesize_dataset` with `camera_model_id=EQUIRECTANGULAR`
  (full registration, near-zero reprojection error) -- this is COLMAP's
  own spherical camera support, not the third-party "SphereSfM" project
  (see docs/adr/0017), and doesn't apply masks (none are built against
  raw frames in this pipeline yet). Real caveat, not yet validated: SIFT
  feature matching itself isn't sphere-aware, so quality near the poles
  and across the equirectangular seam is unproven on actual photos, only
  on noise-free synthetic correspondences.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from vine360.sfm.colmap_adapter import (
    SfmConfig,
    SfmDiagnostics,
    SfmRegistrationError,
    evaluate_registration_quality,
    run_sfm,
    validate_installation,
)

__all__ = ["SfmRegistrationError", "run_sfm_for_project"]


def _image_set_hash(image_dir: Path, project_root: Path) -> str:
    """A weaker stand-in for the data contract's checksum-based
    image_set_hash (vine360.sfm.colmap_adapter.compute_image_set_hash):
    Views don't carry their own checksum field, only Frame/Source do, so
    this hashes the sorted set of relative image paths -- it detects a
    changed *set* of images, not changed *content* of an unchanged path.
    Revisit if that distinction matters before this is relied on for real
    provenance decisions."""
    paths = sorted(p.relative_to(project_root).as_posix() for p in image_dir.rglob("*.png"))
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def run_sfm_for_project(
    conn: sqlite3.Connection,
    project_root: Path,
    *,
    config: SfmConfig = SfmConfig(),
    image_source: str = "projections",
    source_id: str | None = None,
    progress_callback=None,
) -> tuple[SfmDiagnostics, list[str]]:
    """image_source="frames" requires source_id and runs directly against
    that source's raw equirectangular frames (see module docstring) --
    the caller should set config.camera_model="EQUIRECTANGULAR" for this
    to be meaningful; it isn't forced here so an explicit config always
    wins.

    Raises SfmRegistrationError if COLMAP produces no usable
    reconstruction (a real, correct outcome for e.g. single-panorama,
    zero-parallax view sets -- see docs/adr/0007). Records an sfm_runs row
    on success only."""
    notify = progress_callback or (lambda *a: None)
    project_root = Path(project_root)

    if image_source == "projections":
        image_dir = project_root / "projections"
        mask_dir = project_root / "masks" / "keep"
        mask_dir_arg = mask_dir if any(mask_dir.rglob("*.png")) else None
        missing_message = "no projected views found under projections/; generate projections first"
    elif image_source == "frames":
        if not source_id:
            raise ValueError("source_id is required when image_source='frames'")
        image_dir = project_root / "frames" / source_id
        mask_dir_arg = None  # masks aren't built against raw frames in this pipeline yet
        missing_message = f"no extracted frames found under frames/{source_id}/; extract frames first"
    else:
        raise ValueError(f"unknown image_source: {image_source!r} (expected 'projections' or 'frames')")

    if not any(image_dir.rglob("*.png")):
        raise SfmRegistrationError(missing_message)

    # Each run gets its own sparse/<run_id>/ subtree. A shared sparse_dir
    # across runs let a later run's pycolmap.incremental_mapping silently
    # overwrite an earlier run's numbered model directories (both start
    # numbering candidate reconstructions from 0) -- discovered as a real
    # data-loss bug on a project with 4 historical runs, only the most
    # recent of which still had its files on disk (see docs/adr/0019).
    run_id = f"sfm-{uuid.uuid4().hex[:8]}"
    database_path = project_root / "sfm" / "database.db"
    sparse_dir = project_root / "sfm" / "sparse" / run_id

    notify("Extracting features and matching views…", None, None)
    _reconstruction, diagnostics, model_dir = run_sfm(
        image_dir, database_path, sparse_dir, mask_dir=mask_dir_arg, config=config
    )
    notify("Mapping complete.", None, None)

    warnings = evaluate_registration_quality(diagnostics)

    conn.execute(
        """
        INSERT INTO sfm_runs
            (run_id, image_set_hash, engine_version, config, model_stats, selected_model, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            _image_set_hash(image_dir, project_root),
            json.dumps(validate_installation()),
            json.dumps(
                {
                    "camera_model": config.camera_model,
                    "sequential_overlap": config.sequential_overlap,
                    "image_source": image_source,
                    "source_id": source_id,
                }
            ),
            json.dumps(diagnostics.to_dict()),
            model_dir.relative_to(project_root).as_posix(),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    return diagnostics, warnings
