"""Persists an SfmRun record for a project-level SfM run (handover doc,
section 4, step 5 "Estimate poses"). Bridges colmap_adapter.py (which
knows nothing about sqlite or the project layout) to the project's index
database.

Two real, verified image sources:
- "projections" (default): `project/projections/` (the six-face cubemap
  renders) + `project/masks/keep/` if built -- the doc's originally
  recommended pipeline.
- "frames": raw equirectangular frames directly
  (`project/frames/<frame_set_id>/`), using COLMAP's own native
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


def _image_set_hash(image_names: list[str]) -> str:
    """A weaker stand-in for the data contract's checksum-based
    image_set_hash (vine360.sfm.colmap_adapter.compute_image_set_hash):
    Views don't carry their own checksum field, only Frame/Source do, so
    this hashes the sorted set of image paths the run was given -- it
    detects a changed *set* of images, not changed *content* of an
    unchanged path. Revisit if that distinction matters before this is
    relied on for real provenance decisions."""
    return hashlib.sha256("\n".join(sorted(image_names)).encode("utf-8")).hexdigest()


def run_sfm_for_project(
    conn: sqlite3.Connection,
    project_root: Path,
    *,
    config: SfmConfig = SfmConfig(),
    image_source: str = "projections",
    frame_set_id: str | None = None,
    progress_callback=None,
) -> tuple[SfmDiagnostics, list[str]]:
    """image_source="frames" requires frame_set_id and runs directly
    against that frame set's raw equirectangular frames (see module
    docstring) -- the caller should set config.camera_model=
    "EQUIRECTANGULAR" for this to be meaningful; it isn't forced here so
    an explicit config always wins.

    image_source="projections" runs on every projected view in the
    project, or only frame_set_id's views if given -- one source can have
    several frame sets (e.g. every 0.5s and every 1s), and mixing them
    in one reconstruction would just duplicate near-identical images
    (docs/adr/0034).

    Either way COLMAP is handed an explicit image list (pycolmap's
    `image_names`) built from the database, not a bare directory scan --
    so frames/<id>/thumbs/*.jpg never sneak in, and total_images counts
    images rather than per-frame subdirectories. Each run gets its own
    COLMAP database under its own sparse/<run_id>/, so a scoped run never
    sees an earlier run's images left in a shared database.

    Raises SfmRegistrationError if COLMAP produces no usable
    reconstruction (a real, correct outcome for e.g. single-panorama,
    zero-parallax view sets -- see docs/adr/0007). Records an sfm_runs row
    on success only."""
    notify = progress_callback or (lambda *a: None)
    project_root = Path(project_root)

    source_id = None
    if frame_set_id is not None:
        row = conn.execute("SELECT source_id FROM frame_sets WHERE frame_set_id = ?", (frame_set_id,)).fetchone()
        source_id = row[0] if row else None

    if image_source == "projections":
        image_dir = project_root / "projections"
        if frame_set_id is None:
            rows = conn.execute("SELECT image_path FROM views ORDER BY image_path").fetchall()
            missing_message = "no projected views found under projections/; generate projections first"
        else:
            rows = conn.execute(
                "SELECT v.image_path FROM views v JOIN frames f ON f.frame_id = v.frame_id "
                "WHERE f.frame_set_id = ? ORDER BY v.image_path",
                (frame_set_id,),
            ).fetchall()
            missing_message = f"no projected views found for frame set {frame_set_id}; generate projections first"
        image_names = [Path(r[0]).relative_to("projections").as_posix() for r in rows]
        mask_dir = project_root / "masks" / "keep"
        mask_dir_arg = mask_dir if mask_dir.exists() and any(mask_dir.rglob("*.png")) else None
    elif image_source == "frames":
        if not frame_set_id:
            raise ValueError("frame_set_id is required when image_source='frames'")
        image_dir = project_root / "frames" / frame_set_id
        rows = conn.execute(
            "SELECT path FROM frames WHERE frame_set_id = ? ORDER BY source_time", (frame_set_id,)
        ).fetchall()
        image_names = [Path(r[0]).name for r in rows]
        mask_dir_arg = None  # masks aren't built against raw frames in this pipeline yet
        missing_message = f"no extracted frames found under frames/{frame_set_id}/; extract frames first"
    else:
        raise ValueError(f"unknown image_source: {image_source!r} (expected 'projections' or 'frames')")

    image_names = [name for name in image_names if (image_dir / name).exists()]
    if not image_names:
        raise SfmRegistrationError(missing_message)

    # Each run gets its own sparse/<run_id>/ subtree. A shared sparse_dir
    # across runs let a later run's pycolmap.incremental_mapping silently
    # overwrite an earlier run's numbered model directories (both start
    # numbering candidate reconstructions from 0) -- discovered as a real
    # data-loss bug on a project with 4 historical runs, only the most
    # recent of which still had its files on disk (see docs/adr/0019).
    run_id = f"sfm-{uuid.uuid4().hex[:8]}"
    sparse_dir = project_root / "sfm" / "sparse" / run_id
    database_path = sparse_dir / "database.db"

    notify("Extracting features and matching views…", None, None)
    _reconstruction, diagnostics, model_dir = run_sfm(
        image_dir, database_path, sparse_dir, mask_dir=mask_dir_arg, config=config, image_names=image_names
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
            _image_set_hash(image_names),
            json.dumps(validate_installation()),
            json.dumps(
                {
                    "camera_model": config.camera_model,
                    "sequential_overlap": config.sequential_overlap,
                    "image_source": image_source,
                    "frame_set_id": frame_set_id,
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
