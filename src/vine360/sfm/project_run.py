"""Persists an SfmRun record for a project-level SfM run (handover doc,
section 4, step 5 "Estimate poses"). Bridges colmap_adapter.py (which
knows nothing about sqlite or the project layout) to the project's index
database, running against `project/projections/` as the image directory
and `project/masks/keep/` as the mask directory -- exactly the layout
`vine360.projection.generate` and `vine360.masking.build` produce.
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
    progress_callback=None,
) -> tuple[SfmDiagnostics, list[str]]:
    """Raises SfmRegistrationError if COLMAP produces no usable
    reconstruction (a real, correct outcome for e.g. single-panorama,
    zero-parallax view sets -- see docs/adr/0007). Records an sfm_runs row
    on success only."""
    notify = progress_callback or (lambda *a: None)
    project_root = Path(project_root)
    image_dir = project_root / "projections"
    if not any(image_dir.rglob("*.png")):
        raise SfmRegistrationError("no projected views found under projections/; generate projections first")

    mask_dir = project_root / "masks" / "keep"
    mask_dir_arg = mask_dir if any(mask_dir.rglob("*.png")) else None

    database_path = project_root / "sfm" / "database.db"
    sparse_dir = project_root / "sfm" / "sparse"

    notify("Extracting features and matching views…", None, None)
    _reconstruction, diagnostics = run_sfm(
        image_dir, database_path, sparse_dir, mask_dir=mask_dir_arg, config=config
    )
    notify("Mapping complete.", None, None)

    warnings = evaluate_registration_quality(diagnostics)

    run_id = f"sfm-{uuid.uuid4().hex[:8]}"
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
            json.dumps({"camera_model": config.camera_model, "sequential_overlap": config.sequential_overlap}),
            json.dumps(diagnostics.to_dict()),
            run_id,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    return diagnostics, warnings
