"""Persists Mask records + mask files for generated views (handover doc,
section 4, step 4 "Mask"). Bridges semantics.py/classical_sky.py/
sam3_adapter.py (which know nothing about sqlite or the project layout) to
the project's index database and the `project/masks/` filesystem layout.

The keep-mask is written under `masks/keep/<relative-image-path>.png`,
matching `vine360.masking.semantics.colmap_mask_path`'s convention exactly
-- `masks/keep/` doubles as the COLMAP `mask_path` root once SfM runs
against `project/projections/` as its image directory (see
vine360.sfm.project_run).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from vine360.masking.classical_sky import classify_sky_classical
from vine360.masking.sam3_adapter import Sam3Adapter
from vine360.masking.semantics import (
    MaskBuildConfig,
    build_keep_and_exclude_masks,
    colmap_mask_path,
    is_keep_fraction_anomalous,
    keep_fraction,
    save_mask,
)
from vine360.models import Mask

PROJECTIONS_DIRNAME = "projections"


class MaskBuildError(Exception):
    pass


def build_mask_for_view(
    conn: sqlite3.Connection,
    project_root: Path,
    view_id: str,
    *,
    use_sam3_person: bool = False,
    use_sam3_sky: bool = False,
    sam3_adapter: Sam3Adapter | None = None,
    mask_config: MaskBuildConfig = MaskBuildConfig(),
) -> Mask:
    project_root = Path(project_root)
    row = conn.execute("SELECT image_path FROM views WHERE view_id = ?", (view_id,)).fetchone()
    if row is None:
        raise MaskBuildError(f"unknown view_id: {view_id}")
    image_relative_path = Path(row[0])

    with Image.open(project_root / image_relative_path) as img:
        image = np.asarray(img.convert("RGB"))

    class_masks: list[np.ndarray] = []
    prompts: list[str] = []
    model_name = "classical-sky"
    model_version = "1"

    if use_sam3_person or use_sam3_sky:
        adapter = sam3_adapter or Sam3Adapter()
        sam3_prompts = {}
        if use_sam3_person:
            sam3_prompts["person"] = "person"
        if use_sam3_sky:
            sam3_prompts["sky"] = "sky"
        sam3_masks = adapter.segment(image, sam3_prompts)
        for class_name, mask in sam3_masks.items():
            class_masks.append(mask)
            prompts.append(sam3_prompts[class_name])
        model_name = "sam3"
        model_version = adapter.model_id
        if not use_sam3_sky:
            class_masks.append(classify_sky_classical(image))
            prompts.append("sky(classical-fallback)")
    else:
        class_masks.append(classify_sky_classical(image))
        prompts.append("sky(classical)")

    exclude, keep = build_keep_and_exclude_masks(class_masks, mask_config)

    classes_dir = project_root / "masks" / "classes" / view_id
    for prompt, mask in zip(prompts, class_masks):
        save_mask(mask, classes_dir / f"{prompt.replace('/', '_')}.png")
    save_mask(exclude, classes_dir / "exclude.png")

    relative_to_projections = image_relative_path.relative_to(PROJECTIONS_DIRNAME)
    keep_path = colmap_mask_path(relative_to_projections, project_root / "masks" / "keep")
    save_mask(keep, keep_path)

    kf = keep_fraction(keep)
    mask = Mask(
        view_id=view_id,
        model=model_name,
        model_version=str(model_version),
        prompts=prompts,
        thresholds={},
        morphology={"dilation_px": mask_config.dilation_px, "min_component_px": mask_config.min_component_px},
        keep_fraction=kf,
        edited=False,
        # Auto-flag anomalies as a starting point for review; the user can
        # freely flag/unflag afterward via set_view_flagged without
        # re-running the build.
        flagged_for_review=is_keep_fraction_anomalous(kf) is not None,
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO masks
            (view_id, model, model_version, prompts, thresholds, morphology, keep_fraction, edited,
             flagged_for_review, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            mask.view_id,
            mask.model,
            mask.model_version,
            json.dumps(mask.prompts),
            json.dumps(mask.thresholds),
            json.dumps(mask.morphology),
            mask.keep_fraction,
            int(mask.edited),
            int(mask.flagged_for_review),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    return mask


def set_view_flagged(conn: sqlite3.Connection, view_id: str, flagged: bool) -> None:
    """Toggles flagged_for_review without touching anything else -- the
    user reviewing masks shouldn't have to re-run the (potentially slow,
    SAM-3-loading) build just to mark or clear a flag."""
    cursor = conn.execute(
        "UPDATE masks SET flagged_for_review = ? WHERE view_id = ?", (int(flagged), view_id)
    )
    if cursor.rowcount == 0:
        raise MaskBuildError(f"unknown view_id (no mask built yet?): {view_id}")
    conn.commit()


def build_masks_for_frame_set(
    conn: sqlite3.Connection,
    project_root: Path,
    frame_set_id: str,
    *,
    use_sam3_person: bool = False,
    use_sam3_sky: bool = False,
    mask_config: MaskBuildConfig = MaskBuildConfig(),
    progress_callback=None,
) -> list[Mask]:
    """progress_callback(message, current, total). Loads the SAM 3 model
    (if requested) once for the whole batch, not once per view."""
    notify = progress_callback or (lambda *a: None)
    view_rows = conn.execute(
        "SELECT v.view_id FROM views v JOIN frames f ON v.frame_id = f.frame_id "
        "WHERE f.frame_set_id = ? ORDER BY v.view_id",
        (frame_set_id,),
    ).fetchall()
    if not view_rows:
        raise MaskBuildError(f"no views found for frame set {frame_set_id}; generate projections first")

    adapter = Sam3Adapter() if (use_sam3_person or use_sam3_sky) else None
    total = len(view_rows)
    masks: list[Mask] = []
    for index, (view_id,) in enumerate(view_rows):
        notify(f"Masking view {index + 1}/{total}…", index, total)
        masks.append(
            build_mask_for_view(
                conn,
                project_root,
                view_id,
                use_sam3_person=use_sam3_person,
                use_sam3_sky=use_sam3_sky,
                sam3_adapter=adapter,
                mask_config=mask_config,
            )
        )
    notify(f"Masked {total} views.", total, total)
    return masks
