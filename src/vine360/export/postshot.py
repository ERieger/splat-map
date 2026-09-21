"""Bundles a completed SfM run's poses, images and masks into a
self-contained folder ready to import into Postshot (Jawset) -- or any
other COLMAP-based external trainer.

This step only reformats and copies real, already-computed vine360
artifacts; it runs no new pose estimation or masking. Two things Postshot
specifically needs that vine360's own on-disk layout doesn't already
provide, confirmed against Postshot's own documentation (jawset.com/docs,
"Importing Images" and "Interface/Training Configuration"):

1. **Mask filenames.** vine360 writes keep-masks under
   `masks/keep/<image-relative-path>.png` (see
   `vine360.masking.semantics.colmap_mask_path`) -- COLMAP's own
   `mask_path` convention is to append a full extra ".png" to the
   complete original filename, e.g. `front.png` -> `front.png.png`.
   Postshot's own docs state the opposite requirement: "In order to be
   associated with the correct color image, the mask image must have the
   same base filename" and "Mask images must have the same resolution as
   the associated color image." So this module copies (not embeds) each
   keep-mask, stripping vine360/COLMAP's extra ".png" so the mask sits
   next to the color image with an identical filename. No alpha-channel
   embedding is used or needed: Postshot's masks are separate
   black-and-white images -- exactly vine360's keep.png format already
   (255 = keep, 0 = excluded) -- not something baked into the color
   image's own alpha channel (that phrase in Postshot's release notes
   turned out to refer to a different, internal mechanism; the documented
   *import* workflow is a separate mask image, matched by filename).
2. **Locating the actual selected model.** `pycolmap.incremental_mapping`
   writes *every* candidate reconstruction it finds under
   `sfm/sparse/<key>/`, not only the one vine360 selected as the best.
   `sfm_runs.selected_model` records which one (a real bug -- it used to
   store the run_id string instead -- fixed alongside this feature; see
   docs/adr/0018). Only that model's three canonical COLMAP files
   (cameras/images/points3D) are copied.

**Unverified against a real Postshot install** (Windows-only, not
available in this environment) -- verified only against Postshot's
published documentation. Test importing a small run before relying on
this for a real project. See docs/adr/0018.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pycolmap

from vine360.masking.semantics import colmap_mask_path

_COLMAP_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")


class PostshotExportError(Exception):
    pass


@dataclass
class PostshotExportResult:
    output_dir: Path
    sparse_dir: Path
    images_dir: Path
    masks_dir: Path | None
    num_images: int
    num_masks: int
    warnings: list[str] = field(default_factory=list)


def export_for_postshot(
    conn: sqlite3.Connection,
    project_root: Path,
    output_dir: Path,
    *,
    run_id: str | None = None,
) -> PostshotExportResult:
    """Exports the given sfm_runs row (the most recent one, by default)
    to `output_dir/{sparse,images,masks}/`. Raises PostshotExportError if
    there's no run to export, or its selected model is missing on disk."""
    project_root = Path(project_root)
    output_dir = Path(output_dir)

    if run_id is not None:
        row = conn.execute(
            "SELECT run_id, selected_model, config FROM sfm_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT run_id, selected_model, config FROM sfm_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise PostshotExportError("no SfM run found -- run Pose estimation first")

    found_run_id, selected_model, config_json = row
    if not selected_model:
        raise PostshotExportError(
            f"sfm run {found_run_id!r} has no recorded selected_model "
            "(from before this field was fixed to store the model path -- re-run Pose estimation)"
        )

    model_dir = project_root / selected_model
    missing = [name for name in _COLMAP_MODEL_FILES if not (model_dir / name).exists()]
    if missing:
        raise PostshotExportError(f"selected model directory {model_dir} is missing {missing}")

    config = json.loads(config_json)
    image_source = config.get("image_source", "projections")
    if image_source == "projections":
        source_root = project_root / "projections"
    elif image_source == "frames":
        source_id = config.get("source_id")
        if not source_id:
            raise PostshotExportError(f"sfm run {found_run_id!r} used image_source='frames' with no source_id")
        source_root = project_root / "frames" / source_id
    else:
        raise PostshotExportError(f"unknown image_source in sfm run config: {image_source!r}")

    warnings: list[str] = []
    sparse_out = output_dir / "sparse"
    images_out = output_dir / "images"
    sparse_out.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)
    for name in _COLMAP_MODEL_FILES:
        shutil.copy2(model_dir / name, sparse_out / name)

    reconstruction = pycolmap.Reconstruction(model_dir)
    image_names = sorted(image.name for image in reconstruction.images.values())

    num_images = 0
    for name in image_names:
        src = source_root / name
        if not src.exists():
            warnings.append(f"source image missing, skipped: {src}")
            continue
        dst = images_out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        num_images += 1

    masks_out: Path | None = None
    num_masks = 0
    if image_source == "projections":
        keep_root = project_root / "masks" / "keep"
        if keep_root.exists() and any(keep_root.rglob("*.png")):
            masks_out = output_dir / "masks"
            masks_out.mkdir(parents=True, exist_ok=True)
            for name in image_names:
                keep_src = colmap_mask_path(name, keep_root)
                if not keep_src.exists():
                    continue
                dst = masks_out / name  # strip the extra ".png" COLMAP's own convention adds
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(keep_src, dst)
                num_masks += 1
            if num_masks == 0:
                warnings.append("masks/keep/ exists but no mask matched any exported image")
        else:
            warnings.append("no masks have been built yet -- exporting poses and images only")

    return PostshotExportResult(
        output_dir=output_dir,
        sparse_dir=sparse_out,
        images_dir=images_out,
        masks_dir=masks_out,
        num_images=num_images,
        num_masks=num_masks,
        warnings=warnings,
    )
