"""Bundles vine360 artifacts into a self-contained folder ready to import
into Postshot (Jawset) -- or any other COLMAP-based external trainer.

Two export modes, both real, already-computed vine360 artifacts copied
and reformatted, not regenerated:

- `export_for_postshot`: poses + images + masks, for importing vine360's
  own already-computed COLMAP reconstruction.
- `export_frames_and_masks_for_postshot`: images + masks only, no poses
  -- for letting Postshot run its *own* pose estimation instead (its own
  COLMAP-based SfM step, the same one it runs when you give it a folder
  of images/video rather than an existing COLMAP dataset). This needs
  only that Projection (and optionally Masking) has run -- no SfM run at
  all. Requires the six-face-projection images (pinhole), not the raw
  equirectangular frames: masks are only ever built against projections
  in this pipeline (see vine360.sfm.project_run), and pinhole images are
  the safe, universally-supported choice for an external tool's own SfM
  regardless -- Postshot's equirectangular/360 support, if any, isn't
  confirmed in its own docs (see docs/adr/0018).

Both functions reset their own managed subdirectories (`images/`,
`masks/`, `sparse/`) under `output_dir` before writing (see
`_reset_managed_subdirs`), so exporting to a folder used by a previous
export -- by either mode, or the same mode with a different view/run --
never leaves stale files behind (e.g. a poses-mode `sparse/` lingering
after a later frames-and-masks-only export to the same folder).

Two things Postshot specifically needs that vine360's own on-disk layout
doesn't already provide, confirmed against Postshot's own documentation
(jawset.com/docs, "Importing Images" and "Interface/Training
Configuration"):

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
   "Same base filename" turned out to mean literally that -- confirmed by
   real ambiguous-mask-association failures once exported: every frame's
   projections share the same five leaf filenames (`front.png`,
   `back.png`, ...), which the *directory* they sit in disambiguates but
   the *filename alone* does not, and Postshot's matching goes by
   filename. So every exported file (image and mask) is flattened into a
   single globally-unique filename via `_flatten_to_unique_filename`
   (folding the original relative path into the name itself, e.g.
   `<frame_id>/front.png` -> `<frame_id>__front.png`), rather than relying
   on folder nesting for uniqueness -- see docs/adr/0020.
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
_MANAGED_SUBDIRS = ("images", "masks", "sparse")


class PostshotExportError(Exception):
    pass


def _flatten_to_unique_filename(relative_path: Path) -> str:
    """Turns a nested relative path (e.g. `<frame_id>/front.png`) into a
    single, globally-unique, filesystem-safe flat filename (e.g.
    `<frame_id>__front.png`). Every frame's projections share the same
    five leaf filenames (`front.png`, `back.png`, ...) -- fine for COLMAP
    import, which matches by full relative path, but Postshot's mask
    association goes by filename alone (see the module docstring), so
    relying on folder nesting for uniqueness there silently mismatches
    masks to the wrong frame (or fails to match at all). Also replaces
    ":" (vine360 frame_ids can contain it, e.g. "<source_id>:000042" --
    invalid in a bare Windows filename, though tolerated in a WSL/drvfs
    *directory* name, which is how this went unnoticed until export)."""
    return "__".join(part.replace(":", "_") for part in relative_path.parts)


def _reset_managed_subdirs(output_dir: Path) -> None:
    """Removes any of this module's own output subdirectories already
    present under output_dir, so exporting to a folder used before -- by
    either mode, or a previous run of the same mode with a different view
    set -- never leaves stale files behind. For example: export poses
    mode writes an output_dir/sparse/; exporting frames-and-masks mode to
    that same output_dir afterward doesn't produce a sparse/ of its own,
    but without this, the old one would still be sitting there for
    Postshot (or a person) to mistake for current data. Only ever touches
    these three well-known subdirectory names, never output_dir itself or
    anything else in it."""
    for name in _MANAGED_SUBDIRS:
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)


@dataclass
class PostshotExportResult:
    output_dir: Path
    images_dir: Path
    masks_dir: Path | None
    num_images: int
    num_masks: int
    # None when exporting via export_frames_and_masks_for_postshot --
    # there's no vine360-computed pose model in that mode by design.
    sparse_dir: Path | None = None
    warnings: list[str] = field(default_factory=list)


def export_for_postshot(
    conn: sqlite3.Connection,
    project_root: Path,
    output_dir: Path,
    *,
    run_id: str | None = None,
    progress_callback=None,
) -> PostshotExportResult:
    """Exports the given sfm_runs row (the most recent one, by default)
    to `output_dir/{sparse,images,masks}/`. Raises PostshotExportError if
    there's no run to export, or its selected model is missing on disk.

    progress_callback(message, current, total), if given, is called once
    before the (fast) sparse-model copy and once per image/mask file
    copied -- real images can run into the thousands (a 343-frame,
    five-face project is 1715 files per pass), so a bare "Exporting..."
    spinner gives no sense of progress or remaining time for what can be
    several GB of copying."""
    notify = progress_callback or (lambda *a: None)
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
    if not model_dir.exists():
        raise PostshotExportError(
            f"sfm run {found_run_id!r}'s recorded selected_model ({selected_model!r}) doesn't exist on "
            f"disk at {model_dir}. This run predates a bug fix where selected_model stored the run's own "
            "ID instead of its real model directory -- the underlying reconstruction files can't be "
            "reliably identified for a run recorded that way. Re-run Pose estimation for this project; "
            "runs made after this fix export correctly."
        )
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
    notify("Clearing any previous export in this folder…", None, None)
    _reset_managed_subdirs(output_dir)
    sparse_out = output_dir / "sparse"
    images_out = output_dir / "images"
    sparse_out.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    reconstruction = pycolmap.Reconstruction(model_dir)
    # (original COLMAP image name, flattened export filename), sorted by
    # the original name for a stable, predictable copy order.
    renames = sorted(
        ((image.name, _flatten_to_unique_filename(Path(image.name))) for image in reconstruction.images.values())
    )

    num_images = 0
    for i, (original_name, flat_name) in enumerate(renames):
        notify(f"Copying images ({i + 1}/{len(renames)})…", i + 1, len(renames))
        src = source_root / original_name
        if not src.exists():
            warnings.append(f"source image missing, skipped: {src}")
            continue
        shutil.copy2(src, images_out / flat_name)
        num_images += 1

    # images.bin must reference the same flattened filenames images_out/
    # was just populated with, or Postshot's COLMAP import can't locate them.
    notify("Copying pose data…", None, None)
    for image in reconstruction.images.values():
        image.name = _flatten_to_unique_filename(Path(image.name))
    reconstruction.write(sparse_out)

    masks_out: Path | None = None
    num_masks = 0
    if image_source == "projections":
        keep_root = project_root / "masks" / "keep"
        if keep_root.exists() and any(keep_root.rglob("*.png")):
            masks_out = output_dir / "masks"
            masks_out.mkdir(parents=True, exist_ok=True)
            for i, (original_name, flat_name) in enumerate(renames):
                notify(f"Copying masks ({i + 1}/{len(renames)})…", i + 1, len(renames))
                keep_src = colmap_mask_path(original_name, keep_root)
                if not keep_src.exists():
                    continue
                shutil.copy2(keep_src, masks_out / flat_name)  # same flattened name as its color image
                num_masks += 1
            if num_masks == 0:
                warnings.append("masks/keep/ exists but no mask matched any exported image")
        else:
            warnings.append("no masks have been built yet -- exporting poses and images only")
    notify("Export complete.", None, None)

    return PostshotExportResult(
        output_dir=output_dir,
        sparse_dir=sparse_out,
        images_dir=images_out,
        masks_dir=masks_out,
        num_images=num_images,
        num_masks=num_masks,
        warnings=warnings,
    )


def export_frames_and_masks_for_postshot(
    conn: sqlite3.Connection,
    project_root: Path,
    output_dir: Path,
    *,
    progress_callback=None,
) -> PostshotExportResult:
    """Exports every generated projection view and its keep-mask (if
    built) with no pose data at all, for Postshot to run its own SfM on
    -- see the module docstring for when to use this instead of
    export_for_postshot. Needs only project/views (Projection having
    run); masks are optional. Raises PostshotExportError if no
    projections exist yet.

    progress_callback(message, current, total), if given, is called once
    per view copied -- see export_for_postshot's docstring for why this
    matters at real project scale (thousands of files)."""
    notify = progress_callback or (lambda *a: None)
    project_root = Path(project_root)
    output_dir = Path(output_dir)

    rows = conn.execute("SELECT image_path FROM views").fetchall()
    if not rows:
        raise PostshotExportError("no projected views found -- generate projections first")

    keep_root = project_root / "masks" / "keep"
    has_masks = keep_root.exists() and any(keep_root.rglob("*.png"))

    notify("Clearing any previous export in this folder…", None, None)
    _reset_managed_subdirs(output_dir)  # in particular, drops a stale sparse/ from a prior poses-mode export
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out = output_dir / "masks" if has_masks else None
    if masks_out is not None:
        masks_out.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    num_images = 0
    num_masks = 0
    total = len(rows)
    for i, (image_path,) in enumerate(rows):
        notify(f"Copying images and masks ({i + 1}/{total})…", i + 1, total)
        relative_path = Path(image_path)
        src = project_root / relative_path
        if not src.exists():
            warnings.append(f"source image missing, skipped: {src}")
            continue
        relative_to_projections = relative_path.relative_to("projections")
        flat_name = _flatten_to_unique_filename(relative_to_projections)
        shutil.copy2(src, images_out / flat_name)
        num_images += 1

        if masks_out is not None:
            keep_src = colmap_mask_path(relative_to_projections, keep_root)
            if keep_src.exists():
                shutil.copy2(keep_src, masks_out / flat_name)  # same flattened name as its color image
                num_masks += 1

    if masks_out is None:
        warnings.append("no masks have been built yet -- exporting images only")
    elif num_masks == 0:
        warnings.append("masks/keep/ exists but no mask matched any exported image")
    notify("Export complete.", None, None)

    return PostshotExportResult(
        output_dir=output_dir,
        images_dir=images_out,
        masks_dir=masks_out,
        num_images=num_images,
        num_masks=num_masks,
        warnings=warnings,
    )
