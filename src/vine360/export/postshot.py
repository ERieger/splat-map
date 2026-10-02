"""Bundles vine360 artifacts into a self-contained folder ready to import
into Postshot (Jawset), RealityScan (Epic Games), or any other COLMAP-
based or images-only external trainer.

Three export modes, all real, already-computed vine360 artifacts copied
and reformatted, not regenerated:

- `export_for_postshot`: poses + images + masks, for importing vine360's
  own already-computed COLMAP reconstruction. Writes to a `colmap`-tagged
  format folder (see the directory-layout note below).
- `export_frames_and_masks_for_postshot`: images + masks only, no poses
  -- for letting Postshot run its *own* pose estimation instead (its own
  COLMAP-based SfM step, the same one it runs when you give it a folder
  of images/video rather than an existing COLMAP dataset). Writes to a
  `postshot`-tagged format folder.
- `export_for_realityscan`: images + masks only, no poses -- the same
  "let the external tool align its own photos" shape as the Postshot
  mode above, but with RealityScan's own documented mask convention
  instead of Postshot's (see its own docstring). Writes to a
  `realityscan`-tagged format folder. **Unverified against a real
  RealityScan install** (not available in this environment) -- built
  from RealityScan's published documentation only (docs/adr/0028), same
  honesty-discipline caveat as the Postshot modes (docs/adr/0018).

All three need only that Projection (and optionally Masking) has run --
no SfM run at all -- except `export_for_postshot`, which needs a
completed Pose-estimation run. The two images-only modes require the
six-face-projection images (pinhole), not the raw equirectangular
frames: masks are only ever built against projections in this pipeline
(see vine360.sfm.project_run), and pinhole images are the safe,
universally-supported choice for an external tool's own SfM regardless.
Both take an optional `frame_set_id` to export just one frame set's
views (docs/adr/0034) instead of every view in the project (the
default) -- `export_for_postshot` has no such filter, since narrowing it
would mean editing images out of an already-computed COLMAP
reconstruction, not supported here (docs/adr/0026).

**Directory layout** (docs/adr/0028): every export lives at
`exports/<capture>/<format>/`, where `<capture>` is the per-source (or
"all") folder the caller chose and `<format>` is `colmap`/`postshot`/
`realityscan` depending on which function wrote it -- the GUI
(`ExportPanel`) is what actually constructs this path; the functions
here just receive `output_dir` as given. `_migrate_legacy_export_layout`,
called by all three functions before they write anything, moves a
pre-existing flat `<capture>/{images,masks,sparse}/` folder (from before
this nested layout existed) into the correctly-inferred
`<capture>/<format>/` subfolder the first time any export function is
asked to write into that capture folder again.

All three functions reset their own managed subdirectories (`images/`,
`masks/`, `sparse/`) under `output_dir` before writing (see
`_reset_managed_subdirs`), so exporting to a folder used by a previous
export -- by any mode, or the same mode with a different view/run --
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

**RealityScan-only: optional camera-priors CSV.** `export_for_realityscan`
can optionally also write a `CameraPriors.csv` (position-only, no
orientation) alongside the images, derived from vine360's own
already-completed COLMAP SfM run if one exists -- for import into
RealityScan's real, documented "Camera Priors" pre-alignment feature
(WORKFLOW tab -> Import Metadata -> Trajectory), to help its own
alignment through repetitive/self-similar scene content (the motivating
case: adjacent vineyard rows that look nearly identical to a pure
feature-matcher). Checked against Postshot's own published docs too:
Postshot's `Camera Poses` setting is strictly binary (`Import` a
complete external reconstruction wholesale, or `Estimate` from scratch)
-- no GPS/partial-pose input is surfaced anywhere, so this option does
not exist for `export_frames_and_masks_for_postshot`. See docs/adr/0033.
"""

from __future__ import annotations

import csv
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


def _run_frame_set_id(config: dict) -> str | None:
    """Which frame set an sfm run was scoped to -- None for a project-wide
    "projections" run. Runs recorded before frame sets existed only have
    source_id, which for a migrated legacy frame set *is* its
    frame_set_id (docs/adr/0034), so it doubles as the fallback."""
    return config.get("frame_set_id") or config.get("source_id")


def _select_priors_run(conn: sqlite3.Connection, *, frame_set_id: str | None) -> tuple[str, str, dict] | None:
    """Picks which sfm_runs row to draw camera-position priors from for a
    RealityScan export scoped to frame_set_id (None = all frame sets) --
    maximal-overlap-first, not just "most recent": a run scoped to
    exactly this frame set (either engine) is preferred when frame_set_id
    is given (its every registered image can potentially overlap the
    export); a project-wide "projections" run is the fallback (always
    covers any frame set, since it's project-wide); a run scoped to a
    *different* frame set is never picked when frame_set_id is given
    (guaranteed zero overlap -- picking it would silently write an empty
    CSV). When frame_set_id is None, prefers the most recent project-wide
    "projections" run, falling back to the most recent run of any scope
    (which only partially covers an all-frame-sets export -- acceptable,
    surfaced as a warning by the caller, not an error here).

    Returns (run_id, selected_model, config) or None if no sfm_runs row
    has a selected_model at all. Does NOT check the model files still
    exist on disk -- callers that actually need the model do that
    themselves; this is also used for cheap GUI checkbox-enablement,
    where a full disk check isn't worth the cost on every UI refresh."""
    rows = conn.execute(
        "SELECT run_id, selected_model, config FROM sfm_runs "
        "WHERE selected_model IS NOT NULL ORDER BY created_at DESC"
    ).fetchall()
    candidates = [(run_id, model, json.loads(cfg)) for run_id, model, cfg in rows]

    def is_project_wide(candidate: tuple[str, str, dict]) -> bool:
        config = candidate[2]
        return config.get("image_source", "projections") == "projections" and not config.get("frame_set_id")

    def is_matching(candidate: tuple[str, str, dict]) -> bool:
        return _run_frame_set_id(candidate[2]) == frame_set_id

    if frame_set_id is not None:
        for candidate in candidates:
            if is_matching(candidate):
                return candidate
        for candidate in candidates:
            if is_project_wide(candidate):
                return candidate
        return None

    for candidate in candidates:
        if is_project_wide(candidate):
            return candidate
    return candidates[0] if candidates else None


def realityscan_priors_available(conn: sqlite3.Connection, *, frame_set_id: str | None = None) -> bool:
    """Cheap (DB-only, no disk/pycolmap access) check for whether
    export_for_realityscan's include_camera_priors option has anything
    to draw from for this frame_set_id filter -- what ExportPanel's camera
    priors checkbox is gated on."""
    return _select_priors_run(conn, frame_set_id=frame_set_id) is not None


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


def _write_camera_priors_csv(
    conn: sqlite3.Connection,
    model_dir: Path,
    config: dict,
    output_dir: Path,
    exported_flat_names: set[str],
) -> tuple[Path | None, int, str | None]:
    """Writes output_dir/CameraPriors.csv -- position-only camera priors
    (#name,x,y,alt) for RealityScan's Trajectory/Flight-Log import
    (WORKFLOW -> Import Metadata -> Trajectory), derived from an
    already-computed vine360 COLMAP reconstruction. Position only, no
    orientation: RealityScan's own "Position" absolute-pose state is a
    real, independently-valid prior (not a degraded fallback), and this
    sidesteps needing to reverse-engineer RealityScan's yaw/pitch/roll
    convention with no real install to verify against (adapter-honesty
    discipline -- see the module docstring / docs/adr/0033). Coordinate
    frame is whatever vine360's own SfM run used -- arbitrary/
    uncalibrated is fine, since RealityScan runs its own alignment from
    scratch; the prior only needs internal consistency, not absolute
    correctness.

    Only writes a row for a registered COLMAP image whose corresponding
    flattened export filename is in exported_flat_names (the set of
    files this export run actually copied) -- an sfm run that only
    partially overlaps the exported image set is fine (RealityScan's own
    "Unknown" prior-state default already covers an image with no row).
    Returns (path, num_rows_written, warning_or_None); (None, 0,
    warning) if there was nothing to write.

    image_source="projections": COLMAP's registered image.name IS the
    path relative to projections/, so it maps to a flattened export name
    directly via _flatten_to_unique_filename -- no composition needed.

    image_source="frames": COLMAP's registered image.name is a raw
    equirect frame's filename relative to frames/<frame_set_id>/. Every
    cube-face view rendered from that frame shares the panorama's own
    optical center (zero-translation fixed_rotation, see
    vine360.projection.cubemap), so composing the frame's pose with a
    face's fixed_rotation is a translation no-op -- every view under one
    frame gets that frame's own projection_center() directly, with no
    need to touch fixed_rotation, Pose, or compose() at all (verified
    for real against a synthesized pycolmap reconstruction while
    building this). Frame filenames are matched back to frame_id via the
    frames table's own `path` column (populated at extraction time),
    rather than by parsing the "frame_NNNNNN.png" naming convention."""
    reconstruction = pycolmap.Reconstruction(model_dir)
    image_source = config.get("image_source", "projections")
    rows: list[tuple[str, float, float, float]] = []

    if image_source == "projections":
        for image in reconstruction.images.values():
            flat_name = _flatten_to_unique_filename(Path(image.name))
            if flat_name in exported_flat_names:
                x, y, z = image.projection_center()
                rows.append((flat_name, x, y, z))
    elif image_source == "frames":
        run_frame_set_id = _run_frame_set_id(config)
        if not run_frame_set_id:
            return None, 0, "camera priors: sfm run config missing frame_set_id for image_source='frames' -- skipped"
        frame_id_by_path = {
            Path(path).as_posix(): frame_id
            for frame_id, path in conn.execute(
                "SELECT frame_id, path FROM frames WHERE frame_set_id = ?", (run_frame_set_id,)
            ).fetchall()
        }
        flat_names_by_frame_id: dict[str, list[str]] = {}
        for frame_id, image_path in conn.execute(
            "SELECT v.frame_id, v.image_path FROM views v JOIN frames f ON f.frame_id = v.frame_id "
            "WHERE f.frame_set_id = ?",
            (run_frame_set_id,),
        ).fetchall():
            flat_name = _flatten_to_unique_filename(Path(image_path).relative_to("projections"))
            if flat_name in exported_flat_names:
                flat_names_by_frame_id.setdefault(frame_id, []).append(flat_name)

        for image in reconstruction.images.values():
            full_rel = (Path("frames") / run_frame_set_id / image.name).as_posix()
            frame_id = frame_id_by_path.get(full_rel)
            if frame_id is None:
                continue
            x, y, z = image.projection_center()
            for flat_name in flat_names_by_frame_id.get(frame_id, []):
                rows.append((flat_name, x, y, z))
    else:
        return None, 0, f"camera priors: unrecognized image_source in sfm run config: {image_source!r} -- skipped"

    if not rows:
        return None, 0, "camera priors: the SfM run had no overlap with the exported image set -- no CameraPriors.csv written"

    priors_path = output_dir / "CameraPriors.csv"
    with priors_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#name", "x", "y", "alt"])
        for flat_name, x, y, z in sorted(rows):
            writer.writerow([flat_name, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

    warning = None
    total = len(exported_flat_names)
    if len(rows) < total:
        warning = (
            f"camera priors: wrote {len(rows)}/{total} exported image(s) to CameraPriors.csv "
            f"({total - len(rows)} had no matching pose in the SfM run)"
        )
    return priors_path, len(rows), warning


def _migrate_legacy_export_layout(capture_dir: Path) -> None:
    """Migrates a pre-nested-layout export (images/masks/sparse directly
    under capture_dir, from before the <capture>/<format>/ layout
    existed) into capture_dir/<format>/ -- called once at the top of
    every export function, before it writes its own nested output_dir
    (a sibling of the legacy flat folders, one level down from
    capture_dir). Infers which format the flat folder was from its own
    contents: sparse/ present -> "colmap", images/ or masks/ with no
    sparse/ -> "postshot" -- the only two modes that used to write this
    flat shape (no version of this module ever wrote a flat
    RealityScan-shaped export, so that format is never inferred here).
    Conservative, matching this project's "never guess under ambiguity"
    precedent (vine360.sfm.repair_selected_model,
    QueueManager._resolve_prerequisites): a no-op if there's no legacy
    flat folder, if the inferred nested target already exists (won't
    clobber or merge), or if neither recognized legacy shape matches.

    Special case: the old all-sources default was literally named
    "postshot" at the exports/ root -- every mode shared that one path,
    since per-source capture folders didn't exist before ADR 0026. The
    new all-sources default is "all" (docs/adr/0028), a folder that
    never existed pre-nesting, so a legacy flat export headed for
    exports/all/<format>/ is actually sitting at the sibling
    exports/postshot/ instead -- check there in that one case."""
    legacy_dir = capture_dir
    if capture_dir.name == "all" and capture_dir.parent.name == "exports":
        legacy_default = capture_dir.parent / "postshot"
        if legacy_default.exists():
            legacy_dir = legacy_default

    present = {name: legacy_dir / name for name in _MANAGED_SUBDIRS if (legacy_dir / name).exists()}
    if not present:
        return
    if "sparse" in present:
        inferred_format = "colmap"
    elif "images" in present or "masks" in present:
        inferred_format = "postshot"
    else:
        return

    target = capture_dir / inferred_format
    if target.exists():
        return  # already migrated (or the new format folder was written first) -- don't clobber

    target.mkdir(parents=True)
    for name, path in present.items():
        shutil.move(str(path), str(target / name))
    if legacy_dir != capture_dir and not any(legacy_dir.iterdir()):
        legacy_dir.rmdir()  # clean up the now-empty old exports/postshot/ folder itself


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
    # RealityScan-only camera-priors CSV -- see export_for_realityscan's
    # include_camera_priors parameter. Zero/None for every other export.
    num_priors: int = 0
    priors_path: Path | None = None


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
    _migrate_legacy_export_layout(output_dir.parent)

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
        run_frame_set_id = _run_frame_set_id(config)
        if not run_frame_set_id:
            raise PostshotExportError(f"sfm run {found_run_id!r} used image_source='frames' with no frame_set_id")
        source_root = project_root / "frames" / run_frame_set_id
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
    frame_set_id: str | None = None,
    progress_callback=None,
) -> PostshotExportResult:
    """Exports every generated projection view and its keep-mask (if
    built) with no pose data at all, for Postshot to run its own SfM on
    -- see the module docstring for when to use this instead of
    export_for_postshot. Needs only project/views (Projection having
    run); masks are optional. Raises PostshotExportError if no
    projections exist yet.

    frame_set_id, if given, restricts the export to that one frame set's
    views instead of every view in the project (the default, frame_set_id=None
    -- matches this function's original behavior for callers that don't
    pass it). export_for_postshot (the "poses" mode) has no equivalent
    filter: its image set is whatever the chosen SfM run already covers,
    and narrowing that after the fact would mean editing images out of
    the COLMAP reconstruction itself, which isn't supported here.

    progress_callback(message, current, total), if given, is called once
    per view copied -- see export_for_postshot's docstring for why this
    matters at real project scale (thousands of files)."""
    notify = progress_callback or (lambda *a: None)
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    _migrate_legacy_export_layout(output_dir.parent)

    if frame_set_id is not None:
        rows = conn.execute(
            "SELECT v.image_path FROM views v JOIN frames f ON f.frame_id = v.frame_id "
            "WHERE f.frame_set_id = ?",
            (frame_set_id,),
        ).fetchall()
        if not rows:
            raise PostshotExportError(
                f"no projected views found for frame set {frame_set_id!r} -- generate projections for it first"
            )
    else:
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


def export_for_realityscan(
    conn: sqlite3.Connection,
    project_root: Path,
    output_dir: Path,
    *,
    frame_set_id: str | None = None,
    include_camera_priors: bool = False,
    progress_callback=None,
) -> PostshotExportResult:
    """Exports every generated projection view and its keep-mask (if
    built) with no pose data, for RealityScan to run its own alignment
    on -- same prerequisite, error messages and frame_set_id filter as
    export_frames_and_masks_for_postshot (see its docstring). Masks go
    into a dedicated images/layers/.mask/ subfolder using the *same*
    flattened filename as their color image (no extra suffix -- the
    subfolder itself is what signals "this is a mask" in RealityScan's
    scheme), rather than Postshot's separate top-level masks/ directory
    or flat-adjacent-in-images/ (RealityScan's *other* documented
    convention, used here up through an earlier version of this
    function -- switched away from it so a plain listing of images/
    stays mask-free, e.g. for picking just the images for a separate
    Postshot import, while RealityScan itself still auto-resolves each
    mask by filename the same way it would the flat-adjacent form; see
    docs/adr/0028). **Unverified against a real RealityScan install**
    (not available in this environment) -- built from RealityScan's
    published documentation only, same honesty-discipline caveat as
    export_for_postshot (docs/adr/0018).

    include_camera_priors, when True, also writes output_dir/
    CameraPriors.csv -- position-only camera priors drawn from vine360's
    own already-completed SfM run (if one exists), for RealityScan's
    real "Camera Priors" pre-alignment feature (see
    _write_camera_priors_csv's docstring and docs/adr/0033). Never
    raises on its own: no usable run, a run whose model is missing on
    disk, or a run with no overlap with the exported image set all just
    append a warning and leave the images+masks export itself untouched.

    progress_callback(message, current, total), if given, is called once
    per view copied."""
    notify = progress_callback or (lambda *a: None)
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    _migrate_legacy_export_layout(output_dir.parent)

    if frame_set_id is not None:
        rows = conn.execute(
            "SELECT v.image_path FROM views v JOIN frames f ON f.frame_id = v.frame_id "
            "WHERE f.frame_set_id = ?",
            (frame_set_id,),
        ).fetchall()
        if not rows:
            raise PostshotExportError(
                f"no projected views found for frame set {frame_set_id!r} -- generate projections for it first"
            )
    else:
        rows = conn.execute("SELECT image_path FROM views").fetchall()
        if not rows:
            raise PostshotExportError("no projected views found -- generate projections first")

    keep_root = project_root / "masks" / "keep"
    has_masks = keep_root.exists() and any(keep_root.rglob("*.png"))

    notify("Clearing any previous export in this folder…", None, None)
    _reset_managed_subdirs(output_dir)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    # RealityScan's dedicated-subfolder mask convention: same filename as
    # the color image, no extra suffix -- created lazily, only if a mask
    # actually gets written (mirrors export_frames_and_masks_for_postshot's
    # own masks_out handling).
    masks_out = images_out / "layers" / ".mask"

    warnings: list[str] = []
    num_images = 0
    num_masks = 0
    exported_flat_names: set[str] = set()
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
        exported_flat_names.add(flat_name)

        if has_masks:
            keep_src = colmap_mask_path(relative_to_projections, keep_root)
            if keep_src.exists():
                masks_out.mkdir(parents=True, exist_ok=True)
                shutil.copy2(keep_src, masks_out / flat_name)  # same name as its color image
                num_masks += 1

    if not has_masks:
        warnings.append("no masks have been built yet -- exporting images only")
    elif num_masks == 0:
        warnings.append("masks/keep/ exists but no mask matched any exported image")

    priors_path: Path | None = None
    num_priors = 0
    if include_camera_priors:
        notify("Looking for an SfM run to draw camera priors from…", None, None)
        selection = _select_priors_run(conn, frame_set_id=frame_set_id)
        if selection is None:
            warnings.append("camera priors requested but no usable SfM run was found -- skipped")
        else:
            found_run_id, selected_model, config = selection
            model_dir = project_root / selected_model
            missing = [name for name in _COLMAP_MODEL_FILES if not (model_dir / name).exists()]
            if not model_dir.exists() or missing:
                warnings.append(
                    f"camera priors requested but sfm run {found_run_id!r}'s model is missing on disk -- skipped"
                )
            else:
                priors_path, num_priors, priors_warning = _write_camera_priors_csv(
                    conn, model_dir, config, output_dir, exported_flat_names
                )
                if priors_warning:
                    warnings.append(priors_warning)
    notify("Export complete.", None, None)

    return PostshotExportResult(
        output_dir=output_dir,
        images_dir=images_out,
        masks_dir=masks_out if num_masks else None,
        num_images=num_images,
        num_masks=num_masks,
        warnings=warnings,
        num_priors=num_priors,
        priors_path=priors_path,
    )
