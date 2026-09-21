# 0020. Flatten exported filenames to avoid ambiguous mask association

## Status

Accepted.

## Context

The user reported that exported masks weren't associating correctly in
Postshot -- ambiguous filenames. Investigating confirmed a real gap in
both export modes (docs/adr/0018, docs/adr/0019): every vine360 frame's
six-face projections share the same five leaf filenames (`front.png`,
`back.png`, `left.png`, `right.png`, `down.png`) -- disambiguated on
disk only by which `<frame_id>/` folder they sit in. That's fine for
vine360's own use and for COLMAP import (COLMAP matches `images.bin`
entries by their full relative path, not basename), but Postshot's own
docs are explicit that mask association goes by filename alone: "In
order to be associated with the correct color image, the mask image must
have the same base filename." Nothing in the exported folder previously
made `front.png` unique across hundreds of frames -- only the directory
structure did, which this particular matching rule doesn't consult.

Separately, real vine360 frame_ids can contain `:` (e.g.
`<source_id>:000042`, confirmed on `/mnt/e/TEST`) -- tolerated in a
directory name under WSL's drvfs translation, but not valid in a bare
Windows filename. Flattening a path containing `:` into a single
filename would hit this too if not handled.

## Decision

New `_flatten_to_unique_filename(relative_path)` in
`vine360/export/postshot.py`: joins a relative path's parts with `__`
and replaces `:` with `_` (e.g. `source-a:000042/front.png` ->
`source-a_000042__front.png`) -- a deterministic, collision-free,
Windows-filename-safe flattening of the original nested path. Both
export functions now write every image and mask under this flat name
directly in `images/`/`masks/` (no per-frame subdirectories), instead of
mirroring the original `<frame_id>/<face>.png` structure:

- `export_frames_and_masks_for_postshot`: straightforward -- flatten the
  destination filename for both the image and its mask copy.
- `export_for_postshot`: same, but the flattened name must also be
  written back into the exported COLMAP model, or Postshot's COLMAP
  import can't locate the (renamed) image files at the paths
  `images.bin` references. `pycolmap.Image.name` is a plain mutable
  attribute (confirmed by round-tripping a renamed reconstruction
  through `.write()`/`Reconstruction()`), so this renames every image in
  the loaded reconstruction in place before re-writing the sparse model
  to the export folder, replacing the previous plain `shutil.copy2` of
  the three COLMAP files.

## Consequences

- Verified with real tests: two different frames' same-named face
  (`front.png`) now produce two distinct, non-colliding files in both
  export modes; for the poses mode, the re-written `images.bin` (reloaded
  via `pycolmap.Reconstruction`) references exactly the flattened
  filenames the files were actually written under, not the stale nested
  originals.
- Exported filenames no longer resemble vine360's internal project
  layout at a glance (e.g. `source-a_000042__front.png` instead of
  `source-a:000042/front.png`), but nothing outside this export step
  depends on that mirroring -- it exists purely to satisfy Postshot's
  own matching rule.
- Masks and poses mode already reset their output subdirectories before
  writing (ADR 0019's fix), so this doesn't reintroduce any stale-file
  risk from the rename.
