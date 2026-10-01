# 0026. Source selection for the images+masks export mode only

## Status

Accepted.

## Context

Both export modes previously pulled from every source in the project with no filter:
`export_frames_and_masks_for_postshot`'s `SELECT image_path FROM views` had no `WHERE` clause at
all, and `export_for_postshot`'s (poses mode) image set was whatever the chosen SfM run happened
to cover -- itself often multi-source, since the default six-face-projections Pose engine runs
against `project/projections/` as a whole, not scoped to one source. The user wanted the ability to
export a single chosen source (or "All", matching today's behavior) instead.

Filtering the poses-mode export by source turned out to be a real blocker: doing it correctly means
removing images from an already-computed `pycolmap.Reconstruction` before writing `sparse/`, and
`pycolmap.Reconstruction` has no direct "remove this image" method in its API -- a naive filter
(copying fewer image files while leaving `images.bin` referencing all of them) would produce a
COLMAP dataset with broken references, worse than the current unfiltered behavior. The user
confirmed this isn't worth solving right now: they mainly care about the images+masks-only (raw
frames) export mode, and are fine leaving poses-mode export unfiltered.

## Decision

- `export_frames_and_masks_for_postshot` (`vine360/export/postshot.py`) gained an optional
  `source_id: str | None = None` parameter. `None` (the default) keeps the original all-sources
  behavior; a given source_id filters the `views`/`frames` join to that source only, with an
  actionable error (`no projected views found for source {source_id!r} -- generate projections for
  it first`) if that source has no views yet.
- `export_for_postshot` (poses mode) is **unchanged** -- no source filter, per the user's explicit
  direction not to worry about it.
- `ExportPanel` gained a **Source** combo ("All sources" + one entry per source that has at least
  one projected view, same `JOIN`-based filtering `MasksPanel`/`ProjectionPanel` already use for
  their own source pickers). It's only meaningful for the "Images + masks only" mode -- disabled
  (with an explanatory tooltip) when "Poses + images + masks" is selected, since that mode ignores
  it entirely.
- **Output folder default follows the source selection**: choosing a specific source defaults
  `output_dir` to `exports/<source-file-stem>/` instead of the shared `exports/postshot/`, so
  exporting source B later doesn't wipe source A's output via `_reset_managed_subdirs`. "All"
  keeps using the shared `exports/postshot/` default, matching prior behavior exactly. This is a
  *live* default (`ExportPanel._update_output_dir`) that keeps following mode/source changes right
  up until the user manually picks a folder via "Choose output folder…", at which point it's
  pinned and stops auto-updating -- the same "smart default vs. explicit override" pattern already
  implicit in how output_dir worked before this change.
- The queue's prerequisite auto-insertion (`QueueManager._resolve_prerequisites`,
  `vine360/gui/queue_manager.py`) is now source-aware for `frames_masks`-mode export jobs: given a
  specific `source_id`, it checks/auto-inserts a Projection job for *that* source precisely
  (mirroring the existing Masks-stage logic), instead of falling back to the old project-wide
  "guess only if there's a single unambiguous candidate" path -- which is still used when the
  export job's source is "All" (`source_id` is `None`).

## Consequences

- A user who only wants to hand one source off to Postshot no longer has to manually sort mixed
  images out of a combined export folder.
- Poses-mode export remains a known, documented gap (still tracked, not solved here): it exports
  whatever the chosen SfM run covers, with no way to narrow it further. Revisit only if a real need
  arises to filter it, since the fix requires either rebuilding a `pycolmap.Reconstruction` from
  scratch with a subset of images, or scoping the underlying SfM run itself to one source up front
  (already possible today via the native-equirectangular Pose engine, which *is* single-source by
  construction).
