# 0035. Data manager tab

## Status

Accepted.

## Context

Once a source can have several frame sets (ADR 0034), and each can carry projections, masks,
SfM runs and exports, a project accumulates derived data with no single place to see it or
reclaim disk space. The only delete paths were indirect: re-running a stage, or removing a
whole source on Import.

## Decision

**A "Data manager" sidebar entry after Export.** It isn't a pipeline stage, so it has no status
dot and isn't part of `compute_stage_statuses`. It shows:
- **Sources**: read-only. Adding and removing sources stays on Import, where the source
  cascade already lives.
- **Frame sets**, grouped by source, with frame/view/mask counts and bytes on disk per kind.
  Three deletes: the whole frame set (cascades to its projections and masks), just its
  projections (and their masks, keeping the frames), or just its masks.
- **SfM runs**: delete removes the row and `sfm/sparse/<run_id>/`. Runs from before per-run
  directories (ADR 0019) point into the shared layout, so only their row is removed.
- **Export folders** (`exports/<capture>/<format>/`, plus any pre-0028 flat folder): delete
  refuses any path that doesn't resolve strictly inside `exports/`.

**All logic is in a Qt-free `vine360.data_manager`**, tested directly. Deletes reuse the
existing cascade functions (`clear_frame_set`, `clear_views_for_frame`) rather than
reimplementing them, so the ADR 0017 invariant holds in one place. Disk sizes are walked on a
background thread, since a real project can hold tens of thousands of images on a slow
Windows-mounted drive. The lists appear immediately and the sizes fill in when the walk
finishes.

**Every delete asks for confirmation and is disabled while the queue is running.** The
confirmation notes any queued jobs that still target the frame set.

## Addendum: exports written directly into exports/, and the sidebar dot

- The Data manager row gets a fixed blue dot (`DATA_MANAGER_DOT_COLOR`) rather than a blank
  spacer: not a status, but visually of a piece with the stage rows, and distinct from every
  stage/job status color (purple is already the queue's BLOCKED).
- Choosing `exports/` itself as an export's output folder is allowed, so `list_export_dirs` lists
  such an export as one `"exports"` entry (`in_exports_root=True`); its size and delete cover only
  what an export writes (`images/`, `masks/`, `sparse/`, `CameraPriors.csv`), never the
  `<capture>/` folders beside it.
- Real bug this surfaced: the export functions run `_migrate_legacy_export_layout` on
  `output_dir.parent` -- for `output_dir = exports/` that's the project root, whose own `masks/`
  looked like a flat legacy export and was moved to `<root>/postshot/masks/`, orphaning every
  mask row (shown as 0 bytes here). The migration now never touches a folder holding
  `project.yaml`, and every export function refuses a project root as `output_dir` outright
  (`_reset_managed_subdirs` would otherwise delete its `masks/`). A frame set row whose
  frames/projections/masks have rows but no files now shows "files missing" instead of "0 B".
