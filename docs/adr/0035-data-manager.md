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
