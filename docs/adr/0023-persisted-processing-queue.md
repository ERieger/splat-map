# 0023. A persisted, dependency-aware processing queue

## Status

Accepted.

## Context

Every pipeline stage (Frames, Projection, Masks, Pose, Export) previously ran only when a user
clicked that stage's own "Run now" button, one action at a time. There was no way to line up a
multi-stage or multi-source run and let it go unattended -- confirmed by grep, there was no
queue/batch/pipeline-orchestration concept anywhere in the codebase.

The user asked for a queue, with several concrete requirements worked out during planning:

- Manual single-step running must keep working exactly as before -- the queue is additive, not a
  replacement UI.
- The queue lives in a new right-hand dock panel (`QDockWidget`, `Qt.RightDockWidgetArea`) -- no
  dock widgets existed in the app before this, so it doesn't disturb the existing sidebar+stack
  `QSplitter`.
- The queue is **persisted** to the project's own `index.sqlite` (a new `queue_jobs` table), not
  kept in memory only, so an unattended overnight run survives an app crash/restart.
- Queuing a stage whose input data doesn't exist yet (e.g. a Masks job on a source with no
  projections) should not just fail -- the missing upstream job(s) should be queued automatically
  ahead of it, using the same defaults the manual panel would.
- **Failure isolation is per-chain, not global**: if one job fails, only the jobs that actually
  depend on its output should be blocked. An independent source's queued chain (no shared
  dependency) must keep running. The failed job stays visible with Retry / Skip / Remove actions.

That last requirement is the one with real design weight: it rules out a flat, strictly linear
queue ("run item 1, then 2, then 3, stop on first failure"), because a linear model has no way to
distinguish "the next job needs the failed one's output" from "the next job just happens to be
listed after it."

## Decision

**Dependency edges, not just list order.** Each `QueueJob` (`vine360/gui/queue_manager.py`)
records `depends_on: list[job_id]` -- the specific job(s) it needs `DONE` before it can run, not
merely "whatever's above it in the list." `QueueManager._run_next` scans the whole ordered list
for the first `QUEUED` job whose dependencies are all satisfied, rather than only ever looking at
the front of the list. When a job fails, `_revalidate_dependencies` walks the graph and flips only
its transitive dependents to `BLOCKED`; every other job -- in particular a different source's
queued chain, which shares no dependency edge -- stays eligible to run next. This is the whole
mechanism behind "one source's failure doesn't stall the rest of the queue."

**Prerequisite auto-insertion, never guessed under ambiguity.** `QueueManager._resolve_prerequisites`
checks the project's current DB state (same counts `compute_stage_statuses` already uses) plus
what's already queued. If a stage's input is missing and there's exactly one unambiguous upstream
job to insert (e.g. a single source with extracted frames but no views, when queuing Masks for
it), it's inserted automatically, marked `auto_added=True`, with default params matching what the
manual panel would show. When the right choice would be a guess -- no candidate source, or several
-- the job is added as `BLOCKED` with an explanatory `error_message` instead of guessing. This
follows the same "never guess under ambiguity" precedent already set by
`vine360.sfm.repair_selected_model` (docs/adr/0019).

**Persisted to the project DB, not kept in memory.** A new `queue_jobs` table (added to
`_ensure_schema` in `vine360/project.py`, same idempotent-migration pattern as `views`/`masks`/
`sfm_runs`) stores the full plan: stage, target source, params (JSON), dependency edges (JSON),
status, and timing. On project open, any row loaded with `status='running'` is reset to `failed`
with `error_message="interrupted by app restart"` -- a job that was mid-`ffmpeg`/`pycolmap`/SAM3
when the app last closed can't be trusted to have finished cleanly, but it's always safe to retry:
every stage's worker is already idempotent on a fresh call (`clear_frames_for_source`/
`clear_views_for_frame` cascade-delete on re-extraction/re-projection, `INSERT OR REPLACE` for
masks, a fresh `run_id`/subdirectory per SfM run, managed-subdir reset on export -- see docs/adr/
0017 and 0019), just none of them support *resuming* a half-written call.

**All job parameters captured at "Add to Queue" time, never re-prompted at run time.** Every stage
panel (`FramesPanel`, `ProjectionPanel`, `MasksPanel`, `PoseEstimationPanel`, `ExportPanel`) gained
an "Add to Queue" button beside its existing "Run now" button, reading the exact same form
fields -- including the two real branch points, Pose estimation's engine choice (six-face
projections vs. native equirectangular; SphereSfM stays excluded, same as the manual panel) and
Export's mode choice (import vine360's own SfM run vs. images+masks-only) -- and packaging them
into a `QueueJob` instead of calling `run_in_background` immediately. This is deliberate: it's how
those branches are represented in the interface. The branch choice is made once, with the combo
box the manual flow already has, and the resulting job's `label` renders that choice in plain
text (e.g. "Pose estimation — COLMAP (equirectangular)"), so a queued plan is self-describing
without opening each job. No second parameter-entry UI was built, and nothing pops a dialog (e.g.
Export's output-folder picker) once the queue is running, since it may run unattended.

One specific case needed lazy resolution rather than a captured value: an Export(poses) job added
while its Pose job is still queued (not yet run) can't know a concrete `run_id` yet. Its params
store `run_id=None` and its `depends_on` points at that Pose job; `QueueManager._start_job`
resolves `run_id` as "the most recent `sfm_runs` row" only once that dependency is `DONE` --
correct by construction, since that Pose job will just have completed and be the newest row.

**Execution reuses `run_in_background` and every existing worker function unmodified.**
`QueueManager` runs exactly one job at a time by calling the same `_extract_frames_worker`/
`_generate_views_worker`/`_build_masks_worker`/`_run_sfm_worker`/`_export_postshot_worker`/
`_export_frames_and_masks_worker` functions the manual panels already call, through the same
`run_in_background`. (A deferred `from vine360.gui import main_window as mw` inside
`QueueManager._start_job` avoids a module-load-time import cycle, since `main_window.py` also
imports `QueueManager`.) Each stage panel's manual "Run now" button -- not "Add to Queue" -- is
additionally disabled while `queue_manager.is_running`, so a manual run can't race a queued one on
the same sqlite connection or output directories; this is the only change to the existing
single-worker-per-action pattern, and it deliberately leaves `run_in_background`'s own
multi-worker-per-owner support untouched.

## Consequences

- The right-hand dock (`QueuePanel`) shows every job with a status dot (a new `QUEUED`/`RUNNING`/
  `DONE`/`FAILED`/`BLOCKED`/`SKIPPED` vocabulary, deliberately kept separate from the sidebar's
  `DONE`/`ACTIVE`/`PENDING` stage-status vocabulary even though both render with the same dot
  idiom), drag-to-reorder, and per-row Retry/Skip/Remove. Reordering or removing a job
  re-validates the whole dependency graph, so a `QUEUED` job whose upstream job got moved after it
  or removed correctly flips to `BLOCKED` instead of running against stale/missing data.
- A queue is scoped to one project's `index.sqlite`, matching the app's existing one-project-at-a-
  time model.
- Import (add-source) and Training stay outside the queue: Import needs a file picker per source,
  which doesn't fit an unattended run; Training is adapter-only with no backend to run at all
  (docs/adr/0009).
- This intentionally does not add a generic multi-worker/concurrent-execution model -- the queue
  still runs strictly one job at a time. Running two *independent* chains' jobs concurrently (e.g.
  two sources' Frame extractions in parallel) was considered out of scope; the failure-isolation
  behavior only concerns *ordering and blocking*, not parallel execution.
