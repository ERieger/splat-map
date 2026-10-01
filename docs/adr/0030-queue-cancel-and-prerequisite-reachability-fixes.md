# 0030. Fix: Cancel All didn't clear the queue, and four panels made the queue's own
prerequisite auto-insertion unreachable

## Status

Accepted.

## Context

Three real, user-reported bugs in the persisted processing queue (docs/adr/0023):

**1. "Pressing cancel continues processing and doesn't clear the queue."** `QueueManager.cancel_all()`
only flipped `QUEUED`/`BLOCKED` jobs to `SKIPPED` in place -- it never removed them from `self.jobs`
or the `queue_jobs` table, so they stayed visibly sitting in `QueuePanel`'s list forever, which does
not read as "cleared." A `RUNNING` job was, and still is, left completely untouched: `pause()`'s own
docstring already said so plainly ("Doesn't interrupt a job already running -- run_in_background's
workers have no cancellation hook"), and that's architecturally real, not an oversight --
`LocalRunner` (docs/adr/0002) wraps ffmpeg as a fully synchronous, blocking `subprocess.run` with no
incremental-output or interrupt hook, and pycolmap/SAM3 calls are in-process Python with no process
handle to kill at all. None of this was documented in ADR 0023 itself, had zero test coverage
(confirmed by grep -- no hits for "cancel" or "pause" anywhere in `tests/test_gui_queue_manager.py`),
and was independently flagged as a gap by `docs/critical-review-2026-09-22.md`.

**2 & 3. "Unable to queue dependent steps until they have finished processing -- I should be able to
queue mask on frames that haven't been generated, as loading the source should propagate through"**
and **"Selecting a source that hasn't got a dependency should disable the process now button."**
`QueueManager._resolve_prerequisites` already implements exactly this: queuing Masks for a source
with frames-but-no-views auto-inserts a Projection job ahead of it (and, transitively, a Frames job
if even frames are missing) -- fully implemented and tested at the `QueueManager` level
(`test_masks_job_without_views_auto_inserts_frames_and_projection` already covered it). But it was
unreachable from the actual GUI: `ProjectionPanel.refresh_sources`, `MasksPanel.refresh_sources`,
`PoseEstimationPanel._refresh_sources` (equirectangular-engine combo), and
`ExportPanel._refresh_sources` (`frames_masks`/`realityscan` modes) all populated their source
dropdown with an **INNER JOIN** against the upstream table, silently excluding any source that
didn't have the prerequisite yet -- so the user could never select that source to click "Add to
Queue" on it in the first place. `FramesPanel` already did this correctly: every video source is
listed regardless of readiness (a scalar subquery for the count, not a JOIN that would hide
zero-count rows), and only the immediate "Run now" button -- not "Add to Queue" -- is
readiness-gated.

## Decision

**Cancel All actually empties the pending backlog.** `cancel_all()` now calls the existing
`remove_job()` (which already refuses to touch a `RUNNING` job, and already persists the deletion)
for every `QUEUED`/`BLOCKED` job, instead of relabeling them `SKIPPED` in place. A currently-`RUNNING`
job still can't be interrupted -- that's unchanged, and out of scope here; doing so for real would
mean redesigning `Runner`/`LocalRunner` to track process handles and adding a cross-thread
cancellation signal, a separate, larger architectural change, not attempted in this fix. Instead, a
new `_cancel_requested` flag records that the running job should be removed too, the moment its own
`_on_job_success`/`_on_job_error` fires naturally (the work genuinely completes or fails for real --
nothing is corrupted or half-written) -- so the queue does end up genuinely empty once that job
finishes, honoring "clear the queue" literally without claiming a fake interrupt. `QueuePanel`
also now posts an explicit status message ("Clearing queue -- the job currently running can't be
interrupted and will finish first.") when Cancel is pressed while a job is running, rather than
leaving that gap silent.

Deliberately scoped to the *pending* backlog only: already-terminal jobs (`DONE`/`FAILED`/`SKIPPED`)
are left alone, matching `retry_job`/`skip_job`'s existing habit of only ever touching non-terminal
jobs -- Cancel All clears what's still going to run, not the session's history.

**The four panels' source-selection combos are broadened to match `FramesPanel`'s existing, correct
pattern**, generalized rather than reinvented: every eligible source is listed (INNER JOIN -> LEFT
JOIN, with an explicit `WHERE s.media_type = 'video'` filter replacing what the INNER JOIN was
previously doing implicitly), with each source's relevant prerequisite count tracked per-panel
(`_source_frame_counts`/`_source_view_counts`) and shown in the combo item text even when zero (e.g.
"0 frames, no views yet") -- "loading the source" now really does "propagate through" to the queue.
"Add to Queue" stays enabled based on "any source exists at all," unchanged and still correct, since
`_resolve_prerequisites` always auto-inserts unambiguously once a specific `source_id` is given (the
ambiguity-block path in `_resolve_project_wide_projection_prerequisite` only ever triggers when
`source_id is None`, i.e. `ExportPanel`'s "All sources" choice). The immediate "Run now" button
(`generate_btn`/`build_btn`/`run_btn`/`export_btn`) is where the new gating lands: it's now disabled
whenever the *currently selected* source lacks the immediate prerequisite, with a tooltip pointing at
"Add to Queue" as the alternative -- this is the literal fix for "selecting a source that hasn't got
a dependency should disable the process now button." `PoseEstimationPanel`'s six-face-projections
engine branch and `ExportPanel`'s poses mode are unaffected (their own existing project-wide/
run-scoped checks were already correct and don't filter by a single selected source).

## Consequences

- **New, intentional asymmetry**: "Add to Queue" only requires *a* source to exist somewhere; "Run
  now" requires *the selected* source to already be ready. This is a real behavior change from
  before (where an unready source was often simply absent from the dropdown, so the distinction
  didn't visibly exist) -- worth remembering as intentional the next time this code is touched, not
  something to "fix" back into hiding unready sources again.
- `PoseEstimationPanel._refresh_sources` gained an explicit `AND s.media_type = 'video'` filter that
  the old INNER JOIN on `frames` provided implicitly (only video sources ever get `frames` rows) --
  needed now that the JOIN no longer does that filtering as a side effect, otherwise a non-video
  source with `projection = 'equirectangular'` could wrongly appear and offer to auto-queue a
  video-only frame-extraction job.
- `ExportPanel`'s source combo gained count text ("N views" / "no views yet") it never had before,
  for consistency with the other three panels' item text.
- Real mid-job interruption (a genuine kill of an in-flight ffmpeg/pycolmap/SAM3 call) remains
  unimplemented and undocumented as a plan -- if it's wanted later, it needs its own ADR and touches
  `Runner`/`LocalRunner` (docs/adr/0002), not just `QueueManager`.
