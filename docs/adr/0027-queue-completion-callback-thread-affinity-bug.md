# 0027. Fix: queue completion callbacks silently ran on the wrong thread

## Status

Accepted.

## Context

Found on a real, live project (`/mnt/e/11-9-26_EstoWines_Capture1/A1_InstaLow`): a queued Projection
job on 604 real frames (Insta360_EyeLevel.mp4) genuinely finished -- all 3020 views were correctly
written to the database, matching the final "Projected 604 frames (3020 views)" progress message --
but the Queue dock's row never turned green, the progress bar stayed frozen at 100%, and the queue
never advanced to any later job. The app's log had the answer:

```
sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread.
```

`QueueManager._start_job` (`vine360/gui/queue_manager.py`) wired the background worker's completion
callbacks as bare lambdas:

```python
on_success=lambda result, job_id=job.job_id: self._on_job_success(job_id, result),
on_error=lambda exc, job_id=job.job_id: self._on_job_error(job_id, exc),
```

Qt's automatic cross-thread signal queuing only kicks in when the *connected receiver itself* is a
`QObject` with known thread affinity -- a bound method of a `QObject` subclass (e.g.
`FramesPanel._on_extract_success`, the pattern every manual panel already used) qualifies; a bare
lambda does not, even though this one closes over `self` (a `QObject`). Without that affinity
information, Qt has no reason to queue the call onto the main thread, so it ran `_on_job_success`
synchronously on the background `QThread` that had just finished the real work. `_on_job_success`
immediately touches `self.state.conn` -- a `sqlite3.Connection` created on the main thread -- and
Python's `sqlite3` module refuses cross-thread use by default. The resulting exception aborted the
handler right after `job.status = DONE` was set *in memory* but before `self._persist(job)` (the
line that raised), `self._running_job_id = None`, `notify_change()`, or either `job_finished`/
`queue_changed` ran -- so the job was permanently stuck showing `RUNNING` in the persisted
`queue_jobs` table and the UI, with the queue itself wedged (`is_running` never cleared, so
`Start Queue` stayed disabled and no later job could run) even though its actual work had long
since completed correctly.

## Decision

`_start_job` now connects `on_success`/`on_error`/`on_progress` to real bound methods of `self`
(`_handle_job_success`, `_handle_job_error`, `_handle_job_progress`) instead of lambdas. Bound
methods of a `QObject` subclass carry real thread affinity, so Qt correctly queues their execution
onto the main thread, matching the exact pattern every manual panel's own `on_success=self.
_on_extract_success`-style wiring already relied on. Since only one job runs at a time, the wrapper
methods read `self._running_job_id` instead of needing `job_id` threaded through a closure, which
is what motivated the lambda in the first place.

Added a real regression test, `tests/test_gui_queue_manager.py::
test_real_dispatch_persists_done_status_without_a_cross_thread_error` -- unlike this file's other
tests (which patch `QueueManager._start_job` itself to test the state machine in isolation), this
one drives the real `_start_job` -> `run_in_background` -> `QThread` path end to end with a real
Qt event loop pumped via `QEventLoop`/`QTimer`, specifically because the bug lived in that exact
path and no other test exercised it (docs/critical-review-2026-09-22.md had already flagged this
as a real coverage gap, independently, before this bug surfaced).

## Consequences

- Any project with a job that ran through the queue before this fix may have a `queue_jobs` row
  stuck at `status='running'` with `finished_at` unset, even though the underlying work (frames/
  views/masks/SfM run/export) completed correctly and is safe to use. No data was corrupted --
  only the queue's own bookkeeping for that one job. On next load, `QueueManager.load()`'s existing
  running-on-load handling (docs/adr/0023) will mark such a row `FAILED` ("interrupted by app
  restart") rather than `DONE`, since it can't distinguish "genuinely interrupted" from "finished
  but never got to say so" -- retrying it is safe (every worker is idempotent on rerun) and quick,
  since the underlying data already exists.
- This is the second real bug this session traced to the `STAGE_*`/thread-affinity family of Qt/
  Python integration footguns (see docs/adr/0023's `QUEUE_STAGE_*` aliasing fix for the first). The
  critical review (docs/critical-review-2026-09-22.md) had already flagged the queue's own module
  docstring claim about the `on_progress` deferred-import comment being slightly overstated; the
  underlying pattern -- "a lambda closing over `self` looks like it should behave like a bound
  method to a Qt connection, but doesn't" -- is worth remembering the next time this module's
  dispatch code changes.
