# 0037. Verbose progress, and a per-project activity log

## Status

Accepted.

## Context

A real SphereSfM run on 280 8K frames showed only an indeterminate bar for many minutes: no way
to tell extraction from matching from mapping, or how far each had got. Separately, nothing
recorded what had been run in a project, in what order, or with which settings. For research use,
that provenance matters as much as the outputs themselves.

## Decision

**Counted, step-labelled progress for every long operation.** This is now a standing requirement
for new features (see CLAUDE.md). For SfM:
- **SphereSfM:** `Runner.run` gained `on_output`, and `LocalRunner` streams combined
  stdout/stderr line by line. `spheresfm_adapter.ProgressParser` turns COLMAP 3.8's own log lines
  into counts:
  - `Processed file [i/N]` (feature extraction)
  - `Matching image [i/N]` (matching)
  - `Initializing with image pair` / `Registering image #id (n)` (mapping)

  Each step's full log is also kept in the run folder (`1_feature_extractor.log`,
  `2_sequential_matcher.log`, `3_mapper.log`), which helps when diagnosing a failure.
- **pycolmap engines:**
  - Mapping counts registrations through `incremental_mapping`'s `initial_image_pair_callback`
    and `next_image_callback`.
  - Extraction and matching have no Python callback. While each call runs,
    `colmap_adapter.native_log_lines` temporarily redirects file descriptor 2 to a pipe, parses
    `Processed file [i/N]` and `Processing image [i/N]`, and passes every byte through to the real
    stderr. It holds a process-wide lock, because fd 2 is shared.

Two approaches were tried for real and rejected:
- **Polling pycolmap's database** from another connection crashes the run: pycolmap doesn't use
  WAL, and COLMAP aborts the whole process on "database is locked". SphereSfM (COLMAP 3.8) does
  use WAL, so reading its database is safe, but the log-line approach makes reading it
  unnecessary.
- **Extracting in small batches** to report between them made GPU SIFT registration worse on the
  same synthetic sequence: 3 of 10 runs lost a frame, against 0 of 10 for a single call (CPU was
  0 of 10 either way).

`ProgressArea` (used by every panel and the queue dock) now does three things:
- keeps a timestamped "Details" log of every distinct progress message, collapsed by default and
  kept after the run ends
- restarts its ETA whenever a new step begins, so a multi-step run's ETA describes the current
  step
- shows the job's name in the queue dock, plus "Done:" or "Failed:" when it ends

**Activity log.** A new `activity_log` table in each project's `index.sqlite`, written by
`vine360.activity_log`. Every GUI/queue worker is wrapped in `logged_operation`, which:
- records the operation, a target (frame set, source or output folder), and every argument as
  JSON parameters (excluding `project_root` and `progress_callback`)
- records start and finish time, duration, status (`running`, `done`, `failed`, `interrupted`), a
  one-line result summary or the error, and the origin: "manual", or "queue: <job label>" passed
  by the queue as `log_origin`

Logging writes through its own short-lived connection, because workers run off the GUI thread,
and a logging failure never affects the work itself. On opening a project in the app, entries
still `running` from a previous session are marked `interrupted`, matching how the queue treats
its own RUNNING jobs (ADR 0023). Project creation is logged too.

The new **Activity log** sidebar tab is read-only. It lists entries oldest first (newest first
optionally), refreshes every few seconds while visible, shows the full parameters and result of
the selected entry, and exports to CSV. Runs from before this ADR aren't listed, since nothing
recorded them.

## Consequences

- Test expectations on the tiny synthetic 360 scene are sized from what actually registered
  (at least 5 of 6), because SfM there is legitimately borderline from run to run.
- A pycolmap call's stderr is briefly routed through a pipe. Anything else in the process that
  writes to stderr at the same moment is still passed through, not lost.
