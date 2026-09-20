# 0012. Wire Projection, Masking and SfM into the GUI; unify progress
reporting; derive sidebar status from the database

## Status

Accepted.

## Context

Following a critical review of the running app: Project's "create" vs
"open" controls weren't visually separated (capture mode looked relevant
to opening too); progress bars gave no percentage/ETA even where the
underlying work has a natural count; there was no visible signal that
Import had produced anything; Projection's sidebar dot never left
PENDING because nothing executed it; and Masking/Pose estimation had no
execution at all, per the earlier explicit "no runs yet" instruction that
no longer applied once frame extraction had been proven safe.

## Decision

- **New persistence-layer modules**, each bridging a pure library
  (`vine360.projection.*`, `vine360.masking.*`, `vine360.sfm.colmap_adapter`)
  to the project's index database and filesystem layout, following the
  same shape as `vine360.ingest.frames`:
  - `vine360/projection/generate.py`: `generate_views_for_frame`/
    `_for_source`, writing `View` rows and
    `project/projections/<frame_id>/<face>.png`. Clears a frame's prior
    views (and their masks) before regenerating, same reasoning as
    frames' `clear_frames_for_source`.
  - `vine360/masking/build.py`: `build_mask_for_view`/`_for_source`,
    writing `Mask` rows and mask files under `project/masks/`. The
    keep-mask path uses `colmap_mask_path` directly, so `masks/keep/`
    doubles as COLMAP's `mask_path` once SfM runs.
  - `vine360/sfm/project_run.py`: `run_sfm_for_project`, running against
    `project/projections/` and `project/masks/keep/`, recording an
    `sfm_runs` row on success.
  - Three new tables (`views`, `masks`, `sfm_runs`) added to
    `project.py`'s index schema, per the data contracts already declared
    in `models.py`.
- **A real bug found and fixed while adding the first real test that
  called `validate_installation()`**: `pycolmap.COLMAP_version` and
  `COLMAP_build` are plain `str` attributes, not callables --
  `colmap_adapter.py` had been calling them as functions since ADR 0007,
  uncaught because nothing had exercised that function until now.
- **Unified progress-callback signature**: every function that can report
  progress now calls `progress_callback(message, current, total)`
  (`current`/`total` are `None` for phases without a natural count).
  `sha256_file` gained real byte-level progress, so `add_source`'s
  previously-opaque ~25s checksum step on a large file is now a real
  determinate progress bar. The GUI's `ProgressArea` widget (shared by
  every panel) switches between an indeterminate spinner and a
  determinate bar with an elapsed-time-based ETA depending on whether
  `current`/`total` were given.
- **Sidebar stage status is derived from the database**
  (`compute_stage_statuses`), not tracked ad hoc per panel: a stage is
  DONE once it has produced at least one real artifact, ACTIVE once its
  prerequisite exists but it hasn't, else PENDING. Every panel calls
  `state.notify_change()` after a successful action; the main window
  re-queries and repaints every dot. This is what makes Import show DONE
  once a source exists, and Projection actually reach ACTIVE while
  running (previously stuck PENDING forever, since nothing executed it).
- **Projection, Masks and Pose estimation panels** follow the same
  pattern as Frames: a source picker populated from the database (only
  sources with the right prerequisite -- e.g. Masks only lists sources
  that already have views), a background-threaded action button, and a
  `ProgressArea`. Masks lets SAM 3 be toggled per class (person/sky);
  loads the model once for the whole batch, not once per view. Pose
  estimation runs against the whole project's projections/masks at once
  (COLMAP needs real camera-position parallax across the full capture,
  not one frame at a time) and treats `SfmRegistrationError` as an
  informative, non-alarming outcome in its own dialog, not a generic
  crash message -- a single-frame's projected faces share one optical
  center and cannot be 3D-reconstructed regardless of match quality (see
  ADR 0007), which is exactly what the full pipeline smoke test hit and
  correctly reported.
- **Project panel split into two `QGroupBox` sections** ("Create a new
  project" / "Open an existing project"), each showing only its own
  relevant controls, directly addressing the "which buttons are relevant
  to what" review comment.

## Consequences

- Verified with a full offscreen pipeline smoke test (mocked file
  dialogs and message boxes, a real synthetic clip): create project ->
  add source -> extract frames -> generate projections -> build masks
  (classical) -> run SfM, checking `compute_stage_statuses` after each
  step. All transitions were correct, including Pose estimation
  correctly staying ACTIVE (not DONE) after a real, expected
  `SfmRegistrationError` from the zero-parallax synthetic test clip.
- Training stays untouched (adapter-only, per ADR 0009) -- this work
  goes up to Pose estimation and stops there, as asked.
- No GUI automated test harness (`pytest-qt` or similar) exists yet;
  verification here was manual/offscreen smoke tests during development,
  same gap noted in docs/status.md previously.
- SAM 3 availability (`sam3_validate_installation()`, which imports
  `torch`/`transformers` eagerly if installed -- roughly 140MB to 700MB+
  RSS observed) is checked lazily on the Masks panel's first `on_shown()`,
  not at app construction, so users who never open that panel don't pay
  the cost at startup.
