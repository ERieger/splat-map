# 0019. Per-run SfM output directories; a repair tool for runs made before this fix

## Status

Accepted.

## Context

Follow-up to ADR 0018's `selected_model` fix: the user asked whether
past runs could be migrated to the fix rather than re-run. Checking
against the real project at `/mnt/e/TEST` (4 historical `sfm_runs` rows)
surfaced a second, more serious bug behind the first: `run_sfm_for_project`
wrote every run's output to the *same* `project/sfm/sparse/` directory.
`pycolmap.incremental_mapping` numbers its own candidate reconstructions
starting from 0 on every call, so a later run's output can land on the
same directory name an earlier run used, silently overwriting it.

Confirmed for real: loading each of the 5 numbered directories under
`/mnt/e/TEST/sfm/sparse/` and comparing their live diagnostics
(`num_reg_images`, `num_points3D`, `compute_mean_reprojection_error`)
against each run's recorded `model_stats` showed all 5 directories
belonged to the single most recent run (`sfm-e40feae0`, which itself
found 5 disconnected reconstruction fragments and kept the largest,
matching `num_connected_models: 5` in its own diagnostics). The three
earlier runs' models are not recoverable by any means -- their files no
longer exist anywhere on disk. This directly contradicted this project's
own stated design (ADR 0017: "`sfm_runs` are kept as historical
provenance, not overwritten") -- the *database row* was preserved, but
the *files it pointed to* were not.

## Decision

- `run_sfm_for_project` now generates `run_id` before computing paths and
  writes each run to its own `project/sfm/sparse/<run_id>/` subtree, so
  no future run can collide with a past one's files. (`database.db`
  stays shared -- COLMAP's own incremental feature/match cache, not
  reconstruction output, and not implicated in this bug.)
- New `vine360/sfm/repair_selected_model.py`: for any `sfm_runs` row
  whose `selected_model` doesn't resolve to a directory on disk, it
  searches every remaining `sfm/sparse/**/cameras.bin` (old shared layout
  and new per-run layout both), and accepts a match only when exactly one
  candidate's live diagnostics (registered images, point count, mean
  reprojection error) exactly equal the row's recorded `model_stats`. A
  row with zero or multiple equally-exact matches is left untouched and
  reported as unrecoverable, rather than guessed. Run for real against
  `/mnt/e/TEST`: recovered `sfm-e40feae0` (-> `sfm/sparse/4`); the three
  older runs correctly reported unrecoverable (their files are gone).
- GUI: Export panel gained a "Repair runs from before this feature"
  button running this same function in the background, so this doesn't
  stay a one-off script only run from a shell for this one project.

## Consequences

- Verified with real tests (`tests/test_repair_selected_model.py`, real
  `pycolmap.synthesize_dataset` reconstructions throughout): a
  recoverable run gets fixed, an unrecoverable one is left alone with a
  clear reason, an ambiguous double-match is refused rather than guessed,
  and an already-correct row is left alone. `tests/test_sfm_project_run.py`
  gained a regression test asserting two successive real calls to
  `run_sfm_for_project` use two distinct `sparse_dir` values.
- Going forward, no project should ever need this repair tool again --
  it exists specifically for runs made before this fix. It's intentionally
  conservative (never guesses under ambiguity) rather than maximally
  recovering everything it possibly could.
- This does not change `run_sfm_for_project`'s public signature or
  `sfm_runs` schema -- `selected_model`'s meaning (docs/adr/0018) is
  unchanged, just now reliably preserved across runs.
- **Later removed from the GUI.** The "Repair runs from before this
  feature (one-time)" button (`ExportPanel`) was removed at the user's
  request once its one-time job was actually done: with only two real
  projects, the repair was run for real against both (`/mnt/e/TEST` had
  3 unrecoverable runs and 1 already-fine run, matching this ADR's
  original finding exactly, nothing newly fixed; `/mnt/e/11-9-26_
  EstoWines_Capture1/A1_InstaLow` had no `sfm_runs` rows at all yet), so
  a standing UI affordance for a fix that's now fully applied wasn't
  worth keeping. `vine360.sfm.repair_selected_model.repair_selected_model`
  itself (and its tests) are untouched -- only the GUI button and its
  handler code in `main_window.py` were removed; the function remains
  callable directly if ever needed again.
