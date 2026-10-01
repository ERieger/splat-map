# 0025. Remove Training from the desktop app's scope

## Status

Accepted.

## Context

The user asked to remove Training from the application's scope. Since ADR 0009, the GUI had two
permanently-non-functional placeholder panels -- "Training preset" (a preset combo with no wiring)
and "Training monitor" (a disabled "Start Training" button, a progress bar stuck at 0) -- occupying
two sidebar stages between Pose estimation and Export. ADR 0009's decision to keep training
adapter-only (no backend installed, no run) still stands; this ADR only concerns whether that
adapter-only state is represented in the desktop app's sidebar at all.

## Decision

- Removed `build_training_preset_panel`/`build_training_monitor_panel` and the `STAGE_TRAINING_PRESET`/
  `STAGE_TRAINING_MONITOR` sidebar-index constants from `vine360/gui/main_window.py`, along with
  their entries in the panel/label lists built in `Vine360MainWindow.__init__`. The sidebar now
  has 7 stages (Project, Import, Frames, Projection, Masks, Pose estimation, Export) instead of 9;
  `STAGE_EXPORT` shifted from `8` to `6`.
- `vine360/training/` (the library module: `config.py`, `adapter.py`, `nerfstudio_adapter.py`) is
  untouched -- it's real, tested library code (`tests/test_training_adapter.py`), not GUI
  scaffolding. This decision is scoped to the desktop app's sidebar, not the library's contents;
  nothing here reopens or changes ADR 0009's adapter-only decision.

## Consequences

- Fewer permanently-disabled controls in the app -- every sidebar stage now does something real.
- If training is wired up for real later (ADR 0009's "Consequences" section already anticipates
  this as a possible next step), it gets a fresh GUI stage added back at that point, informed by
  whatever the real backend's actual UI needs turn out to be, rather than resurrecting these two
  placeholder panels as-is.
- `docs/status.md`'s "Exact next task" list and "Known gaps" section were updated to drop the
  now-stale "wire Training for real" framing.
