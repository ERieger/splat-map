# 0024. A `models/` project directory, and migrating layout directories on open

## Status

Accepted.

## Context

The user asked for a `models/` directory to exist in every project, to hold the trained 3DGS
model output (e.g. `.ply`/`.splat`, checkpoints) that comes back from Postshot or another
external trainer after training on an already-exported bundle. This is distinct from `exports/`
(`LAYOUT_DIRS`, `vine360/project.py`, ADR 0003), which holds what vine360 sends *to* the trainer,
not what comes back from it. No vine360 code reads or writes into `models/` -- it's purely a place
for the user to put their own output.

Adding a new entry to `LAYOUT_DIRS` exposed a real gap: directories were only ever created once,
inline in `create_project`. Unlike the SQLite schema -- which already has a migrate-on-open
mechanism (`_ensure_schema`/`_ensure_column`, called both when a project is created and every time
an existing one is opened, so an older project missing a table/column doesn't break silently) --
there was no equivalent for `LAYOUT_DIRS`. A project created before this change would never get
`models/` just by being opened again in a newer vine360.

## Decision

- Added `"models"` to `LAYOUT_DIRS`, after `"exports"` (the two directories concerned with
  pipeline output -- what goes out, what comes back).
- Extracted the directory-creation loop into `_ensure_layout_dirs(root)` -- idempotent
  (`mkdir(parents=True, exist_ok=True)` per entry, safe to call unconditionally) -- and call it
  from both `create_project` (replacing the inline loop) and `open_index_db`, mirroring
  `_ensure_schema`'s create-and-open dual-call pattern. An older project gets any new
  `LAYOUT_DIRS` entry (not just `models/`, any future addition too) the next time it's opened, no
  manual migration step required.

## Consequences

- Every project, new or previously created, has a `models/` directory the next time it's opened.
- Adding a future `LAYOUT_DIRS` entry no longer needs its own bespoke migration -- `_ensure_layout_dirs`
  already covers it.
- `models/` is intentionally inert: no vine360 code path reads from or writes to it. If a later
  milestone wants to track model files formally (e.g. record them in the index, like `sfm_runs`
  tracks SfM output), that's a new decision, not implied by this one.
