# 0029. Drop `sources/`, `datasets/`, `cache/` from the project layout

## Status

Accepted.

## Context

A review of the project's fixed `LAYOUT_DIRS` (`vine360/project.py`) against real code usage and
the two real projects on disk (`/mnt/e/TEST`, `/mnt/e/11-9-26_EstoWines_Capture1/A1_InstaLow`)
found three directories created for every project but never written to by any code path, and
confirmed empty in both real projects:

- `sources/` — sources are recorded by reference (checksum + path), never copied into the project
  (ADR 0003); nothing ever writes here.
- `datasets/` — reserved per the handover doc's original layout diagram (ADR 0003), presumably for
  prepared training datasets, but nothing in `vine360/training/` (or anywhere else) ever reads or
  writes it.
- `cache/` — zero references anywhere in the codebase; reserved but never implemented.

The user confirmed these should be dropped: they don't plan to use on-disk source copies (sources
stay referenced elsewhere, as designed), Training is now explicitly out of scope for the app
(ADR 0025), and `cache/` can be added back later if a real caching need ever comes up.

`runs/` (also currently empty/unused, reserved for Training output) and `models/` (intentionally
inert by design, ADR 0024) were **not** part of this request and stay as-is.

## Decision

- Removed `"sources"`, `"datasets"`, `"cache"` from `LAYOUT_DIRS`. `_ensure_layout_dirs` only ever
  creates missing entries, never removes ones no longer listed, so this only affects projects
  created or opened after this change -- it doesn't retroactively delete these folders from
  existing projects on its own.
- The two real projects' now-orphaned empty `sources/`, `datasets/`, `cache/` directories were
  removed directly, once, by hand (there are only two, and both were confirmed empty first) --
  the same "run it once for real rather than build permanent migration machinery" approach already
  used for the export-layout migration (docs/adr/0028's Consequences section).

## Consequences

- A freshly created project now gets `frames/`, `projections/`, `masks/{classes,keep}/`,
  `sfm/sparse/`, `runs/`, `exports/`, `models/` -- every directory that's either actively used by
  real code or, for `runs/`/`models/`, still deliberately reserved.
- If a real caching need comes up later, re-adding `"cache"` to `LAYOUT_DIRS` is a one-line change;
  `_ensure_layout_dirs`'s existing migrate-on-open mechanism (ADR 0024) means every existing
  project picks it up automatically the next time it's opened, no special-casing needed.
- Nothing in `vine360.ingest.sources` or any other module referenced these three directory names
  for real file I/O, so this is a pure removal with no code path to update elsewhere.
