# 0003. Project filesystem layout and source-by-reference indexing

## Status

Accepted.

## Context

The handover doc, section 5, specifies a fixed project directory layout
and requires "Artifacts remain inspectable; database provides status,
provenance and recovery" (filesystem artifacts plus a SQLite index) and
"Never delete source media automatically."

## Decision

- `create_project` creates exactly the directories listed in the
  handover's layout diagram (`sources/`, `frames/`, `projections/`,
  `masks/classes/`, `masks/keep/`, `sfm/sparse/`, `datasets/`, `runs/`,
  `exports/`, `cache/`). `sfm/database.db` is left for the COLMAP adapter
  (M4) to create, since it is COLMAP's own database, not ours.
- `index.sqlite` at the project root holds `sources` and `frames` tables
  now. `views`, `masks`, `sfm_runs` and `training_runs` tables are added
  when their milestones (M2-M5) need them, not now, per the handover's
  instruction not to implement later milestones early. The dataclass
  shapes for those entities are already declared in `vine360/models.py` so
  the contract is visible and stable ahead of time.
- **Sources are recorded by reference**, not copied into `sources/`: an
  `add_source` call stores the resolved absolute path, a sha256 checksum,
  and probed metadata. It does not duplicate the (potentially many-GB)
  original video into the project directory.

## Consequences

- This avoids doubling storage for every ingested clip, which matters
  given section 10's "Large storage use" risk, and keeps "never delete
  source media automatically" trivially true (there's nothing of the
  source under project control to delete).
- The tradeoff: if the original file is moved, renamed or the drive is
  disconnected, the project can no longer re-run ingest-dependent stages
  against it until the path is repaired. The checksum recorded at ingest
  time is what makes that drift detectable rather than silent.
- If review of the real reference workflow (M6) shows researchers expect
  the project folder to be self-contained and portable (e.g. copying it to
  another machine), this decision should be revisited -- e.g. an explicit
  `--copy-into-project` opt-in -- rather than changing the default
  silently.
