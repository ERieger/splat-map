# 0001. Build M0/M1 on the standard library instead of Typer/Pydantic

## Status

Accepted.

## Context

The handover doc (section 3) recommends "Python 3.12, PySide6, Pydantic,
SQLModel or SQLite, and subprocess adapters for FFmpeg and COLMAP" as a
pragmatic starting stack, with Typer for the CLI.

The development machine available for M0/M1 has Python 3.12.3 and the
system-installed `python3-pytest` and `python3-yaml` packages, but no `pip`,
no `venv` module (`ensurepip` is missing), and no passwordless `sudo`, so
third-party packages (Typer, Pydantic, PySide6, SQLModel) cannot be
installed. `ffmpeg` and `colmap` binaries are also not installed.

## Decision

Implement M0 and M1 using only the standard library plus the two
system-provided packages:

- `argparse` instead of Typer for the CLI.
- `dataclasses` instead of Pydantic for typed configuration and data
  contracts (`vine360/config.py`, `vine360/models.py`).
- `sqlite3` (stdlib) for the project index, as the doc already allows
  "SQLModel **or** SQLite".
- `PyYAML` (already present) for `project.yaml`.

All external-tool calls go through a `Runner` interface (`vine360/runners`)
so subprocess handling is centralized and swappable regardless of which
higher-level framework wraps it later.

## Consequences

- M0/M1 are fully runnable and testable on this machine today; the test
  suite is not blocked on network access.
- The public shapes (typed dataclasses with `to_dict`/`from_dict`, a small
  `Runner` ABC, argparse subcommands mirroring one CLI action each) were
  chosen so that swapping in Pydantic (`BaseModel` with matching field
  names) and Typer (thin wrappers calling the same functions) later is a
  mechanical, low-risk change rather than a rewrite.
- PySide6 is required regardless once M6 (desktop UI) starts, at which
  point this machine (or the target Windows 11 / WSL2 machine) will need
  working `pip`/`venv` access anyway. Revisit this ADR when that happens.
