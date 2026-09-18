# 0006. Use real numpy/Pillow/pycolmap/ffmpeg once pip access exists, rather
than staying stdlib-only

## Status

Accepted. Supersedes the stdlib-only scope of ADR 0001 for anything
touching image or geometry data; ADR 0001's choice of argparse/dataclasses
over Typer/Pydantic for the CLI/config layer stands.

## Context

ADR 0001 chose the standard library for M0/M1 because `pip` could not be
bootstrapped at all in that session. Continuing into M2 (projection) needs
real image I/O and array math; hand-rolling a PNG codec and vectorized
geometry in pure Python is not a reasonable substitute for numpy/Pillow,
and the handover doc's own recommended stack already calls for exactly
this kind of library.

`pip` turned out to be bootstrappable after all, without root: `venv
--without-pip` + `get-pip.py` (see ADR 0005), run against a venv outside
`/mnt/e`. That removes the original constraint for anything that isn't the
CLI/config layer itself.

## Decision

Install real dependencies into `~/.venvs/vine360` and use them directly:

- **numpy** and **Pillow** for `vine360/projection/*` (geometry, cubemap
  presets, image resampling).
- **pycolmap** (COLMAP's official Python bindings, wheel-installable, no
  system `colmap` binary or `sudo apt install colmap` required) for the
  SfM adapter -- see ADR 0007.
- **static-ffmpeg** (bundles real `ffmpeg`/`ffprobe` binaries, fetched on
  first use, no `sudo apt install ffmpeg` required) so the M1 ingest
  adapters can be exercised for real rather than only unit-tested at the
  command-construction level.
- **torch** + **transformers** + **huggingface_hub** for the SAM 3
  masking adapter -- see ADR 0008.

The CLI and config layer (ADR 0001's actual subject: argparse over Typer,
dataclasses over Pydantic) is unchanged; that choice wasn't about missing
image/ML libraries, and revisiting it isn't necessary to make M2+ real.

## Consequences

- `tests/test_ingest_integration.py`'s `requires_ffmpeg` skip now resolves
  to "run for real" whenever `ffmpeg`/`ffprobe` are on `PATH` -- put
  static-ffmpeg's bundled binaries there (see docs/status.md) or install
  system ffmpeg.
- `pyproject.toml`'s dependency list needs to grow to cover these once M2+
  code is meant to be pip-installed normally; for now they're tracked as
  "installed into the dev venv" rather than declared project dependencies,
  since exact version pins should wait until the adapters they back
  (SfM, training) are further along and version sensitivity is better
  understood.
