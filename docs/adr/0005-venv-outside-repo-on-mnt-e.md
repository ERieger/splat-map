# 0005. Virtual environment lives outside the repo, under $HOME

## Status

Accepted.

## Context

The repo lives on a Windows drive mounted into WSL2 as `/mnt/e` (DrvFs).
`python3 -m venv` needs to create symlinks (`bin/lib64` etc.) as part of
its layout, and DrvFs does not support symlinks for a normal (non-elevated)
process: `python3 -m venv /mnt/e/Software/SplatMap/.venv` fails with
`Operation not permitted: 'lib' -> '.../lib64'`.

Separately, the system Python here has no `ensurepip`, so `venv` alone
can't bootstrap `pip` either; `get-pip.py` (a standalone bootstrap script)
has to be run manually inside the created environment.

## Decision

- The venv lives at `~/.venvs/vine360`, on the native Linux filesystem,
  not inside the repo.
- It was created with `python3 -m venv --without-pip`, then bootstrapped
  with `bootstrap.pypa.io/get-pip.py` run through that venv's own
  interpreter.
- The repo itself stays dependency-free of the venv: nothing under
  `/mnt/e/Software/SplatMap` assumes the venv's location, and `.gitignore`
  doesn't need a `.venv/` entry for this repo path since the venv is never
  created there.

## Consequences

- Anyone continuing this project on this machine (or a similarly-mounted
  drive) should create their venv under their home directory (or any
  native filesystem path), not under the repo path on `/mnt/e`, `/mnt/c`,
  etc.
- Commands in docs/README now assume `$HOME/.venvs/vine360/bin/python3`
  (or an equivalent activated venv) rather than a project-local `.venv`.
- This also means `pip install -e .` for the package itself hasn't been
  exercised from within `/mnt/e` -- editable installs also rely on
  symlink-like mechanisms in some pip versions. Tests instead run with
  `pythonpath = ["src"]` (already set in `pyproject.toml`), which avoids
  the issue entirely.
