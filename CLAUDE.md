# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Vine360: a local-first orchestration app that turns 360°/conventional capture into masked,
pose-estimated, trainable 3D Gaussian Splatting projects (for vineyard research). The core is a
typed library/CLI (`vine360`); a PySide6 desktop app (`vine360.gui`) calls the same library
functions the CLI calls rather than shelling out to the CLI. The full product/technical spec is
`Vineyard_360_3DGS_Software_Handover.docx`; `docs/status.md` tracks current progress and the exact
next task; `docs/adr/` (25+ entries) records every non-trivial decision — check there before
assuming why something is built a certain way, and add a new ADR for a similarly non-trivial choice.

## Session guidance

Do not interrupt a currently processing session. If a Claude Code session in this repo is actively
running (mid-task, running tests, running the app, etc.), let it finish its current step rather
than stopping or restarting it.

Do not interrupt a processing instance of the application. If a running `vine360.gui` instance is
actively working (e.g. mid-extraction, mid-projection, mid-masking, running SfM, exporting, or
running queued jobs), do not kill or restart that process -- let it finish or reach a natural
stopping point first.

Updating source control means committing **and** pushing to the remote. When asked to commit /
update source control, commit on `main` and then `git push origin main`
(`origin` = https://github.com/ERieger/splat-map.git) -- don't leave commits local-only. Check
`git fetch` / `git status -sb` first so a push isn't rejected for being behind the remote.

## Environment setup

Python 3.12+. On this dev machine the system Python has neither working `pip` nor `venv`
(externally-managed environment, no `python3-venv`, no sudo):

```
python3 -m venv --without-pip ~/.venvs/vine360   # NOT under /mnt/e or any Windows-mounted drive --
                                                   # venv needs symlinks DrvFs can't make (docs/adr/0005)
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
~/.venvs/vine360/bin/python3 /tmp/get-pip.py
~/.venvs/vine360/bin/python3 -m pip install -e ".[dev,sfm,masking]"
~/.venvs/vine360/bin/python3 -m pip install static-ffmpeg PySide6
```

If `pip`/`venv` already work normally: `python3 -m venv .venv && source .venv/bin/activate && pip
install -e ".[dev,sfm,masking]" static-ffmpeg ffmpeg PySide6`.

`pyproject.toml` extras: `dev` (pytest), `sfm` (pycolmap), `masking` (torch/transformers/SAM 3 —
large, gated weights, see below), `ingest-ffmpeg` (bundles ffmpeg/ffprobe binaries, no system
install needed). PySide6 (GUI) isn't yet in an extra; install it directly.

SAM 3 (`vine360.masking.sam3_adapter`) requires manual Meta approval on Hugging Face
(https://huggingface.co/facebook/sam3), then `huggingface-cli login` with the approved token.

## Common commands

```
VENV=~/.venvs/vine360/bin/python3
export PATH="$HOME/.venvs/vine360/lib/python3.12/site-packages/static_ffmpeg/bin/linux:$PATH"

# Tests (ffmpeg/pycolmap/SAM3-dependent tests skip automatically when unavailable)
$VENV -m pytest
$VENV -m pytest tests/test_export_postshot.py -q          # one file
$VENV -m pytest tests/test_export_postshot.py::test_name   # one test
$VENV -m pytest -m 'not network'                            # skip real network calls (e.g. HF)

# CLI (project + ingest only -- see Architecture)
$VENV -m vine360 version
$VENV -m vine360 project create ./myproject --name "Block 7" --capture-mode 360
$VENV -m vine360 ingest add-source ./myproject --source /path/to/clip.mp4
$VENV -m vine360 ingest extract-frames ./myproject --source-id <id> --interval 1.0

# Desktop app -- xcb, not wayland (window-stacking/dropdown reliability under WSLg, docs/adr/0011)
export LD_LIBRARY_PATH="$HOME/.local/lib/xcb-cursor:$LD_LIBRARY_PATH"
export QT_QPA_PLATFORM=xcb
PYTHONPATH=src $VENV -m vine360.gui.main_window

# ...or just use the launcher, which does all of the above (and falls back to wayland
# if libxcb-cursor is missing):
./run-gui.sh
```

`~/.local/lib/xcb-cursor` needs `libxcb-cursor.so.0` extracted without root (`apt-get download
libxcb-cursor0` + `dpkg-deb -x`, see docs/adr/0011). Without it, fall back to
`QT_QPA_PLATFORM=wayland` (works, but with known combo-box/stacking quirks under WSLg/Weston).

No linter/formatter is configured in this repo.

## Architecture

**Two layers, split deliberately by dependency weight** (docs/adr/0001, 0006): `vine360.cli` /
`vine360.config` are stdlib-only (argparse, dataclasses, PyYAML) — no numpy/Pillow needed just to
create a project or report a version. Everything from M2 onward (`projection`, `masking`, `sfm`,
`export`, `training`) uses real numpy/Pillow/scipy/pycolmap/transformers. **The CLI only covers
project creation and ingest** (`vine360/cli.py`) — projection, masking, SfM and export are real,
tested library code called directly by the GUI, not exposed as CLI subcommands.

**Runner/Adapter pattern** (docs/adr/0002): every external engine (ffmpeg, COLMAP) is invoked
through the `Runner` abstract interface (`vine360/runners/base.py`), never via a bare
`subprocess` call from adapter code — `LocalRunner` (`vine360/runners/local.py`) is the only
implementation, fully synchronous (`subprocess.run` under the hood, no incremental-output hook).
COLMAP itself goes through `pycolmap` (COLMAP's official Python bindings), not a subprocess
`colmap` binary (docs/adr/0007) — `vine360/sfm/colmap_adapter.py`'s function/argument names were
confirmed against the actually-installed pycolmap source, not assumed from memory. **Adapter
honesty discipline**: every adapter's docstring states plainly whether it was verified against
real installed source/a real call, or is unverified — e.g. `training/nerfstudio_adapter.py` is an
explicitly unverified scaffold (no install available to check against); don't treat its command
shapes as trustworthy without re-verifying first. `sfm/spheresfm_adapter.py` *is* verified against
a real SphereSfM build (`~/.local/spheresfm/bin/colmap`, or `$VINE360_SPHERESFM_COLMAP`;
docs/adr/0036) — and pycolmap must never load a SphereSfM model (its `SPHERE` camera model id
collides with a different pycolmap model); 360-frame runs of either engine are read/exported
through `sfm/frame_poses.py`.

**Project layout & index** (docs/adr/0003, 0024, 0029): a project is a directory (`project.yaml` +
`index.sqlite`) with fixed subdirectories (`frames/`, `projections/`, `masks/{classes, keep}/`,
`sfm/sparse/`, `runs/`, `exports/`, `models/`). `sources/`, `datasets/` and `cache/` were dropped
(ADR 0029) — never written to by any code, and sources are recorded by reference (checksum + path),
never copied into the project. `_ensure_layout_dirs` (mirroring `_ensure_schema`'s migrate-on-open
pattern) creates any missing directory on every `create_project`/`open_index_db` call, so a new
`LAYOUT_DIRS` entry reaches an existing project automatically. The SQLite schema
(`sources → frame_sets → frames → views → masks`, plus `sfm_runs`) is created by `_ensure_schema` in
`vine360/project.py`, called on **both** `create_project` and every `open_index_db` — an older
project missing a table/column (e.g. `views`, or `masks.flagged_for_review`) gets migrated via
`_ensure_column` rather than breaking silently. `sfm_runs.selected_model` records which
`sfm/sparse/<run_id>/<key>/` subdirectory holds that run's actual COLMAP model —
`pycolmap.incremental_mapping` writes every candidate reconstruction it finds, not just the
selected one, and each run gets its own subdirectory specifically so a later run can't overwrite
an earlier one's files (docs/adr/0019) — this field is load-bearing, not incidental.

**Frame sets** (docs/adr/0034): one source can have several extraction configs side by side
(e.g. every 0.5s and every 1s). Each is a frame set with a deterministic id
`<source_id>~<tag>` (`frame_set_id_for`), and every stage after Frames — projection, masks,
SfM, export, queue jobs, GUI dropdowns — is scoped by `frame_set_id`, not `source_id`. Frames
extracted before frame sets existed were migrated into a set whose id *is* the source_id, so
their paths are unchanged.

**Cascade-delete invariant**: re-running an earlier pipeline stage must delete every downstream
derived artifact (frames → views → masks), both DB rows and on-disk files, or you get orphaned
references and stale directories (the exact bug fixed in docs/adr/0017/0019 — a real
data-integrity gap, not hypothetical). Any change to `clear_frame_set` /
`clear_frames_for_source` / `clear_views_for_frame` / `vine360.cleanup` / `vine360.data_manager`
/ mask rebuilding needs to preserve this.

**Pipeline stages** (mirrored by the GUI sidebar, `vine360/gui/main_window.py`'s `STAGE_*`
constants): Project → Import → Frames → Projection → Masks → Pose estimation → Export, plus a
non-stage Data manager tab (docs/adr/0035) for listing and deleting derived data. Training
has no GUI stage (removed from the app's scope, docs/adr/0025) — the adapter-only library code
(`vine360/training/`) still exists per docs/adr/0009 (no backend is installed or run), it's just
not surfaced here. `compute_stage_statuses` computes each stage's done/active/pending status from
the DB, including staleness detection for Pose (an old `sfm_runs` row can outlive the views/masks
it was built from changing underneath it — docs/adr/0017).

**GUI** (`vine360/gui/main_window.py`, one large file): dashboard/sidebar layout, chosen over a
linear wizard after a direct side-by-side comparison (see `gui/dashboard_preview.py` /
`wizard_preview.py`, kept as the historical previews). Long-running work runs off the GUI thread
via `run_in_background`/`_BackgroundWorker` (a `QThread` wrapper with real progress signals
`(message, current, total)`) — never block the GUI thread on ffmpeg/COLMAP/SAM3 calls. Shared
widgets worth knowing about before adding a new panel: `ProgressArea` (determinate/indeterminate
progress + ETA), `PreviewGallery` (image thumbnails, never raises on a missing file),
`TemporalFrameSelector` (frame scrubber), `StageIndicator` (small step row for a panel with more
than one sequenced action — see docs/adr/0022 for which panels qualify and which don't).

**Export** (`vine360/export/postshot.py`): bundles already-computed poses/images/masks for import
into Postshot or another COLMAP-based external trainer — two modes (with vine360's own SfM poses,
or images+masks only so the external tool runs its own SfM). Verified against Postshot's published
docs only, not a real install (Windows-only, unavailable here) — see docs/adr/0018/0020.

## Testing conventions

Real fixtures over mocks wherever feasible: real `ffmpeg`/`ffprobe` via `LocalRunner` (skipped
with `@requires_ffmpeg` from `tests/conftest.py` when unavailable), real `pycolmap` reconstructions
via `pycolmap.synthesize_dataset`/`pytest.importorskip("pycolmap")` rather than hand-built fixtures,
real SAM 3 calls gated behind the `network` marker/availability checks. When a mock is used (e.g.
mocking `colmap_adapter.run_sfm` in `test_sfm_project_run.py`), it's specifically to isolate the
*calling* module's own DB/file wiring from a heavy call that's already covered by a real test
elsewhere — the comment at the top of the mocking test says which one. GUI widget tests
(`tests/test_gui_stage_indicator.py`) run under `QT_QPA_PLATFORM=offscreen`, set at the top of the
test module itself.
