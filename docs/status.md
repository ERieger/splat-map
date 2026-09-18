# Status

## Completed work

**M0 -- Repository** (exit criteria: CLI starts; tests run; versions are
reported; no processing yet)

- Package scaffold at `src/vine360/` matching the handover doc's expected
  structure (`cli.py`, `config.py`, `project.py`, `models.py`, `runners/`,
  `ingest/`, plus placeholder packages for `projection/`, `masking/`,
  `sfm/`, `training/`, `export/`).
- `vine360 version` reports vine360 version, Python version, platform, and
  found/missing status + version for ffmpeg, ffprobe, colmap, git and
  nvidia-smi (`vine360/runners/probe.py`), without raising on missing
  tools.
- `Runner` interface (`vine360/runners/base.py`) with a `LocalRunner`
  implementation; all subprocess calls go through it.
- `pytest` test harness (`pyproject.toml` sets `pythonpath = ["src"]`);
  `tests/conftest.py` provides a `requires_ffmpeg` skip marker.
- 4 ADRs recorded in `docs/adr/`.

**M1 -- Ingest** (exit criteria: a short X6 clip produces deterministic
frames and a source map)

- `vine360 project create` / `project info`: writes `project.yaml` plus
  the full directory layout from section 5, and an `index.sqlite` index.
- `vine360 ingest add-source`: probes a media file with ffprobe, computes
  a sha256 checksum, infers media type/projection, and requires explicit
  confirmation before treating a 2:1 still image as equirectangular.
- `vine360 ingest extract-frames`: deterministic ffmpeg-based frame
  extraction by interval or target count, with thumbnails, checksums, and
  a persisted source-to-frame map.
- `vine360 ingest manifest`: writes `exports/ingest_manifest.json`
  (sources, frames, source-frame map, dependency versions used).

## Decisions

- Built M0/M1 on the standard library instead of Typer/Pydantic --
  [docs/adr/0001-stdlib-first-cli.md](adr/0001-stdlib-first-cli.md).
- Runner + adapter contract for external tools --
  [docs/adr/0002-runner-and-adapter-contract.md](adr/0002-runner-and-adapter-contract.md).
- Project layout, deferred index tables, sources-by-reference (not copied)
  -- [docs/adr/0003-project-layout-and-index.md](adr/0003-project-layout-and-index.md).
- Assumptions adopted for open questions in section 12 (input format,
  equirectangular detection, GPS/EXIF scope, compute environment) --
  [docs/adr/0004-m1-scope-assumptions.md](adr/0004-m1-scope-assumptions.md).

## Blockers

- **This development machine currently has no `pip`, no working `venv`
  (`ensurepip` is missing), no passwordless `sudo`, and no `ffmpeg` or
  `colmap` installed.** All 28 non-integration tests pass here; the one
  ffmpeg-dependent integration test (`tests/test_ingest_integration.py`)
  skips automatically rather than failing. It still needs to be run for
  real once `ffmpeg`/`ffprobe` are available (e.g. `sudo apt-get install
  ffmpeg`, once sudo access exists), and again against an actual Insta360
  X6 export rather than a synthetic `testsrc` clip.
- No GPU/COLMAP work was attempted -- correctly out of scope for M0/M1.
- Photo EXIF/GPS parsing for drone stills is not implemented (see ADR
  0004); only ffprobe's container-level tags are read.
- Near-duplicate frame filtering (called for in section 4, step 2) is not
  implemented; frame extraction is otherwise deterministic and tested.

## Assumptions that materially affect later work (flagging per section 11's
instruction to report these)

- Projection geometry: not yet touched (M2). No assumption made yet beyond
  what ADR 0004 states about equirectangular *detection*, which is
  separate from projection.
- Operating-system support: implemented and tested on WSL2/Linux, matching
  the handover's "secondary environment." Native Windows execution and the
  eventual WSL-bridging `Runner` are not yet built or tested.
- Licensing: no third-party ML/CV packages were added yet (stdlib +
  PyYAML only), so there is nothing new to audit beyond what M0/M1 itself
  introduces.
- 3DGS backend selection: not made (M5); section 12 leaves this open.

## Exact next task

M2 -- Projection: implement equirectangular-to-perspective projection
(start with the six-face cubemap preset at 90° FOV per section 4, step 3),
implementing the pose-composition invariant
`T_world_face = T_world_panorama · T_panorama_face` from section 4, with a
documented coordinate convention (camera-to-world vs. world-to-camera).
Add acceptance test A01 (round-trip projection of known points on a
synthetic equirectangular grid, landing within one pixel) and A02 (pose
conversion survives internal/COLMAP/trainer round-trips within tolerance)
before touching COLMAP or trainer adapters.
