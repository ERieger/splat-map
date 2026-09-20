# Status

## Completed work

**M0 -- Repository.** CLI scaffold, dependency probe (ffmpeg/ffprobe/
colmap/git/nvidia-smi, never raises on a missing tool), `Runner`
abstraction, pytest harness. Unchanged since the first pass; see the ADRs
below for what's since layered on top.

**M1 -- Ingest.** `vine360 project create/info`, `ingest add-source`
(ffprobe metadata, checksum, equirectangular confirmation gate),
`ingest remove-source` (added 2026-09-20 -- deletes the registered source
and its extracted frames, never the original file),
`ingest extract-frames` (deterministic ffmpeg extraction + thumbnails +
source-frame map; now clears a source's prior frames before re-extracting
-- see ADR 0011, this was a real "UNIQUE constraint failed" bug found via
use), `ingest manifest`. The real end-to-end integration test now
actually runs (previously skipped for lack of ffmpeg) against
`static-ffmpeg`'s bundled real binaries.

**M2 -- Projection (`vine360/projection/`), fully implemented as library
code, not yet CLI-wired.**
- `pose.py`: camera-to-world `Pose`, `compose()` implementing
  `T_world_face = T_world_panorama . T_panorama_face`, and COLMAP
  qvec/tvec conversion.
- `geometry.py`: equirectangular <-> direction, perspective camera <->
  direction, the six-face rotation math.
- `cubemap.py`: the six-face 90-degree preset, polar faces excluded by
  default per the handover doc's own guidance.
- `render.py`: real numpy/Pillow bilinear resampling from an
  equirectangular image to a perspective face, including seam wraparound.
- Verified two ways: unit tests to 1e-9 precision (`test_projection_
  geometry.py`, `test_projection_pose.py`), and a visual sanity check
  (rendered a labeled synthetic panorama, inspected the front/right/up
  face outputs directly) that showed exactly the expected marker
  centering and polar radial-line convergence.
- A01 (round-trip projection within one pixel) passes against the actual
  transform; the image-based tests use a documented, wider pixel
  tolerance to account for marker-block quantization under bilinear
  resampling, not transform error (see the comment in
  `test_projection_render.py`).

**M3 -- Masking (`vine360/masking/`), fully implemented, not yet
CLI-wired.**
- `semantics.py`: mask value conventions, small-component removal +
  dilation (deliberately reordered from the doc's literal step order --
  see ADR 0008), keep/exclude mask derivation, `keep_fraction` anomaly
  flagging, COLMAP mask path convention.
- `classical_sky.py`: a weights-free sky heuristic (brightness/blue-
  dominance/top-of-frame), explicitly a stand-in for SAM 3, not a
  substitute for it. No equivalent exists (or can reasonably exist) for
  person masks.
- `sam3_adapter.py`: the real backend against `transformers`' `Sam3Model`/
  `Sam3Processor`. Confirmed against a live call to the actual (gated,
  currently unauthenticated) `facebook/sam3` repo -- caught and fixed a
  real bug where `transformers` surfaces the gating failure as a plain
  `OSError`, not `huggingface_hub`'s `GatedRepoError`.

**M4 -- SfM (`vine360/sfm/colmap_adapter.py`), fully implemented, not yet
CLI-wired.**
- Uses `pycolmap` (COLMAP's own Python bindings) instead of a subprocess
  `colmap` binary -- see ADR 0007. `extract_and_match` (masked feature
  extraction + sequential matching) and `map_and_diagnose` (incremental
  mapping + the section-7 diagnostics) are separate, independently
  testable functions; `run_sfm` composes them for the real-files case.
- Real, unmocked tests: A04 ("no keypoints inside a zero-valued mask
  region") against actual COLMAP database output; a documented,
  *asserted* failure case showing that views rendered from a single
  panorama (zero camera-position baseline) correctly cannot be
  3D-reconstructed, no matter how good the 2D matching is; and a real
  multi-position reconstruction via `pycolmap.synthesize_dataset`
  (genuine parallax, genuine ground truth) proving `map_and_diagnose` and
  `evaluate_registration_quality` work correctly end-to-end.

**M5 -- Training (`vine360/training/`), adapter architecture only, per
an explicit choice not to install a backend or spend GPU time yet.**
- `config.py` (typed run config + `runs/<run-id>/` layout wiring),
  `adapter.py` (the backend contract), `nerfstudio_adapter.py` (a
  concrete implementation). **Unlike every other adapter in this
  project, `nerfstudio_adapter.py`'s CLI flags are unverified** -- no
  Nerfstudio install exists to confirm them against, unlike ffmpeg/
  pycolmap/SAM 3, which were each checked against real installed source
  or a real live call. See ADR 0009.

**M6 -- Desktop UI (`vine360/gui/`), started.** Built two full PySide6
previews to resolve the layout question directly rather than guessing:
`wizard_preview.py` (linear QWizard) and `dashboard_preview.py`
(persistent sidebar + stage panels). The user compared both running
side by side and chose the dashboard -- "I prefer the dashboard. I like
the more visual approach" (saved to memory). `main_window.py` is the real
app going forward: its Project and Import panels are genuinely wired to
`vine360.project`/`vine360.ingest.sources` (create/open a project on
disk, register a real source with ffprobe metadata + checksum + a
capture-group tag), verified end-to-end with an offscreen smoke test
(mocked file dialogs, a real synthetic clip, a real add_source call).
Frames/Projection/Masks/Pose/Training panels show real preset
enums/computed values (e.g. the actual cubemap face count) but their
action buttons are disabled with an explanatory tooltip -- wiring those
to actually execute is intentionally deferred (see "Exact next task").
The GUI calls the same typed library functions the CLI calls (not
CLI-as-subprocess) -- this still satisfies the doc's "every UI action
maps to a reproducible command" intent, since the action is backed by one
well-defined function either surface can call; it just means the CLI
isn't literally shelled out to.

## Environment notes (this dev machine)

- No system `pip`/`venv` (Debian's `python3-venv`/`python3-pip` aren't
  installed, and there's no passwordless sudo to add them) -- worked
  around with `venv --without-pip` + a manually bootstrapped `get-pip.py`,
  in a venv living outside the repo (`~/.venvs/vine360`) because `/mnt/e`
  is a Windows-mounted drive that can't hold the symlinks `venv` needs.
  See ADR 0005.
- Real internet access exists, which is what made the rest of this
  possible: `numpy`/`Pillow`/`scipy`/`pycolmap`/`torch`/`transformers`/
  `static-ffmpeg` all installed and were exercised for real. See ADR 0006.
- This machine has a real NVIDIA GPU passed through into WSL2 (confirmed
  via `nvidia-smi` and `torch.cuda.is_available()`), matching the
  handover doc's own "Secondary environment: WSL2" line -- this looks
  like the actual target dev machine, not a disposable sandbox.
- `facebook/sam3` requires **manual** Meta approval (confirmed live via
  `HfApi().model_info(...).gated == "manual"`), not just a click-through
  license. **Approved and authenticated as of 2026-09-19** (`hf auth
  login` -- the modern replacement for the now-deprecated
  `huggingface-cli`); real SAM 3 inference now runs on this machine's GPU
  and is covered by a real, passing test
  (`test_segment_finds_sky_on_a_real_image`). Surfaced and fixed one real
  gap: `Sam3ImageProcessor` needs `torchvision`, not installed
  automatically alongside `torch`/`transformers`.
- Real capture equipment: an Insta360 (ground) and an **Antigravity A1**,
  which is itself an 8K 360-degree drone, not a conventional perspective
  drone as the handover doc's "mixed capture" milestone assumed -- see
  ADR 0010. Real sample footage exists at
  `/mnt/e/11-9-26_EstoWines_Capture1/Equi/` (multi-GB clips, a `.gpx`
  flight-telemetry sidecar); nothing has been run against it yet, per an
  explicit "no runs yet" instruction.

## Decisions (see docs/adr/ for full reasoning)

0001: stdlib-first CLI/config (argparse/dataclasses over Typer/Pydantic).
0002: Runner + adapter contract for external tools.
0003: project layout, deferred index tables, sources-by-reference.
0004: M1 scope assumptions (input format, EXIF/GPS, compute environment).
0005: venv lives outside the repo, under `$HOME` (DrvFs symlink limits).
0006: real numpy/Pillow/pycolmap/ffmpeg once pip access existed after all.
0007: pycolmap (not a subprocess `colmap` binary) for the SfM adapter.
0008: SAM 3 as the real masking backend; classical sky as an interim
      fallback; reordered mask-build pipeline (dedupe before dilate).
      Updated 2026-09-19: real access confirmed, real inference tested.
0009: training stays adapter-only; Nerfstudio adapter is unverified.
0010: real capture kit is two 360-degree platforms (Insta360 ground +
      Antigravity A1 aerial), not 360 + conventional; capture_group is
      now a real, wired parameter on add_source.
0011: fixed re-extraction's UNIQUE constraint bug and a QThread
      use-after-free crash in the GUI's background-worker helper; GUI now
      runs under QT_QPA_PLATFORM=xcb (not wayland) for more reliable
      window stacking/dropdowns, using a user-space-extracted
      libxcb-cursor0 (no root needed).

## Blockers / known gaps

- **Nerfstudio adapter is unverified** (see ADR 0009) -- run its `--help`
  output against a real install before trusting its flags.
- **Projection/masking/SfM are not yet wired into the project CLI, GUI,
  or index database for execution.** `vine360/project.py`'s
  `index.sqlite` only has `sources`/`frames` tables (per ADR 0003's "add
  tables when their milestone needs them"); M2-M4 are complete, tested
  library code, and the GUI's Frames/Projection/Masks/Pose panels show
  real preset values, but nothing actually runs extraction/projection/
  masking/SfM from the CLI or GUI yet -- their action buttons are
  present but disabled. This is explicitly the next task, not an
  oversight -- see below.
- Near-duplicate frame filtering (M1, called for in section 4) still
  isn't implemented.
- Photo EXIF/GPS parsing for drone stills (ADR 0004) still isn't
  implemented -- and a GPX-sidecar reader (see ADR 0010) is now a better
  first georeferencing step than EXIF parsing, given the real sample
  data has a `.gpx` file, not JPEG GPS tags.
- The GUI (`vine360/gui/`) has no automated tests yet (an offscreen smoke
  test was run manually during development, not added to the pytest
  suite -- PySide6 GUI testing needs `pytest-qt` or similar, not yet
  added as a dependency).
- No pipeline stage has been run against real footage. Real EstoWines
  sample data exists (see above) but per an explicit "no runs yet"
  instruction, only its GPX file and directory listing have been
  inspected -- no video has been opened or processed.

## Exact next task

Wire frame extraction, projection, masking and SfM into both the CLI and
the GUI's disabled action buttons, so the pipeline is actually drivable
end-to-end rather than only reachable from library calls in tests:

1. Add `views` and `masks` tables to `project.py`'s index schema (fields
   per the data contracts already declared in `models.py`).
2. `vine360 projection generate --project <path> --frame-id <id>
   [--preset six-face] [--fov 90] [--face-size N]` (and the GUI's
   "Generate Projections" button): renders faces for a frame, writes
   them under `project/projections/`, records `View` rows.
3. `vine360 masking build --project <path> --view-id <id> [--sky
   classical|sam3] [--person sam3]` (and "Build Masks"): runs the
   requested backends, writes class/keep/exclude masks under
   `project/masks/`, records `Mask` rows, surfaces
   `is_keep_fraction_anomalous` warnings.
4. `vine360 sfm run --project <path>` (and "Run SfM"): gathers all views
   + their keep masks, calls `extract_and_match` + `map_and_diagnose`
   against `project/sfm/database.db` / `project/sfm/sparse/`, prints
   `evaluate_registration_quality` warnings, records an `sfm_runs` row.
5. Long-running steps (extraction, SfM, eventual training) need to run
   off the GUI's main thread (e.g. `QThread`/`QRunnable`) so the
   dashboard stays responsive and can show live progress/logs -- not
   needed for the CLI, but required before the GUI's action buttons can
   be safely enabled.
6. Only once frame extraction can run for real: validate against the
   real Insta360/Antigravity A1 sample footage -- explicitly on request,
   not proactively, given the multi-GB file sizes.
