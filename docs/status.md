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
0012: wired Projection, Masking and SfM into the GUI (new
      vine360.projection.generate / vine360.masking.build /
      vine360.sfm.project_run persistence-layer modules, three new index
      tables); unified every progress_callback to (message, current,
      total) with real ETAs; sidebar stage status is now derived live
      from the database (compute_stage_statuses) instead of tracked ad
      hoc; Project panel split into clearly separated create/open
      sections. Also fixed a real latent bug this surfaced:
      pycolmap.COLMAP_version/COLMAP_build are str attributes, not
      callables -- validate_installation() had been calling them as
      functions since ADR 0007, uncaught until now.

## Blockers / known gaps

- **Nerfstudio adapter is unverified** (see ADR 0009) -- run its `--help`
  output against a real install before trusting its flags.
- Near-duplicate frame filtering (M1, called for in section 4) still
  isn't implemented.
- Photo EXIF/GPS parsing for drone stills (ADR 0004) still isn't
  implemented -- and a GPX-sidecar reader (see ADR 0010) is now a better
  first georeferencing step than EXIF parsing, given the real sample
  data has a `.gpx` file, not JPEG GPS tags.
- The GUI (`vine360/gui/`) has no automated test *suite* yet -- every
  verification so far (including the full Project-through-SfM pipeline)
  was a manual offscreen smoke test run during development, not added to
  pytest. `pytest-qt` or similar isn't a dependency yet.
- No pipeline stage has been run against real footage. Real EstoWines
  sample data exists (see above); every test and smoke-test run so far
  uses synthetic clips/images, per the standing preference not to run
  multi-GB real files without it being explicitly asked for.
- Training (preset selection UI, monitor UI) remains untouched --
  adapter-only, per the explicit standing decision in ADR 0009. Not a gap;
  the boundary of "wire up to training" was intentional.

## Exact next task

With Project through Pose estimation now real and wired, reasonable next
directions (none started):

1. Add a `pytest-qt`-based automated GUI test suite -- codify the manual
   offscreen smoke tests (create -> import -> frames -> projection ->
   masks -> SfM, plus the remove-source/custom-interval/re-extraction
   paths) as real pytest tests instead of ad hoc scripts.
2. Wire Training for real (backend install + execution), which was
   explicitly deferred, not attempted, in this pass -- would need to
   revisit ADR 0009's "adapter only" decision first.
3. Georeferencing (M8): a GPX-sidecar reader (see ADR 0010) is a better
   first step than the still-unimplemented EXIF/GPS parsing, given real
   sample data already has `.gpx` files.
4. Near-duplicate frame filtering (M1 gap, still open).
5. Validate the real, wired pipeline against actual Insta360/Antigravity
   A1 footage -- explicitly on request given multi-GB file sizes.
