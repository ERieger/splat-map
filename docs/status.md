# Status

## Completed work

**M0 -- Repository.** CLI scaffold, dependency probe (ffmpeg/ffprobe/
colmap/git/nvidia-smi, never raises on a missing tool), `Runner`
abstraction, pytest harness. Unchanged since the first pass; see the ADRs
below for what's since layered on top.

**M1 -- Ingest.** `vine360 project create/info`, `ingest add-source`
(ffprobe metadata, checksum, equirectangular confirmation gate),
`ingest extract-frames` (deterministic ffmpeg extraction + thumbnails +
source-frame map), `ingest manifest`. The real end-to-end integration
test now actually runs (previously skipped for lack of ffmpeg) against
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
  license. Real SAM 3 inference is therefore still untested; the adapter
  is otherwise built and its failure path is verified against the real
  endpoint.

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
0009: training stays adapter-only; Nerfstudio adapter is unverified.

## Blockers / known gaps

- **SAM 3 weights**: manual approval pending (see above). Once approved
  and `huggingface-cli login` is run, replace
  `test_masking_sam3_adapter.py`'s access-denied test with a real
  segmentation assertion, and double check
  `post_process_instance_segmentation`'s actual output against real
  inference (currently confirmed only via source inspection).
- **Nerfstudio adapter is unverified** (see ADR 0009) -- run its `--help`
  output against a real install before trusting its flags.
- **Projection/masking/SfM are not yet wired into the project CLI or
  index database.** `vine360/project.py`'s `index.sqlite` only has
  `sources`/`frames` tables (per ADR 0003's "add tables when their
  milestone needs them"); M2-M4 are complete, tested library code, but
  there's no `vine360 projection generate` / `masking build` / `sfm run`
  CLI command yet, and no `views`/`masks`/`sfm_runs` tables. This is
  explicitly the next task, not an oversight -- see below.
- Near-duplicate frame filtering (M1, called for in section 4) still
  isn't implemented.
- Photo EXIF/GPS parsing for drone stills (ADR 0004) still isn't
  implemented.
- No real Insta360 X6 footage or real vineyard imagery has been used
  anywhere; every test uses synthetic data (labeled panoramas, procedural
  textures, `pycolmap.synthesize_dataset`). Section 12's open question
  about a committable reference dataset is still open.

## Exact next task

Wire M2-M4 into the project CLI and index database, so the pipeline is
actually drivable end-to-end from `vine360` commands rather than only
from library calls in tests:

1. Add `views` and `masks` tables to `project.py`'s index schema (fields
   per the data contracts already declared in `models.py`).
2. `vine360 projection generate --project <path> --frame-id <id>
   [--preset six-face] [--fov 90] [--face-size N]`: renders faces for a
   frame, writes them under `project/projections/`, records `View` rows.
3. `vine360 masking build --project <path> --view-id <id> [--sky
   classical|sam3] [--person sam3]`: runs the requested backends, writes
   class/keep/exclude masks under `project/masks/`, records `Mask` rows,
   surfaces `is_keep_fraction_anomalous` warnings.
4. `vine360 sfm run --project <path>`: gathers all views + their keep
   masks, calls `extract_and_match` + `map_and_diagnose` against
   `project/sfm/database.db` / `project/sfm/sparse/`, prints
   `evaluate_registration_quality` warnings, records an `sfm_runs` row.
5. Only after that: M6 (desktop UI) becomes "wrap the CLI in a wizard",
   which is the doc's own intended shape (section 3: "every UI action
   maps to a reproducible command").
