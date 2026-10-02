# 0040. SfM advanced options

## Status

Accepted.

## Context

The Pose estimation panel only offered an engine and a frame set. Everything else was fixed in
code: the camera model followed the engine, matching was always sequential with an overlap of 10,
and every SIFT, matching and mapper setting was left at the engine's default. Real vineyard
capture is likely to need tuning: low-texture canopy, mostly forward motion down a row, rows walked
up and back, and 5.7K equirectangular frames that COLMAP downscales to 3200 px by default.
Neither engine let you change any of this.

## Decision

**One flat, frozen `SfmConfig` in a stdlib-only module, `vine360/sfm/options.py`, shared by
both engines.** It holds 45 options in five groups: Camera, Features, Matching, Mapping, and
Quality checks. The GUI, the queue and the activity log can build, validate and serialize it
without importing pycolmap. `colmap_adapter` re-exports it, so existing imports keep working.

**Every default is the behaviour from before options existed.** The tests check this against the real engines:
- `SfmConfig()` produces option objects identical to pycolmap 4.2's own defaults (`.todict()`
  equality).
- For SphereSfM, every tunable flag now passed with default options equals the binary's own
  `-h` default.

Where the two engines' defaults differ (max image size, max matches per pair, min model size), the
value 0 means "engine default" and the flag or attribute isn't set.

**The GUI form is generated from a declarative `OPTION_SPECS` table.** Each spec gives the
label, kind, range, help text, the engine variants it applies to, and `enabled_when`
dependencies. `gui/sfm_options.py`'s `SfmOptionsEditor` builds the "Advanced options" section from
it:
- Rows the chosen engine can't use are hidden.
- Rows that only matter alongside another choice (e.g. overlap with the sequential matcher) are
  disabled until that choice is made.
- There are presets (Default / Fast preview / Thorough) and "Reset to defaults".
- Validation errors appear inline and disable Run and Add to Queue.

Adding an option means adding a field and a spec. No GUI code changes.

**Recorded and reused.** Each `sfm_runs` row stores the full option set in `config["options"]`.
The worker and queued jobs carry only the non-default options, so the activity log stays readable.
Older jobs without options run as before. The panel starts from the project's latest run's options,
so tuning carries over between runs and sessions. The last-run line and queued job labels list
what was changed.

**What's exposed:**
- **Camera:** camera model (six-face projections only), camera sharing mode, and refinement of
  focal length, principal point and distortion (COLMAP only; SPHERE intrinsics stay fixed).
- **Features:** use GPU, max image size, max features, peak/edge thresholds, upright,
  affine-shape and DSP SIFT.
- **Matching:** strategy (sequential / exhaustive / vocabulary tree), sequential overlap, quadratic
  overlap, loop detection, block size, vocab-tree candidates and tree file, ratio test, max
  distance, cross-check, max matches, guided matching, and verification inliers / error / ratio.
- **Mapping:** mapper (incremental, or global/GLOMAP on pycolmap only), min matches, multiple
  models, min model size, initial-pair inliers / angle / forward motion, registration inliers /
  ratio, filtering error / angle, and random seed.
- **Quality-check thresholds.**

All names were read off the installed pycolmap 4.2.1 classes and SphereSfM's own `<command> -h`.

## Verified for real, not assumed

- **Global mapping (`pycolmap.global_mapping`):** registers the synthetic 360 sequence (6/6,
  ~0.15 px). It has no Python callback, so progress comes from its log: five named stages
  ("=== Running rotation averaging ===" ...), reported as stage *k*/5.
- **Exhaustive matching:** both engines log "Processing block [i/N, j/M]" and report it as
  counted blocks. SphereSfM's matcher log file is named after the matcher actually used.
- **Affine-shape / DSP SIFT:** COLMAP switches to a CPU "Covariant SIFT" extractor by itself, so
  progress reports extraction as CPU. SphereSfM gets `use_gpu 0` for these.

## Left out, and why

- **GPU bundle adjustment** (`ba_use_gpu`): this pycolmap's Ceres is built without CUDA/cuDSS. It
  logs "Falling back to CPU" and runs on the CPU anyway, so a checkbox would be misleading.
- **Spatial matching:** needs pose priors (GPS); vine360 doesn't import any yet (ADR 0033/0036).
- **Vocabulary trees:** none are bundled, and the engines need different formats. pycolmap 4.x
  reads FAISS trees and SphereSfM (COLMAP 3.8) reads the older FLANN ones. The user picks a file;
  without one, vocab-tree matching and loop detection are blocked by validation, not left to fail
  mid-run.
- **Presets** only make speed/thoroughness trade-offs that follow directly from what each option
  does. None is tuned on real vineyard footage yet; their tooltip says so.

## Consequences

- `_run_sfm_worker` takes an explicit `options: dict | None` argument after `engine`, and
  queued pose jobs carry `params["options"]`.
- `run_sfm_for_project` validates options up front (`ValueError`, before any files or rows are
  written). Its quality warnings use the configured thresholds.
