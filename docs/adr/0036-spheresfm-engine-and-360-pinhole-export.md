# 0036. SphereSfM engine, and pinhole export of 360 SfM runs

## Status

Accepted. Supersedes the "unverified scaffold" status of `vine360/sfm/spheresfm_adapter.py`
(ADR 0017). Extends 0018/0028 (export) and 0033 (RealityScan camera priors).

## Context

SphereSfM (https://github.com/json87/SphereSfM) is a COLMAP 3.8 fork with a spherical `SPHERE`
camera model plus sphere-aware two-view geometry and bundle adjustment, which suits raw 360
footage. There's no wheel or prebuilt binary. It was built from source on this machine
(commit 6b40b2d, CPU-only, Ubuntu 24.04 packages) and installed to `~/.local/spheresfm`.

Separately, both engines that reconstruct raw 360 frames (SphereSfM, and pycolmap's own
`EQUIRECTANGULAR` model) had no usable trainer export. The Postshot poses export copied raw
equirectangular frames with a 360 camera model, which Postshot can't train on, and it carried no
masks, because vine360's masks are built on projected views, not on raw frames.

## Decision

**SphereSfM runs through its CLI, via `Runner`** (ADR 0002): `feature_extractor` (SPHERE,
params `1,W/2,H/2` as in the README), then `sequential_matcher`, then `mapper`
(`--Mapper.sphere_camera 1`, intrinsics fixed), then `model_converter --output_type TXT`.
- Every flag used was checked against the real binary's `-h`. A test re-checks them whenever
  the binary is present.
- Sphere-aware verification lives in `two_view_geometry.cc` and is chosen per camera pair, so
  it applies to `sequential_matcher` too, not only the README's `spatial_matcher` (which needs
  pose priors vine360 doesn't have yet).
- Images go in through an explicit `--image_list_path` built from the DB (no `thumbs/`).
- Everything stays under `sfm/sparse/<run_id>/`: the database, every candidate model, and each
  model's `txt/` (ADR 0019/0034 layout).
- The run row records `config.engine = "spheresfm"`. Rows without `engine` are pycolmap runs.

**Finding the binary.** `find_binary()` checks, in order:
1. `$VINE360_SPHERESFM_COLMAP`
2. `~/.local/spheresfm/bin/colmap`
3. `~/build/spheresfm/src/exe/colmap`

It only accepts a binary whose `help` lists `sphere_cubic_reprojecer` (sic). It's deliberately
not on `PATH`, so it can't be confused with a stock COLMAP.

**pycolmap never loads a SphereSfM model.** SphereSfM's `SPHERE` is camera model id 11, but
pycolmap 4.2's id 11 is `RAD_TAN_THIN_PRISM_FISHEYE`, so loading it would silently misread the
camera. Models are converted to TXT with SphereSfM's own converter and parsed by
`spheresfm_adapter.read_text_model`.

**360 runs export as vine360's own projected views** (`vine360/sfm/frame_poses.py`). Every view
shares its frame's optical centre (ADR 0033), so a view's pose is the frame's pose rotated by
the view's `fixed_rotation`:

    R_view = F · R_faceᵀ · Mᵀ · R_frame,   t_view = −R_view · C,   F = diag(1, −1, 1)

F converts vine360's y-up camera frames to COLMAP's y-down ones. M maps an engine's 360 camera
frame onto vine360's panorama frame. **M was measured, not assumed:** it was fitted from real
reconstructions of a synthetic ray-cast 360 room (`tests/synthetic_360.py`) with known
ground-truth poses. Both engines came out as exactly `M = F` (median angular residual 0.04°).

The result is a standard **PINHOLE** model:
- one camera per distinct view intrinsics
- one image per projected view, named like a six-face run's images
- the run's sparse points, with tracks made by projecting each point into the views of the
  frames that observed it (in front of the camera and inside the image, the same rule as
  SphereSfM's `ExportPerspectiveCubic`)

`export_for_postshot` copies those views plus their `masks/keep` masks, through the same
flattening path as six-face exports. A 360 run whose frame set has no projections fails with
"generate projections for this frame set first". The GUI disables Export with that hint, and
the queue auto-inserts the Projection job. RealityScan camera priors for 360 runs read frame
centres through the same module, so they work for SphereSfM too.

**How it's verified** (`tests/test_frame_poses.py`, real reconstructions, nothing mocked):
- Every sparse-point observation's equirect pixel, carried into each view purely through
  vine360's projection geometry, lands where the derived pinhole pose projects the 3D point.
  Over about 30k observations per engine the error is a median of 0.11 px and a 95th
  percentile of 0.49 px (for 256 px faces).
- SphereSfM's own `sphere_cubic_reprojecer` composes face poses independently. Its
  front/right/back/left faces match vine360's to within 0.1°.
- Recovered camera centres match the ground truth up to a similarity transform.
- The export writes a PINHOLE model that pycolmap reads back, with consistent image and mask
  names.

## Consequences / known limits

- **GPU (added after first acceptance).** Both SIFT paths now use the GPU on this machine:
  - **pycolmap:** the `pycolmap-cuda12` wheel (the `sfm-cuda` extra) replaces `pycolmap`. Its
    default device choice picks the GPU, so vine360 needed no code change. Measured on six 2K
    synthetic frames: 0.82 s on GPU vs 3.37 s on CPU.
  - **SphereSfM:** rebuilt with CUDA 12.8 (NVIDIA's WSL-Ubuntu `cuda-toolkit-12-8`, no driver
    package), `-DCMAKE_CUDA_ARCHITECTURES=89`. Its CUDA code already uses texture objects, so it
    builds unchanged against CUDA 12. The adapter only passes `--SiftExtraction.use_gpu 1` and
    `--SiftMatching.use_gpu 1` when the binary's version banner says "with CUDA" (`has_cuda`).
    A CPU-only build would otherwise fall back to OpenGL SiftGPU, which needs a display.
  - The mapper's bundle adjustment stays on the CPU either way.
- **No masking during 360 SfM**, on either engine. A static camera mask (`--ImageReader.
  camera_mask_path`) covering the operator or tripod at the nadir is a possible follow-up.
- **No GPS/POS priors yet.** `spatial_matcher` and `--ImageReader.pose_path` would pair
  naturally with the Antigravity A1's GPX sidecars (ADR 0010).
- `LocalRunner` is synchronous, so progress is per step (extract, match, map, read), with an
  indeterminate bar within each step and no cancellation, like other external calls.
- A run's views are those of its frame set at export time. Re-projecting the frame set with
  different faces changes which views export. Poses stay correct, because they're derived from
  the frame, not stored per view.
