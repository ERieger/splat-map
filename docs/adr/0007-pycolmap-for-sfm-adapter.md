# 0007. SfM adapter uses pycolmap bindings, not a subprocess `colmap` CLI

## Status

Accepted.

## Context

The handover doc calls for "COLMAP adapter; HLOC later" and "Run COLMAP
through a stable adapter" (section 3), with the Adapter rule elsewhere
describing engines as subprocess commands (matching how the ffmpeg adapter
works). The system `colmap` binary isn't installed here and installing it
needs `sudo apt install colmap` (or a from-source build with its own
dependency chain) -- unavailable in this session (no passwordless sudo).

`pycolmap`, COLMAP's own official Python bindings, is a normal pip wheel
(confirmed: `pip install pycolmap` succeeded, no system package or sudo
needed) and exposes the same underlying COLMAP C++ implementation
in-process: `extract_features`, `match_sequential`, `incremental_mapping`,
full `ImageReaderOptions`/`IncrementalPipelineOptions` control, and
`Reconstruction` objects with the diagnostics section 7 asks for
(`compute_mean_reprojection_error`, `compute_mean_track_length`, etc.).

## Decision

`vine360/sfm/colmap_adapter.py` calls `pycolmap` directly rather than
shelling out to a `colmap` binary through `vine360.runners.Runner`. This
is still "COLMAP" (same registration algorithm, same file formats, same
mask convention) -- just invoked as a library instead of a CLI.

Every API call used was confirmed against the actually-installed
`pycolmap` 4.2.0 by introspection (`help()`, `dir()`) before being wired
up, not assumed from COLMAP's own CLI docs, which describe flags for the
binary, not the Python bindings' argument names.

`ImageReaderOptions.mask_path` uses exactly the same "append `.png` to the
full image subpath" convention as
`vine360.masking.semantics.colmap_mask_path` -- confirmed by a real,
un-mocked test (`tests/test_sfm_integration.py::test_a04...`) that no
extracted keypoint lands inside a masked-out region.

A real, load-bearing finding from building this adapter, worth keeping in
mind for anyone testing or extending it: **views generated from a single
panorama's projection (any cubemap/projection preset, all sharing one
optical center) have zero camera-position baseline and cannot be
3D-reconstructed**, no matter how good the 2D feature matches are --
COLMAP's `IncrementalMapperOptions.init_min_tri_angle` (default 16 deg)
correctly rejects a zero-baseline initial pair. Real parallax needs
distinct capture positions (successive video frames from a moving camera,
or separate photographs), matching how the real pipeline will actually
feed it: temporally-sequenced frames from M1, not several views of one
frame.

## Consequences

- No `colmap` binary or `sudo` install is required to develop or run this
  adapter -- only `pip install pycolmap`, matching the "no admin access
  needed" bar this project has held to since ADR 0005/0006.
- If a future milestone needs COLMAP CLI-only features that pycolmap
  doesn't expose (unlikely given the bindings cover extraction, matching,
  mapping and reconstruction I/O), revisit this decision and add a
  subprocess-based adapter behind the same `Runner` interface used
  elsewhere, without changing `colmap_adapter.py`'s public functions.
- `extract_and_match` and `map_and_diagnose` are kept as separate
  functions (not fused into one `run_sfm` call only) specifically so each
  is independently testable: extraction/matching against real rendered
  images with masks, and mapping/diagnostics against a dataset with real
  parallax (`pycolmap.synthesize_dataset` in tests, real video frames in
  production).
