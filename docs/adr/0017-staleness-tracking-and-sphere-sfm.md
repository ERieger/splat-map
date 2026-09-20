# 0017. Staleness tracking after re-running earlier stages; SphereSfM and
native-equirectangular SfM engine options

## Status

Accepted.

## Context

Two requests: (1) re-running an earlier pipeline stage should visibly
downgrade later stages' sidebar status (yellow/grey) rather than leaving
them looking falsely DONE; (2) add "SphereSfM" as an SfM option.

Investigating (1) surfaced a real, pre-existing data-integrity bug:
`clear_frames_for_source` (called whenever frames are re-extracted, or a
source removed) deleted only `frames` rows/files -- any `views`/`masks`
built from the *old* frames became orphans: dangling `frame_id`
references, and stale `project/projections/<old-frame-id>/` directories
left on disk. It had never been exercised by re-extracting a source that
already had projections/masks built.

Investigating (2): "SphereSfM" (https://github.com/json87/SphereSfM) is
a genuine, separate fork of COLMAP -- its own `SPHERE` camera model, a
`--Mapper.sphere_camera` flag, a custom `sphere_cubic_reprojecter` tool.
It has no pip package or prebuilt binary; it must be compiled from C++
source, which this environment can't do. Separately, and this was the
useful finding: mainline COLMAP -- and therefore the `pycolmap` already
installed here -- has its own native `EQUIRECTANGULAR` camera model
(`pycolmap.CameraModelId.EQUIRECTANGULAR`, params just `w,h`). Confirmed
for real via `pycolmap.synthesize_dataset(camera_model_id=
EQUIRECTANGULAR)`: full registration, near-zero reprojection error. This
is a different, already-available, already-verified capability -- not
SphereSfM's `SPHERE` model or its sphere-aware matching/mapping, which
have no equivalent in stock COLMAP.

## Decision

- `clear_frames_for_source` (`vine360/ingest/frames.py`) now cascades
  fully: deletes masks for affected views, deletes the views, removes
  their `projections/<frame_id>/` directories, then removes the frames --
  mirroring `clear_views_for_frame`'s existing per-frame cascade.
- `views` and `masks` gained an `updated_at` column (migrated via the
  existing `_ensure_column` mechanism from ADR 0015). `compute_stage_
  statuses` now treats Pose estimation specially: since `sfm_runs` rows
  are kept as provenance history rather than overwritten, an old run can
  still exist after masks/views change underneath it. If the latest
  `views`/`masks.updated_at` is newer than the latest `sfm_runs.
  created_at`, Pose shows ACTIVE (stale, should re-run) instead of DONE.
  Frames/Projection/Masks don't need this special handling -- their old
  output is now genuinely deleted by the cascade fix above, so the
  existing count-based status logic already reflects reality.
- `run_sfm_for_project` gained an `image_source` parameter:
  `"projections"` (default, unchanged behavior) or `"frames"` (runs
  directly against a source's raw equirectangular frames with COLMAP's
  native `EQUIRECTANGULAR` model, no masking support yet on this path).
- New `vine360/sfm/spheresfm_adapter.py`: command-builder functions for
  SphereSfM's actual CLI shape (per its README, not source-verified --
  same honesty standard as the Nerfstudio adapter, ADR 0009).
  `validate_installation()` can only report whether *some* `colmap`
  binary is on PATH, not whether it's actually a SphereSfM build (both
  are named `colmap`).
- Pose Estimation panel gained an "Engine" selector: the two real options
  route through `run_sfm_for_project` with the right `image_source`/
  `camera_model`; SphereSfM shows an explanatory note and keeps its Run
  button disabled, matching how the Training tab already handles its own
  unverified adapter.

## Consequences

- Verified with real, unmocked tests: re-extracting a source that already
  had a view and a mask built now correctly leaves zero views/masks and
  removes the stale projections directory; a simulated "mask rebuilt
  after a successful SfM run" scenario correctly flips Pose from DONE
  back to ACTIVE; and the new `image_source="frames"` path runs real
  COLMAP feature extraction/matching against a real equirectangular image
  with the `EQUIRECTANGULAR` camera model (correctly reporting the same
  honest zero-parallax failure as the projections path for a
  single-position test clip -- proving the pipeline runs cleanly, not
  that registration itself succeeds without real parallax).
- This is the first automated (non-GUI-manual-smoke-test) coverage of
  `vine360.gui.main_window` -- `compute_stage_statuses` is pure logic
  over a `sqlite3.Connection` and needed no `QApplication`/display to
  test directly.
- Real-photo feature-matching quality for the native-equirectangular path
  (pole distortion, seam wraparound) remains unproven -- only synthetic,
  noise-free correspondences and a real-but-unregistrable single-position
  clip have been tested. This is exactly the gap SphereSfM's own
  sphere-aware matching claims to address, which is why it stays a
  documented option rather than being dismissed outright.
