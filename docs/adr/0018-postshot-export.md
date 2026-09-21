# 0018. Export stage for Postshot (and other COLMAP-based external trainers)

## Status

Accepted.

## Context

Training itself stays adapter-only (docs/adr/0009); the user wants to hand
this project's already-computed poses (and masks) to real external 3DGS
training software instead -- specifically Postshot (Jawset, commercial,
Windows) and, separately, LichtFeld Studio (open-source). Investigating
what that import actually requires, against each tool's own documentation
(not memory), surfaced two real gaps:

1. **A real, pre-existing bug**: `run_sfm_for_project` recorded
   `sfm_runs.selected_model` as the run's own `run_id` string, not the
   COLMAP model directory it actually wrote. This field exists
   specifically to answer "which subdirectory under `sfm/sparse/` holds
   the selected reconstruction" -- and it's a real question, not a
   formality: `pycolmap.incremental_mapping(output_path=...)` writes
   *every* candidate reconstruction it finds to `sfm/sparse/<key>/`, not
   only the best one (confirmed against the real project at `/mnt/e/TEST`,
   which has five numbered subdirectories from four `sfm_runs` rows).
   Picking the wrong one would silently hand Postshot a worse or
   unrelated reconstruction.
2. **A format mismatch, not a missing feature**: the user asked whether an
   "embed alpha masks" pipeline stage was needed. Postshot's own docs
   (Interface/Training Configuration, "Image Masks") say masks are
   **separate black-and-white images matched by filename**, "the same
   base filename" as the color image, same resolution -- not an
   alpha channel baked into the color image (a phrase from Postshot's
   release notes that turned out to describe something else internal, not
   the documented import mechanism). vine360's own `masks/keep/*.png` is
   already exactly this format (255 = keep, 0 = excluded) -- the only real
   gap is the *filename*: COLMAP's own `mask_path` convention (which
   vine360's masks already follow, since `masks/keep/` doubles as COLMAP's
   `mask_path` root) appends an extra `.png` to the full original filename
   (`front.png` -> `front.png.png`), which does not match Postshot's "same
   base filename" rule. No alpha-embedding stage is needed; a renaming
   copy is.

## Decision

- `colmap_adapter.map_and_diagnose`/`run_sfm` now return the actual
  written model directory as a third tuple element, and
  `run_sfm_for_project` stores it (relative to `project_root`, e.g.
  `"sfm/sparse/0"`) as `selected_model` -- fixing the bug above. All
  callers/tests updated for the new 3-tuple.
- New `vine360/export/postshot.py`: `export_for_postshot(conn,
  project_root, output_dir, run_id=None)` bundles the given (default:
  most recent) SfM run into `output_dir/{sparse,images,masks}/`:
  - copies just the three canonical COLMAP files from the run's real
    `selected_model` directory;
  - copies every image the reconstruction actually registered (read back
    via `pycolmap.Reconstruction(model_dir).images`) from
    `projections/` or `frames/<source_id>/` depending on the run's
    recorded `image_source`, preserving the relative paths COLMAP's
    `images.bin` expects;
  - if masks were built (`image_source == "projections"` only -- no
    masks exist against raw frames yet), copies each keep-mask to the
    *same relative path and filename as its color image*, stripping the
    COLMAP-only extra `.png`.
  - Raises `PostshotExportError` with an actionable message for: no SfM
    run yet, a run whose `selected_model` predates this fix (needs
    re-running), or a missing/incomplete model directory. Warns (doesn't
    fail) on a missing source image or an empty mask set.
- GUI: `build_export_panel()`'s disabled stub replaced with a real
  `ExportPanel` -- pick an SfM run (listed with its registration stats),
  choose an output folder (defaults to `project/exports/postshot/`), and
  export in the background, same `run_in_background` pattern as every
  other stage. Success message includes the exact Postshot import steps
  (which folder is the COLMAP dataset, which is the images root, which
  files to drop on the Image Masks list).

## Consequences

- Verified with real tests (`tests/test_export_postshot.py`): real
  `pycolmap.synthesize_dataset` + `map_and_diagnose` reconstructions,
  real file copies, real re-load of the exported sparse folder via
  `pycolmap.Reconstruction` -- not mocked, since this module's whole job
  is file wiring around an already-written model.
- **Unverified against an actual Postshot install** (Windows-only
  commercial software, unavailable in this environment) -- verified only
  against Postshot's published documentation, which itself doesn't fully
  specify COLMAP import folder layout or confirm equirectangular/360
  camera model support (marketing/review sources claim it; Postshot's own
  "Importing Images" page doesn't mention it at all). The default
  six-face-projection engine's pinhole-camera poses are the safe,
  portable choice for this export regardless.
- Masks are only ever exported for the `"projections"` engine; the
  native-equirectangular engine (docs/adr/0017) has no masks to export
  yet, and the export function reports this rather than failing.
