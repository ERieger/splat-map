# 0047. Georeferencing (M8) with AprilTag ground control points

## Status

Accepted. Adds a Georeference stage between Pose estimation (ADRs 0036, 0046) and Export
(ADRs 0018, 0028, 0033). Follows the adapter-honesty (0002, 0007), verbose-progress and
activity-log (0037) and migrate-on-open (0003, 0012) rules.

## Context

Every SfM run is in an arbitrary model frame: unknown scale, orientation and origin. Vineyard
research needs metres in the survey's projected CRS. Each capture now has surveyed ground control
points (GCPs), each marked with a printed AprilTag. The survey files sit in each capture folder's
`gcp/` subfolder.

The handover (design session, 2026-10-25) fixed the approach below. Where the real files or the
installed libraries disagreed with the handover, the files and libraries win. Those differences
are listed under "Deviations from the handover".

## Decision

### Approach

1. **Detect tags on projected pinhole views, never on equirectangular frames.** Tag detectors
   assume a pinhole image. Every view already has intrinsics and a `fixed_rotation`, and
   `sfm/frame_poses.py` gives a COLMAP-convention pose for every view of a registered frame from
   either engine. So detection works the same for every engine.
2. **Find each tag centre in model space by multi-view triangulation:** DLT plus RANSAC over every
   registered view that saw the tag. Per-view PnP isn't used.
3. **Solve a 7-DOF similarity `X_local = s·R·X_model + t`** in closed form (Umeyama with scale,
   reflection-safe). No optimiser is used. At least 3 control points are required, with a warning
   below 5.
4. **Export a local metric frame, never raw MGA.** Northings near 6.1e6 lose centimetres in
   float32, which gsplat, Postshot and SuperSplat all use. Each control-point set stores an origin
   (E0, N0, H0). The exported model is in CRS − origin. A `georef.json` sidecar carries the
   origin, EPSG, s/R/t and the residual report, and `georef.prj` carries the CRS WKT. H0 = 0
   (heights near 360 m are fine in float32). The default origin floors the minimum E/N of the
   control/check points to 100 m. For Capture 2 that is (298900, 6126100, 0).
5. **v1 takes projected coordinates only** (E, N, H), with an EPSG code stored as metadata. It
   does no pyproj, lat/lon, datum or height conversion. The height datum is a free-text field
   (`ellipsoidal` / `AHD` / `unknown`) and is recorded but never converted.
6. **The transform is applied at export time** to the pinhole `pycolmap.Reconstruction`: the
   one `build_view_reconstruction` builds for a 360 run, or the run's own model for a projections
   run. It uses `pycolmap.Sim3d(scale, Rotation3d, translation)` and
   `Reconstruction.transform(new_from_old_world)`. Both names and the argument order were checked
   against the installed pycolmap 4.2.1 (`Sim3d.__init__` overload 2, `Reconstruction.transform`
   docstring), and a test pins them against a hand-computed point.
7. **New package `vine360/georef/`**, a new queue stage `georef` between `pose` and `export`, and
   four new index tables (below). Export gains an optional `georef_solution_id`.

### Answers from Elijah (2026-10-25)

- **Datum:** GDA2020 / MGA zone 54, **EPSG:7854**. The UI defaults to it, but EPSG remains a
  required input per import and is never inferred from the numbers.
- **Heights:** **unknown**. Stored as `"unknown"`.
- **Tag numbering:** the printed tags **match the point names** (`020` → tag 20). A per-point
  override still exists.
- **Tag size:** **"Some are 400, some are 250, size may change."** The edge length is therefore a
  **per-point** value (`control_points.tag_size_m`). The set-level `tag_size_m` is only an
  optional default for new points. See "Deviations".
- Tag family: tag36h11 (default, not contradicted). Check points: none by default; the user marks
  them in the table.

### Data model (`index.sqlite`, created by `_ensure_schema`)

- `control_point_sets(set_id, label, source_path, checksum, epsg, height_datum, tag_family,
  tag_size_m NULL, origin_east, origin_north, origin_height, imported_at)`
- `control_points(set_id, name, east, north, height, tag_id NULL, use, sigma_h NULL, sigma_v NULL,
  tag_size_m NULL, note NULL)`, primary key (set_id, name). `use` is `control` | `check` |
  `ignore`.
- `tag_observations(view_id, tag_family, tag_id, corners, center, decision_margin, hamming,
  detector, detector_version, params, detected_at)`, primary key (view_id, tag_family, tag_id).
- `georef_solutions(solution_id, run_id, set_id, transform, tag_positions, report, params,
  created_at)`.

**Cascades (the cascade-delete invariant, ADRs 0017/0019):**

- Every path that deletes `views` rows deletes their `tag_observations` first:
  `clear_frame_set`, `clear_views_for_frame`, and the Data manager's view deletion. Re-projecting
  a frame therefore invalidates its detections.
- `delete_sfm_run` deletes the run's `georef_solutions`.
- Deleting a control-point set deletes its points and its solutions.

### Pixel and pose conventions (measured, not assumed)

- **pupil-apriltags coordinates are COLMAP's**: pixel *i* spans [i, i+1), so its centre is at
  i + 0.5. Measured on area-sampled synthetic tags at known sub-pixel offsets, the centre error
  was below 0.05 px in that convention. vine360 view intrinsics are already in that convention
  (ADR 0036), so `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]` from `views.intrinsics` applies to
  detections unchanged.
- **Corners are at the outer edge of the black square** (the printed size we're given). The order
  is fixed in the tag's own frame: bottom-left, bottom-right, top-right, top-left for an upright
  tag. Corner *k* is therefore the same physical corner in every view. The measured corners sit
  0.1–0.3 px outward of the true edge, which is fine for a scale cross-check.
- `corner_scale` is defined as **tag_size_m / mean triangulated edge length (model units)**, in
  metres per model unit. That makes it directly comparable with the fitted `s`. The handover wrote
  it inverted (edge / size), which isn't comparable with `s`.

### Detector

**pupil-apriltags 1.0.4.post11** (cp312 manylinux wheel, installed cleanly) is in a new `georef`
extra. Its `Detector(families, nthreads, quad_decimate, quad_sigma, refine_edges,
decode_sharpening, debug)` and `detect(img: uint8 ndarray)` were checked against the installed
package with `inspect.signature`. No OpenCV is used. Test images come from the official
AprilRobotics `apriltag-imgs` tag36h11 PNGs (BSD-2-Clause, commit f3fd9a7, `fixtures/apriltag/`),
because the detector can't render tags.

### Queue and stage

`STAGE_GEOREF = "georef"` has two job kinds, in `params["kind"]`:

- `detect_tags`: targets a frame set. Its prerequisite is projection of that set (auto-queued as
  usual).
- `solve`: targets a run plus a control set. It is BLOCKED (never guessed) if the control set
  doesn't exist. It auto-queues `detect_tags` for each of the run's frame sets that has no
  detections or queued detection yet.

Sidebar status:

- pending: no control set and no observations;
- active: a control set or observations exist;
- done: a solution exists for the newest run.

### RealityScan

With a solution, `CameraPriors.csv` positions are in the local metric frame. The export also writes
`GroundControl.csv` (`#name,x,y,alt`, local frame, control and check points only). **The GCP
format is not verified.** RealityScan's help pages (rshelp.capturingreality.com, "Ground Controls
Import" and "Customize Import Formats") now redirect to the product homepage. Search snippets of
those pages only confirm three things: `#` lines are skipped, the column order is chosen in the
import dialog (or `groundcontrol.xml`), and samples use name, X, Y, altitude. Pick that layout and
a local Euclidean system on import. Same honesty caveat as ADR 0033.

### Non-destructive storage and selection (addendum, 2026-10-25)

- **An SfM run's model directory is never written.** Every solve inserts a new
  `georef_solutions` row with a new `solution_id`; re-solving never updates or deletes one. The
  solve then writes the transformed pinhole model to
  `sfm/georef/<run_id>/<solution_id>/` (`cameras.bin`, `images.bin`, `points3D.bin`,
  `georef.json`, `georef.prj`). For a 360 run that model is `build_view_reconstruction`'s output;
  for a projections run it is the run's own model, loaded into memory and written elsewhere. The
  path is stored in `georef_solutions.model_dir`, and `sfm/georef/` is a layout dir.
- **Export only copies.** With a solution selected, Export reads that solution's `model_dir`;
  with "None" it reads the run's `selected_model` as before. Both go through the same copy code,
  and `export_info.json` records `georef_solution_id` (or `"none"`). The file is deterministic, so
  a "None" export is byte-identical before and after a solve, which a test pins. `georef.json`,
  `georef.prj` and `export_info.json` are managed files, removed on re-export, so a later "None"
  export to the same folder can't inherit a stale georef.
- **Choosing a solution.**
  - The Export dropdown lists "None (original SfM frame)" first, then the run's solutions, newest
    first. Each is labelled "<created_at> RMSE 3D <x> m (<n> control pts)".
  - The choice is kept per run in `project.yaml` under `georef.active_solution` (new
    `Project.georef` field, omitted from the YAML while empty). "None" clears it.
  - Queued export jobs carry `georef_solution_id` explicitly, so a queued export doesn't depend on
    whatever choice is active later.
- **Deleting a solution** is a Data manager action, with confirmation. It removes the row and
  `sfm/georef/<run_id>/<solution_id>/` only, and forgets the solution as the run's active choice.
  Deleting a run removes its solutions and `sfm/georef/<run_id>/`. Deleting a control-point set
  removes its points and its solutions.
- **`transform_splat_ply` never writes in place.** The default output is
  `<stem>.georef-<solution_id[:8]>.ply` next to the input; output == input is refused, and output
  goes through a `.partial` file. The action is in the Export panel ("Georeference a trained
  splat…"), using the selected solution. It is only meaningful for a splat trained on a "None"
  export. **Unverified that Postshot keeps the imported COLMAP frame** in its output .ply; check on
  the first real splat.

### pupil-apriltags teardown bug (found while building this)

The full test suite segfaulted in 2 of 3 runs, in `quick_decode_uninit`. The cause is pupil-apriltags'
`Detector.__del__`: it calls `tag36h11_destroy(family)` **before**
`apriltag_detector_destroy(detector)`, and the latter's `apriltag_detector_clear_families` then
dereferences the freed family. That is a use-after-free, which crashes only when the freed memory
has been reused, hence the intermittency. AprilTag's documented order is detector first, then
family. `TagDetector.close()` (also called from `__del__` and the context manager) destroys them
in that order and nulls the library object's pointer, so the library's own `__del__` does nothing.
After the fix, 4 of 4 full-suite runs completed without a crash.

### Smaller implementation choices

- **`tag_detection_runs(view_id, tag_family, n_tags, detected_at)`.** Records every view a
  detection pass covered, including empty ones. Without it, "detected, nothing found" looks the
  same as "never detected" to the queue and the sidebar.
- **Detector defaults.** `quad_decimate` is 1.0, not the library's 2.0, because tags in a
  1024 px vineyard view are often 20–40 px across. Only `hamming == 0` and
  `decision_margin >= 20` detections are kept. If one id appears twice in a view, both are dropped.
- **Triangulation.** After DLT, each tag gets a few Gauss-Newton iterations on reprojection error
  (cheap, and more accurate than linear DLT alone). RANSAC runs over all view pairs, or 500 random
  pairs when there are more.
- **The keep-mask filter isn't implemented.** The handover listed "skip detections inside masked
  pixels" as optional and low priority.

## Deviations from the handover

1. **Real GCP files come in three shapes, not one.**
   - Capture 1's two files are **UTF-16 with a BOM, CRLF, and a header row**
     (`Point ID,Easting,Northing,Elevation[,GNSS Vector Observation.H. Precision,…V. Precision]`).
   - Capture 2's file (the handover's sample) and Flinders Bridge's file are UTF-8 with no header
     and a trailing comma.

   The reader therefore detects the BOM/encoding and the header. It maps header columns by name
   where it can, and otherwise by position.
2. **Precision is two values, H and V.** The single `sigma` became `sigma_h` and `sigma_v` (both
   optional, metres). A headerless file's 5th and 6th numeric columns are read as H and V.
3. **Per-point tag size** (Elijah's answer above). The handover had one size per set.
4. **Name → tag id.**
   - An all-digit name gives `tag_id = int(name)`, as the handover says.
   - New: a short alphabetic prefix followed by digits (`GCP000`, Flinders Bridge) gives
     `int(digits)`.
   - Either way the result must be a valid id for the family (tag36h11: 0–586). Otherwise it is
     None, which keeps `PRS778941614462` untagged.
   - The name string is preserved everywhere else.
5. **The reference station is ~10.4 km away, not 5.5 km.** PRS778941614462 is ΔE ≈ 5.45 km and
   ΔN ≈ 8.85 km from the other points' median. The handover appears to have quoted ΔE only. The
   generic far-point rule (more than 500 m from the median of the others → `use = "ignore"` with
   a note and a warning) handles it either way.
6. **Two `gcp/` READMEs warn that Capture 1 and Capture 2 reuse point ids at different places**
   (17–72 m apart). Sets are imported per file and solutions name their set, so the two are never
   mixed. The GUI shows the source path of each set.
7. **GUI sidebar indices shift.** Georeference becomes row 6, Export 7, Data manager 8, Activity
   log 9. Only the named constants are used, so nothing else depends on the numbers.

## Consequences

- No real end-to-end check is possible yet. The EstoWines Capture 1 footage has no tags, and
  Capture 2's footage isn't downloaded. Everything is verified synthetically:
  - similarity recovery to 1e-9;
  - triangulation with injected outliers;
  - tag detection on a view projected from a synthetic equirect panorama;
  - a pycolmap reconstruction transform;
  - a splat PLY transform.

  The first real check is Capture 2.
- `transform_splat_ply` rotates positions and Gaussian orientations and adds ln(s) to the log
  scales. **It does not rotate higher-order SH coefficients.** Degree-0 colour is unaffected, but
  view-dependent colour from degrees 1–3 will be slightly wrong after a large rotation. Fine for
  measurement, not for photoreal relighting.
- Raw MGA values never reach a float32 consumer. Anything that needs true CRS coordinates adds
  `georef.json`'s origin back.
