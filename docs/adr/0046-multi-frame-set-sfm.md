# 0046. Pose estimation over several chosen frame sets at once

## Status

Accepted. Extends frame sets (ADR 0034) and the SphereSfM engine (ADR 0036).

## Context

This project has two capture platforms for the same blocks: an Antigravity A1 flying overhead
and an Insta360 on the ground (ADR 0010). Reconstructing both captures together puts every
camera in one coordinate frame, and gives the splat both canopy-top and in-row viewpoints.
Separate runs can't do that; they would have to be aligned to each other afterwards.

Before this ADR, a Pose run could only be scoped to:

- **one frame set** (any engine), or
- **"All frame sets"** (the perspective-projections engine only). "All" includes every frame
  set in the project, so it also mixes a source's 0.5 s and 1 s sets. Those are near-duplicate
  images of the same moments.

The two raw-360 engines (COLMAP equirectangular and SphereSfM) only took one frame set. SphereSfM
is the better engine for this footage.

## Decision

**Any engine can run on a chosen list of frame sets** (`run_sfm_for_project(frame_set_ids=[...])`).
In the Pose panel, the frame-set dropdown has a **"Several frame sets…"** entry once there are at
least two sets to choose from. It shows a checklist and needs at least two sets checked.

How each image source handles several sets:

- **Projections:** the views of every chosen set are passed in. `projections/` is already
  keyed by frame_id, so no paths change.
- **Raw frames:** `image_dir` becomes `frames/`, and image names become
  `<frame_set_id>/frame_NNNNNN.png`, because every set numbers its frames from 1. A
  single-set run keeps bare names and `frames/<id>/` exactly as before, so existing runs, tests
  and SphereSfM's verified behaviour don't change. `project_run.frame_image_name` is the one
  place this naming rule is defined, and both the run and `frame_poses.load_frame_model` use it.

**One camera per frame set on the raw-frame engines.** The two sources are different cameras and
may record at different resolutions. (The EstoWines A1 and Insta360 frames happen to share
7680×3840, but a SPHERE camera is only right for one image size, so this can't be assumed.)

- **SphereSfM:** SPHERE's params are the image centre and are passed on the command line. So
  `feature_extractor` runs once per set into the shared database, each call with its own
  `camera_params` and its own camera. Steps 2–4 (matching, mapping, conversion) are unchanged.
  Each call gets its own log (`1_feature_extractor_<k>.log`) and image list
  (`image_list_<k>.txt`); `image_list.txt` still lists every image. Progress is counted across
  all sets ("set 2/2: frame 140/631").
- **pycolmap EQUIRECTANGULAR:** camera mode `auto`/`single` becomes `per_folder`, and a frame
  set is a folder here. The recorded `options.camera_mode` shows that.
- **Projections:** left as the user sets it. The faces are pinhole renders whose size and field of
  view the user picked per set, so the "Shared intrinsics" option already covers this case.

**Recorded scope.** `sfm_runs.config` gains `frame_set_ids` (the full list) and `source_ids`.
`frame_set_id`/`source_id` keep their single-set meaning and are `None` for a multi-set run. That
way older readers treat a multi-set run as unscoped, and never as a run of only its first set.
`project_run.run_frame_set_ids(config)` is the reader every consumer uses (frame poses, export,
priors selection). The Data manager and queue read the same keys without importing pycolmap.

**Downstream:**

- Postshot/COLMAP export of a multi-set raw-frame run converts and exports the projected views
  of every set in the run.
- RealityScan camera priors for one frame set prefer a run of exactly that set, then a
  multi-set run that includes it, then a project-wide run.
- A queued multi-set Pose job requires frames (raw-frame engines) or views (projections) for
  each set. It auto-inserts each missing prerequisite under the usual never-guess rules, and
  blocks if any one set can't be prepared.
- Exporting a multi-set run requires views for each of its sets.

**Warnings, not refusals.**

- `multi_set_matcher_warning`: sequential matching pairs each image only with its neighbours in
  the image list, and that list is one set after another. So two sets meet only at the single
  boundary between them, and usually come out as separate models. The panel shows this while
  the sets are being checked, and the run returns it in its warnings. Exhaustive matching, a
  vocabulary tree, or loop detection clear it.
- Checking two sets from the same clip shows a near-duplicate warning in the panel.

Neither is refused. Sequential matching with high overlap can still work on small sets, and the
user may have a reason to combine two sets from one clip.

## Consequences

- Combining aerial and ground footage is now a one-run operation for every engine. Whether
  SIFT matches a top-down A1 view to an in-row Insta360 view on real vineyard footage is
  **unmeasured**. The synthetic tests prove only the plumbing:
  `tests/test_multi_frame_set_sfm.py` renders two walks through the same room, the second
  0.8 m higher and at 768 px instead of 1024 px. Both engines register frames from both sets
  into one model, each set gets its own camera, and export covers both sets' views. On real
  footage, low oblique drone passes along the rows are the likely way to get matches between
  the two captures.
- A run's scope is now a list, so anything new that reads `sfm_runs.config` should use
  `run_frame_set_ids` rather than `frame_set_id`.
- Exhaustive matching grows with the square of the image count. On thousands of frames, a
  vocabulary tree (needs a tree file) is the practical choice.
