# 0034. Frame sets: several extraction configs per source

## Status

Accepted. Extends 0017/0019 (cascade delete, per-run SfM directories), 0026/0028 (export
source filter, nested export layout) and 0031 (time range).

## Context

`extract_frames` always cleared a source's previous frames before writing new ones
(`clear_frames_for_source`), so a source could only ever have one extraction. In practice
the user wants to compare configs side by side -- e.g. the same clip at every 0.5s *and* every
1s -- and carry each through projection, masking, SfM and export independently. Every
downstream stage, the queue, and every GUI selector was keyed on `source_id`, so "two
extractions of one source" had nowhere to live.

## Decision

**A frame set is one extraction config of one source.** New `frame_sets` table
(`frame_set_id, source_id, label, extraction_settings, created_at`) and a
`frames.frame_set_id` column. Frames live under `frames/<frame_set_id>/` with frame ids
`<frame_set_id>:NNNNNN`. Projection, masking, six-face/equirectangular SfM and the two
images-only exports are all scoped by `frame_set_id` (`generate_views_for_frame_set`,
`build_masks_for_frame_set`, `run_sfm_for_project(frame_set_id=...)`,
`export_*(frame_set_id=...)`). Every GUI stage's dropdown lists frame sets ("clip.mp4 — every
0.5s").

**Ids are deterministic per config: `<source_id>~<tag>`** (`i0.5`, `i1_r10-60`, `n200` for a
count-mode run), via `frame_set_id_for`. Two consequences:
- Re-extracting the same config replaces only that frame set (cascading to its views and
  masks, rows and files); a different interval or time range adds a new set. Thumbnails don't
  count toward identity.
- The queue can target a frame set *before* it exists. A queued Frames job stores its target
  frame set id, so a Projection job added afterwards depends on it. Auto-inserting a missing
  Frames prerequisite is only done for the balanced-preset set, since any other config would
  mean inventing settings nobody chose. Anything else is BLOCKED, not guessed (same rule as
  ADR 0023/0030).

**Legacy migration: id = source_id.** On open (`_migrate_legacy_frame_sets`, part of
`_ensure_schema`), each source's pre-existing frames become one frame set whose id *is* the
source_id. Every path and id already built from the source_id -- `frames/<source_id>/`,
`<source_id>:NNNNNN`, `projections/<frame_id>/`, `masks/keep/<frame_id>/`, a "frames"-engine
SfM run's recorded `source_id`, queued jobs' `target_source_id` -- is therefore already
correct, with no file moved. `frame_set_id_for` reuses an existing set whose settings match,
so re-extracting a legacy set's own config replaces it rather than duplicating it. Checked
against a copy of a real project's database (280 frames, 1680 views, 1086 masks).

**SfM gets an explicit image list and its own database per run.** `run_sfm_for_project`
hands pycolmap `image_names` (built from the DB) instead of a bare directory scan. Checked
against the installed pycolmap 4.2.0 `extract_features(image_names=...)` signature, with a
real-call test that extraction really is limited to the subset. The database moved from the
shared `sfm/database.db` into `sfm/sparse/<run_id>/database.db`: a scoped run would otherwise
see an earlier run's images left in the shared database. Two existing bugs were fixed along
the way. The equirectangular engine's directory scan would have picked up `thumbs/*.jpg`.
Six-face `total_images` counted per-frame *directories* under `projections/`, not images.

**Six-face SfM can run project-wide ("All frame sets") or on one frame set.** If a source has
two frame sets, "All" feeds COLMAP near-duplicate images of the same moments. The Pose panel
says so; it doesn't forbid it.

**Mask files now cascade too.** `clear_views_for_frame` used to delete mask *rows* but leave
`masks/keep/<frame_id>/` and `masks/classes/<view_id>/` on disk. It now removes them via the
new stdlib-only `vine360.cleanup` (shared with `clear_frame_set` and the Data manager, ADR
0035).

## Consequences

- Export folder defaults use `<source stem>_<tag>` for a frame set (`exports/clip_i0.5/
  postshot/`); a legacy set keeps the bare stem, so its exports land where they always did.
- `_select_priors_run` (RealityScan camera priors) matches by frame set. It never picks a
  run scoped to a *different* frame set, of either engine.
- Old persisted queue jobs reload with `target_frame_set_id = target_source_id` (Projection/
  Masks) or read `params["source_id"]` as the frame set (Pose/Export). Both resolve to the
  legacy frame set.
