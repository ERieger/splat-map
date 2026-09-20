# 0013. Per-face view selection; make existing project state visible on open

## Status

Accepted.

## Context

Further review after ADR 0012 raised three issues:

1. "Opening project doesn't correctly load state. I think I already
   generated frames but that isn't loading." Investigated with a real
   offscreen test that creates a project, extracts frames, closes the
   connection, and reopens the same project in a **fresh** window
   instance (and separately, reopens it in the *same* running window).
   Both scenarios loaded the data correctly (`compute_stage_statuses`,
   the Frames source combo, etc. all reflected the prior work). The
   actual problem: nothing in the UI *displayed* existing counts --
   `refresh_sources()` on Frames/Projection/Masks always showed just a
   filename, with no indication a source already had N frames/views/
   masks. Correctly-loaded data with no visible confirmation looks
   identical to data that failed to load.
2. "Make it clearer that the video source for projection is the frames
   not the raw video." The Projection panel's combo was labeled "Video
   source", and its description didn't say projection operates on
   already-extracted frame images, not the original video file.
3. "Add individual options for the various views." Projection only
   offered a single "include polar faces" checkbox -- no way to pick,
   say, only front+right, or only front+up.

## Decision

- `vine360.projection.cubemap.six_face_preset` gained a `face_names`
  parameter (an explicit subset, in the doc's front/right/back/left/up/
  down order) that overrides `include_polar_faces` when given. Threaded
  through `generate_views_for_frame`/`_for_source`. The Projection panel
  now shows six individual checkboxes (front/right/back/left checked,
  up/down unchecked, by default) instead of one polar toggle.
- Frames/Projection/Masks panels' source combos now show existing
  artifact counts computed live from the database (e.g. "3 frames, 6
  views already generated" / "not masked yet"), via `JOIN`/`LEFT JOIN`
  aggregate queries -- not new state to track, just displaying what
  `compute_stage_statuses` already proved was loading correctly.
- Projection panel's combo relabeled to "Source (its extracted frames)"
  and its header text now explicitly says it renders from
  already-extracted frames and does not touch the raw video again;
  Masks panel's combo relabeled to "Source (its generated views)" for
  the same reason.

## Consequences

- Verified with a real offscreen test: generating with only `front` and
  `up` checked produced exactly `2 faces x 3 frames = 6` views (not the
  default 4-face set), and reopening a project with prior frames/views
  in a fresh window correctly showed "3 frames already extracted" /
  "3 frames, 6 views already generated" / "6 views, not masked yet" in
  the respective panels.
- The underlying "state loading" mechanism (`AppState`, `on_project_
  changed`/`on_shown` refresh hooks, `compute_stage_statuses`) needed no
  changes -- it was already correct. This is a reminder that a
  correctness bug report can sometimes be a visibility gap; verifying
  with a real reproduction before changing the suspected code paid off
  here (no changes were made to the actual load path).
