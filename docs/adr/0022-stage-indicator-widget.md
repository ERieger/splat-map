# 0022. Per-panel stage indicator for tabs with more than one action

## Status

Accepted.

## Context

The user asked for a stage indicator on tabs that have multiple actions,
so it's clearer which step you're on within that tab (distinct from the
sidebar's own per-*stage* status dots, which track progress across the
whole pipeline, not within a single panel). Auditing every panel's
buttons found two real candidates -- panels with genuinely distinct,
sequenced actions rather than alternative/independent ones:

- **Masks**: "Build Masks" (a batch action for the whole source) then a
  separate flag/review workflow (Previous/Next Flagged, Flag for
  Review) once masks exist -- a real build-then-review sequence.
- **Export**: configure (mode, SfM run if applicable, output folder)
  then "Export for Postshot" -- plus a separate one-time "Repair" action
  (left out of the stepper; it's project maintenance, not part of the
  normal export flow).

Project (create/open), Import (add/remove source) and Pose Estimation
(configure engine, run) were considered and left out: their multiple
buttons are alternatives or CRUD operations on a list, not a sequence
within completing that tab.

## Decision

New `StageIndicator(QWidget)`: a small horizontal row of
labelled status dots (one per stage, connected by "→"), reusing the
sidebar's own `_status_dot`/`STATUS_COLOR` (DONE/ACTIVE/PENDING)
vocabulary rather than inventing a second visual language for the same
idea, per the existing "dashboard, favor visual elements" direction.
`set_stages(list[str])` updates all dots/labels at once and records the
statuses on `.statuses` for inspection (tests, mainly).

- **MasksPanel**: `["Build masks", "Review flagged"]`. Computed from
  `_flat_views` (already queried for the frame/face selectors) --
  Build=ACTIVE until any mask exists for the selected source, then DONE;
  Review=PENDING until masks exist, ACTIVE while any view is still
  flagged, DONE once none are. Updated wherever `_flat_views` already
  gets refreshed (`_refresh_frame_list`, after a build, after toggling a
  flag) -- no new refresh triggers needed.
- **ExportPanel**: `["Configure", "Export"]`. Configure=DONE once the
  current mode's requirements are met (output folder chosen, and an SfM
  run selected if in poses mode); Export=ACTIVE once configured but not
  yet exported, DONE after a successful export *of the current
  configuration* -- changing the mode or output folder resets this back
  to ACTIVE, since a prior export no longer describes the new choice.

## Consequences

- First automated (non-manual-smoke-test) coverage of real QWidget panel
  behavior in this project (`tests/test_gui_stage_indicator.py`, run
  under the offscreen Qt platform) -- everything else in `vine360.gui`
  so far was either pure logic or a manual smoke test (docs/status.md's
  noted pytest-qt gap still stands; this doesn't add that framework,
  just tests these specific widgets directly under `QApplication`).
- Purely additive UI, no behavior change to the underlying build/export
  functions -- verified the app still launches cleanly and existing
  panels are unaffected.
