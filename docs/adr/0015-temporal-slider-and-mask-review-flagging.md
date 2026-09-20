# 0015. Temporal slider for frame selection; explicit review flagging for masks

## Status

Accepted.

## Context

The Projection/Masks preview selectors were plain dropdowns -- workable
for a handful of frames, but a real capture easily has 100+ frames, and
a 100-entry combo box is a poor way to scrub through a timeline.
Separately, `is_keep_fraction_anomalous` already auto-detects likely-bad
masks after a build, but there was no way for a user to flag a mask they
personally judged as needing another look, or to navigate directly
between flagged views once several exist.

## Decision

- New `TemporalFrameSelector` widget (a `QSlider` + a label showing
  "Frame i/N — t=X.XXs — M views"), shared by Projection and Masks. Both
  panels' "which frame" control is now this slider instead of a combo.
- `Mask` gained a `flagged_for_review: bool` field, distinct from the
  existing `edited` field (flagged means "needs a look"; edited means "a
  human already corrected it"). Persisted as a new `masks.flagged_for_
  review` column. Since `CREATE TABLE IF NOT EXISTS` (ADR 0012's schema
  migration) does nothing for a column missing from an *existing* table,
  `project.py` gained `_ensure_column` (a `PRAGMA table_info` + `ALTER
  TABLE ADD COLUMN` check), called for this column on every open --
  verified against a simulated older `masks` table missing the column.
- `build_mask_for_view` auto-sets `flagged_for_review` from `is_keep_
  fraction_anomalous` at build time (a real, empirically-hit case during
  testing: a view with 98.7% kept genuinely tripped the "too much kept"
  threshold). `set_view_flagged(conn, view_id, flagged)` lets the user
  freely toggle it afterward without re-running the (potentially slow,
  SAM-3-loading) build.
- Masks panel now has: the frame slider, a small "View" combo for
  picking a face within that frame (labeled with its keep-fraction and a
  🚩 marker if flagged), a checkable "🚩 Flag for Review" button
  reflecting/toggling the current view, and "Previous/Next Flagged"
  buttons that cycle through every flagged view for the selected source
  (wrapping around; showing an info dialog if none are flagged).
  Projection panel's frame slider similarly drives its existing face
  preview gallery.

## Consequences

- Verified with a real offscreen pipeline run: scrubbing the Projection
  slider correctly changed which frame's 4 real thumbnails were shown;
  in Masks, manually flagging a view persisted to the database
  immediately, and next/prev-flagged navigation correctly cycled between
  it and a genuinely auto-flagged view (98.7% keep-fraction) from the
  same build -- confirming the auto-flagging path fires for real, not
  only in unit tests with contrived inputs.
- `TemporalFrameSelector.set_frames` always resets to index 0 and always
  emits `frame_changed` (even calling it with signals blocked
  internally, then a manual emit) so callers get one predictable refresh
  per repopulation, rather than relying on Qt's implicit "did the index
  actually change" signal semantics -- this was deliberately chosen to
  avoid the same class of "signal didn't fire because the value looked
  unchanged" surprise hit earlier in this project.
