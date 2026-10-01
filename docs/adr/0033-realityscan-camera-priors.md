# 0033. Optional camera-priors CSV export for RealityScan

## Status

Accepted.

## Context

While diagnosing a real RealityScan export's ground-level vineyard-row footage completely failing to
register (0 of 1200 images, versus 1118/1200 for the same project's aerial pass — confirmed directly
from RealityScan's own `Registration.csv` output on a live project), the root cause identified was the
classic repetitive-scene structure-from-motion problem: adjacent vineyard rows look nearly identical
to a pure feature-matcher, which can't reliably tell "this post in row 5" from "the same-looking post
in row 8." The user asked whether either external tool vine360 exports to (Postshot, RealityScan)
supports importing position "priors" to help guide its own alignment through exactly this kind of
ambiguity.

**Research (against each tool's real, official public documentation) confirmed this is real, but
RealityScan-only:**

- **RealityScan** has a genuine, documented pre-alignment feature called **Camera Priors**
  (rshelp.capturingreality.com), imported via WORKFLOW tab → Import Metadata → **Trajectory** (a
  customizable-column-layout CSV, reader `RealityScan.Import.CSVFlightLog`). Documented fields include
  position (`X, Y, Altitude`), orientation (`Yaw, Pitch, Roll`), and full intrinsics/distortion —
  closely mirroring this codebase's own `Registration.csv` *output* shape. Critically, a per-image
  "Absolute pose" prior state of **"Position" alone (no orientation)** is documented as a real,
  independently-valid prior — not a degraded fallback — refined (not necessarily locked) during
  alignment, with a continuous "hardness" dial the user tunes inside RealityScan itself. This is a
  distinct mechanism from Ground Control Points (which anchor 3D scene points, not cameras).
- **Postshot** (jawset.com/docs) has no equivalent. Its `Camera Poses` setting is strictly binary:
  `Import` (trust a complete external COLMAP/Bundler/etc. reconstruction wholesale, skipping its own
  tracking entirely) or `Estimate` (run its own COLMAP-based tracking from scratch). No GPS, partial
  pose, or prior-position input is surfaced anywhere in its documented UI, CLI, or release notes.
  **This feature is therefore RealityScan-only** — `export_frames_and_masks_for_postshot` is
  deliberately untouched.

No RealityScan-authored guidance specifically discusses row-crop/repetitive-scene use of Camera
Priors — applying the general mechanism to this specific problem is this project's own domain
reasoning, not a documented Epic recipe.

The only position data vine360 can offer today is its own already-completed COLMAP SfM run (when one
exists) — there is no GPS/GPX ingestion yet (a separate, unstarted future milestone; see ADR 0004/0010).
This is sufficient: RealityScan runs its own alignment from scratch on the same image set, so an
internally-consistent but otherwise arbitrary/uncalibrated coordinate frame is exactly what a
"Position" prior needs — it only has to be self-consistent, not absolutely correct.

## Decision

**Position-only, no orientation, deliberately.** Shipping orientation too would require
reverse-engineering RealityScan's exact yaw/pitch/roll axis/order convention with no real install
available to verify against — and a wrong convention could actively corrupt alignment rather than help
it. This follows this project's established adapter-honesty discipline (see e.g. docs/adr/0018/0028):
don't ship an unverified transform presented as correct. A camera's world position (`x, y, alt`) has
no such convention ambiguity.

**No pose-composition math needed, for either of vine360's two SfM "engines."** vine360's own SfM can
run against either the six-face projected images (`"projections"` engine) or the raw equirectangular
frames (`"frames"` engine). Verified for real against the installed `pycolmap` (a synthesized
reconstruction, cross-checked `Image.projection_center()` against the manual qvec/tvec route to
1e-6): because every cube face shares the panorama's own optical center — `fixed_rotation`
(`vine360.projection.cubemap`) is rotation-only, zero translation relative to the panorama — composing
a frame's world pose with a face's fixed rotation is a translation no-op. **Every face view under one
frame gets exactly that frame's own `projection_center()`.** This means priors can be drawn from
*either* engine's completed run with no `Pose`/`compose()` involvement at all — which matters
concretely here: the six-face engine is the one more likely to have *also* failed on the same
repetitive content this feature exists to work around, while the native-equirectangular engine (whole
panoramas, far more context/overlap per frame) is plausibly the one actually likely to succeed and
produce a useful prior. Supporting only the six-face case would likely make this feature useless for
the exact scenario that motivated it.

**Run selection prefers maximal overlap, not just "most recent."** `_select_priors_run` (`postshot.py`)
prefers a `"frames"`-engine run scoped to exactly the export's `source_id` (guaranteed overlap) over a
project-wide `"projections"` run (always covers any source) over any other run — and never picks a
`"frames"` run scoped to a *different* source when `source_id` is given, since that's guaranteed zero
overlap and would silently produce an empty or useless CSV.

**Every failure mode is non-fatal.** No usable `sfm_runs` row, a run whose model is missing on disk, or
a run with no overlap with the exported image set all degrade to a warning appended to
`PostshotExportResult.warnings` — the images+masks export itself always succeeds regardless of whether
priors could be produced. The GUI checkbox (`ExportPanel.priors_checkbox`) is additionally gated
up front by `realityscan_priors_available` (a cheap DB-only check), so the common case never even
reaches export time with a doomed request.

**File**: `output_dir/CameraPriors.csv`, header `#name,x,y,alt` — the `#name` column matches the exact
flattened filename already on disk in `images/` (reusing the existing `_flatten_to_unique_filename`),
so a row is directly resolvable to its image file. Only images the export actually copied get a row;
an sfm run that only partially overlaps the exported set is fine — RealityScan's own "Unknown"
absolute-pose default already covers an image with no corresponding row.

## Consequences

- `export_for_realityscan` gains `include_camera_priors: bool = False`; `PostshotExportResult` gains
  `num_priors: int = 0` and `priors_path: Path | None = None` (safe defaults, every existing
  construction site already uses keyword args — non-breaking).
- `ExportPanel` gains a checkbox, gated by `realityscan_priors_available(conn, source_id=...)`,
  force-disabled and unchecked for any non-RealityScan mode.
- `export_frames_and_masks_for_postshot` and the Postshot queue path are explicitly untouched.
- Same "unverified against a real RealityScan install" caveat as ADR 0018/0028 applies here too — test
  actually importing a real `CameraPriors.csv` via WORKFLOW → Import Metadata → Trajectory before
  relying on this for a real project; whether it measurably helps RealityScan register the vineyard-row
  footage that motivated this feature can't be confirmed without that real install.
