# 0010. Real capture kit is two 360-degree platforms, not 360 + conventional

## Status

Accepted.

## Context

The handover doc's P1 "Mixed capture" milestone (M7) and its terminology
throughout ("drone perspective images", "conventional drone photographs")
assume a conventional (rectilinear) drone camera alongside a 360-degree
ground camera. The actual capture kit is:

- An **Insta360** camera, ground-based -- matches the doc's assumed
  primary device.
- An **Antigravity A1** -- confirmed via web search to be an 8K
  **360-degree drone** (Insta360's Antigravity brand), not a conventional
  perspective drone. It outputs equirectangular video (`.insv`, exported
  to standard video up to 7680x3840) through the same kind of pipeline as
  the Insta360 footage, just airborne. A real sample capture
  (`VID_20260911_110250_002.mp4`, 7.65 GB, plus a `.gpx` sidecar created
  by "Antigravity Studio" with real per-second flight telemetry) confirms
  this.

So for this project, "mixed capture" means combining a ground-level 360
platform and an aerial 360 platform -- both equirectangular, both needing
the same projection pipeline -- not stitching true perspective drone
photos into an equirectangular-derived reconstruction.

## Decision

- `Source.capture_group` (already in the data contract, but previously
  hardcoded to `None` in `add_source`) is now a real parameter:
  `add_source(..., capture_group="antigravity-a1-aerial")`, exposed on the
  CLI as `vine360 ingest add-source ... --capture-group <tag>`. This is
  the mechanism for telling later milestones (M4's camera groups, M7's
  mixed-capture reconciliation) which physical rig a source came from,
  regardless of whether the two rigs are 360-vs-360 or 360-vs-perspective.
- `CaptureMode` (360 / conventional / mixed) is unchanged -- it still
  describes what kind of *media* a project accepts, which stays a
  meaningful, general concept for other users of this software who might
  have a true conventional camera. Nothing about this project's actual
  kit requires removing that option.
- No change was made to `infer_media_type`/`_resolve_projection`: both
  devices export standard 2:1-aspect equirectangular video, which the
  existing aspect-ratio-based equirectangular detection already handles
  correctly without caring which physical camera produced it.

## Consequences

- M7 (mixed capture) should be read, for this project, as "reconcile
  ground-360 and aerial-360 camera groups sharing the SfM problem" rather
  than "combine perspective and equirectangular projections" -- the
  camera-group mechanism is the same either way, but test fixtures and
  UI copy should reflect two 360 rigs, not a 360 + conventional pairing.
- The GPX sidecar file found with the real sample data is a much simpler
  source of GPS/position priors than the JPEG EXIF parsing ADR 0004
  deferred -- a GPX reader is a better first georeferencing (M8) step
  than EXIF parsing when that work starts.
- No frame extraction, projection, masking, or SfM has been run against
  the real sample footage -- it's large (7.65 GB+ per clip) and running
  anything against it needs to be an explicit ask, not an assumption.
