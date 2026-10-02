# 0038. Layered masks, an overexposure layer, and review that acts

## Status

Accepted.

## Context

A real tunnel capture (FMC-Tunnel1) had a blown-out exit with blooming around it. Postshot trained
that glare into wispy floaters along the whole length of the tunnel. Masking it needed a new
exclusion class, and three problems with the existing design got in the way:

- **No independent classes.** The keep-mask was baked in one pass (union → remove small → dilate →
  invert), with one `masks` row per view. Per-class PNGs were written, but nothing ever read them
  back. Changing one class, such as a different clip level, meant rebuilding every class,
  including slow SAM 3 classes.
- **Classical sky always ran.** It treats any pixel brighter than 180 in the top 60% of a view as
  sky. In a tunnel, that cuts out white walls and ceilings.
- **Review was hard to use.**
  - Changing frame reset the View combo to its first item ("back"), so one perspective couldn't
    be scrubbed through time.
  - "Flag for review" auto-flagged views keeping < 30% or > 98% of their pixels. That is most
    down and side faces of any capture with nothing masked in them. Nothing downstream acted on
    a flag.

## Decision

**Layers.** Each class is a finished binary PNG at `masks/classes/<view_id>/<layer>.png`
(255 = exclude). Morphology is applied per layer before saving, so the file on disk is exactly
what gets merged. Each layer has a `mask_layers` row (`view_id, layer, method, params, enabled,
edited, coverage, updated_at`). The layers are `sky` (classical | sam3), `person` (sam3),
`overexposed` and `excluded_view`.

**The composite stays where every consumer already looks.**
- `compose_view` ORs the enabled layers into `masks/keep/…png.png` and `exclude.png`, and upserts
  the one-per-view `masks` row.
- The composite is rebuilt whenever a layer changes, not deferred to export, because pycolmap SfM
  reads `masks/keep` too.
- SfM and all three exports also call `ensure_composites_current` first. It picks up layer PNGs
  edited outside vine360 (file mtime newer than the row), marks them `edited`, and recomposes any
  view whose composite is older than its newest layer or is missing.
- Stage status, Pose staleness (`masks.updated_at`), Data manager counts and exports are
  unchanged.
- Regeneration skips `edited` layers unless told to overwrite them.

**Cascades.** Layer files live in the directory the cascade deletes already remove
(`cleanup.remove_mask_files`). `mask_layers` rows are deleted alongside `masks` rows in
`clear_views_for_frame`, `clear_frame_set` and `delete_masks_for_frame_set`.

**Legacy projects.** On opening the Masks panel, `register_legacy_layers` renames old class PNGs
to layer names and registers them. Each row is dated by the file's own mtime, so registering
doesn't make the composite, or a Pose run built on it, look stale.

**Overexposure (`masking/overexposure.py`).** Masking is driven by *clipping*, not brightness:
- The core is pixels with `min(R,G,B) >= 250`. Cores smaller than 0.2% of the view are dropped,
  which removes glints.
- Bloom is pixels with luma ≥ 230 that lie within `bloom_radius_px` (40) of a core *and* belong
  to a bright component touching it. The result is then dilated by 6 px.
- A correctly exposed white wall (~200–245, textured) is never masked on its own, and the halo
  can't run along an adjoining bright wall.
- On 400 random FMC-Tunnel1 views this masked the clipped exit and its bloom edge, and left the
  white and yellow brick walls alone. It took ~0.13 s per 1024² view.

**Sky is optional** (default on, so existing behaviour is unchanged).

**Review acts.**
- The selected face sticks across frame changes. ←/→ step frames and ↑/↓ step faces.
- One overlay preview tints each enabled layer in its own colour.
- Per-view layer checkboxes and a set-wide "merged" toggle switch layers without rebuilding them.
- "Flag for review" is replaced by:
  - **"Exclude this view"**: a full-frame `excluded_view` layer. The all-excluded keep-mask means
    COLMAP finds no features and trainers get no supervision from that view.
  - **A suspicious-views list**: a layer whose coverage differs by more than 15 percentage points
    from the median of the same face in the ±3 neighbouring frames.

`masks.flagged_for_review` stays in the schema but is no longer written.

## Consequences

- Layer morphology is per class rather than applied once to the union. For dilation this is
  equivalent. Tiny specks are removed per class, so two sub-threshold specks of different classes
  no longer combine.
- Legacy class PNGs are pre-morphology, so a recompose of a legacy view can differ by a few edge
  pixels. Regenerating that layer gives an exact result.
- SAM 3 sky and person each run a separate inference per view, as the adapter already did per
  prompt.
- There is no in-app brush editor. Hand edits are made to the layer PNG in an external editor and
  picked up automatically.
