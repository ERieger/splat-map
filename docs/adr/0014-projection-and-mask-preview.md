# 0014. Image previews for projections and masks

## Status

Accepted.

## Context

Projection and Masks generated real files and database rows, but the GUI
only reported counts ("Generated 12 views.") -- there was no way to
actually look at what got produced without leaving the app to browse the
project folder. Requested directly: "add ability to preview masks and
projection frames in the app."

## Decision

- A shared `PreviewGallery` widget (a horizontally scrollable row of
  image thumbnails, `QPixmap`-loaded from disk) is used by both panels.
  It never raises on a missing/unreadable file -- shows a grey
  placeholder instead, since a stale db row pointing at a since-moved
  file shouldn't crash a preview.
- **Projection panel**: a "Preview frame" combo (populated from that
  source's frames, showing each frame's timestamp and current view
  count) drives a gallery of that frame's generated face images,
  captioned by face name. Refreshes on source/frame selection change and
  after a successful generation.
- **Masks panel**: a "Preview view" combo (populated from that source's
  views, showing each view's keep-fraction once masked) drives a gallery
  showing the original view image and, if built, its keep-mask side by
  side. Refreshes the same way.
- Both comboboxes' population avoids firing their `currentIndexChanged`
  handler mid-rebuild (`blockSignals`) and explicitly triggers a preview
  refresh once population is complete, rather than relying on incidental
  signal firing during `clear()`/`addItem()` calls.

## Consequences

- Verified with a real offscreen pipeline run: after generating
  projections for a 3-frame clip, the first frame's preview showed 4 real
  thumbnails (the default front/right/back/left faces); after masking,
  the view preview correctly grew from 1 item (original only) to 2
  (original + keep-mask) and the view combo showed the real computed
  keep-fraction (69%) for that view.
- No overlay/blend view (e.g. mask tinted over the original image) was
  built -- side-by-side is simpler and was enough to verify masks visually
  in testing; worth adding if side-by-side proves hard to read in
  practice.
