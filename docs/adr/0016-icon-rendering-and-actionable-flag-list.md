# 0016. Fix unrenderable emoji icons; make the auto-flagged list actionable

## Status

Accepted.

## Context

Reported directly: "the icons don't render in the app." Confirmed the
cause: this machine has no emoji font installed at all --
`fc-match "Noto Color Emoji"` falls back to plain DejaVu Sans, and
`fc-list :charset=1F6A9` (the flag emoji used on the "Flag for Review"
button) and `:charset=23EE`/`23ED` (the previous/next-track arrows) all
return nothing. `⚠` (U+26A0, used in two plain warning labels) *is*
covered by DejaVu Sans and was left alone.

Separately: "when I ran the mask the first time there was a list of
items that got flagged to check -- can they be automatically flagged?"
Auto-flagging was already working (`is_keep_fraction_anomalous` sets
`Mask.flagged_for_review` at build time, per ADR 0015) -- confirmed again
here with a real offscreen run. The actual gap: the list shown after a
build was a static, unclickable text dump.

## Decision

- Never rely on emoji glyphs rendering via whatever font happens to be
  installed. Added `_flag_icon()` (a small `QPainter`-drawn flag pixmap,
  the same technique already used for the sidebar status dots) and used
  Qt's built-in `QStyle.SP_ArrowLeft`/`SP_ArrowRight` standard icons for
  the Previous/Next Flagged buttons -- neither depends on any font at
  all. The "Flag for Review" button, and each face combo item, use the
  drawn flag icon (`QIcon`) instead of an emoji character in the text.
- The post-build flagged list (`flags_summary_label` + new `flags_list`,
  a real `QListWidget`) is double-click-actionable: double-clicking an
  entry jumps the frame slider and view combo straight to it, instead of
  only being reachable one-at-a-time via Previous/Next Flagged.
- Also removed an unused `QSizePolicy` import found while auditing this
  section of the file.

## Consequences

- Verified the flag pixmap actually contains non-transparent drawn
  pixels (70 out of 14x14) and that all three affected buttons report a
  non-null icon -- confirming the fix doesn't just look plausible in
  source but produces real, renderable icon data regardless of the
  system's font/emoji support.
- Verified end-to-end with a real offscreen pipeline run: after masking,
  the flags list contained a real auto-flagged entry, and double-clicking
  it correctly navigated to that exact view.
- If a genuinely rich icon set becomes worth the effort later (many more
  icons than the handful here), consider bundling a proper icon font or
  SVG icon set as an application resource rather than drawing each one by
  hand -- but for this small number of icons, hand-drawn `QPainter`
  pixmaps and Qt's standard icons are simpler and have zero external
  dependency.
