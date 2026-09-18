# 0004. Assumptions adopted for M1, pending answers in section 12

## Status

Proposed -- these are assumptions, not confirmed answers. See "Open
questions for Elijah" (handover doc, section 12) and docs/status.md.

## Context

Section 12 leaves several inputs open that affect ingest behavior. M1 had
to pick concrete defaults to be buildable and testable now.

## Decision

- **Input format**: the handover's own text names "exported 2:1
  equirectangular media" as "the safest MVP target," so `add_source`
  targets exported equirectangular MP4/MOV video and 2:1 still images
  first. Raw Insta360 `.insv` camera files are accepted by extension but
  not specially handled (no un-stitching step) -- if raw files turn out to
  be the required input, ingest needs a stitching adapter before M1's
  frame extraction is actually usable end-to-end.
- **Equirectangular detection**: aspect ratio alone (2:1 within 2%) decides
  whether to *ask* for confirmation; it never silently classifies an image
  as equirectangular, per section 4's explicit requirement. Video sources
  at 2:1 are still required to pass `--media-type video` or
  `--confirm-equirectangular` is not asked for video (only for still
  images), because the CLI has no interactive prompt in M1 -- the
  confirmation is a caller-supplied flag, not a runtime prompt.
- **GPS/EXIF**: only ffprobe's own container-level tags are read (covers
  video formats, including whatever Insta360 X6 exports embed as location
  tags). JPEG EXIF GPS parsing for drone stills is not implemented; if
  georeferencing (M8) needs it before then, add it as its own adapter
  rather than folding ad hoc EXIF parsing into `media_probe.py`.
- **Compute environment**: this M0/M1 work targets the WSL2/Linux side per
  the handover doc's own "Secondary environment" line, matching the
  machine this was built and tested on.

## Consequences

Any of these can be overridden without a redesign: `add_source` already
takes an explicit `media_type_override`, and the ffprobe-based metadata
path is isolated in `vine360/ingest/media_probe.py` if EXIF parsing needs
to be added later.
