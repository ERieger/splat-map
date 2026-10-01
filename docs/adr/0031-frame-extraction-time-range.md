# 0031. Optional start/end time range for frame extraction

## Status

Accepted.

## Context

`extract_frames` (`vine360/ingest/frames.py`) always sampled the *entire* source video at a fixed
interval -- there was no way to limit extraction to a sub-range. For a real capture where the actual
useful footage sits somewhere in the middle of a longer recording (dead time before/after the actual
capture pass, camera setup, walking to the start position), this wastes ffmpeg decode time and disk
space on frames nobody wants downstream, and forces the user to live with a larger, noisier frame set
than necessary. This extends `extract_frames`'s design (docs/adr/0021 is the most recent ADR to touch
this function's internals, for its progress-reporting fix) rather than superseding it.

## Decision

**A new pure, testable resolution function**, `resolve_time_range(duration_seconds, *, start_time,
end_time) -> tuple[float, float | None]`, mirrors `resolve_interval_seconds`'s existing separation of
validation/resolution logic from ffmpeg execution. It takes user-facing absolute `start_time`/
`end_time` (seconds from the start of the source file; either/both may be `None`, meaning "from the
start"/"to the end") and returns ffmpeg-shaped `(effective_start, duration_limit)`.

**Flag choice: `-ss` before `-i` (input option) + `-t` after `-i` (output option), not `-ss`+`-to`.**
Once `-ss` is given as an input option, ffmpeg's output timeline resets to 0 relative to the seek
point -- a `-to` given alongside it would then need to already be range-relative rather than absolute,
which is confusing to reason about and to test (and easy to get subtly wrong at the call site). `-t
<duration>` sidesteps that ambiguity entirely: it's just "how much to decode from wherever `-ss`
landed," with no dependency on which timeline it's being measured against. `-ss` as an input option
(fast keyframe-based seek, not frame-exact) is acceptable here since this is already deterministic
interval sampling, not frame-exact single-frame extraction -- the existing whole-file path isn't
frame-exact either.

**`end_time` beyond the source's real duration is clamped, not rejected.** ffmpeg would just run to
EOF anyway if asked for more than exists; erroring on "asked for slightly more than exists" would be
unfriendly UX for a value that may have been derived from a slightly-stale duration estimate.
`start_time` at or past the real duration, and a negative `start_time`, are still hard errors
(`ValueError`) -- there's no sensible clamp for those.

**`resolve_interval_seconds` and the progress poller's `expected_frame_count` both use the *range's*
duration, not the whole file's**, so a `target_count` request means "N frames within the requested
range" and the progress bar's denominator stays accurate for a ranged extraction.

**`Frame.source_time` becomes `range_start + index * interval`, not `index * interval`.** This is the
one correctness-critical change: every downstream consumer (the temporal frame selector, SfM/pose
frame ordering) relies on `source_time` meaning "absolute seconds in the original source file," not
"seconds since this particular extraction started." A naive unfixed version would silently reset every
ranged extraction's frames back to appearing to start at t=0, corrupting that meaning for anything
built from them afterward.

**`ExtractionSettings`** (`vine360/config.py`) gains two new optional fields, `start_time_seconds` and
`end_time_seconds`, storing the *raw user-facing* values (not the clamped/resolved ones) -- matching
its existing "recorded verbatim... so extraction is reproducible" contract. No schema migration is
needed: `frames.extraction_settings` is a write-only JSON blob column, and nothing reconstructs an
`ExtractionSettings` object from a stored row, so older rows missing these keys are harmless.

**Exposed identically across CLI and GUI**, matching the app's existing "one typed library function,
called the same way by both surfaces" architecture: `vine360 ingest extract-frames` gains
`--start-time`/`--end-time` (independent optional flags, not part of the existing `--interval`/
`--count` mutually-exclusive group -- they compose with either); the GUI's Frames panel gains a
"Limit to a time range" checkbox plus Start/End spin boxes, disabled unless checked, mirroring the
panel's existing preset-vs-custom-interval enable-when-relevant pattern. The queue's
`_default_params_for_prerequisite(STAGE_FRAMES)` deliberately does *not* apply any range when
auto-inserting a Frames job as a prerequisite for a later stage (e.g. queuing Masks on a source with
no frames yet) -- a downstream job's range preference isn't something prerequisite auto-insertion
tries to infer; an auto-inserted Frames job always extracts the whole video.

## Consequences

- A partial-range re-extraction can *shrink* an existing frame set for a source that previously had a
  full-video extraction. This is already handled correctly by `clear_frames_for_source`'s existing
  unconditional destroy-and-rebuild-for-that-source-id contract (every call to `extract_frames`, range
  or not, already wipes and rebuilds frames/views/masks for the source first) -- nothing new needed
  there, just newly reachable via a smaller range than before.
- `build_frame_extraction_command(..., start_time=0.0, duration_limit=None)` (the "no range" case)
  produces the exact same argv as calling it with no range kwargs at all -- `start_time=0.0` is falsy
  and deliberately does not emit `-ss`, so every existing caller that never passes a range is
  byte-for-byte unaffected.
- The `Runner`/`LocalRunner` interface (docs/adr/0002) needed no changes -- it's a plain
  `Sequence[str]` in, `Result` out, with no awareness of flag position or meaning.
