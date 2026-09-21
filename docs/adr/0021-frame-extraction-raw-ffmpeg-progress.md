# 0021. Real progress during raw ffmpeg frame extraction

## Status

Accepted.

## Context

The user asked whether frame extraction has a proper progress bar while
one was actively running. Checking `extract_frames`
(`vine360/ingest/frames.py`) confirmed a real gap: it has two phases --
the raw ffmpeg extraction (one blocking `runner.run(command)` call) and a
per-frame thumbnail-generation loop after it. Only the second phase ever
reported real `(current, total)` progress (docs/adr from that earlier
fix); the first phase always reported `(None, None)` for however long it
took. For a real high-resolution/360 source, that first phase is very
likely the *slower* of the two (decoding and writing potentially
hundreds of full-resolution frames), so the indeterminate spinner during
exactly the longest wait looked identical to a hang -- the same failure
mode the thumbnail loop was already fixed for, just left in the other
phase.

`Runner.run()` (`vine360/runners/base.py`) is a deliberately simple,
fully synchronous interface -- it returns only after the subprocess
exits, with no hook for incremental output. Changing that interface
would ripple into every adapter built on it (ffmpeg, COLMAP, and any
future trainer). ffmpeg's `image2` muxer (the `frame_%06d.png` output
pattern already used here) writes each numbered frame to disk as soon as
it's decoded, though, so the output directory's file count is a real,
if approximate, progress signal available without touching that call at
all.

## Decision

`extract_frames` now starts a small daemon thread
(`_poll_output_frame_count`) immediately before the blocking ffmpeg call
and stops it in a `finally` once that call returns. The thread polls
`output_dir.glob("frame_*.png")`'s count every `poll_interval_seconds`
(default 0.5s, a new keyword argument mainly so tests can shrink it well
below a real ffmpeg call's duration) and reports `(count, expected_total)`
through the same `progress_callback` the thumbnail loop already uses --
`expected_total` is computed from the already-known
`duration_seconds / interval`, the same arithmetic `resolve_interval_seconds`
uses. No change to `Runner` or any other adapter.

## Consequences

- Verified with a pure unit test (`_poll_output_frame_count` run directly
  against a temp directory, files created on a timer from the test
  itself -- no ffmpeg involved) and a real-ffmpeg integration test
  extension confirming the raw-extraction phase now reports at least one
  non-`None` progress event even for a fast synthetic clip (via a tiny
  `poll_interval_seconds` so the poller has time to fire before the
  subprocess returns).
- This is an approximate signal, not an exact one -- the last file may
  still be mid-write when counted, undercounting by at most one, and it
  says nothing about ffmpeg's own internal decode progress if it ever
  buffers output. Good enough to distinguish "still working, N of M so
  far" from "looks hung," which is the actual problem being solved.
- Doesn't touch `add_source`'s checksum progress or the thumbnail loop --
  both already report real progress by other means.
