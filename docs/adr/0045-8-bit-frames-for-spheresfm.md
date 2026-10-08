# 0045. Always extract 8-bit frames; refuse 16-bit frames before SphereSfM runs

## Status

Accepted. Amends frame extraction (`vine360.ingest.frames`) and the SphereSfM pipeline (ADR 0036).

## Context

A queued SphereSfM pose run on the EstoWines Insta360 frame set (888 frames) failed after
1 h 36 min. Feature extraction took 96 minutes and logged `ERROR: Failed to read image file
format.` for every one of the 888 frames, yet it exited 0. The sequential matcher then aborted
on the empty database with `cache.h:132] Check failed: max_num_elems > 0 (0 vs. 0)` in
`FeatureMatcherCache::Setup`, which says nothing about the real cause. The A1 frame set in the
same project ran fine (338/343 registered).

Cause: the Insta360 source is 10-bit HEVC (`yuv420p10le`); the A1 source is 8-bit (`yuvj420p`).
With no `-pix_fmt`, ffmpeg's PNG encoder kept the extra precision and wrote **16-bit RGB PNGs**
(`rgb48be`, ~109 MB per 8K frame, 77 GB for the set). SphereSfM's COLMAP 3.8 can't read 16-bit
RGB PNGs through FreeImage. This was checked against the real build: on the same frame scaled
to 1920×960, the 16-bit copy fails and the 8-bit copy gives 12,452 features.

Nothing else noticed. Pillow silently reduces 16-bit PNGs to 8-bit when it opens them, so
projection and masks worked, and the views and masks built from those frames are already 8-bit.

## Decision

- **Frame extraction always writes 8-bit RGB** (`-pix_fmt rgb24`). Nothing downstream uses more
  than 8 bits per channel: Pillow, SAM 3, COLMAP/pycolmap SIFT and Postshot import all work on
  8-bit. It also cuts frame storage to roughly a quarter for 10-bit sources.
- **SphereSfM preflight** (`spheresfm_adapter.check_readable_images`): before a run directory
  is created, the first and last frames' PNG headers are read (25 bytes each). A bit depth
  above 8 raises `SphereSfmError` at once, telling the user to re-extract the frame set.
- **Post-extraction check**: `ProgressParser` counts "Failed to read image file format" lines.
  If every frame was unreadable, the run stops after step 1 with that message and a pointer to
  the log, instead of handing the matcher an empty database. A partial count is reported in
  the progress log as frames skipped.

## Consequences

- Frame sets already extracted as 16-bit must be re-extracted (or converted to 8-bit) before
  SphereSfM can use them. Re-extracting cascades as usual: views and masks are rebuilt too.
- The pycolmap engine wasn't re-checked for 16-bit input. 8-bit extraction makes that moot for
  new frame sets.
- Tests: `test_png_bit_depth_reads_the_header`, `test_16_bit_frames_are_refused_before_the_run_starts`,
  and `test_the_real_binary_cant_read_16_bit_pngs_and_the_parser_counts_it` (real SphereSfM,
  skipped when absent). The last one records the failure itself, so a future SphereSfM build
  that reads 16-bit PNGs will show up as a failing test.
