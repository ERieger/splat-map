# 0032. Parallel frame projection across worker processes

## Status

Accepted.

## Context

`generate_views_for_source` (`vine360/projection/generate.py`) rendered every extracted frame's
perspective faces strictly sequentially, one frame at a time, even though each frame's rendering
(`project_equirect_to_face`, `vine360/projection/render.py`) is a pure numpy/PIL computation that
reads only that frame's own equirect image and writes only that frame's own view files -- no
dependency on any other frame. On a real capture (hundreds of frames) this is the slowest stage in
the pipeline after frame extraction, and the work is embarrassingly parallel across frames.

This is the first use of `multiprocessing`/`concurrent.futures` anywhere in this codebase.
Deliberately not routed through the `Runner`/`Adapter` pattern (docs/adr/0002): that abstraction
exists to keep Windows/WSL/Linux *external tool invocation* (ffmpeg, COLMAP) behind one interface --
this parallelizes the project's own pure Python/numpy code, a different concern entirely.

A concrete memory constraint shapes the design: `render.py`'s `_bilinear_sample_equirect` does
`equirect.astype(np.float64)` on the **whole** equirect image once per face rendered. For an 8K
source (the real capture resolution used elsewhere in this codebase), that's roughly a 700MB float64
allocation per face call, so each worker peaks around ~1GB of RAM while rendering one frame. Running
one worker per CPU core unconditionally could multiply that into many GB on a high-core-count
machine -- a real risk, not a theoretical one, so the worker count needed to be capped by design
rather than defaulting to "use every core."

## Decision

**New pure, module-level function** `_render_and_save_frame_views(project_root, frame_id,
equirect_relpath, faces) -> list[View]` holds the entire CPU-bound half of frame projection (open
image, render each face, save, build `View` objects) with **no `sqlite3.Connection` involved at
all**. `generate_views_for_frame` (the existing single-frame entry point, public signature and
behavior unchanged) now calls this plus a new `_insert_view_rows(conn, views)` helper that does the
existing per-view `INSERT` + `commit`. Keeping the render function connection-free is what makes it
safe to run inside a worker process: no sqlite3 object ever crosses a process boundary, and
`ProcessPoolExecutor` needs a plain, picklable, module-level callable anyway (`FaceSpec` and `View`
are plain dataclasses with only str/int/dict/numpy-array fields -- confirmed picklable, and `spawn`
never transmits a pickle across machines, only to a same-machine sibling process, so there's no
cross-OS `Path` pickling concern either).

**`generate_views_for_source` gains `max_workers: int | None = None`.** `None` (the default) means
sequential, in-process, byte-identical to the previous behavior and progress-message format -- every
existing caller that doesn't pass it is completely unaffected. `max_workers >= 2` on a source with
more than one frame switches to a `ProcessPoolExecutor`. A source with only one frame (or
`max_workers <= 1`) always takes the sequential path regardless of what was requested, since paying a
process pool's real startup cost (new interpreter boot, re-importing numpy/PIL per worker) for a
single unit of work has no benefit.

**`spawn`, chosen explicitly over the platform default**, for two independent reasons: it's the only
option on Windows at all, and it avoids inheriting the parent's open `sqlite3.Connection` file
descriptor into child processes (even though children never touch it, an inherited fd is an
unnecessary hazard to carry). Separately, `spawn` is exec-based even on POSIX/WSL2 -- a child process
never duplicates the parent's live Qt/QThread state, unlike `fork`, which would risk copying a
half-initialized Qt internal state into the child. This matters concretely here because the GUI
always calls into this function from inside `run_in_background`'s background `QThread`, never the
main thread -- `spawn`'s exec semantics make starting a process pool from that thread unhazardous.

**Every frame's stale views/masks are cleared up front, sequentially, before any rendering starts**
(parallel or not) -- this ordering is a correctness requirement, not an optimization: it must
complete before any new file is written for any frame, avoiding a race between a worker writing frame
X's new files and frame X's old directory not being cleared yet.

**Progress reporting intentionally differs in shape between the two paths.** Sequential:
`"Projecting frame {index+1}/{total}…"`, 0-based `current` reported *before* that frame is processed
(unchanged from before). Parallel: `"Projecting frame {completed}/{total}…"`, 1-based, reported
*after* each frame completes, via `as_completed()`'s completion order (which is not submission
order under real concurrency, so "index of the frame currently starting" is meaningless there). This
asymmetry is deliberate and documented here specifically so a future reader doesn't try to unify the
two formats.

**A worker exception propagates naturally** via `future.result()`, matching the sequential path's
existing behavior of letting PIL/numpy errors propagate uncaught -- no new wrapping. Any frame whose
worker had already started rendering (and written files) but whose result was never reached because
an earlier future's exception stopped the loop first leaves orphaned PNG files with no DB rows for
that frame. This is **not a new gap**: `clear_views_for_frame`'s unconditional `shutil.rmtree` on the
next rerun already removes any such stray files -- the existing cascade-delete safety net already
covers it.

**GUI**: `ProjectionPanel` gained a "Speed up with parallel processing" checkbox (default checked)
plus an adjustable worker-count spin box next to it, ranged `1..os.cpu_count()` and seeded with
`min(os.cpu_count() or 1, 4)` as its initial value -- a capped *default*, not a hard ceiling, because
of the ~1GB-per-worker figure above; unchecking the box passes `None` (sequential) regardless of the
spin box's value. The initial revision of this feature shipped with only the checkbox and no
worker-count control at all (a deliberate scope decision at the time, made before any real-machine
numbers existed) -- the spin box was added immediately after, once real usage surfaced that the flat
default of 4 left real throughput on the table on a higher-core, high-RAM machine (a 6-core/12-thread
Ryzen 5 7600X with 64GB RAM, where even 12 workers at ~1GB each is nowhere near the RAM ceiling): the
cap only ever protected against a plausible worst case (many cores, little RAM), not every real
machine, so a fixed number could never be right for everyone -- letting the user see and adjust the
actual value is more honest than guessing a single "better" constant. The resolved worker count is
captured as a concrete value at "Add to Queue"/"Run now" time, matching ADR 0023's "every job's
params are captured in full when added, never re-derived at run time" principle. An auto-inserted
Projection prerequisite job (e.g. when queuing Masks on a source with frames-but-no-views)
deliberately stays on the sequential default -- it's an implicit job the user never directly
configured, and shouldn't silently start consuming several GB of RAM without an explicit choice.

## Consequences

- `_generate_views_worker` (`main_window.py`) still doesn't thread `include_polar_faces` through at
  all -- an existing, unrelated gap noticed while touching this function's signature, not fixed here.
- **Explicitly out of scope**: `render.py`'s `.astype(np.float64)` cast happens once *per face*, not
  once *per frame* -- it recomputes the same whole-image float64 copy for every face of the same
  frame. Casting once per frame instead would directly cut the per-worker memory figure this ADR's
  worker cap is built around, which would in turn let the default cap safely go higher. This is a
  distinct change to the pure render path (`render.py`), deliberately not bundled into this one -- a
  candidate follow-up, not a requirement of this change.
- No CLI wiring needed or added: projection has no CLI subcommand at all (per CLAUDE.md's
  architecture section), so this is GUI/queue-only, matching the existing scope of that stage.
- **Verified for real, not just in theory**: a manual benchmark (12 synthetic 4K frames, `face_size=1024`)
  measured 16.3s sequential vs. 9.0s with `max_workers=4` -- a genuine wall-clock speedup, not just a
  correctness exercise. That same benchmark also surfaced a real, standard Python `spawn` constraint:
  calling `generate_views_for_source(..., max_workers=N)` from a plain script whose own top-level code
  isn't guarded by `if __name__ == "__main__":` raises `RuntimeError` ("An attempt has been made to
  start a new process before the current process has finished its bootstrapping phase..."). This is
  not a bug in this change -- it's an inherent property of the `spawn` start method (the child
  re-executes whatever the calling process's own `__main__` module is, as part of the child's
  bootstrap, regardless of which module the actual target function lives in) -- but it's a real
  caveat for any future caller. `vine360/gui/main_window.py` already has this guard
  (`if __name__ == "__main__": raise SystemExit(main())`), so the GUI's own usage is unaffected;
  confirmed pytest is unaffected too (its own entry point is properly guarded, and the new tests in
  `tests/test_projection_generate.py` that exercise `max_workers >= 2` pass). Noted directly in
  `generate_views_for_source`'s docstring as well.
