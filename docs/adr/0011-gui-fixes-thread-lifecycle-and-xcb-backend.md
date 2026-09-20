# 0011. Real bugs found via use: QThread lifecycle crash, frame re-extraction
collision, and switching to the xcb Qt backend

## Status

Accepted.

## Context

Three real problems surfaced from actually using the app against real data:

1. **"UNIQUE constraint failed" on re-extracting frames.** `Frame.frame_id`
   is deterministic (`f"{source_id}:{index:06d}"`), and `extract_frames`
   never cleared a source's previous frames before inserting new ones. Any
   second extraction of the same source (a completely reasonable action --
   "try a different interval") collided with the first extraction's rows,
   and could also leave stale files behind if the second extraction
   produced fewer frames than the first.

2. **A process abort ("QThread: Destroyed while thread is still running")**,
   found while smoke-testing the new remove-source/custom-interval
   features. `run_in_background`'s original design stored the in-flight
   `(thread, worker)` pair in a single attribute on the calling panel.
   `worker.finished`/`worker.failed` (our own signal, meaning "the
   callable returned") fire and call `thread.quit()` -- but `quit()` only
   *requests* the thread stop; it does not block until it actually has.
   If a second background call started on the same panel before the first
   `QThread` had genuinely stopped (confirmed only by Qt's own
   `QThread.finished` signal), the second call overwrote the attribute,
   dropping the only Python reference to a still-running `QThread` and
   aborting the process.

3. **Window-layering complaints**: message boxes sometimes appearing
   behind the main window, and combo-box dropdowns "getting stuck". The
   app was running under `QT_QPA_PLATFORM=wayland` (WSLg's native Wayland
   compositor, Weston) because `xcb` failed at startup with `xcb-cursor0
   or libxcb-cursor0 is needed`, and that system library couldn't be
   `apt install`-ed without root. Wayland deliberately gives client
   applications less control over window stacking than X11, and Qt's
   Wayland popup-grab handling has known rough edges -- both symptoms are
   consistent with the Wayland backend specifically, not the app's own
   dialog/combo-box code.

## Decision

1. `vine360/ingest/frames.py` gained `clear_frames_for_source(conn,
   project_root, source_id)`, which deletes a source's `frames` rows and
   its output directory. `extract_frames` now calls it unconditionally
   before extracting, making re-extraction with different settings safe
   and idempotent (the prior extraction's artifacts are simply replaced,
   since nothing else in this codebase yet depends on frame rows as
   "consumed" -- see docs/adr/0003's immutability note for when that
   changes).
2. `run_in_background` now keeps a **list** of `(thread, worker)` pairs
   per owner (`owner._bg_pairs`), and only removes an entry once
   `QThread.finished` fires -- the real "the OS thread has stopped"
   signal, not our own `worker.finished`/`failed`. This makes concurrent
   or rapid-fire background operations on the same panel safe against the
   premature-garbage-collection crash.
3. **The GUI now runs under `QT_QPA_PLATFORM=xcb`** (X11 via Xwayland),
   not `wayland`. The missing `libxcb-cursor0` was obtained without root
   via `apt-get download libxcb-cursor0` (downloading a `.deb` doesn't
   need root) and `dpkg-deb -x` (extracting one doesn't either), landing
   at `~/.local/lib/xcb-cursor/libxcb-cursor.so.0*`. Launch with:
   ```
   LD_LIBRARY_PATH="$HOME/.local/lib/xcb-cursor:$LD_LIBRARY_PATH" \
   QT_QPA_PLATFORM=xcb PYTHONPATH=src python3 -m vine360.gui.main_window
   ```

## Consequences

- `vine360.ingest.sources.remove_source` also had to reuse
  `clear_frames_for_source` (imported lazily to avoid a circular import
  with `frames.py`, which already imports from `sources.py`) so removing
  a source cleans up its frames the same way.
- Also added: a "Remove Selected Source" button (Import panel, confirms
  before deleting, never touches the original file) and a "Custom" frame
  preset with a spin box (Frames panel), both requested alongside the bug
  reports above.
- If message-box/dropdown issues persist even under `xcb`, that would
  point at something in the app rather than the Wayland backend
  specifically, and is worth a fresh look; but this is the standard,
  well-supported Qt/X11 path and a reasonable first real fix to try.
- `~/.local/lib/xcb-cursor` is outside the repo and machine-specific --
  anyone else setting this up on a similarly rootless WSLg machine needs
  to repeat the `apt-get download`/`dpkg-deb -x` steps themselves; not
  worth automating into the repo for a one-machine workaround.
