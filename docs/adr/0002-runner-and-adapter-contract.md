# 0002. Runner interface and adapter contract for external engines

## Status

Accepted.

## Context

The handover doc requires that "Windows paths and WSL/Linux execution"
stay "behind a runner interface", and that every external engine (ffmpeg,
COLMAP, a 3DGS trainer, a segmentation model) implement a small internal
contract: "validate installation, report version and capabilities,
translate a typed configuration into a command, stream structured
progress, capture stdout and stderr, expose expected artifacts, and
classify failures." It explicitly forbids UI code from constructing engine
commands directly.

## Decision

- `vine360.runners.base.Runner` is the single abstraction point for
  executing an external command. `LocalRunner` is the only implementation
  needed for M0/M1 (it runs directly on the current OS); a WSL-bridging
  runner is a plausible future `Runner` implementation but is not built
  until something in scope needs it.
- Every adapter separates **command construction** (a pure function taking
  typed arguments, returning `list[str]`) from **execution** (a function
  that takes a `Runner` and actually calls it). This is what makes
  `build_ffprobe_command`, `build_frame_extraction_command` and
  `build_thumbnail_command` unit-testable without ffmpeg installed.
- `vine360.runners.probe` implements "validate installation, report
  version and capabilities" for the tools vine360 itself depends on
  (ffmpeg, ffprobe, colmap, git, nvidia-smi), returning a status list that
  never raises on a missing tool -- a missing dependency is reportable
  state, not a crash. This is what backs `vine360 version`.

## Consequences

- COLMAP, SAM 3 and the 3DGS trainer adapters (M3-M5) should follow the
  same split (pure command builder + `Runner`-based execution) rather than
  introducing a different pattern per engine.
- Only ffmpeg/ffprobe have adapters so far; the full adapter contract
  (streamed structured progress, artifact expectations, failure
  classification) is partially realized -- ffmpeg calls here are
  synchronous and classify failure only as "ok/not ok" with the process's
  stderr attached. Streaming progress and artifact declarations should be
  added when COLMAP and training adapters need them (M4/M5), since a
  short-lived ffmpeg extraction doesn't yet demand it.
