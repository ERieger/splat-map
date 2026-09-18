# 0009. Training: adapter architecture only, no backend install or run

## Status

Accepted.

## Context

Section 4, step 7 ("Train") and M5 call for training a 3DGS backend
(Nerfstudio Splatfacto or a gsplat-based backend). The handover's own
instructions caution against downloading large weights or starting long
GPU jobs without being asked; when given the explicit choice ("adapter
only, don't launch training" vs. "install backend and run a real short
training job"), the answer was adapter-only.

## Decision

- `vine360/training/config.py`: typed `TrainingConfig` matching the data
  contract's Training run fields, plus `prepare_run_dir`/`write_run_config`
  wiring into the section-5 project layout (`runs/<run-id>/config.yaml`,
  `checkpoints/`, `logs/`).
- `vine360/training/adapter.py`: the `TrainingAdapter` contract (validate
  installation, build train/export commands, run through a `Runner`),
  matching the pattern already used for ffmpeg.
- `vine360/training/nerfstudio_adapter.py`: a concrete implementation.
  **This one is unverified**, unlike every other adapter in this project.
  ffmpeg's commands were confirmed by actually running them (ADR 0006);
  pycolmap's and SAM 3's exact API calls were confirmed by reading their
  real installed source (ADR 0007, ADR 0008). Nerfstudio was not
  installed, so its CLI shape here (`ns-train splatfacto --data ...`,
  `ns-export gaussian-splat --load-config ...`) reflects published
  documentation, not a confirmed real invocation.

## Consequences

- No GPU time was used and no large ML training stack was installed for
  this milestone.
- Before this adapter backs a real training run, run `ns-train splatfacto
  --help` and `ns-export gaussian-splat --help` against an actual
  Nerfstudio install and correct `nerfstudio_adapter.py`'s flag names if
  they've drifted from what's assumed here -- do not trust it the way the
  other adapters can be trusted.
- If gsplat (rather than Nerfstudio) turns out to be the better fit once
  a real vineyard sample is available (section 12's open question), add a
  `GsplatAdapter` implementing the same `TrainingAdapter` contract; nothing
  else in the codebase depends on which concrete adapter is used.
