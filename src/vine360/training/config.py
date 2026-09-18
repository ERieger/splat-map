"""Training run configuration (handover doc, section 4, step 7 "Train";
data contract "Training run": run_id, backend_version, dataset_hash,
config, checkpoint, metrics, output_path)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    backend: str  # e.g. "nerfstudio-splatfacto"
    dataset_path: Path
    max_iterations: int = 30000
    extra_args: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "dataset_path": str(self.dataset_path),
            "max_iterations": self.max_iterations,
            "extra_args": self.extra_args,
        }


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    config_path: Path
    checkpoints_dir: Path
    logs_dir: Path


def prepare_run_dir(project_root: Path, run_id: str | None = None) -> tuple[str, RunPaths]:
    """Creates project/runs/<run-id>/{checkpoints,logs} per the section 5
    project layout. Does not write config.yaml itself -- call
    `write_run_config` once the config is finalized."""
    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    run_dir = Path(project_root) / "runs" / run_id
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return run_id, RunPaths(
        run_dir=run_dir,
        config_path=run_dir / "config.yaml",
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
    )


def write_run_config(paths: RunPaths, config: TrainingConfig) -> None:
    with open(paths.config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
