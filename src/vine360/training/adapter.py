"""Training backend adapter contract (handover doc, section 3's Adapter
rule; section 4, step 7 "Train"). See docs/adr/0009: this is deliberately
adapter-only -- no training backend is installed or run by this codebase,
per an explicit choice not to spend GPU time / large installs on it yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from vine360.runners.base import Runner, RunResult
from vine360.training.config import RunPaths, TrainingConfig


class TrainingAdapter(ABC):
    @abstractmethod
    def validate_installation(self) -> dict:
        """Reports whether the backend's CLI is on PATH and its version,
        without raising -- a missing backend is reportable state."""
        raise NotImplementedError

    @abstractmethod
    def build_train_command(self, config: TrainingConfig, paths: RunPaths) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def build_export_command(self, checkpoint_config_path: Path, output_dir: Path) -> list[str]:
        raise NotImplementedError

    def run_training(self, config: TrainingConfig, paths: RunPaths, runner: Runner) -> RunResult:
        command = self.build_train_command(config, paths)
        return runner.run(command, cwd=paths.run_dir)

    def run_export(self, checkpoint_config_path: Path, output_dir: Path, runner: Runner) -> RunResult:
        command = self.build_export_command(checkpoint_config_path, output_dir)
        return runner.run(command)
