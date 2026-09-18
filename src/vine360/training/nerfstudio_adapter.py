"""Nerfstudio Splatfacto adapter.

**UNVERIFIED, unlike this project's other adapters.** ffmpeg, pycolmap and
SAM 3's exact API calls were each confirmed against real installed
binaries/source or a real (if gated) live endpoint before being wired up.
Nerfstudio was deliberately not installed here (an explicit choice to keep
this milestone adapter-only, with no GPU training run) -- see
docs/adr/0009. The CLI shape below (`ns-train splatfacto --data ...
--output-dir ... --max-num-iterations ...`, `ns-export gaussian-splat
--load-config ... --output-dir ...`) reflects Nerfstudio's published
documentation (docs.nerf.studio) as of this project's context, not a
confirmed real invocation. **Run `ns-train splatfacto --help` and
`ns-export gaussian-splat --help` against a real install and correct this
file before depending on it for an actual training run.**
"""

from __future__ import annotations

import shutil
from pathlib import Path

from vine360.training.adapter import TrainingAdapter
from vine360.training.config import RunPaths, TrainingConfig


class NerfstudioAdapter(TrainingAdapter):
    method_name = "splatfacto"

    def validate_installation(self) -> dict:
        path = shutil.which("ns-train")
        return {"ns_train_found": path is not None, "path": path}

    def build_train_command(self, config: TrainingConfig, paths: RunPaths) -> list[str]:
        command = [
            "ns-train",
            self.method_name,
            "--data",
            str(config.dataset_path),
            "--output-dir",
            str(paths.run_dir),
            "--max-num-iterations",
            str(config.max_iterations),
            "--viewer.quit-on-train-completion",
            "True",
        ]
        for key, value in config.extra_args.items():
            command += [f"--{key}", str(value)]
        return command

    def build_export_command(self, checkpoint_config_path: Path, output_dir: Path) -> list[str]:
        return [
            "ns-export",
            "gaussian-splat",
            "--load-config",
            str(checkpoint_config_path),
            "--output-dir",
            str(output_dir),
        ]
