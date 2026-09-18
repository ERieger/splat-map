"""Dependency probing: "validate installation, report version and
capabilities" (handover doc, Adapter rule), applied here to the tools
vine360 itself shells out to rather than to a specific engine adapter.

M0's exit criterion is "versions are reported" -- this module is what the
`vine360 version` command reports. It must never raise just because a tool
is missing; a missing COLMAP or ffmpeg is expected and reportable state,
not a crash.
"""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass

from vine360 import __version__
from vine360.runners.base import Runner
from vine360.runners.local import LocalRunner


@dataclass
class DependencyStatus:
    name: str
    found: bool
    version: str | None
    path: str | None


def _probe(name: str, version_command: list[str], runner: Runner) -> DependencyStatus:
    path = shutil.which(name)
    if path is None:
        return DependencyStatus(name=name, found=False, version=None, path=None)
    result = runner.run(version_command)
    output = (result.stdout or result.stderr or "").strip()
    version = output.splitlines()[0] if output else None
    return DependencyStatus(name=name, found=result.ok, version=version, path=path)


def probe_dependencies(runner: Runner | None = None) -> list[DependencyStatus]:
    runner = runner or LocalRunner()
    return [
        _probe("ffmpeg", ["ffmpeg", "-version"], runner),
        _probe("ffprobe", ["ffprobe", "-version"], runner),
        _probe("colmap", ["colmap", "-h"], runner),
        _probe("git", ["git", "--version"], runner),
        _probe("nvidia-smi", ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], runner),
    ]


def report() -> dict:
    return {
        "vine360_version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": [d.__dict__ for d in probe_dependencies()],
    }
