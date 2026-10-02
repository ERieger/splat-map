"""The runner interface that keeps Windows/WSL/Linux execution details out
of every other module (handover doc, section 3: "Keep Windows paths and
WSL/Linux execution behind a runner interface").

No adapter (ffmpeg, ffprobe, and later COLMAP / trainers) should call
subprocess directly -- everything goes through a Runner so command
construction, execution and logging stay separable and testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass
class RunResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Runner(ABC):
    """Executes an external command and returns its captured output.

    Every external engine (ffmpeg, COLMAP, a 3DGS trainer, ...) is invoked
    through a Runner rather than constructing subprocess calls itself, per
    the handover's "Adapter rule".
    """

    @abstractmethod
    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_output: "Callable[[str], None] | None" = None,
    ) -> RunResult:
        """on_output, if given, is called with each line of the command's
        combined stdout/stderr as it's produced -- what lets a long external
        run (SphereSfM, ffmpeg) report live progress instead of nothing
        until it exits."""
        raise NotImplementedError
