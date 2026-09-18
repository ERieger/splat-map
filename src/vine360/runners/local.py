from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Sequence

from vine360.runners.base import Runner, RunResult


class LocalRunner(Runner):
    """Runs a command directly on the current OS (native Windows or Linux/WSL).

    A WSL-bridging runner (invoking WSL from a native Windows process) is a
    reasonable future implementation of the same Runner interface, but
    nothing in the handover's M0/M1 scope requires it yet.
    """

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> RunResult:
        command = list(command)
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd) if cwd else None,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            returncode = completed.returncode
            stdout, stderr = completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as exc:
            returncode = -1
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\ntimed out after {timeout}s"
        duration = time.monotonic() - started
        return RunResult(
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
        )
