from __future__ import annotations

import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Callable, Sequence

from vine360.runners.base import Runner, RunResult


class LocalRunner(Runner):
    """Runs a command directly on the current OS (native Windows or Linux/WSL).

    A WSL-bridging runner (invoking WSL from a native Windows process) is a
    reasonable future implementation of the same Runner interface, but
    nothing in the handover's M0/M1 scope requires it yet.
    """

    MAX_KEPT_LINES = 2000

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> RunResult:
        command = list(command)
        if on_output is not None:
            return self._run_streaming(command, cwd=cwd, env=env, timeout=timeout, on_output=on_output)
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

    def _run_streaming(self, command, *, cwd, env, timeout, on_output) -> RunResult:
        """stderr is merged into stdout (tools like COLMAP log progress to
        stderr), read line by line and handed to on_output as it arrives.
        Only the last MAX_KEPT_LINES lines are kept in the RunResult -- a
        long mapper run can print hundreds of thousands. A timeout is only
        checked as lines arrive."""
        started = time.monotonic()
        kept: deque[str] = deque(maxlen=self.MAX_KEPT_LINES)
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        timed_out = False
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip("\n")
            kept.append(line)
            try:
                on_output(line)
            except Exception:
                pass  # a progress callback must never break the run itself
            if timeout is not None and time.monotonic() - started > timeout:
                process.kill()
                timed_out = True
                break
        returncode = process.wait()
        output = "\n".join(kept)
        return RunResult(
            command=command,
            returncode=-1 if timed_out else returncode,
            stdout=output,
            stderr=f"timed out after {timeout}s" if timed_out else "",
            duration_seconds=time.monotonic() - started,
        )
