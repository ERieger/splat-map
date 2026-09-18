import sys

from vine360.runners.local import LocalRunner


def test_local_runner_captures_stdout_and_returncode():
    runner = LocalRunner()
    result = runner.run([sys.executable, "-c", "print('hi')"])
    assert result.ok
    assert result.returncode == 0
    assert result.stdout.strip() == "hi"


def test_local_runner_captures_nonzero_returncode_and_stderr():
    runner = LocalRunner()
    result = runner.run([sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"])
    assert not result.ok
    assert result.returncode == 3
    assert "boom" in result.stderr
