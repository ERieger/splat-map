import shutil

import pytest

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

requires_ffmpeg = pytest.mark.skipif(
    not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed in this environment"
)
