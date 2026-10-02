import shutil

import pytest

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

requires_ffmpeg = pytest.mark.skipif(
    not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed in this environment"
)


def _have_spheresfm() -> bool:
    try:
        from vine360.sfm.spheresfm_adapter import find_binary
    except Exception:
        return False
    return find_binary() is not None


HAVE_SPHERESFM = _have_spheresfm()

requires_spheresfm = pytest.mark.skipif(
    not HAVE_SPHERESFM,
    reason="no SphereSfM build found (see vine360.sfm.spheresfm_adapter.candidate_binaries / VINE360_SPHERESFM_COLMAP)",
)
