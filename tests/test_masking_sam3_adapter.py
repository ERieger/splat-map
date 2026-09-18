import numpy as np
import pytest

from vine360.masking.sam3_adapter import Sam3AccessDeniedError, Sam3Adapter, validate_installation

pytestmark = pytest.mark.skipif(
    not validate_installation().torch_installed or not validate_installation().transformers_installed,
    reason="torch/transformers not installed",
)


def test_validate_installation_reports_versions():
    status = validate_installation()
    assert status.torch_installed
    assert status.transformers_installed
    assert status.torch_version is not None
    assert status.transformers_version is not None


def test_construction_never_touches_network():
    # Should not raise or attempt any download; only segment() does.
    Sam3Adapter()


@pytest.mark.network
def test_segment_raises_access_denied_when_ungated_access_missing():
    """Real integration check against the live, gated facebook/sam3 repo:
    without an approved + authenticated Hugging Face account, this must
    fail with a clearly classified error, not an unhandled exception.
    Once access is granted and `huggingface-cli login` has been run, this
    test's assumption (that access is currently denied) no longer holds
    and it should be replaced with a real segmentation assertion."""
    adapter = Sam3Adapter()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(Sam3AccessDeniedError):
        adapter.segment(image, {"sky": "sky"})
