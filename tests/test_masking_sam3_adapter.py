import numpy as np
import pytest

from PIL import Image, ImageDraw

from vine360.masking.sam3_adapter import Sam3Adapter, validate_installation

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
def test_segment_finds_sky_on_a_real_image():
    """Real integration check against the live facebook/sam3 model, now
    that access has been approved and this environment is authenticated
    (`hf auth login`) -- see docs/adr/0008 and docs/status.md. Downloads
    (or reuses the cached) ~3.4GB model.safetensors on first run."""
    image = Image.new("RGB", (256, 256), (135, 206, 235))  # sky blue
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 150, 256, 256], fill=(34, 90, 34))  # green ground, bottom ~41%
    image = np.asarray(image)

    adapter = Sam3Adapter()
    masks = adapter.segment(image, {"sky": "sky"})

    coverage = masks["sky"].mean() / 255
    assert 0.4 < coverage < 0.75, f"expected sky coverage near the true ~59% sky area, got {coverage:.0%}"
    # the sky region should dominate the top rows and be largely absent from the bottom rows
    assert masks["sky"][10:20, :].mean() > masks["sky"][230:250, :].mean()
