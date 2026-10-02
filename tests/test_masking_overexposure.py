"""Synthetic-image tests for vine360.masking.overexposure: what it must
catch (a clipped light source and its bloom) and, just as important, what
it must leave alone (white-but-unclipped walls, small glints)."""

import numpy as np

from vine360.masking.overexposure import OverexposureConfig, classify_overexposed

NO_DILATION = OverexposureConfig(dilation_px=0)


def _distance_from(shape, center):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return np.hypot(yy - center[0], xx - center[1])


def test_clipped_disc_and_its_bloom_are_masked():
    image = np.full((200, 200, 3), 90, dtype=np.uint8)
    distance = _distance_from(image.shape[:2], (100, 100))
    image[distance < 30] = 255
    image[(distance >= 30) & (distance < 50)] = 240  # bright bloom ring

    mask = classify_overexposed(image, NO_DILATION) > 127

    assert mask[100, 100]
    assert mask[100, 140], "bloom ring next to the clipped core is part of the mask"
    assert not mask[100, 170], "mid-grey beyond the bloom is kept"


def test_white_walls_are_not_masked():
    image = np.full((200, 200, 3), 225, dtype=np.uint8)  # a plain white wall
    rng = np.random.default_rng(0)
    # A brighter, textured white wall: bright, but not every channel clipped.
    image[:, 100:] = np.clip(245 + rng.integers(-6, 6, size=(200, 100, 3)), 0, 255).astype(np.uint8)

    assert not (classify_overexposed(image) > 127).any()


def test_small_glints_are_ignored():
    image = np.full((200, 200, 3), 120, dtype=np.uint8)
    image[50:54, 50:54] = 255  # 16 px, well under 0.2% of 40000 px
    assert not (classify_overexposed(image) > 127).any()


def test_bloom_stops_at_the_radius_instead_of_running_along_a_bright_wall():
    image = np.full((200, 400, 3), 60, dtype=np.uint8)
    image[:, :60] = 255  # clipped opening on the left
    image[:, 60:] = 235  # a long bright wall touching it
    config = OverexposureConfig(bloom_radius_px=20, dilation_px=0)

    mask = classify_overexposed(image, config) > 127

    assert mask[100, 30]
    assert mask[100, 75], "within the bloom radius of the opening"
    assert not mask[100, 200], "the rest of the bright wall is kept"


def test_bloom_must_be_connected_to_the_core():
    image = np.full((200, 200, 3), 60, dtype=np.uint8)
    image[90:110, 20:60] = 255  # clipped core
    image[90:110, 70:80] = 240  # bright patch within the radius, separated by dark pixels
    mask = classify_overexposed(image, OverexposureConfig(bloom_radius_px=40, dilation_px=0)) > 127
    assert mask[100, 40]
    assert not mask[100, 75]


def test_nothing_clipped_returns_empty_mask():
    image = np.full((50, 50, 3), 200, dtype=np.uint8)
    mask = classify_overexposed(image)
    assert mask.shape == (50, 50) and mask.dtype == np.uint8 and not mask.any()
