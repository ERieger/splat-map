"""Synthetic-image tests for vine360.masking.overexposure: what it must
catch (a clipped light source and the halo fading away from it) and, just
as important, what it must leave alone (white-but-unclipped walls, small
glints, evenly lit or textured bright surfaces next to an opening)."""

import numpy as np

from vine360.masking.overexposure import OverexposureConfig, classify_overexposed

NO_DILATION = OverexposureConfig(dilation_px=0)


def _distance_from(shape, center):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return np.hypot(yy - center[0], xx - center[1])


def _grey(value):
    return np.stack([value] * 3, axis=-1).clip(0, 255).astype(np.uint8)


def test_clipped_disc_and_its_fading_halo_are_masked():
    distance = _distance_from((400, 400), (200, 200))
    # A clipped disc on a mid-grey wall, with bloom fading from 245 to the
    # wall's 150 over 120 px -- the shape real tunnel-exit glare has.
    luma = np.where(distance < 40, 255.0, np.maximum(150.0, 245.0 - (distance - 40) * (95.0 / 120.0)))
    mask = classify_overexposed(_grey(luma), NO_DILATION) > 127

    assert mask[200, 200]
    assert mask[200, 300], "halo 60 px out (luma ~198) is part of the mask"
    assert not mask[200, 390], "the plain wall beyond the halo is kept"


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


def test_evenly_lit_bright_wall_beside_an_opening_is_kept():
    image = np.full((200, 400, 3), 60, dtype=np.uint8)
    image[:, :60] = 255  # clipped opening on the left
    image[:, 60:] = 225  # a long, evenly lit white wall touching it

    mask = classify_overexposed(image, NO_DILATION) > 127

    assert mask[100, 30]
    assert not mask[100, 120], "a flat bright wall isn't bloom: it doesn't fade away from the light"


def test_bright_textured_surface_beside_an_opening_is_kept():
    rng = np.random.default_rng(1)
    image = np.full((300, 300, 3), 255, dtype=np.uint8)
    # A blown sky above a sunlit, textured hillside: bright, but full of detail.
    hillside = 215 + rng.normal(0, 25, size=(200, 300))
    image[100:] = _grey(np.repeat(np.repeat(hillside[::3, ::3], 3, 0), 3, 1)[:200, :300])

    mask = classify_overexposed(image, NO_DILATION) > 127

    assert mask[50, 150]
    assert mask[150:, :].mean() < 0.15, "texture stops the halo flood"


def test_bloom_stops_at_the_radius():
    distance = _distance_from((400, 400), (200, 200))
    luma = np.where(distance < 30, 255.0, np.maximum(150.0, 245.0 - (distance - 30) * 0.3))
    mask = classify_overexposed(_grey(luma), OverexposureConfig(bloom_radius_px=40, dilation_px=0)) > 127

    assert mask[200, 250], "within the bloom radius"
    assert not mask[200, 300], "still bright and fading, but beyond the radius"


def test_bloom_must_be_connected_to_the_core():
    image = np.full((200, 200, 3), 60, dtype=np.uint8)
    image[80:120, 20:60] = 255  # clipped core
    image[80:120, 100:110] = 240  # bright patch within the radius, separated by dark pixels
    mask = classify_overexposed(image, NO_DILATION) > 127
    assert mask[100, 40]
    assert not mask[100, 105]


def test_bloom_radius_zero_masks_only_the_core():
    image = np.full((100, 100, 3), 60, dtype=np.uint8)
    image[40:60, 40:60] = 255
    image[40:60, 60:70] = 245
    mask = classify_overexposed(image, OverexposureConfig(bloom_radius_px=0, dilation_px=0)) > 127
    assert mask[50, 50] and not mask[50, 65]


def test_nothing_clipped_returns_empty_mask():
    image = np.full((50, 50, 3), 200, dtype=np.uint8)
    mask = classify_overexposed(image)
    assert mask.shape == (50, 50) and mask.dtype == np.uint8 and not mask.any()


def test_tiny_image_does_not_crash():
    image = np.full((3, 3, 3), 255, dtype=np.uint8)
    assert (classify_overexposed(image) > 127).all()
