"""A weights-free mask for blown-out light sources and their bloom -- e.g.
the overexposed end of a tunnel, or a sun-washed opening in a building --
which 3DGS training otherwise tries to explain with floaters strung along
every ray that saw the glare (docs/adr/0038).

The point is to exclude *clipped* regions without excluding merely *white*
ones. A white wall in a usable exposure sits around 200-245 and still has
some texture/channel variation; a blown-out light source is a large,
connected region where all three channels are saturated, surrounded by a
bright halo (bloom) that fades away from it. So:

1. Core: pixels where min(R, G, B) >= clip_threshold -- every channel
   clipped, not just bright.
2. Drop cores smaller than min_core_fraction of the view -- specular glints
   on a wall, a sunlit post -- they don't produce floaters worth losing
   pixels over.
3. Bloom: pixels with luma >= bloom_threshold that lie within
   bloom_radius_px of a surviving core *and* belong to a bright connected
   region that touches it. A white wall is never masked on its own account,
   and the halo can't run the length of an adjoining bright wall.
4. Dilate the result by dilation_px to catch the soft edge of the bloom.

Known limits: a white surface that is itself fully clipped over a large
area (direct sun on white render) *is* masked -- but clipped pixels carry no
information for training anyway. Bloom fainter than bloom_threshold is left
in; raise bloom_radius_px / lower bloom_threshold for heavier blooming.
Verified only against synthetic images in tests/test_masking_overexposure.py
and by eye on real tunnel views; there is no ground truth for "bloom".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from vine360.masking.semantics import MASK_FALSE, MASK_TRUE


@dataclass(frozen=True)
class OverexposureConfig:
    clip_threshold: int = 250
    min_core_fraction: float = 0.002
    bloom_threshold: int = 230
    bloom_radius_px: int = 40
    dilation_px: int = 6


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _components_touching(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(candidate)
    if count == 0:
        return np.zeros_like(candidate)
    hit = np.unique(labeled[seed & candidate])
    hit = hit[hit != 0]
    return np.isin(labeled, hit)


def classify_overexposed(image: np.ndarray, config: OverexposureConfig = OverexposureConfig()) -> np.ndarray:
    """image: (H, W, 3) uint8 RGB. Returns an (H, W) uint8 mask, 255=overexposed."""
    height, width = image.shape[:2]
    core = image.min(axis=-1) >= config.clip_threshold

    labeled, count = ndimage.label(core)
    if count == 0:
        return np.zeros((height, width), dtype=np.uint8)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    min_area = config.min_core_fraction * height * width
    big = np.flatnonzero(sizes >= min_area)
    if big.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    core = np.isin(labeled, big)

    mask = core
    if config.bloom_radius_px > 0:
        bright = _luma(image.astype(np.float32)) >= config.bloom_threshold
        near = ndimage.distance_transform_edt(~core) <= config.bloom_radius_px
        mask = _components_touching((bright & near) | core, core)

    if config.dilation_px > 0:
        structure = np.ones((2 * config.dilation_px + 1, 2 * config.dilation_px + 1), dtype=bool)
        mask = ndimage.binary_dilation(mask, structure=structure)
    return np.where(mask, MASK_TRUE, MASK_FALSE).astype(np.uint8)
