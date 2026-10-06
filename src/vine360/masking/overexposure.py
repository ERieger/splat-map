"""A weights-free mask for blown-out light sources and their bloom -- e.g.
the overexposed end of a tunnel, or a sun-washed opening in a building --
which 3DGS training otherwise tries to explain with floaters strung along
every ray that saw the glare (docs/adr/0038, 0042, 0043).

The point is to exclude *clipped* regions without excluding merely *white*
ones. A white wall in a usable exposure sits around 200-245 and still has
some texture/channel variation; a blown-out light source is a large,
connected region where all three channels are saturated, surrounded by a
bright halo (bloom) that fades away from it. So:

1. Core: pixels where min(R, G, B) >= clip_threshold -- every channel
   clipped, not just bright.
2. Drop cores smaller than min_core_fraction of the view (0.02%, ~14x14 px
   at 1024^2) -- specular glints on a wall -- they don't produce floaters
   worth losing pixels over. Clipped fragments within _CORE_GROUP_PX of each
   other count as one core, so a blown area broken up by the scene in front
   of it (sky through leaves, a far tunnel end behind a railing) isn't
   dropped piece by piece -- as long as at least _CORE_MIN_DENSITY of the
   group is clipped, so specks scattered through sunlit texture don't
   add up to a core. ADR 0038's 0.2% also dropped distant tunnel ends
   and light fixtures, and with them the glow around them (ADR 0043).
3. Glare: pixels with luma >= _GLARE_THRESHOLD (230) within
   _GLARE_RADIUS_PX (40, capped at bloom_radius_px) of a core, in a bright
   component touching it -- the washed-out rim of the opening and whatever
   is seen, barely, through it. (This was all of "bloom" in ADR 0038.)
4. Halo: flood outward from core + glare, on a smoothed quarter-
   resolution luma image, into pixels that are
   - at least bloom_threshold bright,
   - within bloom_radius_px of a core,
   - smooth (local luma std <= bloom_max_texture: glare washes detail out,
     while a sunlit hillside or a lit textured wall keeps it), and
   - darker than the neighbour the flood came from by at least
     _BLOOM_MIN_FALLOFF -- bloom fades away from the light, so the flood
     follows that falloff and stops on a plateau. An evenly lit white wall
     beside an opening isn't swallowed just for being bright and close.
   Then fill any holes the flood encloses.
5. Dilate the result by dilation_px (a disc, not a square) to catch the
   soft edge of the bloom.

The first version (ADR 0038) stopped at step 3. On real tunnel footage the
halo is a long, gentle ramp (~235 down to ~190 over 150+ px on a ~170 wall)
broken up by brick texture, so it masked little more than the clipped
opening itself (ADR 0042).

Known limits: a white surface that is itself fully clipped over a large
area (direct sun on white render) *is* masked -- but clipped pixels carry no
information for training anyway. Tuned and checked by eye against real
FMC-Tunnel1 views (tunnel exits, light fixtures, sunlit stairs under a
blown sky) and synthetic images in tests/test_masking_overexposure.py;
there is no ground truth for "bloom".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from vine360.masking.semantics import MASK_FALSE, MASK_TRUE

# Bumped whenever the algorithm changes, so a layer's method_version says
# which one built it.
METHOD_VERSION = "3"

# Bloom is low-frequency, so it's flooded at reduced resolution: fast, and
# the averaging suppresses codec noise and brick mortar lines.
_BLOOM_SCALE = 4
_BLOOM_SMOOTHING_SIGMA = 2.0  # in reduced-resolution pixels
_BLOOM_MIN_FALLOFF = 0.1  # luma drop required per reduced-resolution step
_TEXTURE_WINDOW_PX = 9
_GLARE_THRESHOLD = 230
_GLARE_RADIUS_PX = 40
_CORE_GROUP_PX = 4
# A group must be mostly clipped: real blown areas measured 0.35-0.87 on
# FMC-Tunnel1, while clipped specks scattered through sunlit texture form
# big but sparse groups.
_CORE_MIN_DENSITY = 0.3


@dataclass(frozen=True)
class OverexposureConfig:
    clip_threshold: int = 250
    min_core_fraction: float = 0.0002
    bloom_threshold: int = 185
    bloom_radius_px: int = 160
    bloom_max_texture: float = 12.0
    dilation_px: int = 6


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _disc(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return xx * xx + yy * yy <= radius * radius


def _downsample(values: np.ndarray, scale: int) -> np.ndarray:
    height, width = values.shape[0] // scale, values.shape[1] // scale
    return values[: height * scale, : width * scale].reshape(height, scale, width, scale).mean(axis=(1, 3))


def _components_touching(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    labeled, count = ndimage.label(candidate)
    if count == 0:
        return np.zeros_like(candidate)
    hit = np.unique(labeled[seed & candidate])
    hit = hit[hit != 0]
    return np.isin(labeled, hit)


def _glare(luma: np.ndarray, core: np.ndarray, radius: float) -> np.ndarray:
    near = ndimage.distance_transform_edt(~core) <= radius
    return _components_touching(((luma >= _GLARE_THRESHOLD) & near) | core, core)


def _bloom(luma: np.ndarray, core: np.ndarray, config: OverexposureConfig) -> np.ndarray:
    """Full-resolution boolean glare + halo region grown from core (core included)."""
    seed = _glare(luma, core, min(_GLARE_RADIUS_PX, config.bloom_radius_px))
    height, width = core.shape
    scale = _BLOOM_SCALE
    if height < scale or width < scale:
        return seed
    mean = ndimage.uniform_filter(luma, _TEXTURE_WINDOW_PX)
    mean_sq = ndimage.uniform_filter(luma * luma, _TEXTURE_WINDOW_PX)
    texture = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

    small_luma = ndimage.gaussian_filter(_downsample(luma, scale), _BLOOM_SMOOTHING_SIGMA)
    small_texture = ndimage.gaussian_filter(_downsample(texture, scale), _BLOOM_SMOOTHING_SIGMA)
    small_core = _downsample(core.astype(np.float32), scale) > 0  # any clipped pixel: keeps thin tubes
    small_seed = _downsample(seed.astype(np.float32), scale) >= 0.5
    if not small_seed.any():
        return seed
    radius = config.bloom_radius_px / scale
    allowed = (
        (small_luma >= config.bloom_threshold)
        & (small_texture <= config.bloom_max_texture)
        & (ndimage.distance_transform_edt(~small_core) <= radius)  # reach is from the clipped core itself
    )

    region = small_seed.copy()
    neighbourhood = np.ones((3, 3), dtype=bool)
    for _ in range(int(np.ceil(radius)) * 2):  # a path can wind; the distance cap still bounds it
        brightest_neighbour = ndimage.maximum_filter(np.where(region, small_luma, -np.inf), footprint=neighbourhood)
        grow = allowed & ~region & (small_luma <= brightest_neighbour - _BLOOM_MIN_FALLOFF)
        if not grow.any():
            break
        region |= grow

    full = np.zeros((height, width), dtype=bool)
    upsampled = np.kron(region, np.ones((scale, scale), dtype=bool))
    full[: upsampled.shape[0], : upsampled.shape[1]] = upsampled
    return ndimage.binary_fill_holes(full | seed)


def classify_overexposed(image: np.ndarray, config: OverexposureConfig = OverexposureConfig()) -> np.ndarray:
    """image: (H, W, 3) uint8 RGB. Returns an (H, W) uint8 mask, 255=overexposed."""
    height, width = image.shape[:2]
    core = image.min(axis=-1) >= config.clip_threshold

    if not core.any():
        return np.zeros((height, width), dtype=np.uint8)
    groups, count = ndimage.label(ndimage.binary_dilation(core, structure=_disc(_CORE_GROUP_PX)))
    sizes = np.bincount(groups[core], minlength=count + 1)  # clipped pixels per group
    sizes[0] = 0
    density = sizes / np.maximum(np.bincount(groups.ravel(), minlength=count + 1), 1)
    min_area = config.min_core_fraction * height * width
    big = np.flatnonzero((sizes >= min_area) & (density >= _CORE_MIN_DENSITY))
    if big.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    core &= np.isin(groups, big)

    mask = core
    if config.bloom_radius_px > 0:
        mask = _bloom(_luma(image.astype(np.float32)), core, config)

    if config.dilation_px > 0:
        mask = ndimage.binary_dilation(mask, structure=_disc(config.dilation_px))
    return np.where(mask, MASK_TRUE, MASK_FALSE).astype(np.uint8)
