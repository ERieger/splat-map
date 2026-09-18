"""A weights-free sky heuristic, usable while SAM 3 access is pending (or as
a cross-check once it isn't -- handover doc, section 10, "Sky quality":
"Benchmark text-prompted sky masks against a semantic segmentation model;
select per dataset or combine conservatively").

This is intentionally coarse: bright, blue-or-white pixels confined to the
upper portion of the frame. It will misclassify a bright white trellis post
or a pale gravel row as sky, and will miss sky seen low on the horizon
through gaps in canopy. It exists so the masking pipeline (semantics,
morphology, keep-mask export) can be built and tested end-to-end without
SAM 3 weights, not as a substitute for real segmentation quality.

There is no classical fallback for person masks -- there is no
brightness/color heuristic that reliably separates a person from vineyard
foliage, trellis or equipment, so person exclusion genuinely requires
SAM 3 (or an equivalent model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vine360.masking.semantics import MASK_FALSE, MASK_TRUE


@dataclass(frozen=True)
class ClassicalSkyConfig:
    brightness_threshold: float = 180.0
    blue_dominance: float = 10.0
    top_fraction: float = 0.6  # only rows above this fraction of height are considered


def classify_sky_classical(image: np.ndarray, config: ClassicalSkyConfig = ClassicalSkyConfig()) -> np.ndarray:
    """image: (H, W, 3) uint8 RGB. Returns an (H, W) uint8 mask, 255=sky."""
    rgb = image.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    brightness = rgb.mean(axis=-1)

    is_bright = brightness > config.brightness_threshold
    is_blueish = (b - r > config.blue_dominance) & (b - g > -5.0)
    candidate = is_bright | is_blueish

    height = image.shape[0]
    row_index = np.arange(height).reshape(-1, 1)
    within_top = row_index < height * config.top_fraction

    sky = candidate & within_top
    return np.where(sky, MASK_TRUE, MASK_FALSE).astype(np.uint8)
