"""Mask semantics (handover doc, section 4, step 4 "Mask" and the "Mask
semantics" table).

Every mask here is an 8-bit single-channel image: 255 = the named
condition, 0 = elsewhere. `keep.png` (255 usable, 0 excluded) is the
canonical mask fed to pose estimation and training; `exclude.png` is only
its complement, kept for human review. The COLMAP mask ("Non-zero usable,
0 ignored") is byte-identical to keep.png -- COLMAP just wants it filed
next to a specific path (see `colmap_mask_path`).

This module deliberately stops at producing `keep.png` / `exclude.png` /
the COLMAP mask file; the handover doc's invariant "Never paint excluded
pixels black in the RGB training images... supply a mask through the
backend adapter" is enforced by the M4/M5 adapters that consume these
masks, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

MASK_TRUE = 255
MASK_FALSE = 0


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return mask > 127


def _to_uint8(mask_bool: np.ndarray) -> np.ndarray:
    return np.where(mask_bool, MASK_TRUE, MASK_FALSE).astype(np.uint8)


def combine_class_masks(*class_masks: np.ndarray) -> np.ndarray:
    """ORs any number of class masks (e.g. person, sky) into one exclude mask."""
    if not class_masks:
        raise ValueError("at least one class mask is required")
    combined = np.zeros(class_masks[0].shape, dtype=bool)
    for m in class_masks:
        combined |= _as_bool(m)
    return _to_uint8(combined)


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Removes connected components smaller than min_area_px from the
    True/255 regions of `mask` (handover doc: "remove tiny components")."""
    if min_area_px <= 0:
        return mask
    labeled, num_features = ndimage.label(_as_bool(mask))
    if num_features == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(labeled), labeled, index=range(1, num_features + 1))
    keep_labels = [i + 1 for i, size in enumerate(sizes) if size >= min_area_px]
    cleaned = np.isin(labeled, keep_labels)
    return _to_uint8(cleaned)


def dilate_exclude_mask(exclude_mask: np.ndarray, dilation_px: int) -> np.ndarray:
    """Slightly grows the exclusion mask (handover doc: "Dilate exclusion
    masks slightly") so segmentation-boundary pixels aren't left half in,
    half out of the excluded region."""
    if dilation_px <= 0:
        return exclude_mask
    structure = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), dtype=bool)
    dilated = ndimage.binary_dilation(_as_bool(exclude_mask), structure=structure)
    return _to_uint8(dilated)


def keep_mask_from_exclude(exclude_mask: np.ndarray) -> np.ndarray:
    return _to_uint8(~_as_bool(exclude_mask))


def keep_fraction(keep_mask: np.ndarray) -> float:
    return float(_as_bool(keep_mask).mean())


@dataclass(frozen=True)
class MaskBuildConfig:
    # Small noise specks are removed before dilation grows the exclusion
    # region -- doing it the other way around lets dilation fuse noise
    # into real regions, where a component-size filter can no longer tell
    # them apart. This is a deliberate reordering of the handover doc's
    # step list ("dilate... remove tiny components"); see docs/adr/0008.
    min_component_px: int = 64
    dilation_px: int = 3


def build_keep_and_exclude_masks(
    class_masks: list[np.ndarray], config: MaskBuildConfig = MaskBuildConfig()
) -> tuple[np.ndarray, np.ndarray]:
    exclude = combine_class_masks(*class_masks)
    exclude = remove_small_components(exclude, config.min_component_px)
    exclude = dilate_exclude_mask(exclude, config.dilation_px)
    keep = keep_mask_from_exclude(exclude)
    return exclude, keep


def is_keep_fraction_anomalous(
    fraction: float, *, low: float = 0.3, high: float = 0.98
) -> str | None:
    """Handover doc, section 6: "flag masks with unusually low or high
    retained area." Returns a human-readable reason, or None if unremarkable."""
    if fraction < low:
        return f"only {fraction:.0%} of the view is kept (< {low:.0%}); mask may be over-excluding"
    if fraction > high:
        return f"{fraction:.0%} of the view is kept (> {high:.0%}); segmentation may have missed content"
    return None


def colmap_mask_path(source_relative_path: str | Path, masks_root: Path) -> Path:
    """"Preserve the source image subpath and append .png to the complete
    filename, for example frame.jpg.png."""
    return Path(masks_root) / (str(source_relative_path) + ".png")


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"))
