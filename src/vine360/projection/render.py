"""Resamples an equirectangular image into a perspective face (handover doc,
section 4, step 3). Pure numpy (vectorized, bilinear); Pillow only at the
file I/O boundary so the resampling math stays independently testable on
plain arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from vine360.projection.cubemap import FaceSpec
from vine360.projection.geometry import camera_pixel_grid_to_directions, direction_to_equirect_pixel_batch


def _bilinear_sample_equirect(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """image: (H, W, C). u, v: (..., ) float pixel coords. u wraps
    horizontally (the equirectangular seam); v clamps vertically (poles)."""
    height, width = image.shape[0], image.shape[1]

    u_mod = np.mod(u, width)
    u0 = np.floor(u_mod).astype(np.int64) % width
    u1 = (u0 + 1) % width
    fu = (u_mod - np.floor(u_mod))[..., None]

    v_clamped = np.clip(v, 0.0, height - 1.0)
    v0 = np.floor(v_clamped).astype(np.int64)
    v1 = np.clip(v0 + 1, 0, height - 1)
    fv = (v_clamped - v0)[..., None]

    top = image[v0, u0] * (1 - fu) + image[v0, u1] * fu
    bottom = image[v1, u0] * (1 - fu) + image[v1, u1] * fu
    return top * (1 - fv) + bottom * fv


def project_equirect_to_face(equirect: np.ndarray, face: FaceSpec) -> np.ndarray:
    """equirect: (H, W, C) array (any dtype/channel count). Returns an array
    of the face's (height, width, C), same dtype as the input."""
    if equirect.ndim == 2:
        equirect = equirect[..., None]
        squeeze = True
    else:
        squeeze = False

    directions_cam = camera_pixel_grid_to_directions(face.intrinsics)  # (H, W, 3)
    flat = directions_cam.reshape(-1, 3)
    directions_pano = flat @ face.fixed_rotation.T
    directions_pano = directions_pano.reshape(directions_cam.shape)

    src_height, src_width = equirect.shape[0], equirect.shape[1]
    u, v = direction_to_equirect_pixel_batch(directions_pano, src_width, src_height)

    sampled = _bilinear_sample_equirect(equirect.astype(np.float64), u, v)
    if np.issubdtype(equirect.dtype, np.integer):
        sampled = np.clip(np.round(sampled), 0, np.iinfo(equirect.dtype).max).astype(equirect.dtype)
    else:
        sampled = sampled.astype(equirect.dtype)
    return sampled[..., 0] if squeeze else sampled


def render_face_to_file(equirect_path: Path, face: FaceSpec, output_path: Path) -> None:
    with Image.open(equirect_path) as img:
        equirect = np.asarray(img.convert("RGB"))
    face_image = project_equirect_to_face(equirect, face)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(face_image, mode="RGB").save(output_path)
