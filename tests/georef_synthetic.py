"""Synthetic pinhole cameras for the georef tests (docs/adr/0047): COLMAP
convention (x right, y down, z forward), cam_from_world R, t."""

from __future__ import annotations

import numpy as np


def look_at(center: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """cam_from_world (R, t) of a camera at `center` looking at `target`."""
    center = np.asarray(center, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - center
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, np.array([1.0, 0.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return R, -R @ center


def intrinsics(width: int = 1024, height: int = 1024, f: float = 512.0) -> np.ndarray:
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    cam = R @ X + t
    return (K @ (cam / cam[2]))[:2]


def ring_cameras(n: int, radius: float, height: float, target=(0.0, 0.0, 0.0)):
    out = []
    for k in range(n):
        a = 2 * np.pi * k / n
        center = np.array([radius * np.cos(a), radius * np.sin(a), height]) + np.asarray(target)
        out.append(look_at(center, np.asarray(target, dtype=np.float64)))
    return out


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
