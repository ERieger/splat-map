"""Equirectangular <-> perspective camera geometry (handover doc, section 4,
step 3 "Project").

Frame convention, consistent throughout: x = right, y = up, z = forward, a
proper right-handed basis (det(R) = +1 for every rotation built here).

- Equirectangular pixel (u, v) with u in [0, width), v in [0, height):
  longitude lon in (-pi, pi], latitude lat in [-pi/2, pi/2],
  direction = (cos(lat) sin(lon), sin(lat), cos(lat) cos(lon)).
  lon=0, lat=0 (image horizontal center, vertical center) is +z (forward).

- A perspective face camera looks along +z in its own frame, x right,
  y up (NOT image-row-down -- pixel-to-NDC conversion flips the row axis
  so the resulting camera frame stays a proper rotation). Row 0 is the top
  of the image, as usual.

- `face_rotation(yaw, pitch)` is R_panorama_face: applying it to a camera
  direction expresses that direction in panorama space, and it is built so
  that `face_rotation(yaw, pitch) @ (0, 0, 1) == direction(lon=yaw, lat=pitch)`
  exactly, keeping the cubemap faces and the equirectangular projection
  defined by the same underlying rotation math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def equirect_pixel_to_direction(u: float, v: float, width: int, height: int) -> np.ndarray:
    lon = ((u + 0.5) / width) * 2 * math.pi - math.pi
    lat = math.pi / 2 - ((v + 0.5) / height) * math.pi
    return np.array(
        [math.cos(lat) * math.sin(lon), math.sin(lat), math.cos(lat) * math.cos(lon)]
    )


def direction_to_equirect_pixel(direction: np.ndarray, width: int, height: int) -> tuple[float, float]:
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    lon = math.atan2(d[0], d[2])
    lat = math.asin(max(-1.0, min(1.0, d[1])))
    u = (lon + math.pi) / (2 * math.pi) * width - 0.5
    v = (math.pi / 2 - lat) / math.pi * height - 0.5
    return u, v


def equirect_pixel_to_direction_batch(u: np.ndarray, v: np.ndarray, width: int, height: int) -> np.ndarray:
    """Vectorized form of equirect_pixel_to_direction; returns (..., 3)."""
    lon = ((u + 0.5) / width) * 2 * np.pi - np.pi
    lat = np.pi / 2 - ((v + 0.5) / height) * np.pi
    return np.stack(
        [np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], axis=-1
    )


def direction_to_equirect_pixel_batch(direction: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized form of direction_to_equirect_pixel. direction: (..., 3)."""
    d = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    lon = np.arctan2(d[..., 0], d[..., 2])
    lat = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    u = (lon + np.pi) / (2 * np.pi) * width - 0.5
    v = (np.pi / 2 - lat) / np.pi * height - 0.5
    return u, v


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_fov(width: int, height: int, fov_degrees: float) -> "Intrinsics":
        fx = (width / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
        fy = (height / 2.0) / math.tan(math.radians(fov_degrees) / 2.0)
        return Intrinsics(width=width, height=height, fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0)

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}


def camera_pixel_to_direction(px: float, py: float, intrinsics: Intrinsics) -> np.ndarray:
    x_ndc = (px + 0.5 - intrinsics.cx) / intrinsics.fx
    y_ndc = (intrinsics.cy - (py + 0.5)) / intrinsics.fy  # row-down pixels -> up-positive camera y
    direction = np.array([x_ndc, y_ndc, 1.0])
    return direction / np.linalg.norm(direction)


def direction_to_camera_pixel(direction: np.ndarray, intrinsics: Intrinsics) -> tuple[float, float] | None:
    d = np.asarray(direction, dtype=np.float64)
    if d[2] <= 0:
        return None  # behind the camera
    x_ndc = d[0] / d[2]
    y_ndc = d[1] / d[2]
    px = x_ndc * intrinsics.fx + intrinsics.cx - 0.5
    py = intrinsics.cy - y_ndc * intrinsics.fy - 0.5
    return px, py


def camera_pixel_grid_to_directions(intrinsics: Intrinsics) -> np.ndarray:
    """Vectorized camera_pixel_to_direction over every pixel; returns
    (height, width, 3)."""
    px = np.arange(intrinsics.width)
    py = np.arange(intrinsics.height)
    px_grid, py_grid = np.meshgrid(px, py)
    x_ndc = (px_grid + 0.5 - intrinsics.cx) / intrinsics.fx
    y_ndc = (intrinsics.cy - (py_grid + 0.5)) / intrinsics.fy
    directions = np.stack([x_ndc, y_ndc, np.ones_like(x_ndc)], axis=-1)
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def rotation_y(angle_radians: float) -> np.ndarray:
    c, s = math.cos(angle_radians), math.sin(angle_radians)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_x(angle_radians: float) -> np.ndarray:
    c, s = math.cos(angle_radians), math.sin(angle_radians)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


def face_rotation(yaw_radians: float, pitch_radians: float) -> np.ndarray:
    """R_panorama_face: R_y(yaw) @ R_x(pitch), chosen so this rotation's
    third column (its image of camera-forward (0,0,1)) equals
    `equirect_pixel_to_direction`'s direction(lon=yaw, lat=pitch)."""
    return rotation_y(yaw_radians) @ rotation_x(pitch_radians)
