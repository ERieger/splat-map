"""The single pose convention for the whole project (handover doc, section 4:
"Define one coordinate convention, document whether transforms are
camera-to-world or world-to-camera").

**Convention: every Pose here is camera-to-world.** `Pose(R, t)` maps a
point `p` given in the camera's/panorama's local frame to world
coordinates: `p_world = R @ p + t`. Composing two poses, `compose(A, B)`,
answers "where does B's local frame sit inside A's frame, expressed in
world coordinates" -- this is exactly
`T_world_face = T_world_panorama . T_panorama_face` from the handover doc.

COLMAP stores world-to-camera poses (quaternion + translation such that
`p_camera = R_cw @ p_world + t_cw`), so `to_colmap` / `from_colmap` invert
the rotation and re-derive the translation accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pose:
    """Camera-to-world rigid transform: p_world = R @ p_local + t."""

    R: np.ndarray  # (3, 3), orthonormal, det=+1
    t: np.ndarray  # (3,)

    def __post_init__(self):
        object.__setattr__(self, "R", np.asarray(self.R, dtype=np.float64).reshape(3, 3))
        object.__setattr__(self, "t", np.asarray(self.t, dtype=np.float64).reshape(3))

    @staticmethod
    def identity() -> "Pose":
        return Pose(np.eye(3), np.zeros(3))

    def apply(self, point_local: np.ndarray) -> np.ndarray:
        return self.R @ np.asarray(point_local, dtype=np.float64) + self.t

    def apply_direction(self, direction_local: np.ndarray) -> np.ndarray:
        """Rotates a direction (no translation) into world/parent space."""
        return self.R @ np.asarray(direction_local, dtype=np.float64)

    def inverse(self) -> "Pose":
        r_inv = self.R.T
        return Pose(r_inv, -r_inv @ self.t)


def compose(parent: Pose, child: Pose) -> Pose:
    """T_world_face = T_world_panorama . T_panorama_face, generalized.

    `parent` locates `child`'s frame within `parent`'s parent (typically
    world) frame; the result locates points expressed in `child`'s local
    frame directly in that outer frame.
    """
    return Pose(parent.R @ child.R, parent.R @ child.t + parent.t)


def rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """Shepperd's method; returns (qw, qx, qy, qz), normalized."""
    m = R
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def quaternion_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


def to_colmap(pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    """Camera-to-world Pose -> COLMAP's world-to-camera (qvec, tvec)."""
    world_to_camera = pose.inverse()
    return rotation_matrix_to_quaternion_wxyz(world_to_camera.R), world_to_camera.t


def from_colmap(qvec: np.ndarray, tvec: np.ndarray) -> Pose:
    """COLMAP's world-to-camera (qvec, tvec) -> camera-to-world Pose."""
    world_to_camera = Pose(quaternion_wxyz_to_rotation_matrix(qvec), tvec)
    return world_to_camera.inverse()
