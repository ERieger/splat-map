import math

import numpy as np
import pytest

from vine360.projection.geometry import face_rotation
from vine360.projection.pose import Pose, compose, from_colmap, to_colmap


def test_compose_translation_only():
    parent = Pose(np.eye(3), np.array([10.0, 0.0, 0.0]))
    child = Pose(np.eye(3), np.array([0.0, 5.0, 0.0]))
    result = compose(parent, child)
    np.testing.assert_allclose(result.t, [10.0, 5.0, 0.0])
    np.testing.assert_allclose(result.R, np.eye(3))


def test_compose_matches_t_world_face_formula():
    # T_world_panorama: panorama sits at (1, 2, 3), rotated 45 deg around Y.
    R_wp = np.array(
        [
            [math.cos(math.radians(45)), 0, math.sin(math.radians(45))],
            [0, 1, 0],
            [-math.sin(math.radians(45)), 0, math.cos(math.radians(45))],
        ]
    )
    t_wp = np.array([1.0, 2.0, 3.0])
    T_world_panorama = Pose(R_wp, t_wp)

    # T_panorama_face: the "right" cubemap face, centered at the panorama's
    # optical center (t=0), per the fixed six-face preset.
    R_pf = face_rotation(math.radians(90.0), 0.0)
    T_panorama_face = Pose(R_pf, np.zeros(3))

    T_world_face = compose(T_world_panorama, T_panorama_face)

    # A point straight ahead of the face camera (local +z) must land at the
    # same world point as applying the panorama pose to the face's forward
    # direction in panorama space.
    point_in_face = np.array([0.0, 0.0, 1.0])
    expected = T_world_panorama.apply(T_panorama_face.apply(point_in_face))
    np.testing.assert_allclose(T_world_face.apply(point_in_face), expected, atol=1e-9)


def test_compose_is_associative():
    def random_pose(seed):
        rng = np.random.default_rng(seed)
        axis_angle = rng.normal(size=3)
        angle = np.linalg.norm(axis_angle)
        axis = axis_angle / angle
        # Rodrigues' rotation formula
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        return Pose(R, rng.normal(size=3))

    a, b, c = random_pose(1), random_pose(2), random_pose(3)
    left = compose(compose(a, b), c)
    right = compose(a, compose(b, c))
    np.testing.assert_allclose(left.R, right.R, atol=1e-9)
    np.testing.assert_allclose(left.t, right.t, atol=1e-9)


def test_pose_inverse_round_trip():
    pose = Pose(face_rotation(math.radians(37.0), math.radians(12.0)), np.array([3.0, -1.0, 2.0]))
    identity = compose(pose, pose.inverse())
    np.testing.assert_allclose(identity.R, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(identity.t, np.zeros(3), atol=1e-9)


@pytest.mark.parametrize("yaw_deg,pitch_deg", [(0, 0), (37, 12), (90, -20), (200, 45)])
def test_colmap_round_trip(yaw_deg, pitch_deg):
    pose = Pose(
        face_rotation(math.radians(yaw_deg), math.radians(pitch_deg)),
        np.array([1.5, -2.5, 0.75]),
    )
    qvec, tvec = to_colmap(pose)
    restored = from_colmap(qvec, tvec)
    np.testing.assert_allclose(restored.R, pose.R, atol=1e-9)
    np.testing.assert_allclose(restored.t, pose.t, atol=1e-9)
