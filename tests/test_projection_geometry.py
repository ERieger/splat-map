import math

import numpy as np
import pytest

from vine360.projection.geometry import (
    Intrinsics,
    camera_pixel_to_direction,
    direction_to_camera_pixel,
    direction_to_equirect_pixel,
    equirect_pixel_to_direction,
    face_rotation,
)


def test_equirect_direction_round_trip_grid():
    width, height = 640, 320
    for u in [0, 50, 160, 320, 480, 639]:
        for v in [0, 40, 80, 160, 240, 319]:
            direction = equirect_pixel_to_direction(u, v, width, height)
            u2, v2 = direction_to_equirect_pixel(direction, width, height)
            assert abs(u2 - u) < 1e-6
            assert abs(v2 - v) < 1e-6


def test_equirect_center_is_forward():
    width, height = 640, 320
    direction = equirect_pixel_to_direction(width / 2 - 0.5, height / 2 - 0.5, width, height)
    np.testing.assert_allclose(direction, [0.0, 0.0, 1.0], atol=1e-9)


def test_equirect_top_and_bottom_are_poles():
    width, height = 640, 320
    top = equirect_pixel_to_direction(0, 0, width, height)
    bottom = equirect_pixel_to_direction(0, height - 1, width, height)
    # top of the image is near the +y pole, bottom near -y
    assert top[1] > 0.99
    assert bottom[1] < -0.99


def test_camera_direction_round_trip():
    intrinsics = Intrinsics.from_fov(256, 256, 90.0)
    for px in [0, 64, 128, 192, 255]:
        for py in [0, 64, 128, 192, 255]:
            direction = camera_pixel_to_direction(px, py, intrinsics)
            result = direction_to_camera_pixel(direction, intrinsics)
            assert result is not None
            px2, py2 = result
            assert abs(px2 - px) < 1e-6
            assert abs(py2 - py) < 1e-6


def test_camera_center_pixel_is_forward():
    intrinsics = Intrinsics.from_fov(256, 256, 90.0)
    direction = camera_pixel_to_direction(127.5, 127.5, intrinsics)
    np.testing.assert_allclose(direction, [0.0, 0.0, 1.0], atol=1e-9)


def test_direction_behind_camera_returns_none():
    intrinsics = Intrinsics.from_fov(256, 256, 90.0)
    assert direction_to_camera_pixel(np.array([0.0, 0.0, -1.0]), intrinsics) is None


@pytest.mark.parametrize(
    "yaw_deg,pitch_deg,expected",
    [
        (0.0, 0.0, [0.0, 0.0, 1.0]),
        (90.0, 0.0, [1.0, 0.0, 0.0]),
        (180.0, 0.0, [0.0, 0.0, -1.0]),
        (270.0, 0.0, [-1.0, 0.0, 0.0]),
        (0.0, 90.0, [0.0, 1.0, 0.0]),
        (0.0, -90.0, [0.0, -1.0, 0.0]),
    ],
)
def test_face_rotation_forward_matches_equirect_direction(yaw_deg, pitch_deg, expected):
    R = face_rotation(math.radians(yaw_deg), math.radians(pitch_deg))
    forward_in_panorama = R @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(forward_in_panorama, expected, atol=1e-9)


def test_face_rotation_is_a_proper_rotation():
    for yaw_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
        R = face_rotation(math.radians(yaw_deg), math.radians(15))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
