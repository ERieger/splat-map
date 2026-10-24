"""vine360.georef.apriltag_adapter against the real pupil-apriltags install
(docs/adr/0047): the AprilRobotics reference tag36h11 images
(fixtures/apriltag/), first flat, then rendered into a synthetic
equirectangular panorama and projected into a pinhole view with vine360's
own projection code -- the end-to-end proof that detection on projected
views works and that detections share the views' pixel convention."""

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pupil_apriltags")

from PIL import Image

from vine360.georef.apriltag_adapter import TagDetector
from vine360.projection.cubemap import FaceSpec
from vine360.projection.geometry import (
    Intrinsics,
    direction_to_camera_pixel,
    equirect_pixel_to_direction_batch,
    face_rotation,
)
from vine360.projection.render import project_equirect_to_face

TAGS = Path(__file__).resolve().parent.parent / "fixtures" / "apriltag"


def _tag_cells(tag_id: int) -> np.ndarray:
    """10x10 cells, 1.0 = white (including the white outer border)."""
    return (np.asarray(Image.open(TAGS / f"tag36_11_{tag_id:05d}.png").convert("L")) > 127).astype(np.float64)


def _area_sampled(cells: np.ndarray, x0: float, y0: float, cell: float, shape=(600, 700), ss=8) -> np.ndarray:
    """The tag drawn with its top-left cell corner at (x0, y0), pixel i
    spanning [i, i+1) -- supersampled, so edges land at exact sub-pixel
    positions."""
    h, w = shape
    xs = (np.arange(w * ss) + 0.5) / ss
    ys = (np.arange(h * ss) + 0.5) / ss
    u = np.floor((xs - x0) / cell).astype(int)
    v = np.floor((ys - y0) / cell).astype(int)
    inside = (u[None, :] >= 0) & (u[None, :] < 10) & (v[:, None] >= 0) & (v[:, None] < 10)
    values = cells[np.clip(v, 0, 9)][:, np.clip(u, 0, 9)]
    img = np.where(inside, values, 1.0).reshape(h, ss, w, ss).mean(axis=(1, 3))
    return (img * 255).round().astype(np.uint8)


@pytest.fixture(scope="module")
def detector():
    with TagDetector("tag36h11") as d:
        yield d


def test_close_tears_down_in_apriltags_order_and_is_idempotent():
    d = TagDetector("tag36h11")
    d.detect(np.full((64, 64), 255, np.uint8))
    d.close()
    assert d._detector.tag_detector_ptr is None and d._detector.tag_families == {}
    d.close()  # and the library's own __del__ must now be a no-op
    del d


@pytest.mark.parametrize("tag_id", [0, 20, 22, 25, 31])
def test_reference_tags_are_decoded(detector, tag_id):
    image = _area_sampled(_tag_cells(tag_id), 150.0, 100.0, 37.0)
    detections = detector.detect(image)
    assert [d.tag_id for d in detections] == [tag_id]
    assert detections[0].family == "tag36h11" and detections[0].hamming == 0


def test_pixel_convention_and_corner_order(detector):
    x0, y0, cell = 150.3, 100.7, 37.3
    (d,) = detector.detect(_area_sampled(_tag_cells(20), x0, y0, cell))
    # Centre in COLMAP convention (pixel i centred at i + 0.5).
    np.testing.assert_allclose(d.center, [x0 + 5 * cell, y0 + 5 * cell], atol=0.25)
    # Corners on the black square's outer edge (cells 1..9): BL, BR, TR, TL.
    expected = [[x0 + cell, y0 + 9 * cell], [x0 + 9 * cell, y0 + 9 * cell], [x0 + 9 * cell, y0 + cell], [x0 + cell, y0 + cell]]
    np.testing.assert_allclose(d.corners, expected, atol=0.5)


def test_corner_order_follows_the_tag_not_the_image(detector):
    image = _area_sampled(_tag_cells(20), 150.0, 100.0, 37.0, shape=(700, 700))
    (upright,) = detector.detect(image)
    (turned,) = detector.detect(np.ascontiguousarray(np.rot90(image)))  # 90 deg counter-clockwise
    # rot90: (x, y) -> (y, W - x); corner k is the same physical corner either way.
    mapped = np.stack([upright.corners[:, 1], 700 - upright.corners[:, 0]], axis=1)
    np.testing.assert_allclose(turned.corners, mapped, atol=0.5)


def _equirect_with_tag(tag_id: int, direction: np.ndarray, distance: float, tag_size: float, width: int = 4096):
    """A white panorama (vine360 frame: x right, y up, z forward) with a flat
    tag facing the camera at `direction`, its top towards +y."""
    height = width // 2
    forward = direction / np.linalg.norm(direction)
    right = np.cross([0.0, 1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    cell = tag_size / 8.0  # the black square is the inner 8 of 10 cells
    cells = _tag_cells(tag_id)
    ss = 2
    v, u = np.mgrid[0 : height * ss, 0 : width * ss]
    dirs = equirect_pixel_to_direction_batch((u + 0.5) / ss - 0.5, (v + 0.5) / ss - 0.5, width, height)
    along = dirs @ forward
    facing = along > 0.5  # the tag is a few degrees across; nothing behind or beside it can hit it
    t = distance / np.where(facing, along, 1.0)
    hit = dirs * t[..., None] - forward * distance
    col = np.floor((hit @ right + 5 * cell) / cell).astype(np.int64)
    row = np.floor((5 * cell - hit @ up) / cell).astype(np.int64)
    inside = facing & (col >= 0) & (col < 10) & (row >= 0) & (row < 10)
    values = np.ones(dirs.shape[:2])
    values[inside] = cells[row[inside], col[inside]]
    values = values.reshape(height, ss, width, ss).mean(axis=(1, 3))
    gray = (values * 255).round().astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2), forward, right, up, cell


def test_detects_on_a_view_projected_from_an_equirect_panorama(detector):
    yaw, pitch = math.radians(30.0), math.radians(-20.0)
    tag_direction = np.array([math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
    tag_size, distance = 0.4, 3.0
    pano, forward, right, up, cell = _equirect_with_tag(20, tag_direction, distance, tag_size)

    # A view pointed 15 deg away from the tag, so it lands off-centre.
    intrinsics = Intrinsics.from_fov(1024, 1024, 90.0)
    face = FaceSpec("test", intrinsics, face_rotation(yaw - math.radians(15.0), pitch + math.radians(10.0)))
    view = project_equirect_to_face(pano, face)
    detections = detector.detect(view)
    assert [d.tag_id for d in detections] == [20]
    d = detections[0]

    def expected_pixel(point_pano: np.ndarray) -> np.ndarray:
        # vine360 pixel index -> COLMAP coordinate (+0.5), the views' convention (ADR 0036).
        px, py = direction_to_camera_pixel(face.fixed_rotation.T @ point_pano, intrinsics)
        return np.array([px + 0.5, py + 0.5])

    center = forward * distance
    np.testing.assert_allclose(d.center, expected_pixel(center), atol=1.0)
    half = 4 * cell
    corners_3d = [center - right * half - up * half, center + right * half - up * half,
                  center + right * half + up * half, center - right * half + up * half]
    np.testing.assert_allclose(d.corners, [expected_pixel(c) for c in corners_3d], atol=1.5)
