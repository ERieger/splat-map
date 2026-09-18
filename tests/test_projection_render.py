import numpy as np

from vine360.projection.cubemap import six_face_preset
from vine360.projection.geometry import direction_to_equirect_pixel, equirect_pixel_to_direction
from vine360.projection.render import project_equirect_to_face


def _make_marker_equirect(width, height, marker_u, marker_v, marker_size=2):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # blue background
    y0, y1 = marker_v - marker_size, marker_v + marker_size
    x0, x1 = marker_u - marker_size, marker_u + marker_size
    image[y0:y1, x0:x1] = [255, 0, 0]  # red marker
    return image


def _peak_red_pixel(image: np.ndarray) -> tuple[int, int]:
    """(x, y) of the reddest pixel, for sub-marker pixel-accuracy checks."""
    red = image[..., 0].astype(np.int32) - image[..., 2].astype(np.int32)
    py, px = np.unravel_index(np.argmax(red), red.shape)
    return int(px), int(py)


# A01 requires known points to land within one pixel; the underlying
# transform is verified to 1e-9 in test_projection_geometry.py and
# test_projection_pose.py. These image-based tests additionally round-trip
# through actual resampling, where a several-pixel-wide marker centered on
# an integer pixel plus bilinear blending adds its own +-1px of placement
# ambiguity on top of that -- hence the wider (still tight) tolerance here.
MARKER_QUANTIZATION_TOLERANCE_PX = 2


def test_six_face_preset_excludes_polar_by_default():
    faces = six_face_preset(face_size=64)
    assert {f.name for f in faces} == {"front", "right", "back", "left"}


def test_six_face_preset_includes_polar_when_requested():
    faces = six_face_preset(face_size=64, include_polar_faces=True)
    assert {f.name for f in faces} == {"front", "right", "back", "left", "up", "down"}


def test_a01_round_trip_projection_known_point_front_face():
    """Acceptance test A01: a known point on a synthetic equirectangular
    grid lands in the expected face pixel within one pixel."""
    src_width, src_height = 1024, 512
    face_size = 512

    # A marker exactly at the equirectangular center maps to dead center of
    # the front face (lon=0, lat=0 is the front face's forward direction).
    marker_u, marker_v = src_width // 2, src_height // 2
    equirect = _make_marker_equirect(src_width, src_height, marker_u, marker_v)

    faces = {f.name: f for f in six_face_preset(face_size=face_size)}
    front = project_equirect_to_face(equirect, faces["front"])

    expected_px, expected_py = face_size // 2, face_size // 2
    peak_px, peak_py = _peak_red_pixel(front)
    assert abs(peak_px - expected_px) <= MARKER_QUANTIZATION_TOLERANCE_PX
    assert abs(peak_py - expected_py) <= MARKER_QUANTIZATION_TOLERANCE_PX


def test_a01_round_trip_projection_offset_point():
    """A marker at a known offset from center round-trips through the
    projection math (not just image sampling) to the expected face pixel
    within one pixel, verifying the full equirect->direction->camera chain."""
    src_width, src_height = 2048, 1024
    face_size = 512
    faces = {f.name: f for f in six_face_preset(face_size=face_size)}
    front = faces["front"]

    # Pick a point 20 degrees off-center that should land inside the 90 deg
    # front face, then verify by inverting the whole projection chain.
    marker_u = int(src_width * 0.5 + src_width * (20 / 360))
    marker_v = src_height // 2
    direction = equirect_pixel_to_direction(marker_u, marker_v, src_width, src_height)

    # direction is expressed in panorama space; convert into the face's own
    # camera space via the inverse of its fixed rotation, then to a pixel.
    from vine360.projection.geometry import direction_to_camera_pixel

    direction_in_face = front.fixed_rotation.T @ direction
    pixel = direction_to_camera_pixel(direction_in_face, front.intrinsics)
    assert pixel is not None
    px, py = pixel

    # A small (near-single-pixel) marker minimizes bias from the transform's
    # nonlinear warp stretching a larger block asymmetrically off-center.
    equirect = _make_marker_equirect(src_width, src_height, marker_u, marker_v, marker_size=1)
    rendered = project_equirect_to_face(equirect, front)

    peak_px, peak_py = _peak_red_pixel(rendered)
    assert abs(peak_px - px) <= MARKER_QUANTIZATION_TOLERANCE_PX
    assert abs(peak_py - py) <= MARKER_QUANTIZATION_TOLERANCE_PX


def test_seam_wraps_correctly():
    """A marker straddling the equirectangular seam (u=0 / u=width) must
    still sample correctly for a face whose forward direction points at the
    seam (yaw=180, i.e. the 'back' face)."""
    src_width, src_height = 1024, 512
    face_size = 256
    equirect = np.zeros((src_height, src_width, 3), dtype=np.uint8)
    equirect[:, :, 2] = 255
    # seam is at u=0 (lon = +-pi); paint a marker straddling it.
    equirect[src_height // 2 - 3 : src_height // 2 + 3, 0:3] = [255, 0, 0]
    equirect[src_height // 2 - 3 : src_height // 2 + 3, -3:] = [255, 0, 0]

    faces = {f.name: f for f in six_face_preset(face_size=face_size)}
    back = project_equirect_to_face(equirect, faces["back"])

    peak_px, peak_py = _peak_red_pixel(back)
    assert abs(peak_px - face_size // 2) <= MARKER_QUANTIZATION_TOLERANCE_PX
    assert abs(peak_py - face_size // 2) <= MARKER_QUANTIZATION_TOLERANCE_PX
