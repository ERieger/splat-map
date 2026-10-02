"""The 26-direction view catalog (docs/adr/0041): naming, rotations,
backward compatibility with the original six cubemap faces, footprint
outlines, and that a tilted diagonal view actually renders the panorama
region it claims to."""

import json
import math

import numpy as np
import pytest
from PIL import Image

from vine360.config import CaptureMode
from vine360.project import create_project, open_index_db
from vine360.projection.cubemap import (
    ALL_DIRECTION_NAMES,
    ALL_FACE_NAMES,
    DIRECTION_CATALOG,
    DIRECTION_PRESETS,
    DIRECTIONS_BY_NAME,
    direction_center_pixel,
    direction_preset,
    direction_sort_key,
    footprint_outline,
    six_face_preset,
)
from vine360.projection.generate import generate_views_for_frame, generate_views_for_frame_set
from vine360.projection.geometry import direction_to_equirect_pixel, equirect_pixel_to_direction, face_rotation
from vine360.projection.render import project_equirect_to_face

FORWARD = np.array([0.0, 0.0, 1.0])


def _direction(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon, lat = math.radians(lon_deg), math.radians(lat_deg)
    return np.array([math.cos(lat) * math.sin(lon), math.sin(lat), math.cos(lat) * math.cos(lon)])


def test_catalog_has_26_unique_names_in_ring_order():
    assert len(DIRECTION_CATALOG) == 26
    assert len(set(ALL_DIRECTION_NAMES)) == 26
    assert ALL_DIRECTION_NAMES[:8] == [
        "front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left",
    ]
    assert ALL_DIRECTION_NAMES[8] == "front-down"
    assert ALL_DIRECTION_NAMES[16] == "front-up"
    assert ALL_DIRECTION_NAMES[-2:] == ["up", "down"]


@pytest.mark.parametrize("tilt", [30.0, 45.0, 60.0])
def test_every_direction_looks_where_its_yaw_and_pitch_say(tilt):
    faces = {f.name: f for f in direction_preset(32, 90.0, ALL_DIRECTION_NAMES, ring_tilt_degrees=tilt)}
    for spec in DIRECTION_CATALOG:
        axis = faces[spec.name].fixed_rotation @ FORWARD
        expected = _direction(spec.yaw_degrees, spec.pitch_degrees(tilt))
        np.testing.assert_allclose(axis, expected, atol=1e-12)


def test_non_polar_directions_have_zero_roll():
    """Camera-right stays horizontal (y == 0) so the horizon is level in
    every view, tilted or not."""
    for face in direction_preset(32, 90.0, [d.name for d in DIRECTION_CATALOG if not d.is_polar]):
        right = face.fixed_rotation @ np.array([1.0, 0.0, 0.0])
        assert abs(right[1]) < 1e-12


def test_every_rotation_is_proper():
    for face in direction_preset(32, 90.0, ALL_DIRECTION_NAMES):
        R = face.fixed_rotation
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_legacy_six_faces_keep_their_exact_rotations():
    legacy = {
        "front": (0.0, 0.0), "right": (90.0, 0.0), "back": (180.0, 0.0),
        "left": (270.0, 0.0), "up": (0.0, 90.0), "down": (0.0, -90.0),
    }
    faces = {f.name: f for f in six_face_preset(64, include_polar_faces=True)}
    assert set(faces) == set(ALL_FACE_NAMES)
    for name, (yaw, pitch) in legacy.items():
        np.testing.assert_allclose(
            faces[name].fixed_rotation, face_rotation(math.radians(yaw), math.radians(pitch)), atol=1e-12
        )


def test_ring_tilt_only_moves_the_tilted_rings():
    a = {f.name: f.fixed_rotation for f in direction_preset(32, 90.0, ALL_DIRECTION_NAMES, ring_tilt_degrees=30.0)}
    b = {f.name: f.fixed_rotation for f in direction_preset(32, 90.0, ALL_DIRECTION_NAMES, ring_tilt_degrees=60.0)}
    for spec in DIRECTION_CATALOG:
        same = np.allclose(a[spec.name], b[spec.name])
        assert same == (spec.ring in ("horizon", "pole-up", "pole-down")), spec.name


def test_direction_preset_returns_catalog_order_and_validates():
    faces = direction_preset(32, 90.0, ["down", "front-left-down", "front"])
    assert [f.name for f in faces] == ["front", "front-left-down", "down"]
    with pytest.raises(ValueError):
        direction_preset(32, 90.0, ["front", "sideways"])
    with pytest.raises(ValueError):
        direction_preset(32, 90.0, ["front-down"], ring_tilt_degrees=95.0)


def test_six_face_preset_accepts_new_names():
    assert [f.name for f in six_face_preset(32, face_names=["front-right-down"])] == ["front-right-down"]


def test_presets_only_name_catalog_directions():
    for label, names in DIRECTION_PRESETS.items():
        assert names, label
        assert set(names) <= set(ALL_DIRECTION_NAMES), label
    assert len(DIRECTION_PRESETS["Full sphere (26)"]) == 26


def test_direction_sort_key_follows_catalog_and_puts_unknowns_last():
    assert sorted(["down", "front-down", "right", "zzz", "front"], key=direction_sort_key) == [
        "front", "right", "front-down", "down", "zzz",
    ]


def test_direction_center_pixel_matches_equirect_convention():
    spec = DIRECTIONS_BY_NAME["front-right-down"]
    u, v = direction_center_pixel(spec, 720, 360, ring_tilt_degrees=45.0)
    assert (u, v) == pytest.approx(direction_to_equirect_pixel(_direction(45.0, -45.0), 720, 360))


def test_footprint_outline_of_a_level_view_is_one_closed_loop_around_its_center():
    lines = footprint_outline(DIRECTIONS_BY_NAME["right"], 90.0, 720, 360)
    assert len(lines) == 1
    loop = np.array(lines[0])
    np.testing.assert_allclose(loop[0], loop[-1])
    cu, cv = direction_center_pixel(DIRECTIONS_BY_NAME["right"], 720, 360)
    assert loop[:, 0].min() < cu < loop[:, 0].max()
    assert loop[:, 1].min() < cv < loop[:, 1].max()
    # A 90 deg view spans 90 deg of longitude at the horizon -> a quarter of the width.
    assert loop[:, 0].max() - loop[:, 0].min() == pytest.approx(180.0, abs=2.0)


def test_footprint_outline_splits_at_the_back_seam():
    lines = footprint_outline(DIRECTIONS_BY_NAME["back"], 90.0, 720, 360)
    assert len(lines) == 2
    for line in lines:
        us = [u for u, _ in line]
        assert max(us) - min(us) < 360  # no polyline streaks across the image


def test_footprint_outline_of_down_pole_spans_full_width_below_its_tilt():
    lines = footprint_outline(DIRECTIONS_BY_NAME["down"], 90.0, 720, 360)
    assert len(lines) == 1
    us = [u for u, _ in lines[0]]
    vs = [v for _, v in lines[0]]
    assert max(us) - min(us) > 700
    assert min(vs) >= 360 * 0.5  # entirely below the horizon


def _marker_equirect(width, height, lon_deg, lat_deg, size=3):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    u, v = direction_to_equirect_pixel(_direction(lon_deg, lat_deg), width, height)
    u, v = int(round(u)), int(round(v))
    image[v - size:v + size, u - size:u + size] = [255, 0, 0]
    return image


def test_tilted_diagonal_view_renders_the_marker_at_its_center():
    """front-left-down at a 35 deg tilt looks at lon -45 (left is -x), lat -35."""
    equirect = _marker_equirect(1440, 720, -45.0, -35.0)
    face = direction_preset(256, 90.0, ["front-left-down"], ring_tilt_degrees=35.0)[0]
    rendered = project_equirect_to_face(equirect, face)
    red = rendered[..., 0].astype(int) - rendered[..., 2].astype(int)
    py, px = np.unravel_index(np.argmax(red), red.shape)
    assert abs(px - 128) <= 3 and abs(py - 128) <= 3


def test_tilted_view_sees_wall_and_floor_together():
    """The motivating tunnel case: a lower-ring view at 45 deg and 90 deg
    FOV spans from the horizon (top edge) down to straight below (bottom
    edge)."""
    face = direction_preset(64, 90.0, ["left-down"])[0]
    top_center = face.fixed_rotation @ np.array([0.0, 1.0, 1.0]) / math.sqrt(2)
    bottom_center = face.fixed_rotation @ np.array([0.0, -1.0, 1.0]) / math.sqrt(2)
    assert top_center[1] == pytest.approx(0.0, abs=1e-12)  # horizon
    assert bottom_center[1] == pytest.approx(-1.0, abs=1e-12)  # straight down


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    yield root, conn
    conn.close()


def _insert_frame(conn, root, frame_id="frame-1", source_id="source-1"):
    conn.execute(
        "INSERT INTO sources (source_id, path, checksum, media_type, projection, width, height, timestamps, capture_group) "
        "VALUES (?, '/x.mp4', 'deadbeef', 'video', 'equirectangular', 640, 320, '{}', NULL)",
        (source_id,),
    )
    path = root / "frames" / source_id / "frame_000001.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((320, 640, 3), 128, dtype=np.uint8), "RGB").save(path)
    conn.execute(
        "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
        "VALUES (?, ?, 0.0, '{}', ?, 'cafef00d', ?)",
        (frame_id, source_id, str(path.relative_to(root)), source_id),
    )
    conn.commit()
    return source_id, frame_id


def test_generate_writes_diagonal_views_with_their_tilted_rotation(project):
    root, conn = project
    source_id, frame_id = _insert_frame(conn, root)

    views = generate_views_for_frame_set(
        conn, root, source_id, face_size=32, face_names=["front", "back-right-down"], ring_tilt_degrees=30.0
    )

    assert [v.view_id for v in views] == [f"{frame_id}:front", f"{frame_id}:back-right-down"]
    assert (root / "projections" / frame_id / "back-right-down.png").exists()
    stored = conn.execute(
        "SELECT fixed_rotation FROM views WHERE view_id = ?", (f"{frame_id}:back-right-down",)
    ).fetchone()[0]
    axis = np.asarray(json.loads(stored)["matrix"]) @ FORWARD
    np.testing.assert_allclose(axis, _direction(135.0, -30.0), atol=1e-12)


def test_regenerating_with_a_different_selection_removes_stale_views(project):
    root, conn = project
    _source_id, frame_id = _insert_frame(conn, root)
    generate_views_for_frame(conn, root, frame_id, face_size=32, face_names=["front", "left-down"])
    generate_views_for_frame(conn, root, frame_id, face_size=32, face_names=["front-right"])

    rows = [r[0] for r in conn.execute("SELECT view_id FROM views WHERE frame_id = ?", (frame_id,))]
    assert rows == [f"{frame_id}:front-right"]
    assert not (root / "projections" / frame_id / "left-down.png").exists()
    assert not (root / "projections" / frame_id / "front.png").exists()


def test_equirect_convention_sanity_left_is_negative_x():
    """Guards the naming: 'left' must be yaw 270 == lon -90 == -x."""
    axis = direction_preset(8, 90.0, ["left"])[0].fixed_rotation @ FORWARD
    np.testing.assert_allclose(axis, [-1.0, 0.0, 0.0], atol=1e-12)
    u, _v = direction_to_equirect_pixel(axis, 360, 180)
    assert u < 180  # left half of the panorama
    np.testing.assert_allclose(equirect_pixel_to_direction(89.5, 89.5, 360, 180), [-1.0, 0.0, 0.0], atol=1e-2)
