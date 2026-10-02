"""Perspective view directions rendered from each equirectangular frame
(handover doc, section 4, step 3: "Start with a cubemap-like six-face preset
at 90 FOV... Exclude or carefully handle polar views with little useful
vineyard content").

The original preset was the six cubemap faces (front/right/back/left/up/
down). The catalog now holds 26 directions (docs/adr/0041): yaw every 45
degrees on three rings -- the horizon (pitch 0), a lower ring tilted down
and an upper ring tilted up (pitch -/+ `ring_tilt_degrees`, 45 by default)
-- plus the two poles. A lower-ring diagonal such as `front-left-down`
sees a vine-row tunnel's wall *and* floor in one image, which a level face
plus the straight-down pole can't. The six legacy names keep exactly
their old rotations, so existing view ids (`<frame_id>:front`), queued
jobs and saved selections are unaffected.

Names: the horizon ring is `front`, `front-right`, `right`, ... ; the
lower/upper rings append `-down`/`-up` (`front-right-down`); the poles are
`up`/`down`. Every view keeps zero roll (level horizon).

Up/down poles are excluded by default for the handover doc's reason --
pointed at sky and ground respectively, they add little to a terrestrial
vineyard reconstruction and cost extra processing. Pass
`include_polar_faces=True` to six_face_preset to generate all six.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vine360.projection.geometry import Intrinsics, direction_to_equirect_pixel, face_rotation

DEFAULT_FOV_DEGREES = 90.0
DEFAULT_RING_TILT_DEGREES = 45.0
MIN_RING_TILT_DEGREES = 10.0
MAX_RING_TILT_DEGREES = 80.0

RING_HORIZON = "horizon"
RING_LOWER = "lower"
RING_UPPER = "upper"
RING_POLE_UP = "pole-up"
RING_POLE_DOWN = "pole-down"

# Yaw 0 = front (+z), +90 = right (+x), see geometry.py's frame convention.
_YAW_NAMES = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]


@dataclass(frozen=True)
class DirectionSpec:
    """One entry in the direction catalog. pitch_degrees takes the ring
    tilt as an argument because the lower/upper rings' pitch is
    configurable, while their name is not."""

    name: str
    yaw_degrees: float
    ring: str

    @property
    def is_polar(self) -> bool:
        return self.ring in (RING_POLE_UP, RING_POLE_DOWN)

    def pitch_degrees(self, ring_tilt_degrees: float = DEFAULT_RING_TILT_DEGREES) -> float:
        return {
            RING_HORIZON: 0.0,
            RING_LOWER: -ring_tilt_degrees,
            RING_UPPER: ring_tilt_degrees,
            RING_POLE_UP: 90.0,
            RING_POLE_DOWN: -90.0,
        }[self.ring]


def _build_catalog() -> list[DirectionSpec]:
    catalog = [DirectionSpec(name, i * 45.0, RING_HORIZON) for i, name in enumerate(_YAW_NAMES)]
    catalog += [DirectionSpec(f"{name}-down", i * 45.0, RING_LOWER) for i, name in enumerate(_YAW_NAMES)]
    catalog += [DirectionSpec(f"{name}-up", i * 45.0, RING_UPPER) for i, name in enumerate(_YAW_NAMES)]
    catalog += [DirectionSpec("up", 0.0, RING_POLE_UP), DirectionSpec("down", 0.0, RING_POLE_DOWN)]
    return catalog


DIRECTION_CATALOG: list[DirectionSpec] = _build_catalog()
DIRECTIONS_BY_NAME: dict[str, DirectionSpec] = {d.name: d for d in DIRECTION_CATALOG}
ALL_DIRECTION_NAMES: list[str] = [d.name for d in DIRECTION_CATALOG]

# The original six cubemap faces, in the handover doc's order.
ALL_FACE_NAMES = ["front", "right", "back", "left", "up", "down"]
CARDINAL_NAMES = ["front", "right", "back", "left"]

# Named selections offered by the GUI's direction picker. The tunnel preset
# targets a camera moving along a narrow vine row: level views along the
# row, the four diagonal-down views catching wall + floor together, and
# straight-across-down views of each wall's base.
DIRECTION_PRESETS: dict[str, list[str]] = {
    "Cardinal (4)": CARDINAL_NAMES,
    "Cardinal + diagonals (8)": list(_YAW_NAMES),
    "Vine-row tunnel": [
        "front", "back",
        "front-right-down", "back-right-down", "back-left-down", "front-left-down",
        "right-down", "left-down",
    ],
    "Horizon + lower ring (16)": list(_YAW_NAMES) + [f"{n}-down" for n in _YAW_NAMES],
    "Full sphere (26)": list(ALL_DIRECTION_NAMES),
}


def direction_sort_key(name: str) -> int:
    """Catalog order (horizon ring, lower ring, upper ring, poles) for
    sorting view names in the GUI; unknown names sort last."""
    try:
        return ALL_DIRECTION_NAMES.index(name)
    except ValueError:
        return len(ALL_DIRECTION_NAMES)


@dataclass(frozen=True)
class FaceSpec:
    name: str
    intrinsics: Intrinsics
    fixed_rotation: "object"  # np.ndarray, R_panorama_face

    def fixed_rotation_to_dict(self) -> dict:
        return {"matrix": self.fixed_rotation.tolist()}


def _validate_names(names) -> None:
    unknown = set(names) - set(ALL_DIRECTION_NAMES)
    if unknown:
        raise ValueError(f"unknown view direction(s): {sorted(unknown)}; valid names are {ALL_DIRECTION_NAMES}")


def direction_rotation(spec: DirectionSpec, ring_tilt_degrees: float = DEFAULT_RING_TILT_DEGREES) -> np.ndarray:
    return face_rotation(math.radians(spec.yaw_degrees), math.radians(spec.pitch_degrees(ring_tilt_degrees)))


def direction_preset(
    face_size: int,
    fov_degrees: float = DEFAULT_FOV_DEGREES,
    direction_names: list[str] | None = None,
    *,
    ring_tilt_degrees: float = DEFAULT_RING_TILT_DEGREES,
) -> list[FaceSpec]:
    """FaceSpecs for the named directions (default: the four cardinal
    faces), returned in catalog order regardless of the order given.
    Unknown names, or a ring tilt outside [MIN, MAX]_RING_TILT_DEGREES,
    raise ValueError."""
    names = CARDINAL_NAMES if direction_names is None else direction_names
    _validate_names(names)
    if not MIN_RING_TILT_DEGREES <= ring_tilt_degrees <= MAX_RING_TILT_DEGREES:
        raise ValueError(
            f"ring_tilt_degrees must be within [{MIN_RING_TILT_DEGREES}, {MAX_RING_TILT_DEGREES}], got {ring_tilt_degrees}"
        )
    wanted = set(names)
    intrinsics = Intrinsics.from_fov(face_size, face_size, fov_degrees)
    return [
        FaceSpec(name=spec.name, intrinsics=intrinsics, fixed_rotation=direction_rotation(spec, ring_tilt_degrees))
        for spec in DIRECTION_CATALOG
        if spec.name in wanted
    ]


def six_face_preset(
    face_size: int,
    fov_degrees: float = DEFAULT_FOV_DEGREES,
    *,
    include_polar_faces: bool = False,
    face_names: list[str] | None = None,
    ring_tilt_degrees: float = DEFAULT_RING_TILT_DEGREES,
) -> list[FaceSpec]:
    """face_names, if given, selects an explicit set of directions (any
    catalog name, not only the original six) -- overrides
    include_polar_faces. Unknown names raise ValueError. Without
    face_names: the four cardinal faces, plus up/down when
    include_polar_faces."""
    if face_names is None:
        face_names = ALL_FACE_NAMES if include_polar_faces else CARDINAL_NAMES
    return direction_preset(face_size, fov_degrees, face_names, ring_tilt_degrees=ring_tilt_degrees)


def direction_center_pixel(
    spec: DirectionSpec, eq_width: int, eq_height: int, ring_tilt_degrees: float = DEFAULT_RING_TILT_DEGREES
) -> tuple[float, float]:
    """Where a direction's optical axis lands on an equirect image."""
    axis = direction_rotation(spec, ring_tilt_degrees) @ np.array([0.0, 0.0, 1.0])
    return direction_to_equirect_pixel(axis, eq_width, eq_height)


def footprint_outline(
    spec: DirectionSpec,
    fov_degrees: float,
    eq_width: int,
    eq_height: int,
    *,
    ring_tilt_degrees: float = DEFAULT_RING_TILT_DEGREES,
    samples_per_edge: int = 32,
) -> list[list[tuple[float, float]]]:
    """The outline of what a square view in this direction sees, as
    polylines in equirect pixel coordinates (for drawing the view's
    footprint on a panorama). The view frame's border is walked edge by
    edge and mapped to the panorama; the result is split into separate
    polylines wherever it crosses the left/right (lon = +-180 deg) seam,
    so a caller can draw each one as-is without a line streaking across
    the whole image."""
    rotation = direction_rotation(spec, ring_tilt_degrees)
    half = math.tan(math.radians(fov_degrees) / 2.0)
    t = np.linspace(-half, half, samples_per_edge, endpoint=False)
    # Walk the border clockwise in camera NDC (x right, y up, z = 1).
    edges = [
        np.stack([t, np.full_like(t, half)], axis=-1),  # top, left -> right
        np.stack([np.full_like(t, half), -t], axis=-1),  # right, top -> bottom
        np.stack([-t, np.full_like(t, -half)], axis=-1),  # bottom, right -> left
        np.stack([np.full_like(t, -half), t], axis=-1),  # left, bottom -> top
    ]
    ndc = np.concatenate(edges + [edges[0][:1]])  # close the loop
    cam = np.concatenate([ndc, np.ones((len(ndc), 1))], axis=-1)
    pano = cam @ rotation.T
    pano /= np.linalg.norm(pano, axis=-1, keepdims=True)
    lon = np.arctan2(pano[:, 0], pano[:, 2])
    lat = np.arcsin(np.clip(pano[:, 1], -1.0, 1.0))
    u = (lon + np.pi) / (2 * np.pi) * eq_width - 0.5
    v = (np.pi / 2 - lat) / np.pi * eq_height - 0.5

    polylines: list[list[tuple[float, float]]] = [[(float(u[0]), float(v[0]))]]
    for i in range(1, len(u)):
        if abs(u[i] - u[i - 1]) > eq_width / 2:
            polylines.append([])
        polylines[-1].append((float(u[i]), float(v[i])))
    # A loop that crossed the seam starts and ends on the same side of it --
    # join the trailing piece onto the leading one so the count is minimal.
    if len(polylines) > 1 and abs(polylines[-1][-1][0] - polylines[0][0][0]) < eq_width / 2:
        polylines[0] = polylines.pop() + polylines[0][1:]
    return [p for p in polylines if len(p) >= 2]
