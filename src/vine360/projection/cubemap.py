"""The six-face, 90-degree-FOV cubemap preset (handover doc, section 4,
step 3: "Start with a cubemap-like six-face preset at 90 FOV... Exclude or
carefully handle polar views with little useful vineyard content").

Up/down faces are excluded by default for exactly that reason -- pointed at
sky and ground/canopy-top respectively, they add little to a terrestrial
vineyard reconstruction and cost extra processing. Pass
`include_polar_faces=True` to generate all six.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vine360.projection.geometry import Intrinsics, face_rotation

DEFAULT_FOV_DEGREES = 90.0

# (name, yaw_degrees, pitch_degrees, is_polar)
_FACES = [
    ("front", 0.0, 0.0, False),
    ("right", 90.0, 0.0, False),
    ("back", 180.0, 0.0, False),
    ("left", 270.0, 0.0, False),
    ("up", 0.0, 90.0, True),
    ("down", 0.0, -90.0, True),
]

ALL_FACE_NAMES = [name for name, *_ in _FACES]


@dataclass(frozen=True)
class FaceSpec:
    name: str
    intrinsics: Intrinsics
    fixed_rotation: "object"  # np.ndarray, R_panorama_face

    def fixed_rotation_to_dict(self) -> dict:
        return {"matrix": self.fixed_rotation.tolist()}


def six_face_preset(
    face_size: int,
    fov_degrees: float = DEFAULT_FOV_DEGREES,
    *,
    include_polar_faces: bool = False,
    face_names: list[str] | None = None,
) -> list[FaceSpec]:
    """face_names, if given, selects an explicit subset (in the order the
    handover doc's preset defines: front, right, back, left, up, down) --
    overrides include_polar_faces. Unknown names raise ValueError."""
    if face_names is not None:
        unknown = set(face_names) - set(ALL_FACE_NAMES)
        if unknown:
            raise ValueError(f"unknown face name(s): {sorted(unknown)}; valid names are {ALL_FACE_NAMES}")

    intrinsics = Intrinsics.from_fov(face_size, face_size, fov_degrees)
    faces = []
    for name, yaw_deg, pitch_deg, is_polar in _FACES:
        if face_names is not None:
            if name not in face_names:
                continue
        elif is_polar and not include_polar_faces:
            continue
        rotation = face_rotation(math.radians(yaw_deg), math.radians(pitch_deg))
        faces.append(FaceSpec(name=name, intrinsics=intrinsics, fixed_rotation=rotation))
    return faces
