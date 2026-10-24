"""AprilTag detection on vine360's projected pinhole views (docs/adr/0047).

**Verified against the installed library**: pupil-apriltags 1.0.4.post11
(the `georef` extra; cp312 manylinux wheel, which bundles the AprilTag 3 C
library). `Detector(families, nthreads, quad_decimate, quad_sigma,
refine_edges, decode_sharpening, debug)` and `Detector.detect(img)` --
img a C-contiguous uint8 grayscale array -- were checked with
inspect.signature on that install, and each result's `tag_family`
(bytes), `tag_id`, `hamming`, `decision_margin`, `center` and `corners`
attributes against real detections of the AprilRobotics reference tag36h11
images (fixtures/apriltag/).

Measured conventions (tests/test_georef_apriltag.py pins them):
- pixel coordinates are COLMAP's -- pixel i spans [i, i+1), centre i + 0.5
  -- so a view's stored (fx, fy, cx, cy) apply to them unchanged;
- corners lie on the outer edge of the black square (the printed tag size),
  ordered bottom-left, bottom-right, top-right, top-left in the *tag's own*
  frame, so corner k is the same physical corner in every view; they sit
  ~0.1-0.3 px outward of the true edge.

**Teardown is ours, not the library's.** pupil-apriltags' own
`Detector.__del__` destroys the tag family *before* the detector, and
`apriltag_detector_destroy` then reads the freed family
(`apriltag_detector_clear_families` -> `quick_decode_uninit(fam)`): a
use-after-free that segfaulted 2 of 3 full test-suite runs here, whenever
the freed memory had been reused. AprilTag's own documented order is
detector first, then family. `TagDetector.close()` (also run from its
`__del__`) does that and clears the library object's pointer, so its own
`__del__` is a no-op.

Only detect tags on pinhole images -- never on an equirectangular frame
(the detector's quad fitting assumes straight edges). The rest of
vine360.georef never imports the detector library: it sees `Detection`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DETECTOR_NAME = "pupil-apriltags"


class AprilTagUnavailable(Exception):
    pass


@dataclass(frozen=True)
class DetectorParams:
    """pupil-apriltags' Detector arguments that matter here. quad_decimate
    1.0 (no downsampling) rather than the library's 2.0 default: tags in a
    1024 px view of a vineyard are often only 20-40 px across."""

    quad_decimate: float = 1.0
    quad_sigma: float = 0.0
    refine_edges: bool = True
    decode_sharpening: float = 0.25
    nthreads: int = 4
    max_hamming: int = 0  # reject any detection that needed a corrected bit
    min_decision_margin: float = 20.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict | None) -> "DetectorParams":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (data or {}).items() if k in known})


@dataclass
class Detection:
    tag_id: int
    family: str
    corners: np.ndarray  # (4, 2) pixels, detector order
    center: np.ndarray  # (2,)
    decision_margin: float
    hamming: int
    extra: dict = field(default_factory=dict)


def detector_version() -> str:
    try:
        from importlib.metadata import version

        return version("pupil-apriltags")
    except Exception as exc:  # PackageNotFoundError, or a broken install
        raise AprilTagUnavailable(
            "pupil-apriltags isn't installed -- install vine360's `georef` extra (pip install -e '.[georef]')"
        ) from exc


def is_available() -> bool:
    try:
        import pupil_apriltags  # noqa: F401
    except ImportError:
        return False
    return True


class TagDetector:
    """One reusable detector for one family (building it loads the family's
    code table, so don't build one per image)."""

    def __init__(self, family: str = "tag36h11", params: DetectorParams = DetectorParams()):
        try:
            import pupil_apriltags
        except ImportError as exc:
            raise AprilTagUnavailable(
                "pupil-apriltags isn't installed -- install vine360's `georef` extra (pip install -e '.[georef]')"
            ) from exc
        self.family = family
        self.params = params
        self._detector = pupil_apriltags.Detector(
            families=family,
            nthreads=params.nthreads,
            quad_decimate=params.quad_decimate,
            quad_sigma=params.quad_sigma,
            refine_edges=int(params.refine_edges),
            decode_sharpening=params.decode_sharpening,
        )

    def close(self) -> None:
        """Frees the C detector, then its family -- AprilTag's documented
        order (see the module docstring for the library bug this avoids)."""
        detector = getattr(self, "_detector", None)
        if detector is None or getattr(detector, "tag_detector_ptr", None) is None:
            return
        libc = detector.libc
        libc.apriltag_detector_destroy.restype = None
        libc.apriltag_detector_destroy(detector.tag_detector_ptr)
        detector.tag_detector_ptr = None  # the library's own __del__ now does nothing
        for family, pointer in detector.tag_families.items():
            destroy = getattr(libc, f"{family}_destroy")
            destroy.restype = None
            destroy(pointer)
        detector.tag_families = {}

    def __del__(self):
        self.close()

    def __enter__(self) -> "TagDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def detect(self, image: np.ndarray) -> list[Detection]:
        """image: (H, W) or (H, W, 3/4) uint8. Detections failing the
        hamming / decision-margin limits are dropped."""
        if self._detector.tag_detector_ptr is None:
            raise AprilTagUnavailable("this TagDetector has been closed")
        gray = to_gray(image)
        out = []
        for r in self._detector.detect(gray):
            if r.hamming > self.params.max_hamming or r.decision_margin < self.params.min_decision_margin:
                continue
            family = r.tag_family.decode() if isinstance(r.tag_family, bytes) else str(r.tag_family)
            out.append(
                Detection(
                    int(r.tag_id), family, np.asarray(r.corners, dtype=np.float64),
                    np.asarray(r.center, dtype=np.float64), float(r.decision_margin), int(r.hamming),
                )
            )
        return out


def to_gray(image: np.ndarray) -> np.ndarray:
    """uint8 grayscale, C-contiguous (what the C library needs). RGB uses
    ITU-R 601 luma, the same weights as Pillow's convert("L")."""
    image = np.asarray(image)
    if image.ndim == 3:
        rgb = image[..., :3].astype(np.float64)
        image = rgb @ np.array([0.299, 0.587, 0.114])
    if image.dtype != np.uint8:
        image = np.clip(np.round(image), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def load_gray(path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.ascontiguousarray(np.asarray(img.convert("L")))
