"""Georeferences an already-trained Gaussian splat (docs/adr/0047): the
Georeference panel's fourth step.

A splat trained from a *non-georeferenced* export of an SfM run is in that
run's own model frame, so a solution for that run moves it into the
solution's local frame (CRS - origin) with the same similarity it applied
to the run's model. Non-destructive: the input .ply is only read, and the
result is a new .ply plus ``<output stem>.georef.json`` (and
``.georef.prj`` when the EPSG is known) describing the CRS and origin.

The transform is only right for a splat in that run's model frame. Two
mistakes are easy to make and give a silently wrong result:

- the splat was trained from a georeferenced export, so it is already
  local and would be moved twice;
- it was trained from a different run, or from an images-only export where
  the trainer ran its own SfM (an unrelated, arbitrary frame).

So before writing, check_splat_frame compares the splat with the solution's
materialised model (its 3D points, plus camera centres when there are few
points): after the transform, the splat's median position should sit
inside the model's extent, at a similar spread. A splat that only matches
*without* the transform is reported as already local. On any mismatch,
georeference_splat raises SplatFrameMismatch unless force=True, so the GUI
can ask before writing. The check is a coarse sanity check (medians and
interquartile spreads, robust to the background Gaussians a trainer adds),
not a registration.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vine360.georef.apply import (
    GeorefApplyError,
    _read_ply_header,
    georef_metadata,
    transform_splat_ply,
    write_georef_sidecars,
)
from vine360.georef.similarity import Similarity

MAX_SAMPLES = 200_000
MIN_REFERENCE_POINTS = 20  # fewer model points than this: add the camera centres to the reference
CENTRE_TOLERANCE = 0.5  # splat median within this fraction of the model's extent of the model's median
SPREAD_RATIO_RANGE = (0.1, 10.0)  # splat / model interquartile spread

CHECK_OK = "ok"
CHECK_ALREADY_LOCAL = "already_local"
CHECK_MISMATCH = "mismatch"
CHECK_SKIPPED = "skipped"


class SplatFrameMismatch(GeorefApplyError):
    def __init__(self, check: "SplatFrameCheck"):
        super().__init__(check.message)
        self.check = check


@dataclass
class SplatFrameCheck:
    status: str
    message: str

    def to_dict(self) -> dict:
        return {"status": self.status, "message": self.message}


@dataclass
class SplatResult:
    count: int
    out_path: Path
    sidecars: list[Path]
    check: SplatFrameCheck
    warnings: list[str]


def sample_splat_positions(path: Path, max_samples: int = MAX_SAMPLES) -> np.ndarray:
    """Up to max_samples evenly spaced Gaussian centres (N x 3), read
    without loading the whole file."""
    path = Path(path)
    with path.open("rb") as f:
        _header, count, dtype, offset = _read_ply_header(f)
    if not {"x", "y", "z"} <= set(dtype.names):
        raise GeorefApplyError("PLY has no x/y/z vertex properties")
    if count == 0:
        return np.zeros((0, 3))
    vertices = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    index = np.unique(np.linspace(0, count - 1, min(count, max_samples)).astype(np.int64))
    picked = vertices[index]
    return np.stack([picked["x"], picked["y"], picked["z"]], axis=1).astype(np.float64)


def reference_points(project_root: Path, solution: dict) -> np.ndarray | None:
    """The solution's materialised model in local coordinates: its 3D
    points, plus camera centres when there are few points. None when the
    model can't be read (no pycolmap, or the folder is gone)."""
    if not solution.get("model_dir"):
        return None
    model_dir = Path(project_root) / solution["model_dir"]
    try:
        import pycolmap

        reconstruction = pycolmap.Reconstruction(str(model_dir))
    except Exception:  # noqa: BLE001 -- any failure just skips the check
        return None
    points = [p.xyz for p in reconstruction.points3D.values()]
    if len(points) < MIN_REFERENCE_POINTS:
        for image in reconstruction.images.values():
            if image.has_pose:
                points.append(image.projection_center())
    return np.asarray(points, dtype=np.float64) if points else None


def _matches(positions: np.ndarray, reference: np.ndarray) -> tuple[bool, float, float]:
    """(matches, centre offset / model extent, spread ratio)."""
    lo, hi = np.percentile(reference, [2, 98], axis=0)
    extent = max(float(np.linalg.norm(hi - lo)), 1e-9)
    offset = float(np.linalg.norm(np.median(positions, axis=0) - np.median(reference, axis=0))) / extent

    def spread(points):
        q1, q3 = np.percentile(points, [25, 75], axis=0)
        return max(float(np.linalg.norm(q3 - q1)), 1e-9)

    ratio = spread(positions) / spread(reference)
    return offset <= CENTRE_TOLERANCE and SPREAD_RATIO_RANGE[0] <= ratio <= SPREAD_RATIO_RANGE[1], offset, ratio


def check_splat_frame(positions: np.ndarray, similarity: Similarity, reference: np.ndarray | None) -> SplatFrameCheck:
    if reference is None or len(reference) < 3:
        return SplatFrameCheck(CHECK_SKIPPED, "Frame check skipped: the solution's model couldn't be read.")
    if len(positions) == 0:
        return SplatFrameCheck(CHECK_MISMATCH, "The splat has no Gaussians.")
    moved = similarity.apply(positions)
    ok, offset, ratio = _matches(moved, reference)
    if ok:
        return SplatFrameCheck(CHECK_OK, f"Splat matches the run's frame (centre offset {offset:.0%} of the model's extent, "
                                         f"spread ×{ratio:.2f}).")
    already, _, _ = _matches(positions, reference)
    if already:
        return SplatFrameCheck(
            CHECK_ALREADY_LOCAL,
            "This splat already lines up with the georeferenced model, so it looks like it was trained from a "
            "georeferenced export. Transforming it again would move it twice.",
        )
    return SplatFrameCheck(
        CHECK_MISMATCH,
        f"After the transform the splat doesn't line up with this solution's model (centre offset {offset:.0%} of the "
        f"model's extent, spread ×{ratio:.2f}). It was probably trained from a different SfM run, or from an "
        "images-only export where the trainer ran its own SfM.",
    )


def sidecar_stem(out_path: Path) -> str:
    return f"{Path(out_path).stem}.georef"


def georeference_splat(
    conn: sqlite3.Connection,
    project_root: Path,
    in_path: Path,
    out_path: Path,
    solution_id: str,
    *,
    force: bool = False,
    progress_callback=None,
) -> SplatResult:
    from vine360.georef.control_points import get_control_point_set
    from vine360.georef.solve import load_solution

    notify = progress_callback or (lambda *a: None)
    in_path, out_path = Path(in_path), Path(out_path)
    if out_path.resolve() == in_path.resolve():
        raise GeorefApplyError("refusing to overwrite the input splat -- choose a different output file")
    solution = load_solution(conn, solution_id)
    similarity = Similarity.from_dict(solution["transform"])

    notify("Georeference splat step 1/2: checking the splat's frame against the solution's model", 0, 2)
    check = check_splat_frame(sample_splat_positions(in_path), similarity, reference_points(project_root, solution))
    if check.status in (CHECK_ALREADY_LOCAL, CHECK_MISMATCH) and not force:
        raise SplatFrameMismatch(check)

    def step2(message, current, total):
        notify(f"Georeference splat step 2/2: {message}", current, total)

    count = transform_splat_ply(in_path, out_path, similarity, progress_callback=step2)
    metadata = georef_metadata(solution, get_control_point_set(conn, solution["set_id"]))
    metadata.update({
        "source_splat": str(in_path.resolve()),
        "frame_check": check.to_dict() | ({"overridden": True} if force and check.status != CHECK_OK else {}),
        "note": "Higher-order SH (f_rest_*) are not rotated; degree-0 colour is exact (docs/adr/0047).",
    })
    stem = sidecar_stem(out_path)
    warnings = write_georef_sidecars(out_path.parent, metadata, name=stem)
    sidecars = [p for p in (out_path.with_name(f"{stem}.json"), out_path.with_name(f"{stem}.prj")) if p.exists()]
    notify(f"Georeferenced {count} Gaussians -> {out_path.name} ({check.status})", 2, 2)
    return SplatResult(count, out_path, sidecars, check, warnings)
