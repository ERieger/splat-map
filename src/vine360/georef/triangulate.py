"""A tag's position in model space from every registered view that saw it
(docs/adr/0047): linear DLT triangulation inside RANSAC over view pairs,
refit on the inliers and polished with a few Gauss-Newton steps on the
reprojection error. Pure numpy -- no pycolmap, so it tests without the sfm
extra.

Cameras are COLMAP-convention pinholes: x_cam = R @ X + t, pixel =
K @ x_cam / z, pixel centres at i + 0.5 -- the convention of both vine360's
view intrinsics and pupil-apriltags' detections (measured, ADR 0047).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

DEFAULT_THRESH_PX = 2.0
MIN_INLIER_VIEWS = 3
MIN_RAY_ANGLE_DEGREES = 2.0
MAX_RANSAC_PAIRS = 500


@dataclass
class TagObservation:
    """One view's detection of one tag, with that view's camera."""

    view_id: str
    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3) cam_from_world
    t: np.ndarray  # (3,)
    center: np.ndarray  # (2,) pixels
    corners: np.ndarray | None = None  # (4, 2) pixels, detector order

    @property
    def camera_center(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass
class TagPosition:
    tag_id: int
    xyz: np.ndarray | None  # None when rejected
    n_views: int  # observations offered
    n_inliers: int
    rms_px: float | None
    max_ray_angle_deg: float | None
    mean_edge_model: float | None  # mean edge of the triangulated corner square, model units
    inlier_view_ids: list[str] = field(default_factory=list)
    rejected: str | None = None  # why, when xyz is None

    def to_dict(self) -> dict:
        return {
            "xyz_model": None if self.xyz is None else [float(v) for v in self.xyz],
            "n_views": self.n_views,
            "n_inliers": self.n_inliers,
            "rms_px": self.rms_px,
            "max_ray_angle_deg": self.max_ray_angle_deg,
            "mean_edge_model": self.mean_edge_model,
            "inlier_view_ids": list(self.inlier_view_ids),
            "rejected": self.rejected,
        }

    @classmethod
    def from_dict(cls, tag_id: int, data: dict) -> "TagPosition":
        xyz = data.get("xyz_model")
        return cls(
            tag_id, None if xyz is None else np.asarray(xyz, dtype=np.float64), data["n_views"], data["n_inliers"],
            data.get("rms_px"), data.get("max_ray_angle_deg"), data.get("mean_edge_model"),
            list(data.get("inlier_view_ids", [])), data.get("rejected"),
        )


def _projection_matrix(obs: TagObservation) -> np.ndarray:
    return obs.K @ np.hstack([obs.R, obs.t.reshape(3, 1)])


def triangulate_dlt(projections: list[np.ndarray], pixels: list[np.ndarray]) -> np.ndarray:
    """Linear (DLT) triangulation: rows x*P3 - P1 and y*P3 - P2 per view,
    smallest right singular vector. Each row is normalized, which keeps the
    system well-conditioned without a separate Hartley normalization."""
    rows = []
    for P, (x, y) in zip(projections, pixels):
        for row in (x * P[2] - P[0], y * P[2] - P[1]):
            rows.append(row / (np.linalg.norm(row) or 1.0))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    X = vt[-1]
    if abs(X[3]) < 1e-15:
        return np.full(3, np.inf)
    return X[:3] / X[3]


def reprojection_errors(X: np.ndarray, observations: list[TagObservation], pixels: list[np.ndarray]) -> np.ndarray:
    """Pixel distance per view; inf for a point behind (or on) the camera."""
    errors = np.empty(len(observations))
    for i, (obs, px) in enumerate(zip(observations, pixels)):
        cam = obs.R @ X + obs.t
        if not np.all(np.isfinite(cam)) or cam[2] <= 1e-9:
            errors[i] = np.inf
            continue
        proj = obs.K @ (cam / cam[2])
        errors[i] = float(np.hypot(proj[0] - px[0], proj[1] - px[1]))
    return errors


def refine_point(X: np.ndarray, observations: list[TagObservation], pixels: list[np.ndarray], iterations: int = 10) -> np.ndarray:
    """Gauss-Newton on the summed squared reprojection error."""
    X = np.asarray(X, dtype=np.float64).copy()
    for _ in range(iterations):
        J_rows, r_rows = [], []
        for obs, px in zip(observations, pixels):
            cam = obs.R @ X + obs.t
            if cam[2] <= 1e-9:
                return X
            fx, fy, s = obs.K[0, 0], obs.K[1, 1], obs.K[0, 1]
            u = obs.K[0, 0] * cam[0] / cam[2] + s * cam[1] / cam[2] + obs.K[0, 2]
            v = fy * cam[1] / cam[2] + obs.K[1, 2]
            z2 = cam[2] ** 2
            d_cam = np.array(
                [
                    [fx / cam[2], s / cam[2], -(fx * cam[0] + s * cam[1]) / z2],
                    [0.0, fy / cam[2], -fy * cam[1] / z2],
                ]
            )
            J_rows.append(d_cam @ obs.R)
            r_rows.append([px[0] - u, px[1] - v])
        J = np.vstack(J_rows)
        r = np.concatenate(r_rows)
        try:
            step = np.linalg.lstsq(J, r, rcond=None)[0]
        except np.linalg.LinAlgError:
            return X
        X += step
        if np.linalg.norm(step) < 1e-12 * max(1.0, np.linalg.norm(X)):
            break
    return X


def max_ray_angle_degrees(X: np.ndarray, observations: list[TagObservation]) -> float:
    rays = []
    for obs in observations:
        d = X - obs.camera_center
        n = np.linalg.norm(d)
        if n > 0:
            rays.append(d / n)
    best = 0.0
    for a, b in itertools.combinations(rays, 2):
        best = max(best, math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0)))))
    return best


def _fit(observations: list[TagObservation], pixels: list[np.ndarray]) -> np.ndarray:
    X = triangulate_dlt([_projection_matrix(o) for o in observations], pixels)
    if not np.all(np.isfinite(X)):
        return X
    return refine_point(X, observations, pixels)


def triangulate_tag(
    tag_id: int,
    observations: list[TagObservation],
    *,
    thresh_px: float = DEFAULT_THRESH_PX,
    min_inlier_views: int = MIN_INLIER_VIEWS,
    min_ray_angle_deg: float = MIN_RAY_ANGLE_DEGREES,
    max_pairs: int = MAX_RANSAC_PAIRS,
    seed: int = 0,
) -> TagPosition:
    """RANSAC over view pairs (every pair, or max_pairs random ones):
    triangulate the pair, count views reprojecting within thresh_px, keep
    the largest inlier set (ties: lowest summed error), refit on it.
    Rejected (xyz None, `rejected` says why) with fewer than
    min_inlier_views inliers or near-parallel inlier rays. The four corners
    are then triangulated from the same inlier views, for the tag-size scale
    check (mean_edge_model)."""
    n = len(observations)
    if n < 2:
        return TagPosition(tag_id, None, n, 0, None, None, None, rejected=f"seen in {n} view(s), need at least 2")
    centers = [np.asarray(o.center, dtype=np.float64) for o in observations]

    pairs = list(itertools.combinations(range(n), 2))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=max_pairs, replace=False)]

    best_inliers: np.ndarray | None = None
    best_cost = np.inf
    for i, j in pairs:
        X = triangulate_dlt([_projection_matrix(observations[i]), _projection_matrix(observations[j])], [centers[i], centers[j]])
        if not np.all(np.isfinite(X)):
            continue
        errors = reprojection_errors(X, observations, centers)
        inliers = np.flatnonzero(errors < thresh_px)
        cost = float(np.sum(np.minimum(errors, thresh_px)))
        if best_inliers is None or len(inliers) > len(best_inliers) or (len(inliers) == len(best_inliers) and cost < best_cost):
            best_inliers, best_cost = inliers, cost

    if best_inliers is None or len(best_inliers) < 2:
        return TagPosition(tag_id, None, n, 0 if best_inliers is None else len(best_inliers), None, None, None,
                           rejected="no two views agree on a position")

    # Refit on the inliers, then re-collect inliers under the refit (once).
    inlier_obs = [observations[k] for k in best_inliers]
    X = _fit(inlier_obs, [centers[k] for k in best_inliers])
    errors = reprojection_errors(X, observations, centers)
    refit_inliers = np.flatnonzero(errors < thresh_px)
    if len(refit_inliers) >= len(best_inliers):
        best_inliers = refit_inliers
        inlier_obs = [observations[k] for k in best_inliers]
        X = _fit(inlier_obs, [centers[k] for k in best_inliers])
        errors = reprojection_errors(X, observations, centers)

    inlier_errors = errors[best_inliers]
    rms = float(np.sqrt(np.mean(np.square(inlier_errors)))) if np.all(np.isfinite(inlier_errors)) else None
    angle = max_ray_angle_degrees(X, inlier_obs)
    view_ids = [observations[k].view_id for k in best_inliers]
    if len(best_inliers) < min_inlier_views:
        return TagPosition(tag_id, None, n, len(best_inliers), rms, angle, None, view_ids,
                           rejected=f"only {len(best_inliers)} consistent view(s), need {min_inlier_views}")
    if angle < min_ray_angle_deg:
        return TagPosition(tag_id, None, n, len(best_inliers), rms, angle, None, view_ids,
                           rejected=f"views too close together (max ray angle {angle:.2f}° < {min_ray_angle_deg:g}°)")

    return TagPosition(tag_id, X, n, len(best_inliers), rms, angle, _mean_edge(inlier_obs), view_ids)


def _mean_edge(observations: list[TagObservation]) -> float | None:
    """Mean edge length of the square formed by the four triangulated
    corners -- None if any view lacks corners."""
    if any(o.corners is None for o in observations):
        return None
    corners = []
    for k in range(4):
        pixels = [np.asarray(o.corners, dtype=np.float64)[k] for o in observations]
        corner = _fit(observations, pixels)
        if not np.all(np.isfinite(corner)):
            return None
        corners.append(corner)
    edges = [np.linalg.norm(corners[(k + 1) % 4] - corners[k]) for k in range(4)]
    return float(np.mean(edges))
