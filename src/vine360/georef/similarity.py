"""The 7-DOF similarity between model and local coordinates (docs/adr/0047):
X_local = s * R @ X_model + t, solved in closed form -- Umeyama (1991),
"Least-squares estimation of transformation parameters between two point
patterns", with the reflection fix. No optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_POINTS = 3
RECOMMENDED_POINTS = 5
SUSPECT_LOO_FACTOR = 3.0  # a point whose leave-one-out residual is > 3x the median is "suspect"


@dataclass
class Similarity:
    scale: float
    rotation: np.ndarray  # (3, 3), proper (det +1)
    translation: np.ndarray  # (3,)

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(points, dtype=np.float64) @ self.rotation.T + self.translation

    def to_dict(self) -> dict:
        return {"scale": float(self.scale), "rotation": self.rotation.tolist(), "translation": self.translation.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "Similarity":
        return cls(float(data["scale"]), np.asarray(data["rotation"], dtype=np.float64), np.asarray(data["translation"], dtype=np.float64))


def umeyama(src: np.ndarray, dst: np.ndarray, *, with_scale: bool = True) -> Similarity:
    """The least-squares similarity mapping src (N, 3) onto dst (N, 3), N >= 3
    non-collinear points. Never returns a reflection: when det(U V^T) < 0 the
    last singular direction is flipped (Umeyama's S matrix)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src and dst must both be (N, 3), got {src.shape} and {dst.shape}")
    if len(src) < MIN_POINTS:
        raise ValueError(f"need at least {MIN_POINTS} point pairs, got {len(src)}")
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    covariance = dst_c.T @ src_c / len(src)
    u, d, vt = np.linalg.svd(covariance)
    if d[1] < 1e-12 * max(d[0], 1e-300):
        raise ValueError("points are collinear (or coincident) -- the rotation is undetermined")
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt
    if with_scale:
        variance = (src_c**2).sum() / len(src)
        scale = float(np.trace(np.diag(d) @ sign) / variance)
    else:
        scale = 1.0
    translation = mu_dst - scale * rotation @ mu_src
    return Similarity(scale, rotation, translation)


@dataclass
class Residuals:
    """Per-point dst - fitted(src), and RMSEs. Local frame is E, N, H, so
    horizontal is the first two axes, vertical the third."""

    vectors: np.ndarray  # (N, 3)
    d3: np.ndarray  # (N,)
    horizontal: np.ndarray
    vertical: np.ndarray  # signed

    @property
    def rmse_3d(self) -> float:
        return _rms(self.d3)

    @property
    def rmse_horizontal(self) -> float:
        return _rms(self.horizontal)

    @property
    def rmse_vertical(self) -> float:
        return _rms(self.vertical)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else 0.0


def residuals(similarity: Similarity, src: np.ndarray, dst: np.ndarray) -> Residuals:
    vectors = np.asarray(dst, dtype=np.float64) - similarity.apply(src)
    return Residuals(
        vectors,
        np.linalg.norm(vectors, axis=1),
        np.linalg.norm(vectors[:, :2], axis=1),
        vectors[:, 2],
    )


def leave_one_out(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """(N,) 3D residual of each point under the fit to all the *other*
    points -- NaN where the rest can't be fitted (fewer than 3 left, or
    collinear)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    out = np.full(len(src), np.nan)
    for i in range(len(src)):
        keep = np.arange(len(src)) != i
        try:
            fit = umeyama(src[keep], dst[keep])
        except ValueError:
            continue
        out[i] = float(np.linalg.norm(dst[i] - fit.apply(src[i : i + 1])[0]))
    return out


def suspect_flags(loo: np.ndarray, factor: float = SUSPECT_LOO_FACTOR) -> np.ndarray:
    """True where a leave-one-out residual is > factor x the median of them."""
    finite = loo[np.isfinite(loo)]
    if len(finite) < MIN_POINTS + 1:
        return np.zeros(len(loo), dtype=bool)
    median = float(np.median(finite))
    return np.isfinite(loo) & (loo > factor * median) & (median > 0)
