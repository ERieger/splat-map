"""A tiny ray-cast equirectangular "room" renderer for tests that need real
360 images with real parallax and known ground-truth poses -- what SfM on
raw equirectangular frames (SphereSfM, pycolmap's EQUIRECTANGULAR model)
actually needs to register anything (a single panorama, or several from
one spot, can't be reconstructed -- docs/adr/0007).

World frame uses vine360's own convention (x right, y up, z forward --
vine360.projection.geometry), so a camera's ground-truth rotation maps a
direction from its panorama frame into the world frame directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from vine360.projection.geometry import equirect_pixel_to_direction_batch, rotation_y

ROOM_HALF_EXTENT = np.array([6.0, 3.0, 6.0])  # an axis-aligned box, centred on the origin


def _wall_texture(seed: int, size: int = 512) -> np.ndarray:
    """Multi-scale smoothed noise plus hard-edged blobs: plenty of
    distinctive, repeatable SIFT features, the same from any viewpoint."""
    rng = np.random.default_rng(seed)
    tex = np.zeros((size, size, 3), dtype=np.float64)
    for cells, weight in ((4, 0.35), (16, 0.35), (64, 0.3)):
        grid = rng.random((cells, cells, 3))
        up = np.asarray(
            Image.fromarray((grid * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC), dtype=np.float64
        )
        tex += weight * up
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(60):
        cx, cy, r = rng.integers(0, size), rng.integers(0, size), rng.integers(4, 20)
        tex[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = rng.integers(0, 255, size=3)
    return np.clip(tex, 0, 255).astype(np.uint8)


_TEXTURES = [_wall_texture(seed) for seed in range(6)]


def render_equirect(center: np.ndarray, rotation_world_from_pano: np.ndarray, width: int = 1024) -> np.ndarray:
    """Renders the room as seen from `center`, with the panorama frame
    rotated into the world by `rotation_world_from_pano`."""
    height = width // 2
    v, u = np.mgrid[0:height, 0:width]
    dirs = equirect_pixel_to_direction_batch(u.astype(np.float64), v.astype(np.float64), width, height)
    dirs = dirs @ np.asarray(rotation_world_from_pano).T

    with np.errstate(divide="ignore", invalid="ignore"):
        t_pos = (ROOM_HALF_EXTENT - center) / dirs
        t_neg = (-ROOM_HALF_EXTENT - center) / dirs
    t_axis = np.where(dirs > 0, t_pos, t_neg)  # the wall each axis's ray heads towards
    t_axis = np.where(np.isfinite(t_axis) & (t_axis > 0), t_axis, np.inf)
    axis = np.argmin(t_axis, axis=-1)
    t = np.take_along_axis(t_axis, axis[..., None], axis=-1)[..., 0]
    hit = center + dirs * t[..., None]
    positive = np.take_along_axis(dirs, axis[..., None], axis=-1)[..., 0] > 0
    wall = axis * 2 + positive.astype(int)

    image = np.zeros((height, width, 3), dtype=np.uint8)
    other_axes = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
    for a in range(3):
        a1, a2 = other_axes[a]
        for side in range(2):
            mask = wall == a * 2 + side
            if not mask.any():
                continue
            tex = _TEXTURES[a * 2 + side]
            s = (hit[..., a1][mask] / ROOM_HALF_EXTENT[a1] + 1) / 2
            r = (hit[..., a2][mask] / ROOM_HALF_EXTENT[a2] + 1) / 2
            ti = np.clip((s * (tex.shape[1] - 1)).astype(int), 0, tex.shape[1] - 1)
            tj = np.clip((r * (tex.shape[0] - 1)).astype(int), 0, tex.shape[0] - 1)
            image[mask] = tex[tj, ti]
    return image


def ground_truth_trajectory(n: int = 6, offset=(0.0, 0.0, 0.0)) -> list[tuple[np.ndarray, np.ndarray]]:
    """(center, R_world_from_pano) per frame: a gentle walk with a little
    yaw, like a handheld 360 pass -- real translation between frames.
    `offset` shifts the whole walk, e.g. to stand in for a second capture
    of the same scene from higher up."""
    poses = []
    for i in range(n):
        center = np.array([-1.5 + 0.6 * i, 0.1 * np.sin(i), -0.5 + 0.25 * i]) + np.asarray(offset, dtype=np.float64)
        poses.append((center, rotation_y(np.radians(8.0 * i - 20.0))))
    return poses


def write_frames(directory: Path, n: int = 6, width: int = 1024, offset=(0.0, 0.0, 0.0)) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (center, rotation) in enumerate(ground_truth_trajectory(n, offset)):
        path = directory / f"frame_{i + 1:06d}.png"
        Image.fromarray(render_equirect(center, rotation, width)).save(path)
        paths.append(path)
    return paths


def add_synthetic_frame_set(
    conn, project_root: Path, *, n: int = 6, width: int = 1024, source_id: str = "synth", offset=(0.0, 0.0, 0.0)
) -> str:
    """Registers a synthetic equirectangular source and one frame set of
    `n` rendered frames, shaped exactly like extract_frames writes them
    (frames/<frame_set_id>/frame_NNNNNN.png, frame_id <frame_set_id>:NNNNNN).
    Returns the frame_set_id."""
    import json

    frame_set_id = f"{source_id}~i1"
    conn.execute(
        "INSERT INTO sources (source_id, path, checksum, media_type, projection, width, height, timestamps) "
        "VALUES (?, '/synthetic/room.mp4', 'synthetic', 'video', 'equirectangular', ?, ?, ?)",
        (source_id, width, width // 2, json.dumps({"duration_seconds": float(n)})),
    )
    conn.execute(
        "INSERT INTO frame_sets (frame_set_id, source_id, label, extraction_settings, created_at) "
        "VALUES (?, ?, 'every 1s', ?, '2026-01-01')",
        (frame_set_id, source_id, json.dumps({"mode": "interval", "interval_seconds": 1.0})),
    )
    paths = write_frames(Path(project_root) / "frames" / frame_set_id, n=n, width=width, offset=offset)
    for i, path in enumerate(paths):
        conn.execute(
            "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
            "VALUES (?, ?, ?, '{}', ?, 'synthetic', ?)",
            (f"{frame_set_id}:{i:06d}", source_id, float(i), str(path.relative_to(project_root)), frame_set_id),
        )
    conn.commit()
    return frame_set_id
