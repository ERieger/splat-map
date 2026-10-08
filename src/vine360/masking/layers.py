"""Layered masks (docs/adr/0038): each exclusion class -- sky, person,
overexposure, a whole-view exclusion -- is stored as its own finished binary
layer, and the keep mask every consumer reads (pycolmap SfM, the Postshot /
RealityScan exports, the GUI) is a *composite* of the enabled layers.

Why layers: one class can be regenerated with new parameters (e.g. a lower
clip level for overexposure), switched off for one view or a whole frame
set, or hand-edited outside vine360, without re-running the others -- SAM 3
person/sky in particular is slow.

Layout and records:
- masks/classes/<view_id>/<layer>.png -- the layer itself, 255 = exclude.
  Same directory the pre-layer build already wrote per-class PNGs to, so
  every existing cascade delete (vine360.cleanup.remove_mask_files) removes
  layers too. Morphology is applied per layer *before* it's saved, so the
  PNG on disk is exactly what gets merged.
- mask_layers row per (view_id, layer): method, params, enabled, edited,
  coverage, updated_at.
- The composite: masks/keep/<...>.png.png (COLMAP's mask convention,
  semantics.colmap_mask_path), masks/classes/<view_id>/exclude.png, and the
  one-per-view `masks` row (keep_fraction, updated_at). Rewritten by
  compose_view whenever any of the view's layers changes -- not deferred to
  export, because SfM reads masks/keep too -- so Pose staleness detection
  (masks.updated_at vs sfm_runs) keeps working unchanged.
  ensure_composites_current() re-checks before SfM/export in case a layer
  was edited on disk or a compose was interrupted.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image

from vine360.masking.classical_sky import classify_sky_classical
from vine360.masking.overexposure import METHOD_VERSION as OVEREXPOSURE_METHOD_VERSION
from vine360.masking.overexposure import OverexposureConfig, classify_overexposed
from vine360.masking.semantics import (
    MASK_FALSE,
    MASK_TRUE,
    colmap_mask_path,
    dilate_exclude_mask,
    keep_fraction,
    keep_mask_from_exclude,
    load_mask,
    remove_small_components,
    save_mask,
)

PROJECTIONS_DIRNAME = "projections"

LAYER_SKY = "sky"
LAYER_PERSON = "person"
LAYER_OVEREXPOSED = "overexposed"
LAYER_EXCLUDED_VIEW = "excluded_view"

# File mtimes are compared against the row's updated_at to spot layers
# edited outside vine360; allow for filesystem timestamp granularity
# (DrvFs/NTFS) and the gap between writing the file and the row.
_EDIT_DETECTION_SLACK_SECONDS = 2.0


class MaskLayerError(Exception):
    pass


@dataclass(frozen=True)
class LayerSpec:
    name: str
    label: str
    color: tuple[int, int, int]  # review overlay tint
    default_params: dict = field(default_factory=dict)
    needs_sam3: bool = False


LAYER_SPECS: dict[str, LayerSpec] = {
    LAYER_SKY: LayerSpec(
        LAYER_SKY, "Sky", (40, 120, 255), {"method": "classical", "min_component_px": 64, "dilation_px": 3}
    ),
    LAYER_PERSON: LayerSpec(
        LAYER_PERSON,
        "Person",
        (255, 50, 50),
        {"method": "sam3", "min_component_px": 64, "dilation_px": 3},
        needs_sam3=True,
    ),
    LAYER_OVEREXPOSED: LayerSpec(
        LAYER_OVEREXPOSED,
        "Overexposed",
        (255, 0, 255),
        {
            "method": "classical",
            "clip_threshold": OverexposureConfig.clip_threshold,
            "min_core_fraction": OverexposureConfig.min_core_fraction,
            "bloom_threshold": OverexposureConfig.bloom_threshold,
            "bloom_radius_px": OverexposureConfig.bloom_radius_px,
            "bloom_max_texture": OverexposureConfig.bloom_max_texture,
            "dilation_px": OverexposureConfig.dilation_px,
        },
    ),
    LAYER_EXCLUDED_VIEW: LayerSpec(LAYER_EXCLUDED_VIEW, "Excluded view", (128, 128, 128), {"method": "manual"}),
}


@dataclass(frozen=True)
class LayerBuildSummary:
    views: int
    built: dict[str, int]  # layer -> views (re)built
    skipped_edited: dict[str, int]  # layer -> hand-edited views left alone
    nonempty: dict[str, int]  # layer -> views where the layer excludes anything


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def resolve_params(layer: str, params: dict | None) -> dict:
    spec = LAYER_SPECS.get(layer)
    if spec is None:
        raise MaskLayerError(f"unknown mask layer: {layer}")
    merged = dict(spec.default_params)
    merged.update(params or {})
    return merged


def layer_path(project_root: Path, view_id: str, layer: str) -> Path:
    return Path(project_root) / "masks" / "classes" / view_id / f"{layer}.png"


def _view_image_path(conn: sqlite3.Connection, view_id: str) -> Path:
    row = conn.execute("SELECT image_path FROM views WHERE view_id = ?", (view_id,)).fetchone()
    if row is None:
        raise MaskLayerError(f"unknown view_id: {view_id}")
    return Path(row[0])


def keep_mask_path(project_root: Path, image_relative_path: Path) -> Path:
    return colmap_mask_path(
        Path(image_relative_path).relative_to(PROJECTIONS_DIRNAME), Path(project_root) / "masks" / "keep"
    )


def _cleanup(mask: np.ndarray, params: dict) -> np.ndarray:
    mask = remove_small_components(mask, int(params.get("min_component_px", 0)))
    return dilate_exclude_mask(mask, int(params.get("dilation_px", 0)))


def compute_layer(image: np.ndarray, layer: str, params: dict, sam3_adapter=None) -> np.ndarray:
    """The finished (H, W) uint8 layer for one image, 255 = exclude."""
    method = params.get("method")
    if layer == LAYER_EXCLUDED_VIEW:
        return np.full(image.shape[:2], MASK_TRUE, dtype=np.uint8)
    if layer == LAYER_OVEREXPOSED:
        config = OverexposureConfig(
            clip_threshold=int(params["clip_threshold"]),
            min_core_fraction=float(params["min_core_fraction"]),
            bloom_threshold=int(params["bloom_threshold"]),
            bloom_radius_px=int(params["bloom_radius_px"]),
            bloom_max_texture=float(params.get("bloom_max_texture", OverexposureConfig.bloom_max_texture)),
            dilation_px=int(params["dilation_px"]),
        )
        return classify_overexposed(image, config)
    if layer == LAYER_SKY and method == "classical":
        return _cleanup(classify_sky_classical(image), params)
    if layer in (LAYER_SKY, LAYER_PERSON) and method == "sam3":
        if sam3_adapter is None:
            raise MaskLayerError(f"layer {layer!r} with method sam3 needs a SAM 3 adapter")
        return _cleanup(sam3_adapter.segment(image, {layer: layer})[layer], params)
    raise MaskLayerError(f"layer {layer!r} has no method {method!r}")


def _layer_row(conn: sqlite3.Connection, view_id: str, layer: str):
    return conn.execute(
        "SELECT enabled, edited, updated_at FROM mask_layers WHERE view_id = ? AND layer = ?", (view_id, layer)
    ).fetchone()


def _write_layer(
    conn: sqlite3.Connection,
    project_root: Path,
    view_id: str,
    layer: str,
    mask: np.ndarray,
    method: str,
    method_version: str,
    params: dict,
    *,
    enabled: bool = True,
) -> float:
    save_mask(mask, layer_path(project_root, view_id, layer))
    coverage = float((mask > 127).mean())
    conn.execute(
        """
        INSERT OR REPLACE INTO mask_layers
            (view_id, layer, method, method_version, params, enabled, edited, coverage, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
        """,
        (view_id, layer, method, method_version, json.dumps(params), int(enabled), coverage, _now()),
    )
    return coverage


def compose_view(conn: sqlite3.Connection, project_root: Path, view_id: str, *, commit: bool = True) -> float | None:
    """ORs the view's enabled layers into the keep/exclude composite and
    upserts its `masks` row. With no layer rows left at all, the composite
    and row are removed instead (the view is simply unmasked). Returns the
    keep fraction, or None if the view has no layers."""
    project_root = Path(project_root)
    image_relative_path = _view_image_path(conn, view_id)
    keep_path = keep_mask_path(project_root, image_relative_path)
    exclude_path = project_root / "masks" / "classes" / view_id / "exclude.png"
    rows = conn.execute(
        "SELECT layer, method, params, enabled, edited FROM mask_layers WHERE view_id = ? ORDER BY layer", (view_id,)
    ).fetchall()
    if not rows:
        conn.execute("DELETE FROM masks WHERE view_id = ?", (view_id,))
        for path in (keep_path, exclude_path):
            path.unlink(missing_ok=True)
        if commit:
            conn.commit()
        return None

    with Image.open(project_root / image_relative_path) as img:
        width, height = img.size
    exclude = np.zeros((height, width), dtype=bool)
    enabled_layers: list[str] = []
    for layer, _method, _params, enabled, _edited in rows:
        if not enabled:
            continue
        path = layer_path(project_root, view_id, layer)
        if not path.exists():
            continue  # deleted out from under us; treat as empty
        layer_mask = load_mask(path)
        if layer_mask.shape != exclude.shape:
            raise MaskLayerError(f"layer {path} is {layer_mask.shape}, view is {exclude.shape}")
        exclude |= layer_mask > 127
        enabled_layers.append(layer)

    exclude_u8 = np.where(exclude, MASK_TRUE, MASK_FALSE).astype(np.uint8)
    keep = keep_mask_from_exclude(exclude_u8)
    save_mask(exclude_u8, exclude_path)
    save_mask(keep, keep_path)
    kf = keep_fraction(keep)
    methods = sorted({m for _l, m, _p, en, _e in rows if en})
    conn.execute(
        """
        INSERT OR REPLACE INTO masks
            (view_id, model, model_version, prompts, thresholds, morphology, keep_fraction, edited,
             flagged_for_review, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """,
        (
            view_id,
            "layers:" + ",".join(methods) if methods else "layers",
            "1",
            json.dumps(enabled_layers),
            json.dumps({layer: json.loads(params) for layer, _m, params, en, _e in rows if en}),
            json.dumps({}),
            kf,
            int(any(edited for *_rest, edited in rows)),
            _now(),
        ),
    )
    if commit:
        conn.commit()
    return kf


def frame_set_view_ids(conn: sqlite3.Connection, frame_set_id: str) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT v.view_id FROM views v JOIN frames f ON v.frame_id = f.frame_id "
            "WHERE f.frame_set_id = ? ORDER BY f.source_time, v.view_id",
            (frame_set_id,),
        )
    ]


def build_layers_for_views(
    conn: sqlite3.Connection,
    project_root: Path,
    view_ids: list[str],
    layers: dict[str, dict | None],
    *,
    overwrite_edited: bool = False,
    sam3_adapter=None,
    progress_callback=None,
) -> LayerBuildSummary:
    """(Re)builds the given layers -- {layer: params or None for defaults} --
    for each view, then recomposes that view. Layers not named are left
    exactly as they are. A layer hand-edited on disk is skipped unless
    overwrite_edited. Each image is loaded once for all requested layers;
    SAM 3 (if any layer needs it) is created once for the batch."""
    project_root = Path(project_root)
    notify = progress_callback or (lambda *a: None)
    resolved = {layer: resolve_params(layer, params) for layer, params in layers.items()}
    if not view_ids:
        raise MaskLayerError("no views to mask; generate projections first")
    if sam3_adapter is None and any(p.get("method") == "sam3" for p in resolved.values()):
        from vine360.masking.sam3_adapter import Sam3Adapter

        sam3_adapter = Sam3Adapter()
    device = getattr(sam3_adapter, "device", None)
    names = ", ".join(LAYER_SPECS[layer].label.lower() for layer in resolved)
    where = f", SAM 3 on {device.upper()}" if device else ""

    refresh_edited_from_disk(conn, project_root, view_ids)
    built = {layer: 0 for layer in resolved}
    skipped = {layer: 0 for layer in resolved}
    nonempty = {layer: 0 for layer in resolved}
    total = len(view_ids)
    for index, view_id in enumerate(view_ids):
        notify(f"Masking ({names}{where}): view {index + 1}/{total}", index, total)
        image = None
        # Every layer is computed before any is written: a write opens a
        # transaction that holds the database lock, and holding it across a
        # second layer's SAM 3 inference can time out another connection
        # (docs/adr/0044).
        computed = []
        for layer, params in resolved.items():
            row = _layer_row(conn, view_id, layer)
            if row is not None and row[1] and not overwrite_edited:
                skipped[layer] += 1
                continue
            if image is None:
                with Image.open(project_root / _view_image_path(conn, view_id)) as img:
                    image = np.asarray(img.convert("RGB"))
            mask = compute_layer(image, layer, params, sam3_adapter)
            if params.get("method") == "sam3":
                version = getattr(sam3_adapter, "model_id", "1")
            elif layer == LAYER_OVEREXPOSED:
                version = OVEREXPOSURE_METHOD_VERSION
            else:
                version = "1"
            enabled = bool(row[0]) if row is not None else True
            computed.append((layer, params, mask, version, enabled))
        for layer, params, mask, version, enabled in computed:
            coverage = _write_layer(
                conn, project_root, view_id, layer, mask, params["method"], str(version), params, enabled=enabled
            )
            built[layer] += 1
            if coverage > 0:
                nonempty[layer] += 1
        compose_view(conn, project_root, view_id, commit=False)
        conn.commit()

    parts = [f"{LAYER_SPECS[layer].label.lower()} {built[layer]} built, present in {nonempty[layer]}" for layer in resolved]
    edited_note = sum(skipped.values())
    notify(
        f"Masked {total} views ({'; '.join(parts)})"
        + (f"; left {edited_note} hand-edited layer(s) alone" if edited_note else "")
        + ".",
        total,
        total,
    )
    return LayerBuildSummary(views=total, built=built, skipped_edited=skipped, nonempty=nonempty)


def build_layers_for_frame_set(
    conn: sqlite3.Connection, project_root: Path, frame_set_id: str, layers: dict[str, dict | None], **kwargs
) -> LayerBuildSummary:
    view_ids = frame_set_view_ids(conn, frame_set_id)
    if not view_ids:
        raise MaskLayerError(f"no views found for frame set {frame_set_id}; generate projections first")
    register_legacy_layers(conn, project_root, frame_set_id)
    return build_layers_for_views(conn, project_root, view_ids, layers, **kwargs)


def set_layer_enabled(
    conn: sqlite3.Connection,
    project_root: Path,
    view_ids: list[str],
    layer: str,
    enabled: bool,
    progress_callback=None,
) -> int:
    """Switches an existing layer on/off for the given views (without
    rebuilding it) and recomposes them. Returns how many views changed.
    Recomposing reads and writes every view's PNGs, so a whole frame set
    takes a while -- progress is reported per view."""
    notify = progress_callback or (lambda *a: None)
    action = "Merging" if enabled else "Unmerging"
    label = LAYER_SPECS[layer].label.lower() if layer in LAYER_SPECS else layer
    total = len(view_ids)
    changed = 0
    for index, view_id in enumerate(view_ids):
        notify(f"{action} {label} layer, recomposing keep-masks (CPU): view {index + 1}/{total}", index, total)
        cursor = conn.execute(
            "UPDATE mask_layers SET enabled = ? WHERE view_id = ? AND layer = ? AND enabled != ?",
            (int(enabled), view_id, layer, int(enabled)),
        )
        if cursor.rowcount:
            changed += 1
            compose_view(conn, project_root, view_id, commit=False)
            conn.commit()  # per view, so an interrupted run leaves each view consistent
    conn.commit()
    notify(
        f"{label.capitalize()} layer {'merged into' if enabled else 'removed from'} the keep-mask of "
        f"{changed}/{total} views.",
        total,
        total,
    )
    return changed


def set_view_excluded(conn: sqlite3.Connection, project_root: Path, view_id: str, excluded: bool) -> float | None:
    """Drops a whole view from SfM/training (an all-excluded keep mask: COLMAP
    finds no features in it and exports give it no supervision), or restores
    it. This replaced the old "flag for review" toggle, which recorded an
    opinion nothing downstream ever acted on."""
    if excluded:
        with Image.open(Path(project_root) / _view_image_path(conn, view_id)) as img:
            width, height = img.size
        mask = np.full((height, width), MASK_TRUE, dtype=np.uint8)
        _write_layer(conn, project_root, view_id, LAYER_EXCLUDED_VIEW, mask, "manual", "1", {"method": "manual"})
    else:
        conn.execute("DELETE FROM mask_layers WHERE view_id = ? AND layer = ?", (view_id, LAYER_EXCLUDED_VIEW))
        layer_path(project_root, view_id, LAYER_EXCLUDED_VIEW).unlink(missing_ok=True)
    return compose_view(conn, project_root, view_id)


def delete_layer(conn: sqlite3.Connection, project_root: Path, view_ids: list[str], layer: str) -> int:
    deleted = 0
    for view_id in view_ids:
        cursor = conn.execute("DELETE FROM mask_layers WHERE view_id = ? AND layer = ?", (view_id, layer))
        layer_path(project_root, view_id, layer).unlink(missing_ok=True)
        if cursor.rowcount:
            deleted += 1
            compose_view(conn, project_root, view_id, commit=False)
    conn.commit()
    return deleted


def view_layers(conn: sqlite3.Connection, view_id: str) -> list[dict]:
    return [
        {"layer": layer, "method": method, "params": json.loads(params), "enabled": bool(enabled),
         "edited": bool(edited), "coverage": coverage}
        for layer, method, params, enabled, edited, coverage in conn.execute(
            "SELECT layer, method, params, enabled, edited, coverage FROM mask_layers WHERE view_id = ? ORDER BY layer",
            (view_id,),
        )
    ]


def refresh_edited_from_disk(conn: sqlite3.Connection, project_root: Path, view_ids: list[str]) -> list[tuple[str, str]]:
    """Picks up layer PNGs changed outside vine360 (e.g. touched up in an
    image editor): marks them edited -- so regeneration leaves them alone
    unless told otherwise -- refreshes their coverage, and recomposes the
    view. Returns the (view_id, layer) pairs found."""
    found: list[tuple[str, str]] = []
    touched_views: set[str] = set()
    for view_id in view_ids:
        for layer, updated_at in conn.execute(
            "SELECT layer, updated_at FROM mask_layers WHERE view_id = ?", (view_id,)
        ).fetchall():
            path = layer_path(project_root, view_id, layer)
            if not path.exists():
                continue
            if path.stat().st_mtime > _parse_time(updated_at) + _EDIT_DETECTION_SLACK_SECONDS:
                coverage = float((load_mask(path) > 127).mean())
                conn.execute(
                    "UPDATE mask_layers SET edited = 1, coverage = ?, updated_at = ? WHERE view_id = ? AND layer = ?",
                    (coverage, _now(), view_id, layer),
                )
                found.append((view_id, layer))
                touched_views.add(view_id)
    for view_id in touched_views:
        compose_view(conn, project_root, view_id, commit=False)
    conn.commit()
    return found


def ensure_composites_current(conn: sqlite3.Connection, project_root: Path, frame_set_id: str | None = None) -> int:
    """Called before SfM/export read masks/keep: picks up on-disk layer edits
    and recomposes any view whose composite is older than its newest layer
    or missing. Returns how many views were recomposed. frame_set_id=None
    checks every layered view in the project."""
    project_root = Path(project_root)
    if frame_set_id is None:
        view_ids = [r[0] for r in conn.execute("SELECT DISTINCT view_id FROM mask_layers")]
    else:
        view_ids = frame_set_view_ids(conn, frame_set_id)
    edited_views = {view_id for view_id, _layer in refresh_edited_from_disk(conn, project_root, view_ids)}
    recomposed = len(edited_views)  # refresh_edited_from_disk already recomposed these
    for view_id in view_ids:
        if view_id in edited_views:
            continue
        newest = conn.execute("SELECT MAX(updated_at) FROM mask_layers WHERE view_id = ?", (view_id,)).fetchone()[0]
        if newest is None:
            continue
        row = conn.execute("SELECT updated_at FROM masks WHERE view_id = ?", (view_id,)).fetchone()
        stale = (
            row is None
            or row[0] is None
            or _parse_time(row[0]) < _parse_time(newest) - _EDIT_DETECTION_SLACK_SECONDS
            or not keep_mask_path(project_root, _view_image_path(conn, view_id)).exists()
        )
        if stale:
            compose_view(conn, project_root, view_id, commit=False)
            recomposed += 1
    conn.commit()
    return recomposed


_LEGACY_LAYER_NAMES = {
    "sky(classical).png": (LAYER_SKY, "classical"),
    "sky(classical-fallback).png": (LAYER_SKY, "classical"),
    "sky.png": (LAYER_SKY, "sam3"),
    "person.png": (LAYER_PERSON, "sam3"),
}


def register_legacy_layers(conn: sqlite3.Connection, project_root: Path, frame_set_id: str) -> int:
    """Masks built before layers existed have a `masks` row and per-class
    PNGs but no mask_layers rows. Registers those PNGs as layers (renamed to
    the layer convention) so they show up and can be toggled/regenerated.
    The existing composite is left untouched until a layer changes. The old
    class PNGs are pre-morphology, so a recompose may differ by a few edge
    pixels -- regenerate the layer for an exact match. Returns views migrated."""
    project_root = Path(project_root)
    migrated = 0
    for (view_id,) in conn.execute(
        "SELECT m.view_id FROM masks m JOIN views v ON v.view_id = m.view_id JOIN frames f ON f.frame_id = v.frame_id "
        "WHERE f.frame_set_id = ? AND NOT EXISTS (SELECT 1 FROM mask_layers l WHERE l.view_id = m.view_id)",
        (frame_set_id,),
    ).fetchall():
        classes_dir = project_root / "masks" / "classes" / view_id
        registered = False
        for filename, (layer, method) in _LEGACY_LAYER_NAMES.items():
            source = classes_dir / filename
            if not source.exists():
                continue
            target = layer_path(project_root, view_id, layer)
            if source != target:
                if target.exists():
                    source.unlink()
                    continue
                source.rename(target)
            coverage = float((load_mask(target) > 127).mean())
            # Dated by the file itself, not now(): registering must not make
            # the composite (or a Pose run built on it) look stale.
            file_time = datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).isoformat()
            conn.execute(
                "INSERT OR IGNORE INTO mask_layers "
                "(view_id, layer, method, method_version, params, enabled, edited, coverage, updated_at) "
                "VALUES (?, ?, ?, 'legacy', ?, 1, 0, ?, ?)",
                (view_id, layer, method, json.dumps({"method": method, "legacy": True}), coverage, file_time),
            )
            registered = True
        if registered:
            migrated += 1
    conn.commit()
    return migrated


def layer_summary(conn: sqlite3.Connection, frame_set_id: str) -> dict[str, dict]:
    """{layer: {"views": n, "enabled": n, "nonempty": n, "edited": n}} across the frame set."""
    summary: dict[str, dict] = {}
    for layer, views, enabled, nonempty, edited in conn.execute(
        "SELECT l.layer, COUNT(*), SUM(l.enabled), SUM(CASE WHEN l.coverage > 0 THEN 1 ELSE 0 END), SUM(l.edited) "
        "FROM mask_layers l JOIN views v ON v.view_id = l.view_id JOIN frames f ON f.frame_id = v.frame_id "
        "WHERE f.frame_set_id = ? GROUP BY l.layer",
        (frame_set_id,),
    ):
        summary[layer] = {"views": views, "enabled": enabled or 0, "nonempty": nonempty or 0, "edited": edited or 0}
    return summary


@dataclass(frozen=True)
class SuspiciousView:
    view_id: str
    frame_id: str
    layer: str
    coverage: float
    neighbour_median: float

    @property
    def reason(self) -> str:
        return (
            f"{LAYER_SPECS.get(self.layer, LAYER_SPECS[LAYER_SKY]).label} covers {self.coverage:.0%} here vs "
            f"~{self.neighbour_median:.0%} in the same view of nearby frames"
        )


def find_suspicious_views(
    conn: sqlite3.Connection, frame_set_id: str, *, window: int = 3, threshold: float = 0.15
) -> list[SuspiciousView]:
    """A layer whose coverage jumps compared with the *same face* in the
    `window` frames either side -- e.g. the sky layer suddenly swallowing a
    wall on one frame, or overexposure flickering on. Replaces the old
    absolute keep-fraction rule (< 30% or > 98% kept), which flagged most
    down/side faces of any capture with nothing masked in them. Ordered by
    frame time, then face."""
    series: dict[tuple[str, str], list[tuple[str, str, float]]] = {}
    for view_id, frame_id, layer, coverage in conn.execute(
        "SELECT l.view_id, v.frame_id, l.layer, l.coverage FROM mask_layers l "
        "JOIN views v ON v.view_id = l.view_id JOIN frames f ON f.frame_id = v.frame_id "
        "WHERE f.frame_set_id = ? AND l.enabled = 1 AND l.layer != ? AND l.coverage IS NOT NULL "
        "ORDER BY f.source_time",
        (frame_set_id, LAYER_EXCLUDED_VIEW),
    ):
        face = view_id.split(":")[-1]
        series.setdefault((face, layer), []).append((view_id, frame_id, coverage))

    order = {vid: i for i, vid in enumerate(frame_set_view_ids(conn, frame_set_id))}
    found: list[SuspiciousView] = []
    for (_face, layer), points in series.items():
        for i, (view_id, frame_id, coverage) in enumerate(points):
            neighbours = [c for j, (_v, _f, c) in enumerate(points) if j != i and abs(j - i) <= window]
            if len(neighbours) < 2:
                continue
            med = median(neighbours)
            if abs(coverage - med) > threshold:
                found.append(SuspiciousView(view_id, frame_id, layer, coverage, med))
    found.sort(key=lambda s: (order.get(s.view_id, 0), s.layer))
    return found


def render_overlay(conn: sqlite3.Connection, project_root: Path, view_id: str, alpha: float = 0.55) -> np.ndarray:
    """The view's image with each enabled layer tinted in its LAYER_SPECS
    colour (later layers over earlier ones), for review. (H, W, 3) uint8."""
    project_root = Path(project_root)
    with Image.open(project_root / _view_image_path(conn, view_id)) as img:
        out = np.asarray(img.convert("RGB")).astype(np.float32)
    for row in view_layers(conn, view_id):
        if not row["enabled"]:
            continue
        path = layer_path(project_root, view_id, row["layer"])
        if not path.exists():
            continue
        selected = load_mask(path) > 127
        if row["layer"] == LAYER_EXCLUDED_VIEW:
            out = out * 0.35 + 40
            continue
        color = np.array(LAYER_SPECS[row["layer"]].color, dtype=np.float32)
        out[selected] = out[selected] * (1 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
