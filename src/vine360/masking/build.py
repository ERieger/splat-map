"""Mask stage entry points (handover doc, section 4, step 4 "Mask"),
now built from layers (docs/adr/0038, vine360.masking.layers): each class
-- sky, person, overexposure -- is its own layer under
`masks/classes/<view_id>/<layer>.png`, and the keep-mask under
`masks/keep/<relative-image-path>.png` is their composite, matching
`vine360.masking.semantics.colmap_mask_path`'s convention exactly --
`masks/keep/` doubles as the COLMAP `mask_path` root once SfM runs against
`project/projections/` as its image directory (see vine360.sfm.project_run).

This module keeps the pre-layer call shape (use_sam3_person/use_sam3_sky,
one MaskBuildConfig for sky/person morphology) for the GUI's "Build
Masks", the queue and existing callers.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from vine360.masking.layers import (
    LAYER_OVEREXPOSED,
    LAYER_PERSON,
    LAYER_SKY,
    LayerBuildSummary,
    MaskLayerError,
    build_layers_for_frame_set,
    build_layers_for_views,
)
from vine360.masking.sam3_adapter import Sam3Adapter
from vine360.masking.semantics import MaskBuildConfig
from vine360.models import Mask


class MaskBuildError(MaskLayerError):
    pass


def requested_layers(
    *,
    use_sam3_person: bool = False,
    use_sam3_sky: bool = False,
    mask_sky: bool = True,
    mask_overexposure: bool = False,
    overexposure_params: dict | None = None,
    mask_config: MaskBuildConfig = MaskBuildConfig(),
) -> dict[str, dict]:
    """The {layer: params} a "Build Masks" run with these options builds.
    Sky is classical unless SAM 3 sky is asked for; person has no
    classical fallback (see vine360.masking.classical_sky)."""
    morphology = {"min_component_px": mask_config.min_component_px, "dilation_px": mask_config.dilation_px}
    layers: dict[str, dict] = {}
    if mask_sky or use_sam3_sky:
        layers[LAYER_SKY] = {"method": "sam3" if use_sam3_sky else "classical", **morphology}
    if use_sam3_person:
        layers[LAYER_PERSON] = {"method": "sam3", **morphology}
    if mask_overexposure:
        layers[LAYER_OVEREXPOSED] = dict(overexposure_params or {})
    return layers


def _mask_from_row(conn: sqlite3.Connection, view_id: str) -> Mask:
    row = conn.execute(
        "SELECT model, model_version, prompts, thresholds, morphology, keep_fraction, edited FROM masks "
        "WHERE view_id = ?",
        (view_id,),
    ).fetchone()
    if row is None:
        raise MaskBuildError(f"no mask built for view {view_id}")
    model, version, prompts, thresholds, morphology, kf, edited = row
    return Mask(
        view_id=view_id,
        model=model,
        model_version=version,
        prompts=json.loads(prompts),
        thresholds=json.loads(thresholds),
        morphology=json.loads(morphology),
        keep_fraction=kf,
        edited=bool(edited),
    )


def build_mask_for_view(
    conn: sqlite3.Connection,
    project_root: Path,
    view_id: str,
    *,
    use_sam3_person: bool = False,
    use_sam3_sky: bool = False,
    mask_sky: bool = True,
    mask_overexposure: bool = False,
    overexposure_params: dict | None = None,
    sam3_adapter: Sam3Adapter | None = None,
    mask_config: MaskBuildConfig = MaskBuildConfig(),
) -> Mask:
    if conn.execute("SELECT 1 FROM views WHERE view_id = ?", (view_id,)).fetchone() is None:
        raise MaskBuildError(f"unknown view_id: {view_id}")
    layers = requested_layers(
        use_sam3_person=use_sam3_person,
        use_sam3_sky=use_sam3_sky,
        mask_sky=mask_sky,
        mask_overexposure=mask_overexposure,
        overexposure_params=overexposure_params,
        mask_config=mask_config,
    )
    if not layers:
        raise MaskBuildError("no mask layers selected")
    build_layers_for_views(conn, project_root, [view_id], layers, sam3_adapter=sam3_adapter)
    return _mask_from_row(conn, view_id)


def build_masks_for_frame_set(
    conn: sqlite3.Connection,
    project_root: Path,
    frame_set_id: str,
    *,
    use_sam3_person: bool = False,
    use_sam3_sky: bool = False,
    mask_sky: bool = True,
    mask_overexposure: bool = False,
    overexposure_params: dict | None = None,
    mask_config: MaskBuildConfig = MaskBuildConfig(),
    progress_callback=None,
) -> LayerBuildSummary:
    """progress_callback(message, current, total). Builds every requested
    layer for every view of the frame set in one pass (each image loaded
    once; SAM 3, if requested, loaded once for the whole batch). Layers not
    requested are left as they are -- use vine360.masking.layers to delete
    or disable one."""
    layers = requested_layers(
        use_sam3_person=use_sam3_person,
        use_sam3_sky=use_sam3_sky,
        mask_sky=mask_sky,
        mask_overexposure=mask_overexposure,
        overexposure_params=overexposure_params,
        mask_config=mask_config,
    )
    if not layers:
        raise MaskBuildError("no mask layers selected")
    try:
        return build_layers_for_frame_set(
            conn, project_root, frame_set_id, layers, progress_callback=progress_callback
        )
    except MaskLayerError as exc:
        if isinstance(exc, MaskBuildError):
            raise
        raise MaskBuildError(str(exc)) from exc
