"""Every long-running pipeline operation, as one plain function per job
(docs/adr/0037, 0048): the worker functions the GUI's panels run off the
GUI thread, the queue chains (vine360.gui.queue_manager), and the MCP
server starts (vine360.mcp_server).

Each worker takes the project root first and opens its own sqlite
connection -- it runs on a background thread, and sqlite connections
aren't valid across threads -- and is wrapped in logged_operation, so a run
is recorded in the project's Activity log (order, time, parameters,
outcome) whichever of the three started it. Their parameters are what the
log records, so signatures stay explicit and JSON-friendly.

No Qt here: this module is what lets something other than the desktop app
drive the pipeline. It moved out of vine360.gui.main_window unchanged,
which still imports each worker under its old private name.
"""

from __future__ import annotations

import os
from pathlib import Path

from vine360.activity_log import logged_operation
from vine360.data_manager import (
    delete_export_dir,
    delete_frame_set,
    delete_masks_for_frame_set,
    delete_sfm_run,
    delete_views_for_frame_set,
)
from vine360.export.postshot import (
    export_for_postshot,
    export_for_realityscan,
    export_frames_and_masks_for_postshot,
)
from vine360.ingest.frames import extract_frames
from vine360.ingest.sources import add_source, remove_source
from vine360.masking.build import build_masks_for_frame_set
from vine360.masking.layers import (
    LAYER_OVEREXPOSED,
    LAYER_SPECS,
    build_layers_for_frame_set,
    frame_set_view_ids,
    set_layer_enabled,
)
from vine360.project import open_index_db
from vine360.projection.cubemap import DEFAULT_RING_TILT_DEGREES
from vine360.projection.generate import generate_views_for_frame_set
from vine360.runners.local import LocalRunner
from vine360.sfm.options import SfmConfig
from vine360.sfm.project_run import run_sfm_for_project


def _summarize_source(source, _conn) -> str:
    return f"{Path(source.path).name}: {source.media_type.value}, {source.projection.value} ({source.source_id[:8]})"


def _summarize_frames(frames, _conn) -> str:
    frame_set_id = frames[0].frame_set_id if frames else None
    return f"{len(frames)} frames into frame set {frame_set_id}"


def _summarize_count(noun: str):
    return lambda items, _conn: f"{len(items)} {noun}"


def _summarize_sfm(result, conn) -> str:
    diagnostics, warnings = result
    run_id = conn.execute("SELECT run_id FROM sfm_runs ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
    text = (
        f"run {run_id[0] if run_id else '?'}: {diagnostics.registered_images}/{diagnostics.total_images} registered, "
        f"{diagnostics.num_points3d} points, mean reprojection error {diagnostics.mean_reprojection_error:.2f}px"
    )
    return text + (f"; warnings: {'; '.join(warnings)}" if warnings else "")


def _summarize_export(result, _conn) -> str:
    text = f"{result.num_images} images, {result.num_masks} masks -> {result.output_dir}"
    return text + (f"; warnings: {'; '.join(result.warnings)}" if result.warnings else "")


@logged_operation("source.add", "Add source", target=lambda p: str(p.get("file_path") or p.get("path") or ""), summarize=_summarize_source)
def add_source_worker(
    project_root: Path,
    file_path: Path,
    capture_group: str | None,
    confirm_equirectangular: bool,
    added_at: str,
    progress_callback=None,
):
    """Runs on a background thread -- opens its own sqlite connection
    rather than reusing one created on the GUI thread (sqlite3 connections
    aren't valid across threads)."""
    conn = open_index_db(project_root)
    try:
        return add_source(
            conn,
            file_path,
            LocalRunner(),
            capture_group=capture_group,
            confirm_equirectangular=confirm_equirectangular,
            added_at=added_at,
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


@logged_operation("source.remove", "Remove source", target=lambda p: p.get("source_id"))
def remove_source_worker(project_root: Path, source_id: str) -> str:
    conn = open_index_db(project_root)
    try:
        remove_source(conn, project_root, source_id)
        return source_id
    finally:
        conn.close()


@logged_operation("frames.extract", "Frame extraction", target=lambda p: p.get("frame_set_id") or p.get("source_id"), summarize=_summarize_frames)
def extract_frames_worker(
    project_root: Path,
    source_id: str,
    interval_seconds: float,
    start_time: float | None,
    end_time: float | None,
    generate_thumbnails: bool,
    frame_set_id: str | None = None,
    progress_callback=None,
):
    conn = open_index_db(project_root)
    try:
        return extract_frames(
            conn,
            project_root,
            source_id,
            LocalRunner(),
            interval_seconds=interval_seconds,
            start_time=start_time,
            end_time=end_time,
            generate_thumbnails=generate_thumbnails,
            frame_set_id=frame_set_id,
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


def default_parallel_worker_count() -> int:
    return min(os.cpu_count() or 1, 4)


@logged_operation("projection.generate", "Projection", target=lambda p: p.get("frame_set_id"), summarize=_summarize_count("views"))
def generate_views_worker(
    project_root: Path,
    frame_set_id: str,
    face_size: int,
    fov_degrees: float,
    face_names: list[str],
    max_workers: int | None,
    ring_tilt_degrees: float = DEFAULT_RING_TILT_DEGREES,
    progress_callback=None,
):
    conn = open_index_db(project_root)
    try:
        return generate_views_for_frame_set(
            conn,
            project_root,
            frame_set_id,
            face_size=face_size,
            fov_degrees=fov_degrees,
            face_names=face_names,
            ring_tilt_degrees=ring_tilt_degrees,
            max_workers=max_workers,
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


def _summarize_mask_build(summary, _conn) -> str:
    parts = [
        f"{LAYER_SPECS[layer].label.lower()}: {summary.built[layer]} built, present in {summary.nonempty[layer]}"
        + (f", {summary.skipped_edited[layer]} hand-edited kept" if summary.skipped_edited.get(layer) else "")
        for layer in summary.built
    ]
    return f"{summary.views} views; " + "; ".join(parts)


@logged_operation("masks.build", "Masks", target=lambda p: p.get("frame_set_id"), summarize=_summarize_mask_build)
def build_masks_worker(
    project_root: Path,
    frame_set_id: str,
    use_sam3_person: bool,
    use_sam3_sky: bool,
    mask_sky: bool = True,
    mask_overexposure: bool = False,
    overexposure_clip: int = 250,
    overexposure_bloom_radius: int = LAYER_SPECS[LAYER_OVEREXPOSED].default_params["bloom_radius_px"],
    progress_callback=None,
):
    conn = open_index_db(project_root)
    try:
        return build_masks_for_frame_set(
            conn,
            project_root,
            frame_set_id,
            use_sam3_person=use_sam3_person,
            use_sam3_sky=use_sam3_sky,
            mask_sky=mask_sky,
            mask_overexposure=mask_overexposure,
            overexposure_params={"clip_threshold": overexposure_clip, "bloom_radius_px": overexposure_bloom_radius},
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


@logged_operation(
    "masks.layer",
    "Mask layer",
    target=lambda p: f"{p.get('layer')} @ {p.get('frame_set_id')}",
    summarize=_summarize_mask_build,
)
def build_mask_layer_worker(
    project_root: Path, frame_set_id: str, layer: str, params: dict, overwrite_edited: bool = False, progress_callback=None
):
    """Regenerates one layer for a whole frame set, leaving every other
    layer's files untouched (docs/adr/0038)."""
    conn = open_index_db(project_root)
    try:
        return build_layers_for_frame_set(
            conn, project_root, frame_set_id, {layer: params},
            overwrite_edited=overwrite_edited, progress_callback=progress_callback,
        )
    finally:
        conn.close()


def merge_summary(layer: str, enabled: bool, changed: int) -> str:
    label = LAYER_SPECS[layer].label.lower() if layer in LAYER_SPECS else layer
    return f"{label} layer {'merged into' if enabled else 'removed from'} the keep-mask of {changed} view(s)"


@logged_operation(
    "masks.merge",
    "Mask layer merge (whole frame set)",
    target=lambda p: f"{p.get('layer')} {'on' if p.get('enabled') else 'off'} @ {p.get('frame_set_id')}",
    summarize=lambda summary, _conn: summary,
)
def set_layer_merged_worker(
    project_root: Path, frame_set_id: str, layer: str, enabled: bool, progress_callback=None
) -> str:
    """Switches a layer into/out of every keep-mask in a frame set. No layer
    is rebuilt, but every view is recomposed (PNG reads/writes), which is
    far too slow for the GUI thread on a full frame set."""
    conn = open_index_db(project_root)
    try:
        changed = set_layer_enabled(
            conn, project_root, frame_set_view_ids(conn, frame_set_id), layer, enabled,
            progress_callback=progress_callback,
        )
        return merge_summary(layer, enabled, changed)
    finally:
        conn.close()


def _sfm_target(params: dict) -> str:
    return " + ".join(params.get("frame_set_ids") or []) or params.get("frame_set_id") or "all frame sets"


@logged_operation("sfm.run", "Pose estimation", target=_sfm_target, summarize=_summarize_sfm)
def run_sfm_worker(
    project_root: Path,
    image_source: str,
    frame_set_id: str | None,
    camera_model: str,
    engine: str = "pycolmap",
    options: dict | None = None,
    frame_set_ids: list[str] | None = None,
    progress_callback=None,
):
    """options: the non-default SfmConfig fields (docs/adr/0040) -- absent
    for jobs queued before options existed, which then run as before.
    frame_set_ids: several frame sets in one run (docs/adr/0046), with
    frame_set_id None."""
    config = SfmConfig.from_dict(options).replace(camera_model=camera_model)
    conn = open_index_db(project_root)
    try:
        return run_sfm_for_project(
            conn,
            project_root,
            config=config,
            image_source=image_source,
            frame_set_id=frame_set_id,
            frame_set_ids=frame_set_ids,
            engine=engine,
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


def _summarize_tag_detection(summary, _conn) -> str:
    return summary.describe()


def _summarize_solution(solution_id, conn) -> str:
    from vine360.georef.solve import load_solution

    report = load_solution(conn, solution_id)["report"]
    text = (
        f"{solution_id}: {report['n_control']} control / {report['n_check']} check point(s), "
        f"RMSE 3D {report['rmse']['d3']:.3f} m"
    )
    if report.get("check_rmse"):
        text += f", check RMSE 3D {report['check_rmse']['d3']:.3f} m"
    return text


@logged_operation("georef.import_gcps", "Georeference: import control points", target=lambda p: str(p.get("path")))
def import_control_points_worker(
    project_root: Path,
    path: Path,
    epsg: int,
    height_datum: str,
    tag_family: str,
    tag_size_m: float | None,
    tag_size_table: str | None = None,
):
    from vine360.georef.control_points import import_control_points
    from vine360.georef.tag_sizes import read_tag_size_table

    tag_sizes = read_tag_size_table(Path(tag_size_table), tag_family) if tag_size_table else None
    conn = open_index_db(project_root)
    try:
        return import_control_points(
            conn, Path(path), epsg=epsg, height_datum=height_datum, tag_family=tag_family, tag_size_m=tag_size_m,
            tag_sizes=tag_sizes,
        )
    finally:
        conn.close()


@logged_operation(
    "georef.detect_tags", "Georeference: detect tags", target=lambda p: " + ".join(p.get("frame_set_ids") or []),
    summarize=_summarize_tag_detection,
)
def detect_tags_worker(
    project_root: Path,
    frame_set_ids: list[str],
    tag_family: str = "tag36h11",
    detector_params: dict | None = None,
    progress_callback=None,
):
    from vine360.georef.apriltag_adapter import DetectorParams
    from vine360.georef.observe import detect_tags_for_frame_sets

    conn = open_index_db(project_root)
    try:
        return detect_tags_for_frame_sets(
            conn, project_root, list(frame_set_ids), tag_family=tag_family,
            params=DetectorParams.from_dict(detector_params), progress_callback=progress_callback,
        )
    finally:
        conn.close()


@logged_operation(
    "georef.solve", "Georeference: solve", target=lambda p: f"{p.get('run_id')} ← {p.get('set_id')}",
    summarize=_summarize_solution,
)
def solve_georef_worker(
    project_root: Path, run_id: str, set_id: str, solve_params: dict | None = None, progress_callback=None
) -> str:
    from vine360.georef.solve import SolveParams, solve_georef

    conn = open_index_db(project_root)
    try:
        return solve_georef(
            conn, project_root, run_id, set_id, params=SolveParams(**(solve_params or {})),
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


@logged_operation(
    "georef.splat", "Georeference a trained splat", target=lambda p: str(p.get("out_path")),
    summarize=lambda r, _conn: f"{r.count} Gaussians -> {r.out_path.name} (frame check: {r.check.status})",
)
def georef_splat_worker(
    project_root: Path, in_path: Path, out_path: Path, solution_id: str, force: bool = False, progress_callback=None
):
    from vine360.georef.splat import georeference_splat

    conn = open_index_db(project_root)
    try:
        return georeference_splat(conn, project_root, Path(in_path), Path(out_path), solution_id, force=force,
                                  progress_callback=progress_callback)
    finally:
        conn.close()


def unused_path(path: Path) -> Path:
    """path, or path with -2, -3, ... before the suffix if it exists -- a
    default output never replaces an earlier one."""
    path = Path(path)
    candidate, n = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        n += 1
    return candidate


@logged_operation("export.postshot_poses", "Export: poses + images + masks", target=lambda p: str(p.get("output_dir")), summarize=_summarize_export)
def export_postshot_worker(
    project_root: Path,
    output_dir: Path,
    run_id: str | None,
    georef_solution_id: str | None = None,
    progress_callback=None,
):
    """georef_solution_id: export that solution's materialised local-metre
    model instead of the run's own (docs/adr/0047); None = the original."""
    conn = open_index_db(project_root)
    try:
        return export_for_postshot(
            conn, project_root, output_dir, run_id=run_id, georef_solution_id=georef_solution_id,
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


@logged_operation("export.postshot_images", "Export: images + masks", target=lambda p: str(p.get("output_dir")), summarize=_summarize_export)
def export_frames_and_masks_worker(
    project_root: Path, output_dir: Path, frame_set_id: str | None, progress_callback=None
):
    conn = open_index_db(project_root)
    try:
        return export_frames_and_masks_for_postshot(
            conn, project_root, output_dir, frame_set_id=frame_set_id, progress_callback=progress_callback
        )
    finally:
        conn.close()


@logged_operation("export.realityscan", "Export: RealityScan", target=lambda p: str(p.get("output_dir")), summarize=_summarize_export)
def export_realityscan_worker(
    project_root: Path,
    output_dir: Path,
    frame_set_id: str | None,
    include_camera_priors: bool = False,
    georef_solution_id: str | None = None,
    progress_callback=None,
):
    conn = open_index_db(project_root)
    try:
        return export_for_realityscan(
            conn,
            project_root,
            output_dir,
            frame_set_id=frame_set_id,
            include_camera_priors=include_camera_priors,
            georef_solution_id=georef_solution_id,
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


@logged_operation("data.delete", "Data manager: delete", target=lambda p: p.get("target"))
def data_manager_delete_worker(project_root: Path, action: str, target: str) -> str:
    conn = open_index_db(project_root)
    try:
        if action == "frame_set":
            delete_frame_set(conn, project_root, target)
        elif action == "views":
            delete_views_for_frame_set(conn, project_root, target)
        elif action == "masks":
            delete_masks_for_frame_set(conn, project_root, target)
        elif action == "sfm_run":
            delete_sfm_run(conn, project_root, target)
        elif action == "export":
            delete_export_dir(project_root, target)
        elif action == "gcp_set":  # its points and solutions too (docs/adr/0047)
            from vine360.georef.control_points import delete_control_point_set

            delete_control_point_set(conn, project_root, target)
        elif action == "tag_obs":
            from vine360.georef.observe import delete_tag_observations

            delete_tag_observations(conn, target)
        elif action == "georef_solution":  # the row and its sfm/georef/<run>/<solution>/ only
            from vine360.georef.solve import delete_solution

            delete_solution(conn, project_root, target)
        else:
            raise ValueError(f"unknown delete action {action!r}")
        return action
    finally:
        conn.close()

