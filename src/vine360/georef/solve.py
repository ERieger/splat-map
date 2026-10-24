"""Solves one SfM run's georeference against one control-point set
(docs/adr/0047), composing view_poses -> tag observations -> triangulate ->
similarity, and materialises the result.

Non-destructive (ADR 0047 addendum): every solve is a new georef_solutions
row with a new solution_id -- re-solving never updates or deletes one -- and
writes the transformed model to sfm/georef/<run_id>/<solution_id>/
(cameras.bin, images.bin, points3D.bin, georef.json, georef.prj). The run's
own sfm/sparse/<run_id>/ is only ever read. Export copies the materialised
folder; it never recomputes a transform.

The stored transform maps model -> local, local = (E - E0, N - N0, H - H0)
with the set's origin (never raw CRS values -- float32 consumers).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vine360.georef.control_points import (
    USE_CHECK,
    USE_CONTROL,
    USE_IGNORE,
    get_control_point_set,
    load_control_points,
)
from vine360.georef.similarity import (
    MIN_POINTS,
    RECOMMENDED_POINTS,
    Similarity,
    leave_one_out,
    residuals,
    suspect_flags,
    umeyama,
)
from vine360.georef.triangulate import (
    DEFAULT_THRESH_PX,
    MIN_INLIER_VIEWS,
    MIN_RAY_ANGLE_DEGREES,
    TagObservation,
    TagPosition,
    triangulate_tag,
)
from vine360.georef.view_poses import run_config, run_scope_frame_set_ids, view_poses_for_run

CORNER_SCALE_WARN_FRACTION = 0.03
GEOREF_DIR = Path("sfm") / "georef"


class GeorefSolveError(Exception):
    pass


@dataclass(frozen=True)
class SolveParams:
    thresh_px: float = DEFAULT_THRESH_PX
    min_inlier_views: int = MIN_INLIER_VIEWS
    min_ray_angle_deg: float = MIN_RAY_ANGLE_DEGREES

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def triangulate_run_tags(
    conn: sqlite3.Connection,
    project_root: Path,
    run_id: str,
    tag_family: str,
    params: SolveParams = SolveParams(),
    notify=lambda *a: None,
) -> tuple[dict[int, TagPosition], dict]:
    """Every tag seen in the run's registered views, triangulated. Returns
    (positions by tag id, coverage) -- coverage says how many of the run's
    views had tag detection run on them, for the "detect tags first"
    diagnostic."""
    notify("Georeference step 1/4: loading view poses…", None, None)
    poses = view_poses_for_run(conn, project_root, run_id)
    if not poses:
        raise GeorefSolveError(f"sfm run {run_id!r} registered no projected views -- generate projections first")
    view_ids = list(poses)
    detected = set()
    obs_by_tag: dict[int, list[TagObservation]] = {}
    for start in range(0, len(view_ids), 500):
        chunk = view_ids[start : start + 500]
        marks = ",".join("?" * len(chunk))
        detected |= {
            r[0]
            for r in conn.execute(
                f"SELECT view_id FROM tag_detection_runs WHERE tag_family = ? AND view_id IN ({marks})",
                (tag_family, *chunk),
            )
        }
        for view_id, tag_id, center, corners in conn.execute(
            f"SELECT view_id, tag_id, center, corners FROM tag_observations WHERE tag_family = ? AND view_id IN ({marks})",
            (tag_family, *chunk),
        ):
            pose = poses[view_id]
            obs_by_tag.setdefault(int(tag_id), []).append(
                TagObservation(view_id, pose.K, pose.R, pose.t, np.asarray(json.loads(center)), np.asarray(json.loads(corners)))
            )
    coverage = {"registered_views": len(view_ids), "views_detected": len(detected)}
    if not detected:
        raise GeorefSolveError(
            f"tag detection hasn't run on any of sfm run {run_id!r}'s {len(view_ids)} views -- run Detect tags on "
            f"its frame set(s) ({', '.join(run_scope_frame_set_ids(conn, run_id))}) first"
        )
    positions = {}
    tags = sorted(obs_by_tag)
    for index, tag_id in enumerate(tags, start=1):
        positions[tag_id] = triangulate_tag(
            tag_id, obs_by_tag[tag_id], thresh_px=params.thresh_px, min_inlier_views=params.min_inlier_views,
            min_ray_angle_deg=params.min_ray_angle_deg,
        )
        status = "ok" if positions[tag_id].xyz is not None else positions[tag_id].rejected
        notify(f"Georeference step 2/4 (triangulating, CPU): tag {tag_id} ({index}/{len(tags)}): {status}", index, len(tags))
    return positions, coverage


def solve_georef(
    conn: sqlite3.Connection,
    project_root: Path,
    run_id: str,
    set_id: str,
    *,
    params: SolveParams = SolveParams(),
    progress_callback=None,
) -> str:
    """Fits, stores and materialises a new solution; returns its id. Raises
    GeorefSolveError -- naming the tags seen-but-not-surveyed and
    surveyed-but-not-triangulated, the diagnostic the field actually needs --
    when fewer than 3 control points can be used."""
    notify = progress_callback or (lambda *a: None)
    project_root = Path(project_root)
    cp_set = get_control_point_set(conn, set_id)
    points = load_control_points(conn, set_id)
    positions, coverage = triangulate_run_tags(conn, project_root, run_id, cp_set.tag_family, params, notify)

    notify("Georeference step 3/4: fitting the similarity…", None, None)
    tagged = {p.tag_id: p for p in points if p.tag_id is not None and p.use != USE_IGNORE}
    good = {tag_id: pos for tag_id, pos in positions.items() if pos.xyz is not None}
    control = [p for p in points if p.use == USE_CONTROL and p.tag_id in good]
    check = [p for p in points if p.use == USE_CHECK and p.tag_id in good]
    seen_not_in_table = sorted(t for t in positions if t not in tagged)
    not_triangulated = [
        {
            "name": p.name,
            "tag_id": p.tag_id,
            "use": p.use,
            "n_views": positions[p.tag_id].n_views if p.tag_id in positions else 0,
            "reason": positions[p.tag_id].rejected if p.tag_id in positions else "never detected",
        }
        for p in points
        if p.use != USE_IGNORE and p.tag_id is not None and p.tag_id not in good
    ]
    if len(control) < MIN_POINTS:
        lines = [f"only {len(control)} control point(s) could be used -- at least {MIN_POINTS} are needed."]
        if seen_not_in_table:
            lines.append(f"Tags seen by the detector but not in the control table: {seen_not_in_table}.")
        if not_triangulated:
            lines.append(
                "Control-table tags never triangulated: "
                + ", ".join(f"{d['name']} (tag {d['tag_id']}, {d['n_views']} view(s): {d['reason']})" for d in not_triangulated)
                + "."
            )
        untagged = [p.name for p in points if p.use != USE_IGNORE and p.tag_id is None]
        if untagged:
            lines.append(f"Points with no tag id: {', '.join(untagged)}.")
        if coverage["views_detected"] < coverage["registered_views"]:
            lines.append(
                f"Tag detection has only run on {coverage['views_detected']}/{coverage['registered_views']} of the "
                "run's views."
            )
        raise GeorefSolveError(" ".join(lines))

    origin = np.asarray(cp_set.origin, dtype=np.float64)

    def local(p) -> np.ndarray:
        return np.array([p.east, p.north, p.height]) - origin

    src = np.array([good[p.tag_id].xyz for p in control])
    dst = np.array([local(p) for p in control])
    similarity = umeyama(src, dst)
    fit = residuals(similarity, src, dst)
    loo = leave_one_out(src, dst)
    flags = suspect_flags(loo)

    check_fit = None
    if check:
        check_fit = residuals(similarity, np.array([good[p.tag_id].xyz for p in check]), np.array([local(p) for p in check]))

    warnings = []
    if len(control) < RECOMMENDED_POINTS:
        warnings.append(f"only {len(control)} control points -- {RECOMMENDED_POINTS} or more are recommended")
    corner_scales = []
    for p in control + check:
        size = p.tag_size_m if p.tag_size_m else cp_set.tag_size_m
        edge = good[p.tag_id].mean_edge_model
        if size and edge:
            corner_scales.append(size / edge)
    corner_median = float(np.median(corner_scales)) if corner_scales else None
    corner_diff = None if corner_median is None else (similarity.scale - corner_median) / corner_median
    if corner_diff is not None and abs(corner_diff) > CORNER_SCALE_WARN_FRACTION:
        warnings.append(
            f"fitted scale differs from the tag-size scale by {corner_diff:+.1%} -- check tag sizes and point heights"
        )
    if any(flags):
        warnings.append("suspect point(s): " + ", ".join(p.name for p, f in zip(control, flags) if f))
    if coverage["views_detected"] < coverage["registered_views"]:
        warnings.append(
            f"tag detection has only run on {coverage['views_detected']}/{coverage['registered_views']} of the run's views"
        )

    rows = []
    for i, p in enumerate(control):
        pos = good[p.tag_id]
        rows.append(_point_row(p, pos, fit.vectors[i], loo[i], bool(flags[i])))
    for i, p in enumerate(check or []):
        rows.append(_point_row(p, good[p.tag_id], check_fit.vectors[i], None, False))

    report = {
        "n_control": len(control),
        "n_check": len(check),
        "ignored": [{"name": p.name, "note": p.note} for p in points if p.use == USE_IGNORE],
        "seen_not_in_table": seen_not_in_table,
        "not_triangulated": not_triangulated,
        "untagged": [p.name for p in points if p.use != USE_IGNORE and p.tag_id is None],
        "coverage": coverage,
        "scale": similarity.scale,
        "corner_scale_median": corner_median,
        "corner_scale_n": len(corner_scales),
        "corner_scale_diff": corner_diff,
        "rmse": {"d3": fit.rmse_3d, "horizontal": fit.rmse_horizontal, "vertical": fit.rmse_vertical},
        "check_rmse": None
        if check_fit is None
        else {"d3": check_fit.rmse_3d, "horizontal": check_fit.rmse_horizontal, "vertical": check_fit.rmse_vertical},
        "points": rows,
        "warnings": warnings,
        "epsg": cp_set.epsg,
        "height_datum": cp_set.height_datum,
        "origin": list(cp_set.origin),
    }

    solution_id = f"geo-{uuid.uuid4().hex[:8]}"
    created_at = _utcnow_iso()
    conn.execute(
        "INSERT INTO georef_solutions (solution_id, run_id, set_id, transform, tag_positions, report, params, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            solution_id, run_id, set_id, json.dumps(similarity.to_dict()),
            json.dumps({str(t): pos.to_dict() for t, pos in positions.items()}), json.dumps(report),
            json.dumps({**params.to_dict(), "tag_family": cp_set.tag_family}), created_at,
        ),
    )
    conn.commit()

    notify("Georeference step 4/4: writing the georeferenced model…", None, None)
    try:
        model_dir = materialise_solution(conn, project_root, solution_id)
    except Exception:
        delete_solution(conn, project_root, solution_id)
        raise
    notify(
        f"Georeference done: {len(control)} control / {len(check)} check point(s), scale {similarity.scale:.5g}, "
        f"RMSE 3D {fit.rmse_3d:.3f} m (H {fit.rmse_horizontal:.3f}, V {fit.rmse_vertical:.3f})"
        + (f", check RMSE 3D {check_fit.rmse_3d:.3f} m" if check_fit else "")
        + f" -> {model_dir}",
        1,
        1,
    )
    return solution_id


def _point_row(p, pos: TagPosition, vector: np.ndarray, loo: float | None, suspect: bool) -> dict:
    return {
        "name": p.name,
        "tag_id": p.tag_id,
        "use": p.use,
        "views": pos.n_inliers,
        "views_seen": pos.n_views,
        "rms_px": pos.rms_px,
        "dE": float(vector[0]),
        "dN": float(vector[1]),
        "dH": float(vector[2]),
        "resid_3d": float(np.linalg.norm(vector)),
        "horizontal": float(np.linalg.norm(vector[:2])),
        "vertical": float(vector[2]),
        "loo": None if loo is None or not np.isfinite(loo) else float(loo),
        "flag": "suspect" if suspect else "",
    }


def materialise_solution(conn: sqlite3.Connection, project_root: Path, solution_id: str) -> str:
    """Writes the run's pinhole model, transformed, to
    sfm/georef/<run_id>/<solution_id>/ and records it as the solution's
    model_dir: for a 360 run, build_view_reconstruction's view model (what
    Export writes for it anyway); for a projections run, the run's own model
    -- loaded into memory and written elsewhere, never back."""
    import pycolmap

    from vine360.georef.apply import georef_metadata, transform_reconstruction, write_georef_sidecars
    from vine360.sfm.frame_poses import build_view_reconstruction, is_frame_run

    project_root = Path(project_root)
    solution = load_solution(conn, solution_id)
    run_id = solution["run_id"]
    if is_frame_run(run_config(conn, run_id)):
        reconstruction, _model = build_view_reconstruction(conn, project_root, run_id)
    else:
        selected = conn.execute("SELECT selected_model FROM sfm_runs WHERE run_id = ?", (run_id,)).fetchone()[0]
        reconstruction = pycolmap.Reconstruction(project_root / selected)
    transform_reconstruction(reconstruction, Similarity.from_dict(solution["transform"]))
    relative = (GEOREF_DIR / run_id / solution_id).as_posix()
    out = project_root / relative
    out.mkdir(parents=True, exist_ok=True)
    reconstruction.write(out)
    write_georef_sidecars(out, georef_metadata(solution, get_control_point_set(conn, solution["set_id"])))
    conn.execute("UPDATE georef_solutions SET model_dir = ? WHERE solution_id = ?", (relative, solution_id))
    conn.commit()
    return relative


def load_solution(conn: sqlite3.Connection, solution_id: str) -> dict:
    row = conn.execute(
        "SELECT solution_id, run_id, set_id, transform, tag_positions, report, params, model_dir, created_at "
        "FROM georef_solutions WHERE solution_id = ?",
        (solution_id,),
    ).fetchone()
    if row is None:
        raise GeorefSolveError(f"unknown georeference solution: {solution_id}")
    keys = ("solution_id", "run_id", "set_id", "transform", "tag_positions", "report", "params", "model_dir", "created_at")
    solution = dict(zip(keys, row))
    for key in ("transform", "tag_positions", "report", "params"):
        solution[key] = json.loads(solution[key])
    return solution


def list_solutions(conn: sqlite3.Connection, run_id: str | None = None) -> list[dict]:
    """Newest first; each with its report (for the RMSE in a label)."""
    query = "SELECT solution_id FROM georef_solutions"
    args: tuple = ()
    if run_id is not None:
        query += " WHERE run_id = ?"
        args = (run_id,)
    ids = [r[0] for r in conn.execute(query + " ORDER BY created_at DESC, rowid DESC", args)]
    return [load_solution(conn, i) for i in ids]


def solution_label(solution: dict) -> str:
    report = solution["report"]
    return (
        f"{solution['created_at']}  RMSE 3D {report['rmse']['d3']:.3f} m  ({report['n_control']} control pts)"
    )


def delete_solution(conn: sqlite3.Connection, project_root: Path, solution_id: str) -> None:
    """Removes the row and sfm/georef/<run_id>/<solution_id>/ only -- never
    the run, its model, or any other solution. Also forgets it as a run's
    active choice in project.yaml."""
    row = conn.execute("SELECT run_id, model_dir FROM georef_solutions WHERE solution_id = ?", (solution_id,)).fetchone()
    if row is None:
        raise GeorefSolveError(f"unknown georeference solution: {solution_id}")
    run_id, model_dir = row
    conn.execute("DELETE FROM georef_solutions WHERE solution_id = ?", (solution_id,))
    conn.commit()
    target = Path(project_root) / (model_dir or (GEOREF_DIR / run_id / solution_id))
    georef_root = (Path(project_root) / GEOREF_DIR).resolve()
    if target.exists() and georef_root in target.resolve().parents:
        shutil.rmtree(target)
    if active_solution(project_root, run_id) == solution_id:
        set_active_solution(project_root, run_id, None)


# -- the active choice per run, in project.yaml (ADR 0047 addendum) -----------


def active_solution(project_root: Path, run_id: str) -> str | None:
    from vine360.project import load_project

    try:
        project = load_project(project_root)
    except Exception:
        return None
    return (project.georef.get("active_solution") or {}).get(run_id)


def set_active_solution(project_root: Path, run_id: str, solution_id: str | None) -> None:
    """None ("None (original SfM frame)") clears the run's entry."""
    from vine360.project import load_project, save_project

    project = load_project(project_root)
    active = dict(project.georef.get("active_solution") or {})
    if solution_id is None:
        active.pop(run_id, None)
    else:
        active[run_id] = solution_id
    georef = dict(project.georef)
    if active:
        georef["active_solution"] = active
    else:
        georef.pop("active_solution", None)
    project.georef = georef
    save_project(project_root, project)


def write_report_csv(solution: dict, path: Path) -> None:
    """The per-point table, like the activity log's CSV export."""
    import csv

    columns = ["name", "tag_id", "use", "views", "views_seen", "rms_px", "dE", "dN", "dH", "resid_3d", "horizontal",
               "vertical", "loo", "flag"]
    with Path(path).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in solution["report"]["points"]:
            writer.writerow(["" if row.get(c) is None else row.get(c) for c in columns])
