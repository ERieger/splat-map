"""One-time repair for `sfm_runs` rows recorded before two related bugs
were fixed (see docs/adr/0018, docs/adr/0019):

1. `selected_model` used to store the run's own `run_id` instead of its
   real model directory.
2. Every run wrote into the *same* `project/sfm/sparse/` directory, so a
   later run's `pycolmap.incremental_mapping` (which numbers its own
   candidate reconstructions starting from 0 every time) could silently
   overwrite an earlier run's files on disk.

Because of (2), this can only ever recover a run whose files happen not
to have been overwritten by a later run -- in practice, usually only the
*most recent* run of an affected project. It works by loading every
`sfm/sparse/<key>/` directory still on disk and matching it against a
run's recorded `model_stats` (registered image count, point count, mean
reprojection error) -- only an exact match on all three is accepted, so
this never guesses. A run with no matching directory, or more than one
equally exact match, is reported as unrecoverable rather than fixed
incorrectly.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pycolmap


@dataclass
class RepairResult:
    fixed: dict[str, str] = field(default_factory=dict)  # run_id -> new selected_model
    unrecoverable: dict[str, str] = field(default_factory=dict)  # run_id -> reason
    already_fine: list[str] = field(default_factory=list)


def _candidate_model_dirs(project_root: Path) -> list[Path]:
    sparse_root = project_root / "sfm" / "sparse"
    if not sparse_root.exists():
        return []
    # Both the old shared layout (sfm/sparse/<key>/) and the new
    # per-run layout (sfm/sparse/<run_id>/<key>/) are searched, since a
    # project may have runs recorded under either.
    dirs = []
    for path in sparse_root.rglob("cameras.bin"):
        if (path.parent / "images.bin").exists() and (path.parent / "points3D.bin").exists():
            dirs.append(path.parent)
    return dirs


def repair_selected_model(conn: sqlite3.Connection, project_root: Path) -> RepairResult:
    project_root = Path(project_root)
    result = RepairResult()
    candidates = _candidate_model_dirs(project_root)

    rows = conn.execute("SELECT run_id, selected_model, model_stats FROM sfm_runs").fetchall()
    for run_id, selected_model, model_stats_json in rows:
        if selected_model and (project_root / selected_model).exists():
            result.already_fine.append(run_id)
            continue

        stats = json.loads(model_stats_json)
        matches = []
        for model_dir in candidates:
            try:
                reconstruction = pycolmap.Reconstruction(model_dir)
            except Exception:
                continue
            if (
                reconstruction.num_reg_images() == stats["registered_images"]
                and reconstruction.num_points3D() == stats["num_points3d"]
                and abs(reconstruction.compute_mean_reprojection_error() - stats["mean_reprojection_error"]) < 1e-9
            ):
                matches.append(model_dir)

        if len(matches) == 1:
            new_selected_model = matches[0].relative_to(project_root).as_posix()
            conn.execute(
                "UPDATE sfm_runs SET selected_model = ? WHERE run_id = ?", (new_selected_model, run_id)
            )
            result.fixed[run_id] = new_selected_model
        elif len(matches) == 0:
            result.unrecoverable[run_id] = (
                "no on-disk model directory matches this run's recorded diagnostics -- its files were "
                "likely overwritten by a later run sharing the same sfm/sparse/ directory (fixed for "
                "future runs, see docs/adr/0019); re-run Pose estimation to get an exportable model"
            )
        else:
            result.unrecoverable[run_id] = (
                f"{len(matches)} on-disk model directories match this run's recorded diagnostics "
                "equally well -- can't tell them apart without guessing"
            )

    conn.commit()
    return result
