"""Each pipeline stage's done / active / pending status, derived from the
project's index (docs/adr/0017, 0047) -- the sidebar's status dots, and the
MCP server's project status (docs/adr/0048). Moved out of
vine360.gui.main_window unchanged; no Qt.
"""

from __future__ import annotations

import sqlite3

DONE, ACTIVE, PENDING = "done", "active", "pending"

# The GUI sidebar's stage indices (vine360.gui.main_window adds the two
# non-stage tabs after these).
STAGE_PROJECT = 0
STAGE_IMPORT = 1
STAGE_FRAMES = 2
STAGE_PROJECTION = 3
STAGE_MASKS = 4
STAGE_POSE = 5
STAGE_GEOREF = 6  # docs/adr/0047
STAGE_EXPORT = 7

STAGE_NAMES = {
    STAGE_PROJECT: "project",
    STAGE_IMPORT: "import",
    STAGE_FRAMES: "frames",
    STAGE_PROJECTION: "projection",
    STAGE_MASKS: "masks",
    STAGE_POSE: "pose",
    STAGE_GEOREF: "georeference",
    STAGE_EXPORT: "export",
}


def compute_stage_statuses(conn: sqlite3.Connection | None) -> dict[int, str]:
    """Derives every stage's status directly from the database -- the
    single source of truth, rather than tracking status ad hoc as panels
    act. A stage is DONE once it has produced at least one real artifact,
    ACTIVE once its prerequisite exists but it hasn't yet (or its output
    is stale relative to upstream changes -- see below), else PENDING.

    Frames/Projection/Masks fully invalidate (hard-delete) their old
    output when an earlier stage re-runs (clear_frames_for_source and
    clear_views_for_frame cascade all the way down), so their status
    already falls back to ACTIVE/PENDING correctly from the counts alone.
    Pose estimation (SfM) is different on purpose: its runs are kept as
    provenance history (docs/status.md's data contract), not overwritten,
    so an old sfm_runs row can still exist after its inputs changed. That
    case is detected here by comparing timestamps, and reported as ACTIVE
    ("stale, should re-run") rather than DONE."""
    if conn is None:
        return {
            STAGE_IMPORT: PENDING,
            STAGE_FRAMES: PENDING,
            STAGE_PROJECTION: PENDING,
            STAGE_MASKS: PENDING,
            STAGE_POSE: PENDING,
            STAGE_GEOREF: PENDING,
        }
    n_sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    n_video_sources = conn.execute("SELECT COUNT(*) FROM sources WHERE media_type = 'video'").fetchone()[0]
    n_frames = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    n_views = conn.execute("SELECT COUNT(*) FROM views").fetchone()[0]
    n_masks = conn.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
    n_sfm_runs = conn.execute("SELECT COUNT(*) FROM sfm_runs").fetchone()[0]

    if n_sfm_runs > 0:
        latest_run_at = conn.execute("SELECT MAX(created_at) FROM sfm_runs").fetchone()[0]
        latest_input_at = conn.execute(
            "SELECT MAX(t) FROM (SELECT MAX(updated_at) AS t FROM views UNION ALL SELECT MAX(updated_at) FROM masks)"
        ).fetchone()[0]
        pose_status = ACTIVE if (latest_input_at and latest_input_at > latest_run_at) else DONE
    else:
        pose_status = ACTIVE if n_masks > 0 else PENDING

    # Georeference (docs/adr/0047): done once the newest run has a solution;
    # active once there's something to work with (a control-point set or
    # tag detections); otherwise pending.
    n_sets = conn.execute("SELECT COUNT(*) FROM control_point_sets").fetchone()[0]
    n_detected = conn.execute("SELECT COUNT(*) FROM tag_detection_runs").fetchone()[0]
    newest_run_solved = conn.execute(
        "SELECT COUNT(*) FROM georef_solutions WHERE run_id = "
        "(SELECT run_id FROM sfm_runs ORDER BY created_at DESC, rowid DESC LIMIT 1)"
    ).fetchone()[0]
    georef_status = DONE if newest_run_solved else (ACTIVE if (n_sets or n_detected) else PENDING)

    return {
        STAGE_IMPORT: DONE if n_sources > 0 else ACTIVE,
        STAGE_FRAMES: DONE if n_frames > 0 else (ACTIVE if n_video_sources > 0 else PENDING),
        STAGE_PROJECTION: DONE if n_views > 0 else (ACTIVE if n_frames > 0 else PENDING),
        STAGE_MASKS: DONE if n_masks > 0 else (ACTIVE if n_views > 0 else PENDING),
        STAGE_POSE: pose_status,
        STAGE_GEOREF: georef_status,
    }
