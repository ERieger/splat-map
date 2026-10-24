"""Detects AprilTags on every projected view of one or more frame sets and
stores them in tag_observations (docs/adr/0047). Needs only Projection --
no SfM run -- so it can be queued straight after it.

Re-running replaces each view's rows for that family. tag_detection_runs
records every view a pass covered, including views where nothing was found,
so "detected, no tags" and "not detected yet" stay distinguishable.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from vine360.georef.apriltag_adapter import DETECTOR_NAME, DetectorParams, TagDetector, detector_version, load_gray
from vine360.georef.control_points import DEFAULT_TAG_FAMILY
from vine360.sfm.frame_poses import views_for_frame_set

COMMIT_EVERY = 25  # views between commits: short write locks, little lost on a crash (docs/adr/0044)


class TagDetectionError(Exception):
    pass


@dataclass
class DetectionSummary:
    frame_set_ids: list[str]
    views: int = 0
    views_with_tags: int = 0
    detections: int = 0
    missing_images: int = 0
    views_per_tag: dict[int, int] = field(default_factory=dict)

    def describe(self) -> str:
        tags = ", ".join(f"{t} ({n})" for t, n in sorted(self.views_per_tag.items())) or "none"
        text = (
            f"{self.detections} detection(s) on {self.views_with_tags}/{self.views} view(s); "
            f"tags (views): {tags}"
        )
        if self.missing_images:
            text += f"; {self.missing_images} view image(s) missing on disk"
        return text


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def detect_tags_for_frame_sets(
    conn: sqlite3.Connection,
    project_root: Path,
    frame_set_ids: list[str],
    *,
    tag_family: str = DEFAULT_TAG_FAMILY,
    params: DetectorParams = DetectorParams(),
    progress_callback=None,
) -> DetectionSummary:
    notify = progress_callback or (lambda *a: None)
    project_root = Path(project_root)
    views = views_for_frame_set(conn, list(frame_set_ids))
    if not views:
        raise TagDetectionError(
            f"no projected views for frame set {', '.join(frame_set_ids)} -- generate projections first"
        )
    with TagDetector(tag_family, params) as detector:
        return _detect_views(conn, project_root, views, detector, tag_family, params, frame_set_ids, notify)


def _detect_views(conn, project_root, views, detector, tag_family, params, frame_set_ids, notify) -> DetectionSummary:
    version = detector_version()
    params_json = json.dumps(params.to_dict())
    summary = DetectionSummary(list(frame_set_ids))
    n = len(views)
    notify(f"Detecting {tag_family} tags (CPU, {params.nthreads} threads) on {n} views…", 0, n)
    for index, view in enumerate(views, start=1):
        path = project_root / view.image_path
        conn.execute("DELETE FROM tag_observations WHERE view_id = ? AND tag_family = ?", (view.view_id, tag_family))
        if not path.exists():
            summary.missing_images += 1
            conn.execute("DELETE FROM tag_detection_runs WHERE view_id = ? AND tag_family = ?", (view.view_id, tag_family))
            continue
        detections = detector.detect(load_gray(path))
        now = _utcnow_iso()
        seen = set()
        for d in detections:
            if d.tag_id in seen:
                # The same id twice in one view: a duplicate print or a
                # misdecode -- neither can be trusted as this tag.
                conn.execute(
                    "DELETE FROM tag_observations WHERE view_id = ? AND tag_family = ? AND tag_id = ?",
                    (view.view_id, tag_family, d.tag_id),
                )
                continue
            seen.add(d.tag_id)
            conn.execute(
                "INSERT INTO tag_observations (view_id, tag_family, tag_id, corners, center, decision_margin, hamming, "
                "detector, detector_version, params, detected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    view.view_id, tag_family, d.tag_id, json.dumps(d.corners.tolist()), json.dumps(d.center.tolist()),
                    d.decision_margin, d.hamming, DETECTOR_NAME, version, params_json, now,
                ),
            )
        conn.execute(
            "INSERT OR REPLACE INTO tag_detection_runs (view_id, tag_family, n_tags, detected_at) VALUES (?, ?, ?, ?)",
            (view.view_id, tag_family, len(seen), now),
        )
        summary.views += 1
        if seen:
            summary.views_with_tags += 1
            summary.detections += len(seen)
            for tag_id in seen:
                summary.views_per_tag[tag_id] = summary.views_per_tag.get(tag_id, 0) + 1
        if index % COMMIT_EVERY == 0:
            conn.commit()
        found = f", tags {sorted(seen)}" if seen else ""
        notify(f"Tag detection (CPU): view {index}/{n}{found}", index, n)
    conn.commit()
    notify(f"Tag detection done: {summary.describe()}", n, n)
    return summary


def detection_status(conn: sqlite3.Connection, frame_set_id: str, tag_family: str = DEFAULT_TAG_FAMILY) -> tuple[int, int]:
    """(views detected, views total) for one frame set."""
    total = conn.execute(
        "SELECT COUNT(*) FROM views v JOIN frames f ON f.frame_id = v.frame_id WHERE f.frame_set_id = ?",
        (frame_set_id,),
    ).fetchone()[0]
    done = conn.execute(
        "SELECT COUNT(*) FROM tag_detection_runs r JOIN views v ON v.view_id = r.view_id "
        "JOIN frames f ON f.frame_id = v.frame_id WHERE f.frame_set_id = ? AND r.tag_family = ?",
        (frame_set_id, tag_family),
    ).fetchone()[0]
    return done, total


def tag_counts_by_frame_set(conn: sqlite3.Connection, tag_family: str = DEFAULT_TAG_FAMILY) -> dict[str, dict]:
    """{frame_set_id: {"views_detected", "detections", "tags"}} -- for the
    Data manager and the Georeference panel."""
    out: dict[str, dict] = {}
    for fs_id, views_detected in conn.execute(
        "SELECT f.frame_set_id, COUNT(*) FROM tag_detection_runs r JOIN views v ON v.view_id = r.view_id "
        "JOIN frames f ON f.frame_id = v.frame_id WHERE r.tag_family = ? GROUP BY f.frame_set_id",
        (tag_family,),
    ):
        out[fs_id] = {"views_detected": views_detected, "detections": 0, "tags": []}
    for fs_id, n, tags in conn.execute(
        "SELECT f.frame_set_id, COUNT(*), GROUP_CONCAT(DISTINCT o.tag_id) FROM tag_observations o "
        "JOIN views v ON v.view_id = o.view_id JOIN frames f ON f.frame_id = v.frame_id "
        "WHERE o.tag_family = ? GROUP BY f.frame_set_id",
        (tag_family,),
    ):
        entry = out.setdefault(fs_id, {"views_detected": 0, "detections": 0, "tags": []})
        entry["detections"] = n
        entry["tags"] = sorted(int(t) for t in (tags or "").split(",") if t)
    return out


def delete_tag_observations(conn: sqlite3.Connection, frame_set_id: str) -> int:
    """The Data manager's "delete tag detections" for one frame set."""
    sub = "SELECT v.view_id FROM views v JOIN frames f ON f.frame_id = v.frame_id WHERE f.frame_set_id = ?"
    cursor = conn.execute(f"DELETE FROM tag_observations WHERE view_id IN ({sub})", (frame_set_id,))
    conn.execute(f"DELETE FROM tag_detection_runs WHERE view_id IN ({sub})", (frame_set_id,))
    conn.commit()
    return cursor.rowcount
