"""Builds the ingest manifest (handover doc, M1 deliverable) from the index
database: every source, every frame, and the source-to-frame map, plus the
dependency versions used to produce them -- laying groundwork for the A06
provenance acceptance test without trying to satisfy the full provenance
requirement (which also needs SfM/training run IDs from later milestones).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from vine360 import __version__
from vine360.runners.probe import probe_dependencies


def build_manifest(conn: sqlite3.Connection) -> dict:
    sources = [
        dict(
            zip(
                (
                    "source_id",
                    "path",
                    "checksum",
                    "media_type",
                    "projection",
                    "width",
                    "height",
                    "timestamps",
                    "capture_group",
                ),
                row,
            )
        )
        for row in conn.execute(
            "SELECT source_id, path, checksum, media_type, projection, width, height, timestamps, capture_group "
            "FROM sources ORDER BY source_id"
        )
    ]
    for s in sources:
        s["timestamps"] = json.loads(s["timestamps"])

    frames = [
        dict(
            zip(
                ("frame_id", "source_id", "source_time", "extraction_settings", "path", "checksum"),
                row,
            )
        )
        for row in conn.execute(
            "SELECT frame_id, source_id, source_time, extraction_settings, path, checksum "
            "FROM frames ORDER BY source_id, source_time"
        )
    ]
    for f in frames:
        f["extraction_settings"] = json.loads(f["extraction_settings"])

    source_frame_map: dict[str, list[str]] = {}
    for f in frames:
        source_frame_map.setdefault(f["source_id"], []).append(f["frame_id"])

    return {
        "vine360_version": __version__,
        "dependencies": [d.__dict__ for d in probe_dependencies()],
        "sources": sources,
        "frames": frames,
        "source_frame_map": source_frame_map,
    }


def write_manifest(conn: sqlite3.Connection, project_root: Path) -> Path:
    manifest = build_manifest(conn)
    path = Path(project_root) / "exports" / "ingest_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path
