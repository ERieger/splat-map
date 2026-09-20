"""Source registration (handover doc, section 2 "Project and ingest").

Sources are recorded by absolute path, checksum and probed metadata rather
than copied into the project (docs/adr/0003). Video is the primary MVP
target (equirectangular exports); still images are accepted too, but frame
extraction only applies to video.

`capture_group` tags which physical rig a source came from (e.g.
"insta360-ground" vs "antigravity-a1-aerial") -- both of this project's
real capture devices are 360-degree platforms (a ground-based Insta360 and
an airborne Antigravity A1), not the 360-plus-conventional-drone pairing
the handover doc's "mixed capture" milestone (P1/M7) originally assumed.
See docs/adr/0010.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

from vine360.config import EQUIRECTANGULAR_ASPECT_RATIO, MediaType, Projection
from vine360.ingest.media_probe import MediaMetadata, probe_media
from vine360.models import Source
from vine360.runners.base import Runner

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".insv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

_CHECKSUM_CHUNK_SIZE = 1024 * 1024


class SourceError(Exception):
    pass


class EquirectangularConfirmationRequired(SourceError):
    """Raised when a 2:1 still image is found and the caller has not stated
    whether it is truly equirectangular (handover doc, section 4: "Require
    the user to confirm ... rather than guessing silently").
    """


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHECKSUM_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def infer_media_type(path: Path) -> MediaType:
    ext = path.suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return MediaType.VIDEO
    if ext in IMAGE_EXTENSIONS:
        return MediaType.PHOTO  # refined to EQUIRECT_STILL below once we have dimensions
    raise SourceError(f"unrecognised media extension: {ext}")


def _resolve_projection(media_type: MediaType, metadata: MediaMetadata) -> tuple[MediaType, Projection]:
    if metadata.is_2to1_aspect and media_type in (MediaType.VIDEO, MediaType.PHOTO):
        resolved_type = MediaType.EQUIRECT_STILL if media_type == MediaType.PHOTO else media_type
        return resolved_type, Projection.EQUIRECTANGULAR
    return media_type, Projection.PERSPECTIVE if metadata.width else Projection.UNKNOWN


def add_source(
    conn: sqlite3.Connection,
    file_path: Path,
    runner: Runner,
    *,
    media_type_override: MediaType | None = None,
    confirm_equirectangular: bool = False,
    capture_group: str | None = None,
    added_at: str,
) -> Source:
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise SourceError(f"source file not found: {file_path}")

    metadata = probe_media(file_path, runner)
    declared_type = media_type_override or infer_media_type(file_path)
    resolved_type, projection = _resolve_projection(declared_type, metadata)

    if (
        resolved_type == MediaType.EQUIRECT_STILL
        and media_type_override is None
        and not confirm_equirectangular
    ):
        raise EquirectangularConfirmationRequired(
            f"{file_path} has a 2:1 aspect ratio; pass --confirm-equirectangular "
            "or an explicit --media-type to proceed"
        )

    dimensions = (metadata.width, metadata.height) if metadata.width and metadata.height else None
    source = Source(
        source_id=uuid.uuid4().hex,
        path=str(file_path),
        checksum=sha256_file(file_path),
        media_type=resolved_type,
        projection=projection,
        dimensions=dimensions,
        timestamps={
            "added_at": added_at,
            "duration_seconds": metadata.duration_seconds,
            "codec": metadata.codec,
            "frame_rate": metadata.frame_rate,
            "format_tags": metadata.format_tags,
        },
        capture_group=capture_group,
    )

    conn.execute(
        """
        INSERT INTO sources
            (source_id, path, checksum, media_type, projection, width, height, timestamps, capture_group)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source.source_id,
            source.path,
            source.checksum,
            source.media_type.value,
            source.projection.value,
            dimensions[0] if dimensions else None,
            dimensions[1] if dimensions else None,
            json.dumps(source.timestamps),
            source.capture_group,
        ),
    )
    conn.commit()
    return source


def get_source(conn: sqlite3.Connection, source_id: str) -> Source:
    row = conn.execute(
        "SELECT source_id, path, checksum, media_type, projection, width, height, timestamps, capture_group "
        "FROM sources WHERE source_id = ?",
        (source_id,),
    ).fetchone()
    if row is None:
        raise SourceError(f"unknown source_id: {source_id}")
    (source_id, path, checksum, media_type, projection, width, height, timestamps_json, capture_group) = row
    dimensions = (width, height) if width and height else None
    return Source(
        source_id=source_id,
        path=path,
        checksum=checksum,
        media_type=MediaType(media_type),
        projection=Projection(projection),
        dimensions=dimensions,
        timestamps=json.loads(timestamps_json),
        capture_group=capture_group,
    )


def remove_source(conn: sqlite3.Connection, project_root: Path, source_id: str) -> None:
    """Removes a registered source: its extracted frames (rows + files,
    via clear_frames_for_source) and its `sources` row. Never touches the
    original media file on disk -- sources are recorded by reference, and
    the handover doc explicitly requires "never delete source media
    automatically" (see docs/adr/0003).

    Imports clear_frames_for_source lazily to avoid a circular import
    (frames.py already imports get_source/sha256_file from this module).
    """
    get_source(conn, source_id)  # raises SourceError if unknown

    from vine360.ingest.frames import clear_frames_for_source

    clear_frames_for_source(conn, project_root, source_id)
    conn.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))
    conn.commit()
