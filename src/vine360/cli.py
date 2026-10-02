"""Typed CLI (handover doc: "Package the core as a CLI first so every UI
action maps to a reproducible command"). Built on argparse, not Typer --
see docs/adr/0001-stdlib-first-cli.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from vine360.config import CaptureMode, MediaType
from vine360.ingest.frames import FrameExtractionError, extract_frames
from vine360.ingest.manifest import write_manifest
from vine360.ingest.media_probe import ProbeError
from vine360.ingest.sources import (
    EquirectangularConfirmationRequired,
    SourceError,
    add_source,
    remove_source,
)
from vine360.project import (
    ProjectAlreadyExistsError,
    ProjectNotFoundError,
    create_project,
    load_project,
    open_index_db,
)
from vine360.runners.local import LocalRunner
from vine360.runners.probe import report


def cmd_version(args: argparse.Namespace) -> int:
    print(json.dumps(report(), indent=2))
    return 0


def cmd_project_create(args: argparse.Namespace) -> int:
    try:
        project = create_project(Path(args.path), args.name, CaptureMode(args.capture_mode))
    except ProjectAlreadyExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"created project {project.name!r} ({project.project_id}) at {args.path}")
    return 0


def cmd_project_info(args: argparse.Namespace) -> int:
    try:
        project = load_project(Path(args.path))
    except ProjectNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(project.to_dict(), indent=2))
    return 0


def cmd_ingest_add_source(args: argparse.Namespace) -> int:
    root = Path(args.path)
    try:
        conn = open_index_db(root)
    except ProjectNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        source = add_source(
            conn,
            Path(args.source),
            LocalRunner(),
            media_type_override=MediaType(args.media_type) if args.media_type else None,
            confirm_equirectangular=args.confirm_equirectangular,
            capture_group=args.capture_group,
            added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    except EquirectangularConfirmationRequired as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (SourceError, ProbeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    print(f"added source {source.source_id} ({source.media_type.value}, {source.projection.value})")
    return 0


def cmd_ingest_remove_source(args: argparse.Namespace) -> int:
    root = Path(args.path)
    try:
        conn = open_index_db(root)
    except ProjectNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        remove_source(conn, root, args.source_id)
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(f"removed source {args.source_id} (original media file left untouched)")
    return 0


def cmd_ingest_extract_frames(args: argparse.Namespace) -> int:
    root = Path(args.path)
    try:
        conn = open_index_db(root)
    except ProjectNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        frames = extract_frames(
            conn,
            root,
            args.source_id,
            LocalRunner(),
            interval_seconds=args.interval,
            target_count=args.count,
            start_time=args.start_time,
            end_time=args.end_time,
            generate_thumbnails=not args.skip_thumbnails,
        )
    except (FrameExtractionError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    frame_set_id = frames[0].frame_set_id if frames else None
    print(f"extracted {len(frames)} frames from source {args.source_id} into frame set {frame_set_id}")
    return 0


def cmd_ingest_manifest(args: argparse.Namespace) -> int:
    root = Path(args.path)
    try:
        conn = open_index_db(root)
    except ProjectNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        path = write_manifest(conn, root)
    finally:
        conn.close()
    print(f"wrote manifest to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vine360")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_version = subparsers.add_parser("version", help="report vine360 and dependency versions")
    p_version.set_defaults(func=cmd_version)

    p_project = subparsers.add_parser("project", help="project management")
    project_sub = p_project.add_subparsers(dest="project_command", required=True)

    p_create = project_sub.add_parser("create", help="create a new project")
    p_create.add_argument("path")
    p_create.add_argument("--name", required=True)
    p_create.add_argument(
        "--capture-mode", choices=[m.value for m in CaptureMode], default=CaptureMode.THREE_SIXTY.value
    )
    p_create.set_defaults(func=cmd_project_create)

    p_info = project_sub.add_parser("info", help="show project metadata")
    p_info.add_argument("path")
    p_info.set_defaults(func=cmd_project_info)

    p_ingest = subparsers.add_parser("ingest", help="ingest media into a project")
    ingest_sub = p_ingest.add_subparsers(dest="ingest_command", required=True)

    p_add_source = ingest_sub.add_parser("add-source", help="register a media source")
    p_add_source.add_argument("path", help="project directory")
    p_add_source.add_argument("--source", required=True, help="path to the media file")
    p_add_source.add_argument("--media-type", choices=[m.value for m in MediaType], default=None)
    p_add_source.add_argument(
        "--confirm-equirectangular",
        action="store_true",
        help="confirm that a detected 2:1 still image is truly equirectangular",
    )
    p_add_source.add_argument(
        "--capture-group",
        default=None,
        help="tag for the physical rig this source came from, e.g. 'insta360-ground' or 'antigravity-a1-aerial'",
    )
    p_add_source.set_defaults(func=cmd_ingest_add_source)

    p_remove_source = ingest_sub.add_parser("remove-source", help="remove a registered source (never deletes the original file)")
    p_remove_source.add_argument("path", help="project directory")
    p_remove_source.add_argument("--source-id", required=True)
    p_remove_source.set_defaults(func=cmd_ingest_remove_source)

    p_extract = ingest_sub.add_parser("extract-frames", help="extract deterministic frames from a video source")
    p_extract.add_argument("path", help="project directory")
    p_extract.add_argument("--source-id", required=True)
    group = p_extract.add_mutually_exclusive_group(required=True)
    group.add_argument("--interval", type=float, help="seconds between extracted frames")
    group.add_argument("--count", type=int, help="target number of frames")
    p_extract.add_argument(
        "--start-time", type=float, default=None, help="seconds from the start of the source to begin extraction at"
    )
    p_extract.add_argument(
        "--end-time", type=float, default=None, help="seconds from the start of the source to stop extraction at"
    )
    p_extract.add_argument(
        "--skip-thumbnails",
        action="store_true",
        help="skip generating a thumbnail per extracted frame (currently unused elsewhere in the app)",
    )
    p_extract.set_defaults(func=cmd_ingest_extract_frames)

    p_manifest = ingest_sub.add_parser("manifest", help="write the ingest manifest for a project")
    p_manifest.add_argument("path", help="project directory")
    p_manifest.set_defaults(func=cmd_ingest_manifest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
