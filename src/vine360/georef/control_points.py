"""Surveyed ground control points (docs/adr/0047): reading the survey CSVs
as they really come off the GNSS kit, and storing them in the index.

Three real shapes so far (fixtures/gcp/):
- UTF-8, no header, `name,easting,northing,height,` with a trailing empty
  field and CRLF (EstoWines Capture 2, Flinders Bridge);
- UTF-16 with a BOM, CRLF, a header row `Point ID,Easting,Northing,Elevation`
  (EstoWines Capture 1);
- the same plus `GNSS Vector Observation.H. Precision` and `.V. Precision`
  columns (metres) -- read as sigma_h / sigma_v.

Names stay strings everywhere ("000" is not 0); the only integer parse is
derive_tag_id's default AprilTag id. Coordinates are projected (E, N, H)
exactly as surveyed: the EPSG code and height datum are user-supplied
metadata, never inferred and never converted.
"""

from __future__ import annotations

import codecs
import csv
import hashlib
import io
import math
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

USE_CONTROL = "control"
USE_CHECK = "check"
USE_IGNORE = "ignore"
USES = (USE_CONTROL, USE_CHECK, USE_IGNORE)

HEIGHT_DATUMS = ("unknown", "ellipsoidal", "AHD")
DEFAULT_EPSG = 7854  # GDA2020 / MGA zone 54 -- the UI default; always a user input
DEFAULT_TAG_FAMILY = "tag36h11"
# Number of codes per family (valid ids are 0 .. n-1), from the AprilTag 3
# family sources (tag36h11.c etc.).
TAG_FAMILY_SIZES = {"tag36h11": 587, "tag25h9": 35, "tag16h5": 30, "tagStandard41h12": 2115}
DEFAULT_FAR_POINT_M = 500.0
ORIGIN_ROUNDING_M = 100.0

_NAME_WITH_PREFIX = re.compile(r"^([A-Za-z]{1,4})[-_ ]?(\d+)$")


class ControlPointError(Exception):
    pass


@dataclass
class ControlPoint:
    name: str  # as written in the file, e.g. "020"
    east: float
    north: float
    height: float
    tag_id: int | None  # AprilTag id this point is marked with (None = not tagged)
    use: str  # "control" | "check" | "ignore"
    sigma_h: float | None = None  # metres, if the file has it
    sigma_v: float | None = None
    tag_size_m: float | None = None  # printed black-square edge; per point (250/400 mm tags mixed)
    note: str | None = None  # why it was defaulted to ignore, etc.


@dataclass
class ControlPointFile:
    points: list[ControlPoint]
    checksum: str
    encoding: str
    has_header: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class ControlPointSet:
    set_id: str
    label: str
    source_path: str
    checksum: str
    epsg: int
    height_datum: str
    tag_family: str
    tag_size_m: float | None
    origin: tuple[float, float, float]
    imported_at: str


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def derive_tag_id(name: str, tag_family: str = DEFAULT_TAG_FAMILY) -> int | None:
    """The default AprilTag id for a point name: the number itself for an
    all-digit name ("020" -> 20), or the digits after a short letter prefix
    ("GCP000" -> 0) -- None unless that's a valid id in the family, so a
    long station id like "PRS778941614462" stays untagged."""
    name = name.strip()
    if name.isdigit():
        digits = name
    else:
        match = _NAME_WITH_PREFIX.match(name)
        if not match:
            return None
        digits = match.group(2)
    value = int(digits)
    limit = TAG_FAMILY_SIZES.get(tag_family)
    return value if limit is None or value < limit else None


def _decode(data: bytes) -> tuple[str, str]:
    if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
        return data.decode("utf-16"), "utf-16"
    if data.startswith(codecs.BOM_UTF8):
        return data[len(codecs.BOM_UTF8):].decode("utf-8"), "utf-8-sig"
    if len(data) >= 4 and data[1] == 0 and data[3] == 0:  # BOM-less UTF-16 LE
        return data.decode("utf-16-le"), "utf-16-le"
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return math.isfinite(float(text))


def _column_map(header: list[str]) -> dict[str, int]:
    """Header names -> column index, matched loosely by keyword. Anything
    unmatched falls back to the positional layout."""
    columns: dict[str, int] = {}
    for index, raw in enumerate(header):
        h = raw.strip().lower()
        if "precision" in h or "sigma" in h or "accuracy" in h or "std" in h:
            if re.search(r"(^|[^a-z])(h|horiz\w*)([^a-z]|$)", h) and "sigma_h" not in columns:
                columns["sigma_h"] = index
            elif re.search(r"(^|[^a-z])(v|vert\w*)([^a-z]|$)", h) and "sigma_v" not in columns:
                columns["sigma_v"] = index
        elif "east" in h or h in ("x", "e"):
            columns.setdefault("east", index)
        elif "north" in h or h in ("y", "n"):
            columns.setdefault("north", index)
        elif any(k in h for k in ("elev", "height", "alt")) or h in ("z", "h"):
            columns.setdefault("height", index)
        elif any(k in h for k in ("name", "id", "point", "label")):
            columns.setdefault("name", index)
    positional = {"name": 0, "east": 1, "north": 2, "height": 3}
    for key, index in positional.items():
        columns.setdefault(key, index)
    return columns


def read_control_csv(
    path: Path,
    *,
    tag_family: str = DEFAULT_TAG_FAMILY,
    tag_size_m: float | None = None,
    far_point_m: float = DEFAULT_FAR_POINT_M,
    tag_sizes: dict[int, float] | None = None,
) -> ControlPointFile:
    """Parses a GCP file (see the module docstring for the shapes handled).

    A header is recognized only when the first row's first field and second
    field are both non-numeric -- never assumed. Every tagged point defaults
    to use="control", sized from tag_sizes (a tag-size table, by tag id)
    or else tag_size_m; a point more
    than far_point_m horizontally from the median of the *other* points
    (e.g. the GNSS base station) defaults to "ignore", with a note and a
    warning naming it."""
    path = Path(path)
    data = path.read_bytes()
    text, encoding = _decode(data)
    rows = [
        [cell.strip() for cell in row]
        for row in csv.reader(io.StringIO(text))
        if row and any(cell.strip() for cell in row) and not row[0].lstrip().startswith("#")
    ]
    if not rows:
        raise ControlPointError(f"{path.name}: no rows")
    has_header = not _is_number(rows[0][0]) and (len(rows[0]) < 2 or not _is_number(rows[0][1]))
    columns = _column_map(rows[0]) if has_header else {"name": 0, "east": 1, "north": 2, "height": 3, "sigma_h": 4, "sigma_v": 5}
    body = rows[1:] if has_header else rows
    first_line = 2 if has_header else 1

    points: list[ControlPoint] = []
    seen: set[str] = set()
    for offset, row in enumerate(body):
        line = first_line + offset

        def cell(key: str) -> str:
            index = columns.get(key)
            return row[index] if index is not None and index < len(row) else ""

        name = cell("name")
        if not name:
            raise ControlPointError(f"{path.name} line {line}: empty point name")
        if name in seen:
            raise ControlPointError(f"{path.name} line {line}: duplicate point name {name!r}")
        seen.add(name)
        try:
            east, north, height = (float(cell(k)) for k in ("east", "north", "height"))
        except ValueError:
            raise ControlPointError(
                f"{path.name} line {line}: expected numeric easting, northing, height, got {row!r}"
            ) from None
        sigmas = []
        for key in ("sigma_h", "sigma_v"):
            raw = cell(key)
            sigmas.append(float(raw) if raw and _is_number(raw) else None)
        tag_id = derive_tag_id(name, tag_family)
        size = None if tag_id is None else (tag_sizes or {}).get(tag_id, tag_size_m)
        points.append(ControlPoint(name, east, north, height, tag_id, USE_CONTROL, sigmas[0], sigmas[1], size))

    warnings = _flag_far_points(points, far_point_m)
    untagged = [p.name for p in points if p.tag_id is None and p.use != USE_IGNORE]
    if untagged:
        warnings.append(f"no AprilTag id could be derived for: {', '.join(untagged)} -- set one by hand to use them")
    return ControlPointFile(points, hashlib.sha256(data).hexdigest(), encoding, has_header, warnings)


def _flag_far_points(points: list[ControlPoint], far_point_m: float) -> list[str]:
    warnings = []
    if len(points) < 3:
        return warnings
    for i, point in enumerate(points):
        others = [p for j, p in enumerate(points) if j != i]
        me = sorted(p.east for p in others)[len(others) // 2]
        mn = sorted(p.north for p in others)[len(others) // 2]
        distance = math.hypot(point.east - me, point.north - mn)
        if distance > far_point_m:
            point.use = USE_IGNORE
            point.note = (
                f"{_format_distance(distance)} from the other points -- probably the GNSS base/reference "
                "station, not a ground control point"
            )
            warnings.append(f"{point.name} is {_format_distance(distance)} from the other points; set to ignore")
    return warnings


def _format_distance(metres: float) -> str:
    return f"{metres / 1000:.1f} km" if metres >= 1000 else f"{metres:.0f} m"


def default_origin(points: list[ControlPoint], rounding_m: float = ORIGIN_ROUNDING_M) -> tuple[float, float, float]:
    """(E0, N0, 0): the minimum easting/northing of the points in use,
    floored to rounding_m. H0 stays 0 -- heights of a few hundred metres
    are fine in float32 (docs/adr/0047)."""
    used = [p for p in points if p.use != USE_IGNORE] or points
    if not used:
        raise ControlPointError("no points to derive an origin from")
    east = math.floor(min(p.east for p in used) / rounding_m) * rounding_m
    north = math.floor(min(p.north for p in used) / rounding_m) * rounding_m
    return float(east), float(north), 0.0


# -- the index ----------------------------------------------------------------


def import_control_points(
    conn: sqlite3.Connection,
    path: Path,
    *,
    epsg: int,
    height_datum: str = "unknown",
    tag_family: str = DEFAULT_TAG_FAMILY,
    tag_size_m: float | None = None,
    label: str | None = None,
    origin: tuple[float, float, float] | None = None,
    far_point_m: float = DEFAULT_FAR_POINT_M,
    tag_sizes: dict[int, float] | None = None,
) -> tuple[str, list[str]]:
    """Reads path and stores it as a new control-point set (the source is
    recorded by path + checksum, never copied -- as for media, ADR 0003).
    Returns (set_id, warnings). Every import is a new set: a resurvey or a
    second capture with the same point names (EstoWines Capture 1 vs 2) never
    merges into an earlier one. tag_sizes (georef.tag_sizes) sizes each
    tagged point by its id; tag_size_m covers ids the table doesn't list."""
    if not isinstance(epsg, int) or epsg <= 0:
        raise ControlPointError(f"EPSG code is required (got {epsg!r})")
    if tag_size_m is not None and not tag_size_m > 0:
        raise ControlPointError(f"tag size must be positive metres, got {tag_size_m!r}")
    parsed = read_control_csv(path, tag_family=tag_family, tag_size_m=tag_size_m, far_point_m=far_point_m,
                              tag_sizes=tag_sizes)
    if tag_sizes is not None:
        unlisted = [p.name for p in parsed.points if p.tag_id is not None and p.tag_id not in tag_sizes]
        if unlisted:
            fallback = "left unset" if tag_size_m is None else f"given the default {tag_size_m:g} m"
            parsed.warnings.append(f"not in the tag-size table, {fallback}: {', '.join(unlisted)}")
    origin = origin or default_origin(parsed.points)
    set_id = f"gcp-{uuid.uuid4().hex[:8]}"
    conn.execute(
        "INSERT INTO control_point_sets (set_id, label, source_path, checksum, epsg, height_datum, tag_family, "
        "tag_size_m, origin_east, origin_north, origin_height, imported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            set_id, label or Path(path).name, str(Path(path).resolve()), parsed.checksum, epsg, height_datum,
            tag_family, tag_size_m, *origin, _utcnow_iso(),
        ),
    )
    for index, p in enumerate(parsed.points):
        conn.execute(
            "INSERT INTO control_points (set_id, name, row_index, east, north, height, tag_id, use, sigma_h, sigma_v, "
            "tag_size_m, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (set_id, p.name, index, p.east, p.north, p.height, p.tag_id, p.use, p.sigma_h, p.sigma_v, p.tag_size_m, p.note),
        )
    conn.commit()
    return set_id, parsed.warnings


def list_control_point_sets(conn: sqlite3.Connection) -> list[ControlPointSet]:
    rows = conn.execute(
        "SELECT set_id, label, source_path, checksum, epsg, height_datum, tag_family, tag_size_m, origin_east, "
        "origin_north, origin_height, imported_at FROM control_point_sets ORDER BY imported_at DESC, rowid DESC"
    ).fetchall()
    return [ControlPointSet(*row[:8], (row[8], row[9], row[10]), row[11]) for row in rows]


def get_control_point_set(conn: sqlite3.Connection, set_id: str) -> ControlPointSet:
    match = [s for s in list_control_point_sets(conn) if s.set_id == set_id]
    if not match:
        raise ControlPointError(f"unknown control-point set: {set_id}")
    return match[0]


def load_control_points(conn: sqlite3.Connection, set_id: str) -> list[ControlPoint]:
    rows = conn.execute(
        "SELECT name, east, north, height, tag_id, use, sigma_h, sigma_v, tag_size_m, note FROM control_points "
        "WHERE set_id = ? ORDER BY row_index",
        (set_id,),
    ).fetchall()
    return [ControlPoint(*row) for row in rows]


_UNSET = object()


def apply_tag_sizes(conn: sqlite3.Connection, set_id: str, tag_sizes: dict[int, float]) -> tuple[list[str], list[str]]:
    """Sets each tagged point's tag_size_m from a tag-size table, replacing
    any size already there (including hand edits). Points whose tag id the
    table doesn't list are left as they are. Returns (changed point names,
    unlisted point names)."""
    changed, unlisted = [], []
    for p in load_control_points(conn, set_id):
        if p.tag_id is None:
            continue
        size = tag_sizes.get(p.tag_id)
        if size is None:
            unlisted.append(p.name)
        elif p.tag_size_m != size:
            conn.execute("UPDATE control_points SET tag_size_m = ? WHERE set_id = ? AND name = ?", (size, set_id, p.name))
            changed.append(p.name)
    conn.commit()
    return changed, unlisted


def update_control_point(
    conn: sqlite3.Connection,
    set_id: str,
    name: str,
    *,
    use: str | object = _UNSET,
    tag_id: int | None | object = _UNSET,
    tag_size_m: float | None | object = _UNSET,
) -> None:
    """Edits one point's use / tag id / tag size (the GUI table's editable
    columns). Coordinates are never edited -- re-import a corrected file."""
    updates: dict[str, object] = {}
    if use is not _UNSET:
        if use not in USES:
            raise ControlPointError(f"use must be one of {USES}, got {use!r}")
        updates["use"] = use
    if tag_id is not _UNSET:
        if tag_id is not None and (not isinstance(tag_id, int) or tag_id < 0):
            raise ControlPointError(f"tag id must be a non-negative integer, got {tag_id!r}")
        updates["tag_id"] = tag_id
    if tag_size_m is not _UNSET:
        if tag_size_m is not None and not tag_size_m > 0:
            raise ControlPointError(f"tag size must be positive metres, got {tag_size_m!r}")
        updates["tag_size_m"] = tag_size_m
    if not updates:
        return
    cursor = conn.execute(
        f"UPDATE control_points SET {', '.join(f'{k} = ?' for k in updates)} WHERE set_id = ? AND name = ?",
        (*updates.values(), set_id, name),
    )
    if cursor.rowcount == 0:
        raise ControlPointError(f"no point {name!r} in control-point set {set_id}")
    conn.commit()


def update_control_point_set(
    conn: sqlite3.Connection,
    set_id: str,
    *,
    epsg: int | None = None,
    height_datum: str | None = None,
    origin: tuple[float, float, float] | None = None,
    label: str | None = None,
) -> None:
    get_control_point_set(conn, set_id)
    if epsg is not None:
        conn.execute("UPDATE control_point_sets SET epsg = ? WHERE set_id = ?", (int(epsg), set_id))
    if height_datum is not None:
        conn.execute("UPDATE control_point_sets SET height_datum = ? WHERE set_id = ?", (height_datum, set_id))
    if origin is not None:
        conn.execute(
            "UPDATE control_point_sets SET origin_east = ?, origin_north = ?, origin_height = ? WHERE set_id = ?",
            (*origin, set_id),
        )
    if label is not None:
        conn.execute("UPDATE control_point_sets SET label = ? WHERE set_id = ?", (label, set_id))
    conn.commit()


def delete_control_point_set(conn: sqlite3.Connection, project_root: Path, set_id: str) -> int:
    """Removes the set, its points, and every solution fitted to it (rows
    and materialised models, sfm/georef/<run_id>/<solution_id>/). Never
    touches an SfM run or its model. Returns the number of solutions removed."""
    from vine360.georef.solve import delete_solution

    get_control_point_set(conn, set_id)
    solution_ids = [r[0] for r in conn.execute("SELECT solution_id FROM georef_solutions WHERE set_id = ?", (set_id,))]
    for solution_id in solution_ids:
        delete_solution(conn, project_root, solution_id)
    conn.execute("DELETE FROM control_points WHERE set_id = ?", (set_id,))
    conn.execute("DELETE FROM control_point_sets WHERE set_id = ?", (set_id,))
    conn.commit()
    return len(solution_ids)

