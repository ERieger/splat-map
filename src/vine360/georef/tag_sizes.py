"""Tag-size tables: printed size per AprilTag id (docs/adr/0047).

Tags in one deployment differ in size (250 mm and 400 mm in the Vine360
kit), so a control point's tag_size_m comes from a table keyed by tag id,
with the panel's single default size only as a fallback for ids the
table doesn't list.

The built-in table is the Vine360 kit's own sheet,
data/vine360_apriltag_id_sizes.csv, kept byte-for-byte with its PDF
(data/vine360_apriltag_id_sizes.pdf). Its sizes are the *active* size, the
black square's outer edge, which is what tag_size_m means; the board or
sheet size is not. IDs 028-047 (the A3 design) are listed although the
sheet says their printing and deployment aren't confirmed: a size for an
id that isn't in the field does nothing.

A different table can be chosen per device. The choice is stored next to
the recent-projects list (vine360.recent_projects.config_dir), never in a
project or the repository. A missing or unreadable choice falls back to
the built-in table.

Table format: a CSV with a header, a ``tag_id`` column and one size column,
either ``active_size_mm``, ``size_mm`` or ``size_m``. An optional ``family``
column restricts rows to one tag family. Other columns are ignored.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from vine360.recent_projects import config_dir

BUILTIN_TABLE = Path(__file__).with_name("data") / "vine360_apriltag_id_sizes.csv"
SETTINGS_FILENAME = "georef.json"
_SIZE_COLUMNS = {"active_size_mm": 0.001, "size_mm": 0.001, "size_m": 1.0}


class TagSizeTableError(Exception):
    pass


def read_tag_size_table(path: Path, tag_family: str = "tag36h11") -> dict[int, float]:
    """{tag_id: size in metres} for tag_family's rows."""
    path = Path(path)
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise TagSizeTableError(f"{path.name}: can't read ({exc})") from exc
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise TagSizeTableError(f"{path.name}: empty")
    header = [h.strip().lower() for h in rows[0]]
    if "tag_id" not in header:
        raise TagSizeTableError(f"{path.name}: no tag_id column (header {rows[0]!r})")
    size_key = next((k for k in _SIZE_COLUMNS if k in header), None)
    if size_key is None:
        raise TagSizeTableError(f"{path.name}: no size column -- expected one of {', '.join(_SIZE_COLUMNS)}")
    id_index, size_index = header.index("tag_id"), header.index(size_key)
    family_index = header.index("family") if "family" in header else None
    sizes: dict[int, float] = {}
    for line, row in enumerate(rows[1:], start=2):
        row = [cell.strip() for cell in row]
        if not any(row):
            continue
        if family_index is not None and family_index < len(row) and row[family_index] and row[family_index] != tag_family:
            continue
        try:
            tag_id = int(row[id_index])
            size = float(row[size_index]) * _SIZE_COLUMNS[size_key]
        except (IndexError, ValueError):
            raise TagSizeTableError(f"{path.name} line {line}: expected a tag id and a size, got {row!r}") from None
        if tag_id < 0 or not size > 0:
            raise TagSizeTableError(f"{path.name} line {line}: tag id {tag_id}, size {size} m out of range")
        if tag_id in sizes and sizes[tag_id] != size:
            raise TagSizeTableError(f"{path.name} line {line}: tag {tag_id} listed twice with different sizes")
        sizes[tag_id] = size
    if not sizes:
        raise TagSizeTableError(f"{path.name}: no {tag_family} rows")
    return sizes


def _settings_path() -> Path:
    return config_dir() / SETTINGS_FILENAME


def default_table_path() -> Path:
    """The device's chosen table, or the built-in one."""
    try:
        chosen = json.loads(_settings_path().read_text(encoding="utf-8")).get("tag_size_table")
    except (OSError, ValueError, AttributeError):
        chosen = None
    if chosen and Path(chosen).is_file():
        return Path(chosen)
    return BUILTIN_TABLE


def set_default_table_path(path: Path | None) -> None:
    """Remembers path as this device's table; None goes back to the built-in one."""
    settings_path = _settings_path()
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            settings = {}
    except (OSError, ValueError):
        settings = {}
    if path is None:
        settings.pop("tag_size_table", None)
    else:
        settings["tag_size_table"] = str(Path(path).resolve())
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
