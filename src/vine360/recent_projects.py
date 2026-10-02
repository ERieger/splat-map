"""Per-device list of recently opened projects (docs/adr/0039).

Stored as JSON in the user's config directory, never inside a project or
the repository: ``$VINE360_CONFIG_DIR`` if set, else the OS convention --
``%APPDATA%\\vine360`` on Windows, ``~/Library/Application Support/vine360``
on macOS, ``$XDG_CONFIG_HOME/vine360`` (default ``~/.config/vine360``)
elsewhere. Stdlib-only, like
``vine360.config``, so it carries no Qt dependency and is tested directly.

A missing or corrupt file reads as an empty list -- this is a convenience,
so it must never stop the app from starting.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MAX_RECENT_PROJECTS = 10
RECENT_PROJECTS_FILENAME = "recent_projects.json"


@dataclass(frozen=True)
class RecentProject:
    path: str
    name: str
    opened_at: str

    def exists(self) -> bool:
        return (Path(self.path) / "project.yaml").is_file()


def config_dir(platform: str | None = None) -> Path:
    """The per-user config directory for this OS. ``platform`` defaults to
    ``sys.platform`` and exists so each branch can be tested anywhere."""
    override = os.environ.get("VINE360_CONFIG_DIR")
    if override:
        return Path(override)
    platform = platform or sys.platform
    if platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    elif platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "vine360"


def recent_projects_path() -> Path:
    return config_dir() / RECENT_PROJECTS_FILENAME


def _normalize(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def load_recent_projects() -> list[RecentProject]:
    """Most recently opened first."""
    try:
        data = json.loads(recent_projects_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries: list[RecentProject] = []
    if not isinstance(data, list):
        return entries
    for item in data:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        entries.append(
            RecentProject(
                path=item["path"],
                name=str(item.get("name") or Path(item["path"]).name),
                opened_at=str(item.get("opened_at") or ""),
            )
        )
    return entries[:MAX_RECENT_PROJECTS]


def _save(entries: list[RecentProject]) -> None:
    path = recent_projects_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"path": e.path, "name": e.name, "opened_at": e.opened_at} for e in entries]
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def record_recent_project(root: Path | str, name: str) -> list[RecentProject]:
    """Move (or add) ``root`` to the top of the list. Failing to write is
    swallowed -- the project still opens, it just isn't remembered."""
    key = _normalize(root)
    entry = RecentProject(path=key, name=name, opened_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    entries = [entry] + [e for e in load_recent_projects() if _normalize(e.path) != key]
    entries = entries[:MAX_RECENT_PROJECTS]
    try:
        _save(entries)
    except OSError:
        pass
    return entries


def remove_recent_project(root: Path | str) -> list[RecentProject]:
    key = _normalize(root)
    entries = [e for e in load_recent_projects() if _normalize(e.path) != key]
    try:
        _save(entries)
    except OSError:
        pass
    return entries
