# 0038. Recent projects list

## Status

Accepted.

## Context

Opening a project meant browsing to its folder with a directory picker every time. Which
projects a person has opened is a fact about their machine, not about any project or the
repository, so it can't live in `project.yaml`/`index.sqlite` or anywhere under source control.

## Decision

**A "Recent projects" box at the top of the Project panel**, listing the last 10 projects opened
or created on this device, most recent first. Double-click or "Open" reopens one; "Remove from
list" forgets an entry without touching the folder. An entry whose folder no longer has a
`project.yaml` stays listed, greyed and marked "(missing)", rather than silently disappearing,
since a drive that isn't mounted yet is a normal reason for it.

**Stored as JSON in the user's config directory**: `$VINE360_CONFIG_DIR` if set, else
`$XDG_CONFIG_HOME/vine360`, else `~/.config/vine360/recent_projects.json`. That's outside every
project and the repo, so it is never committed. Paths are stored resolved, so the same project
opened through two spellings is one entry.

**Logic is in the Qt-free, stdlib-only `vine360.recent_projects`**, tested directly. QSettings
was the alternative; a plain module keeps the file location explicit and testable without Qt.
A missing, unreadable or corrupt file reads as an empty list, and a failed write is ignored:
this is a convenience and must never stop the app opening a project.

Tests point `VINE360_CONFIG_DIR` at a temp directory through an autouse fixture in
`tests/conftest.py`, so the test suite never writes to the developer's real list.

## Consequences

- Each device, and each user account on it, has its own list.
- Opening a recent entry goes through the same `load_project` path as "Open Existing Project…",
  including its error dialog.
