#!/usr/bin/env bash
# Launch the vine360 MCP server over stdio (docs/adr/0048) -- what .mcp.json
# registers with Claude Code. Extra arguments are passed through, e.g.
# --project /path/to/project. stdout is the MCP protocol stream: anything
# this script says goes to stderr.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${VINE360_PYTHON:-$HOME/.venvs/vine360/bin/python3}"

if [[ ! -x "$VENV_PY" ]]; then
    echo "error: Python venv not found at $VENV_PY" >&2
    echo "Set up the venv per CLAUDE.md 'Environment setup', or set VINE360_PYTHON." >&2
    exit 1
fi
if ! "$VENV_PY" -c 'import mcp' 2>/dev/null; then
    echo "error: the mcp package isn't installed -- pip install -e \".[mcp]\"" >&2
    exit 1
fi

# Bundled ffmpeg/ffprobe from static-ffmpeg, if installed (frame extraction, adding sources).
FFMPEG_DIR="$("$VENV_PY" -c 'import static_ffmpeg, os; print(os.path.join(os.path.dirname(static_ffmpeg.__file__), "bin", "linux"))' 2>/dev/null || true)"
if [[ -n "$FFMPEG_DIR" && -d "$FFMPEG_DIR" ]]; then
    export PATH="$FFMPEG_DIR:$PATH"
elif ! command -v ffmpeg >/dev/null; then
    echo "warning: ffmpeg not found -- adding sources and frame extraction won't work" >&2
fi

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"
exec "$VENV_PY" -m vine360.mcp_server "$@"
