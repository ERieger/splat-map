#!/usr/bin/env bash
# Launch the Vine360 desktop app (see CLAUDE.md "Common commands").
# Extra arguments are passed through to the app.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="${VINE360_PYTHON:-$HOME/.venvs/vine360/bin/python3}"

if [[ ! -x "$VENV_PY" ]]; then
    echo "error: Python venv not found at $VENV_PY" >&2
    echo "Set up the venv per CLAUDE.md 'Environment setup', or set VINE360_PYTHON." >&2
    exit 1
fi

# Bundled ffmpeg/ffprobe from static-ffmpeg, if installed.
FFMPEG_DIR="$("$VENV_PY" -c 'import static_ffmpeg, os; print(os.path.join(os.path.dirname(static_ffmpeg.__file__), "bin", "linux"))' 2>/dev/null || true)"
if [[ -n "$FFMPEG_DIR" && -d "$FFMPEG_DIR" ]]; then
    export PATH="$FFMPEG_DIR:$PATH"
elif ! command -v ffmpeg >/dev/null; then
    echo "warning: ffmpeg not found -- frame extraction won't work" >&2
fi

# xcb is preferred under WSLg (docs/adr/0011); it needs libxcb-cursor.so.0.
# Fall back to wayland if that library isn't available. QT_QPA_PLATFORM
# set by the caller always wins.
XCB_CURSOR_DIR="$HOME/.local/lib/xcb-cursor"
if [[ -z "${QT_QPA_PLATFORM:-}" ]]; then
    if [[ -e "$XCB_CURSOR_DIR/libxcb-cursor.so.0" ]]; then
        export QT_QPA_PLATFORM=xcb
    else
        echo "note: $XCB_CURSOR_DIR/libxcb-cursor.so.0 missing -- using wayland (docs/adr/0011)" >&2
        export QT_QPA_PLATFORM=wayland
    fi
fi
if [[ "$QT_QPA_PLATFORM" == xcb ]]; then
    export LD_LIBRARY_PATH="$XCB_CURSOR_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"
exec "$VENV_PY" -m vine360.gui.main_window "$@"
