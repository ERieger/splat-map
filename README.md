# Vineyard 360 3DGS

Local-first orchestration app that turns 360° and conventional imagery into
masked, pose-estimated, trainable 3D Gaussian Splatting projects. See
`Vineyard_360_3DGS_Software_Handover.docx` for the full product/technical
spec and `docs/status.md` for current progress.

The processing core is a typed CLI (`vine360`); a desktop UI (`vine360.gui`,
M6, in progress) calls the same functions the CLI calls.

## Requirements

- Python 3.12+
- `pip`/`venv` working in *some* environment reachable from here (see
  "Environment setup" below if, like this project's dev machine, the
  system Python has neither)
- `git` (used for version reporting only)

The M0/M1 CLI/config layer itself is standard-library-only (argparse,
dataclasses) -- see `docs/adr/0001-stdlib-first-cli.md`. M2 onward
(projection, masking, SfM) uses real numpy/Pillow/scipy/pycolmap/
transformers, installed into a venv -- see `docs/adr/0006` through
`0009`.

## Environment setup

If `python3 -m pip` fails with "externally managed environment" or
`ensurepip` is missing (Debian/Ubuntu without `python3-venv`/`python3-pip`
installed, and no sudo access to fix that):

```
python3 -m venv --without-pip ~/.venvs/vine360   # NOT under a Windows-mounted
                                                   # drive (e.g. /mnt/e) -- venv
                                                   # needs symlinks DrvFs can't
                                                   # make; see docs/adr/0005.
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
~/.venvs/vine360/bin/python3 /tmp/get-pip.py

~/.venvs/vine360/bin/python3 -m pip install -e ".[dev,sfm,masking]"
# ingest-ffmpeg extra bundles real ffmpeg/ffprobe binaries (no system
# install / sudo needed):
~/.venvs/vine360/bin/python3 -m pip install static-ffmpeg
```

If you already have a normal working `pip`, just:

```
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sfm,masking]" static-ffmpeg ffmpeg
```

## Running

```
VENV=~/.venvs/vine360/bin/python3   # or just `python3` with an activated venv
export PATH="$HOME/.venvs/vine360/lib/python3.12/site-packages/static_ffmpeg/bin/linux:$PATH"

$VENV -m vine360 version
$VENV -m vine360 project create ./myproject --name "Block 7" --capture-mode 360
$VENV -m vine360 ingest add-source ./myproject --source /path/to/clip.mp4
$VENV -m vine360 ingest extract-frames ./myproject --source-id <id> --interval 1.0
$VENV -m vine360 ingest manifest ./myproject
```

Projection (M2), masking (M3) and SfM (M4) are implemented and tested as
library code (`vine360.projection`, `vine360.masking`, `vine360.sfm`) but
not yet wired into the project CLI/index database -- see docs/status.md
for the exact next task.

SAM 3 (real person/sky segmentation, `vine360.masking.sam3_adapter`)
requires **manual** Meta approval on Hugging Face: request access at
https://huggingface.co/facebook/sam3, then once approved run
`$VENV -m pip install huggingface_hub[cli] && huggingface-cli login` and
paste your token when it prompts (never into chat/logs).

## Desktop app

```
VENV=~/.venvs/vine360/bin/python3
export LD_LIBRARY_PATH="$HOME/.local/lib/xcb-cursor:$LD_LIBRARY_PATH"
export QT_QPA_PLATFORM=xcb   # not wayland -- see docs/adr/0011 (window
                             # stacking / dropdown reliability on WSLg)
PYTHONPATH=src $VENV -m vine360.gui.main_window
```

`~/.local/lib/xcb-cursor` needs the extracted `libxcb-cursor.so.0` (see
docs/adr/0011 for how it was obtained without root: `apt-get download
libxcb-cursor0` + `dpkg-deb -x`, no sudo needed for either). Without it,
`xcb` fails to start and `QT_QPA_PLATFORM=wayland` is the fallback, which
works but has had window-stacking and combo-box quirks under WSLg/Weston.

Project creation, source ingest (add/remove) and frame extraction
(including a custom interval) are wired to real code; Projection, Masks,
Pose estimation and Training stay preset-only (buttons disabled) -- see
docs/status.md.

## Tests

```
export PATH="$HOME/.venvs/vine360/lib/python3.12/site-packages/static_ffmpeg/bin/linux:$PATH"
~/.venvs/vine360/bin/python3 -m pytest
```

Tests requiring ffmpeg, pycolmap, or a real (approved) SAM 3 login skip
automatically when those aren't available.
