# Vineyard 360 3DGS

Local-first orchestration app that turns 360° and conventional imagery into
masked, pose-estimated, trainable 3D Gaussian Splatting projects. See
`Vineyard_360_3DGS_Software_Handover.docx` for the full product/technical
spec and `docs/status.md` for current progress.

The processing core is a typed CLI (`vine360`); a desktop UI comes later
(M6) and will call the same functions the CLI calls.

## Requirements

- Python 3.12+
- `ffmpeg` / `ffprobe` on `PATH` (for ingest; not required to run the CLI
  itself or the non-integration tests)
- `git` (used for version reporting only)

M0/M1 are implemented with the standard library plus PyYAML, so no `pip
install` is required to run them today -- see
`docs/adr/0001-stdlib-first-cli.md` for why, and what changes once GUI/ML
milestones need PySide6, Pydantic, etc.

## Running

```
export PYTHONPATH=src   # or: pip install -e . once pip is available
python3 -m vine360 version
python3 -m vine360 project create ./myproject --name "Block 7" --capture-mode 360
python3 -m vine360 ingest add-source ./myproject --source /path/to/clip.mp4
python3 -m vine360 ingest extract-frames ./myproject --source-id <id> --interval 1.0
python3 -m vine360 ingest manifest ./myproject
```

## Tests

```
python3 -m pytest
```

ffmpeg-dependent tests skip automatically if `ffmpeg`/`ffprobe` are not on
`PATH`.
