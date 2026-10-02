"""On-disk removal of derived artifacts, shared by every cascade-delete
path (re-extracting a frame set, regenerating a frame's views, the Data
manager's deletes -- docs/adr/0017/0034/0035). Stdlib-only on purpose:
vine360.ingest.frames uses it, and that module must not pull in
numpy/Pillow just to clear a directory.

Paths mirror where each stage writes:
- projections/<frame_id>/<face>.png (vine360.projection.generate)
- masks/keep/<frame_id>/<face>.png.png -- COLMAP's mask convention,
  vine360.masking.semantics.colmap_mask_path against the view's path
  relative to projections/
- masks/classes/<view_id>/ (vine360.masking.build)
"""

from __future__ import annotations

import shutil
from pathlib import Path


def _rmtree_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def remove_mask_files(project_root: Path, frame_id: str, view_ids: list[str]) -> None:
    project_root = Path(project_root)
    _rmtree_if_exists(project_root / "masks" / "keep" / frame_id)
    for view_id in view_ids:
        _rmtree_if_exists(project_root / "masks" / "classes" / view_id)


def remove_view_files(project_root: Path, frame_id: str, view_ids: list[str]) -> None:
    """A frame's projected images plus every mask built from them."""
    _rmtree_if_exists(Path(project_root) / "projections" / frame_id)
    remove_mask_files(project_root, frame_id, view_ids)
