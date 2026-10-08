"""Persists an SfmRun record for a project-level SfM run (handover doc,
section 4, step 5 "Estimate poses"). Bridges colmap_adapter.py (which
knows nothing about sqlite or the project layout) to the project's index
database.

Two real, verified image sources:
- "projections" (default): `project/projections/` (the six-face cubemap
  renders) + `project/masks/keep/` if built -- the doc's originally
  recommended pipeline.
- "frames": raw equirectangular frames directly
  (`project/frames/<frame_set_id>/`), using COLMAP's own native
  EQUIRECTANGULAR camera model. Confirmed for real via
  `pycolmap.synthesize_dataset` with `camera_model_id=EQUIRECTANGULAR`
  (full registration, near-zero reprojection error) -- this is COLMAP's
  own spherical camera support, not the third-party "SphereSfM" project
  (see docs/adr/0017), and doesn't apply masks (none are built against
  raw frames in this pipeline yet). Real caveat, not yet validated: SIFT
  feature matching itself isn't sphere-aware, so quality near the poles
  and across the equirectangular seam is unproven on actual photos, only
  on noise-free synthetic correspondences.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from vine360.sfm.colmap_adapter import (
    SfmConfig,
    SfmDiagnostics,
    SfmRegistrationError,
    evaluate_registration_quality,
    run_sfm,
    validate_installation,
)
from vine360.sfm.options import VARIANT_EQUIRECT, VARIANT_PROJECTIONS, VARIANT_SPHERESFM

__all__ = ["ENGINE_PYCOLMAP", "ENGINE_SPHERESFM", "SfmRegistrationError", "run_sfm_for_project"]

# sfm_runs.config["engine"]; rows recorded before the field existed are
# pycolmap runs (run_engine()).
ENGINE_PYCOLMAP = "pycolmap"
ENGINE_SPHERESFM = "spheresfm"


def run_engine(config: dict) -> str:
    return config.get("engine", ENGINE_PYCOLMAP)


def _image_set_hash(image_names: list[str]) -> str:
    """A weaker stand-in for the data contract's checksum-based
    image_set_hash (vine360.sfm.colmap_adapter.compute_image_set_hash):
    Views don't carry their own checksum field, only Frame/Source do, so
    this hashes the sorted set of image paths the run was given -- it
    detects a changed *set* of images, not changed *content* of an
    unchanged path. Revisit if that distinction matters before this is
    relied on for real provenance decisions."""
    return hashlib.sha256("\n".join(sorted(image_names)).encode("utf-8")).hexdigest()


def _quality_warnings(diagnostics: SfmDiagnostics, config: SfmConfig) -> list[str]:
    return evaluate_registration_quality(
        diagnostics,
        min_registered_ratio=config.min_registered_ratio,
        max_mean_reprojection_error=config.max_mean_reprojection_error,
    )


def run_sfm_for_project(
    conn: sqlite3.Connection,
    project_root: Path,
    *,
    config: SfmConfig = SfmConfig(),
    image_source: str = "projections",
    frame_set_id: str | None = None,
    engine: str = ENGINE_PYCOLMAP,
    runner=None,
    progress_callback=None,
) -> tuple[SfmDiagnostics, list[str]]:
    """image_source="frames" requires frame_set_id and runs directly
    against that frame set's raw equirectangular frames (see module
    docstring) -- the caller should set config.camera_model=
    "EQUIRECTANGULAR" for this to be meaningful; it isn't forced here so
    an explicit config always wins.

    image_source="projections" runs on every projected view in the
    project, or only frame_set_id's views if given -- one source can have
    several frame sets (e.g. every 0.5s and every 1s), and mixing them
    in one reconstruction would just duplicate near-identical images
    (docs/adr/0034).

    Either way COLMAP is handed an explicit image list (pycolmap's
    `image_names`) built from the database, not a bare directory scan --
    so frames/<id>/thumbs/*.jpg never sneak in, and total_images counts
    images rather than per-frame subdirectories. Each run gets its own
    COLMAP database under its own sparse/<run_id>/, so a scoped run never
    sees an earlier run's images left in a shared database.

    engine="spheresfm" runs the external SphereSfM build instead of
    pycolmap (raw equirectangular frames only -- docs/adr/0036), through
    `runner` (a LocalRunner by default).

    Raises SfmRegistrationError if COLMAP produces no usable
    reconstruction (a real, correct outcome for e.g. single-panorama,
    zero-parallax view sets -- see docs/adr/0007). Records an sfm_runs row
    on success only, with every option in config["options"] (docs/adr/0040).

    Raises ValueError up front, before any work, if config can't run as
    configured on this engine (SfmConfig.validate -- e.g. a missing
    vocabulary tree file, or the global mapper with SphereSfM)."""
    notify = progress_callback or (lambda *a: None)
    project_root = Path(project_root)
    variant = (
        VARIANT_SPHERESFM if engine == ENGINE_SPHERESFM else VARIANT_EQUIRECT if image_source == "frames" else VARIANT_PROJECTIONS
    )
    problems = config.validate(variant)
    if problems:
        raise ValueError("SfM options can't run as configured: " + "; ".join(problems))

    source_id = None
    if frame_set_id is not None:
        row = conn.execute("SELECT source_id FROM frame_sets WHERE frame_set_id = ?", (frame_set_id,)).fetchone()
        source_id = row[0] if row else None

    if image_source == "projections":
        image_dir = project_root / "projections"
        if frame_set_id is None:
            rows = conn.execute("SELECT image_path FROM views ORDER BY image_path").fetchall()
            missing_message = "no projected views found under projections/; generate projections first"
        else:
            rows = conn.execute(
                "SELECT v.image_path FROM views v JOIN frames f ON f.frame_id = v.frame_id "
                "WHERE f.frame_set_id = ? ORDER BY v.image_path",
                (frame_set_id,),
            ).fetchall()
            missing_message = f"no projected views found for frame set {frame_set_id}; generate projections first"
        image_names = [Path(r[0]).relative_to("projections").as_posix() for r in rows]
        from vine360.masking.layers import ensure_composites_current  # numpy/Pillow; projections path only

        ensure_composites_current(conn, project_root, frame_set_id)  # merge mask layers (docs/adr/0038)
        mask_dir = project_root / "masks" / "keep"
        mask_dir_arg = mask_dir if mask_dir.exists() and any(mask_dir.rglob("*.png")) else None
    elif image_source == "frames":
        if not frame_set_id:
            raise ValueError("frame_set_id is required when image_source='frames'")
        image_dir = project_root / "frames" / frame_set_id
        rows = conn.execute(
            "SELECT path FROM frames WHERE frame_set_id = ? ORDER BY source_time", (frame_set_id,)
        ).fetchall()
        image_names = [Path(r[0]).name for r in rows]
        mask_dir_arg = None  # masks aren't built against raw frames in this pipeline yet
        missing_message = f"no extracted frames found under frames/{frame_set_id}/; extract frames first"
    else:
        raise ValueError(f"unknown image_source: {image_source!r} (expected 'projections' or 'frames')")

    image_names = [name for name in image_names if (image_dir / name).exists()]
    if not image_names:
        raise SfmRegistrationError(missing_message)
    if engine == ENGINE_SPHERESFM:
        if image_source != "frames":
            raise ValueError("the SphereSfM engine only runs on raw equirectangular frames (image_source='frames')")
        return _run_spheresfm(
            conn, project_root, image_dir, image_names, frame_set_id, source_id, config, runner, notify
        )
    if engine != ENGINE_PYCOLMAP:
        raise ValueError(f"unknown SfM engine: {engine!r}")

    # Each run gets its own sparse/<run_id>/ subtree. A shared sparse_dir
    # across runs let a later run's pycolmap.incremental_mapping silently
    # overwrite an earlier run's numbered model directories (both start
    # numbering candidate reconstructions from 0) -- discovered as a real
    # data-loss bug on a project with 4 historical runs, only the most
    # recent of which still had its files on disk (see docs/adr/0019).
    run_id = f"sfm-{uuid.uuid4().hex[:8]}"
    sparse_dir = project_root / "sfm" / "sparse" / run_id
    database_path = sparse_dir / "database.db"

    _reconstruction, diagnostics, model_dir = run_sfm(
        image_dir, database_path, sparse_dir, mask_dir=mask_dir_arg, config=config, image_names=image_names,
        progress_callback=notify,
    )
    notify(
        f"COLMAP done: {diagnostics.registered_images}/{diagnostics.total_images} images registered, "
        f"{diagnostics.num_points3d} points, {diagnostics.num_connected_models} model(s).",
        diagnostics.registered_images,
        diagnostics.total_images,
    )

    warnings = _quality_warnings(diagnostics, config)

    conn.execute(
        """
        INSERT INTO sfm_runs
            (run_id, image_set_hash, engine_version, config, model_stats, selected_model, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            _image_set_hash(image_names),
            json.dumps(validate_installation()),
            json.dumps(
                {
                    "engine": ENGINE_PYCOLMAP,
                    "camera_model": config.camera_model,
                    "sequential_overlap": config.sequential_overlap,
                    "image_source": image_source,
                    "options": config.to_dict(),
                    "frame_set_id": frame_set_id,
                    "source_id": source_id,
                }
            ),
            json.dumps(diagnostics.to_dict()),
            model_dir.relative_to(project_root).as_posix(),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    return diagnostics, warnings


def _run_spheresfm(
    conn: sqlite3.Connection,
    project_root: Path,
    image_dir: Path,
    image_names: list[str],
    frame_set_id: str,
    source_id: str | None,
    config: SfmConfig,
    runner,
    notify,
) -> tuple[SfmDiagnostics, list[str]]:
    """SphereSfM's own CLI pipeline (vine360.sfm.spheresfm_adapter), with
    the same per-run layout as a pycolmap run: everything -- database,
    every candidate model, each model's TXT conversion -- under
    sfm/sparse/<run_id>/ (docs/adr/0019/0034/0036)."""
    from PIL import Image

    from vine360.runners.local import LocalRunner
    from vine360.sfm import spheresfm_adapter as sph

    binary = sph.find_binary()
    if binary is None:
        raise sph.SphereSfmError(
            "no SphereSfM build found -- looked in: " + ", ".join(str(p) for p in sph.candidate_binaries())
            + f" (set {sph.BINARY_ENV_VAR} to its colmap binary)"
        )
    runner = runner or LocalRunner()
    use_gpu = sph.has_cuda(sph.validate_installation()["version"])  # GPU SIFT only on a CUDA build
    match_gpu = use_gpu and config.use_gpu
    extract_gpu = match_gpu and not (config.estimate_affine_shape or config.domain_size_pooling)  # CPU-only SIFT
    matcher_command = sph.MATCHER_COMMANDS[config.matcher]  # also names its log file
    sph.check_readable_images(image_dir, image_names)
    with Image.open(image_dir / image_names[0]) as first:
        width, height = first.size

    run_id = f"sfm-{uuid.uuid4().hex[:8]}"
    run_dir = project_root / "sfm" / "sparse" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    database_path = run_dir / "database.db"
    image_list_path = run_dir / "image_list.txt"
    image_list_path.write_text("\n".join(image_names) + "\n")

    extract_device = "GPU" if extract_gpu else "CPU"
    device = "GPU" if match_gpu else "CPU"
    n = len(image_names)

    def parser(step: int, label: str) -> sph.ProgressParser:
        return sph.ProgressParser(f"SphereSfM step {step}/4 ({label}): ", n, notify)

    notify(f"SphereSfM step 1/4 (features, {extract_device}): starting on {n} frames…", 0, n)
    extraction_parser = parser(1, f"features, {extract_device}")
    sph.run_command(
        runner,
        sph.build_feature_extraction_command(
            binary, database_path, image_dir, image_list_path, width=width, height=height, use_gpu=use_gpu,
            config=config,
        ),
        "feature extraction",
        on_line=extraction_parser.feed,
        log_path=run_dir / "1_feature_extractor.log",
    )
    if extraction_parser.unreadable >= n:
        # Extraction exits 0 even when it read nothing; the matcher would
        # then abort on the empty database with an opaque cache error.
        raise sph.SphereSfmError(
            f"SphereSfM couldn't read any of the {n} frames (\"Failed to read image file format\") -- "
            f"see {run_dir / '1_feature_extractor.log'}"
        )
    if extraction_parser.unreadable:
        notify(f"SphereSfM step 1/4: {extraction_parser.unreadable}/{n} frames were unreadable and skipped", n, n)
    notify(f"SphereSfM step 2/4 (matching, {device}): {config.matcher}, starting…", 0, n)
    sph.run_command(
        runner,
        sph.build_matcher_command(binary, database_path, use_gpu=use_gpu, config=config),
        "matching",
        on_line=parser(2, f"matching, {device}").feed,
        log_path=run_dir / f"2_{matcher_command}.log",
    )
    notify("SphereSfM step 3/4 (mapping, CPU): starting…", 0, n)
    sph.run_command(
        runner,
        sph.build_mapper_command(binary, database_path, image_dir, run_dir, config=config),
        "mapping",
        on_line=parser(3, "mapping, CPU").feed,
        log_path=run_dir / "3_mapper.log",
    )

    notify("SphereSfM step 4/4: reading the reconstruction…", None, None)
    candidates = {}
    for model_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        txt_dir = model_dir / "txt"
        txt_dir.mkdir(exist_ok=True)
        sph.run_command(runner, sph.build_model_converter_command(binary, model_dir, txt_dir), "model conversion")
        candidates[model_dir] = sph.read_text_model(txt_dir)
    if not candidates:
        raise SfmRegistrationError(f"SphereSfM registered no images from frames/{frame_set_id}/")
    model_dir, model = max(candidates.items(), key=lambda item: len(item[1].images))

    track_lengths = [len(point.track) for point in model.points.values()]
    registered = len(model.images)
    diagnostics = SfmDiagnostics(
        total_images=len(image_names),
        registered_images=registered,
        registered_ratio=registered / len(image_names),
        num_connected_models=len(candidates),
        num_points3d=len(model.points),
        mean_reprojection_error=(
            sum(point.error for point in model.points.values()) / len(model.points) if model.points else 0.0
        ),
        mean_track_length=sum(track_lengths) / len(track_lengths) if track_lengths else 0.0,
        mean_observations_per_reg_image=sum(track_lengths) / registered if registered else 0.0,
    )
    notify(
        f"SphereSfM done: {registered}/{n} frames registered, {len(model.points)} points, "
        f"{len(candidates)} model(s).",
        registered,
        n,
    )
    warnings = _quality_warnings(diagnostics, config)

    conn.execute(
        """
        INSERT INTO sfm_runs
            (run_id, image_set_hash, engine_version, config, model_stats, selected_model, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            _image_set_hash(image_names),
            json.dumps({"spheresfm": sph.validate_installation()["version"], "binary": str(binary)}),
            json.dumps(
                {
                    "engine": ENGINE_SPHERESFM,
                    "camera_model": "SPHERE",
                    "sequential_overlap": config.sequential_overlap,
                    "image_source": "frames",
                    "options": config.to_dict(),
                    "frame_set_id": frame_set_id,
                    "source_id": source_id,
                }
            ),
            json.dumps(diagnostics.to_dict()),
            model_dir.relative_to(project_root).as_posix(),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    return diagnostics, warnings
