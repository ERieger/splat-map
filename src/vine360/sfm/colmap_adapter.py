"""COLMAP SfM adapter (handover doc, section 4, step 5 "Estimate poses";
section 3's Adapter rule).

Uses `pycolmap` (COLMAP's official Python bindings) rather than shelling
out to a separately-installed `colmap` binary -- see docs/adr/0007. Every
function/argument name below (`ImageReaderOptions(mask_path=...)`,
`match_sequential`, `incremental_mapping(...) -> dict[int, Reconstruction]`,
`Reconstruction.num_reg_images()` etc.) was confirmed against the actually
installed pycolmap 4.2.0 (`help(pycolmap.extract_features)` and
`dir(pycolmap.Reconstruction())`), not assumed from memory.

`mask_path` uses exactly the COLMAP mask convention already implemented in
`vine360.masking.semantics.colmap_mask_path` (append `.png` to the full
image subpath) -- COLMAP's own `ImageReaderOptions.mask_path` expects that
same layout, so the masking and SfM adapters agree by construction.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pycolmap

from vine360.sfm.options import (
    MAPPER_GLOBAL,
    MATCHER_EXHAUSTIVE,
    MATCHER_SEQUENTIAL,
    MATCHER_VOCAB_TREE,
    SfmConfig,
)


class SfmError(Exception):
    pass


class SfmRegistrationError(SfmError):
    """No reconstruction was produced at all (handover doc, section 4, step
    6 "Validate": "Reject or warn on ... disconnected models")."""


@dataclass
class SfmDiagnostics:
    """Mirrors section 7's QC requirements: "Report registered image ratio,
    number of connected models, observations per image, track length,
    sparse-point count and mean reprojection error."""

    total_images: int
    registered_images: int
    registered_ratio: float
    num_connected_models: int
    num_points3d: int
    mean_reprojection_error: float
    mean_track_length: float
    mean_observations_per_reg_image: float

    def to_dict(self) -> dict:
        return {
            "total_images": self.total_images,
            "registered_images": self.registered_images,
            "registered_ratio": self.registered_ratio,
            "num_connected_models": self.num_connected_models,
            "num_points3d": self.num_points3d,
            "mean_reprojection_error": self.mean_reprojection_error,
            "mean_track_length": self.mean_track_length,
            "mean_observations_per_reg_image": self.mean_observations_per_reg_image,
        }


def validate_installation() -> dict:
    # COLMAP_version/COLMAP_build are plain str attributes, not callables --
    # confirmed by direct inspection after this was found calling them as
    # functions (a latent bug nothing had exercised until
    # tests/test_sfm_project_run.py started calling validate_installation()).
    return {
        "pycolmap_version": pycolmap.__version__,
        "colmap_version": pycolmap.COLMAP_version,
        "colmap_build": pycolmap.COLMAP_build,
        "cuda_available": pycolmap.has_cuda,
    }


def compute_image_set_hash(image_checksums: dict[str, str]) -> str:
    """Deterministic hash of the exact image set + content used for an SfM
    run (data contract: SfmRun.image_set_hash), so a changed frame or a
    changed extraction re-run is detectable rather than silently reused."""
    digest = hashlib.sha256()
    for relative_path in sorted(image_checksums):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(image_checksums[relative_path].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def extract_and_match(
    image_dir: Path,
    database_path: Path,
    *,
    mask_dir: Path | None = None,
    config: SfmConfig = SfmConfig(),
    image_names: list[str] | None = None,
    progress_callback=None,
) -> None:
    """Feature extraction (masked, if mask_dir is given) plus matching
    (sequential by default -- "match temporally near frames first",
    section 4, step 5 -- or exhaustive / vocabulary-tree, config.matcher).
    Separated from mapping so it's independently testable: extraction and
    matching are meaningful (and were tested against real overlapping
    renders, including the masking integration) even for an image set that
    cannot be 3D-reconstructed -- see `map_and_diagnose`'s docstring.

    image_names, if given, restricts extraction to those paths (relative
    to image_dir) instead of every image under it -- pycolmap's own
    `extract_features(image_names=...)`, confirmed against the installed
    pycolmap 4.x signature. Mapping then only sees what was extracted.

    Every SfmConfig field maps onto a pycolmap 4.2 option object here; the
    attribute names were read off the installed classes (`.todict()`),
    not assumed (docs/adr/0040)."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    notify = progress_callback or (lambda *a: None)
    total = len(image_names) if image_names else None
    extract_device, match_device = devices(config)

    # Progress: pycolmap's extractor/matcher have no Python callback, but
    # COLMAP logs "Processed file [i/N]" / "Processing image [i/N]" (or
    # "Processing block [i/N, j/M]" for exhaustive matching) to the
    # process's stderr -- captured and parsed for the duration of each call
    # (native_log_lines). Two approaches that DON'T work, both tried for
    # real: polling the database from another connection aborts the whole
    # process ("database is locked" -- pycolmap doesn't use WAL), and
    # extracting in small batches to report between them made GPU SIFT
    # registration measurably worse (3/10 runs lost a frame vs 0/10 for one
    # call on the same synthetic sequence).
    def on_extract_line(line: str) -> None:
        if match := _PROCESSED_FILE.search(line):
            i, n = int(match.group(1)), int(match.group(2))
            notify(f"COLMAP step 1/3 (features, {extract_device}): frame {i}/{n}", i, n)

    def on_match_line(line: str) -> None:
        if match := _PROCESSING_IMAGE.search(line):
            i, n = int(match.group(1)), int(match.group(2))
            notify(f"COLMAP step 2/3 (matching, {match_device}): frame {i}/{n}", i, n)
        elif match := _PROCESSING_BLOCK.search(line):
            a, rows, b, cols = (int(g) for g in match.groups())
            done, blocks = (a - 1) * cols + b, rows * cols
            notify(f"COLMAP step 2/3 (matching, {match_device}): block {done}/{blocks}", done, blocks)

    notify(
        f"COLMAP step 1/3 (features, {extract_device}): starting on {total or 'all'} frames…",
        0 if total else None,
        total,
    )
    with native_log_lines(on_extract_line if progress_callback else None):
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_dir,
            image_names=list(image_names or []),
            camera_mode=_CAMERA_MODES[config.camera_mode],
            reader_options=reader_options(config, mask_dir),
            extraction_options=extraction_options(config),
            device=pycolmap.Device.auto if extract_device == "GPU" else pycolmap.Device.cpu,
        )

    pairing_name, match_fn, pairing = pairing_options(config)
    notify(f"COLMAP step 2/3 (matching, {match_device}): {pairing_name}…", None, None)
    with native_log_lines(on_match_line if progress_callback else None):
        match_fn(
            database_path=database_path,
            matching_options=matching_options(config),
            pairing_options=pairing,
            verification_options=verification_options(config),
            device=pycolmap.Device.auto if match_device == "GPU" else pycolmap.Device.cpu,
        )


_CAMERA_MODES = {
    "auto": pycolmap.CameraMode.AUTO,
    "single": pycolmap.CameraMode.SINGLE,
    "per_folder": pycolmap.CameraMode.PER_FOLDER,
    "per_image": pycolmap.CameraMode.PER_IMAGE,
}


def devices(config: SfmConfig) -> tuple[str, str]:
    """("GPU"|"CPU" for extraction, same for matching). Affine-shape and
    domain-size-pooling SIFT exist only on the CPU -- COLMAP switches to a
    "Covariant SIFT CPU feature extractor" by itself (seen in its log), so
    say so rather than report GPU."""
    gpu = config.use_gpu and pycolmap.has_cuda
    cpu_only_sift = config.estimate_affine_shape or config.domain_size_pooling
    return ("GPU" if gpu and not cpu_only_sift else "CPU"), ("GPU" if gpu else "CPU")


def reader_options(config: SfmConfig, mask_dir: Path | None = None) -> pycolmap.ImageReaderOptions:
    options = pycolmap.ImageReaderOptions(camera_model=config.camera_model)
    if mask_dir is not None:
        options.mask_path = Path(mask_dir)
    return options


def extraction_options(config: SfmConfig) -> pycolmap.FeatureExtractionOptions:
    options = pycolmap.FeatureExtractionOptions()
    options.use_gpu = devices(config)[0] == "GPU"
    if config.max_image_size > 0:
        options.max_image_size = config.max_image_size
    sift = options.sift
    sift.max_num_features = config.max_num_features
    sift.peak_threshold = config.peak_threshold
    sift.edge_threshold = config.edge_threshold
    sift.upright = config.upright
    sift.estimate_affine_shape = config.estimate_affine_shape
    sift.domain_size_pooling = config.domain_size_pooling
    return options


def matching_options(config: SfmConfig) -> pycolmap.FeatureMatchingOptions:
    options = pycolmap.FeatureMatchingOptions()
    options.use_gpu = devices(config)[1] == "GPU"
    options.guided_matching = config.guided_matching
    if config.max_num_matches > 0:
        options.max_num_matches = config.max_num_matches
    options.sift.max_ratio = config.max_ratio
    options.sift.max_distance = config.max_distance
    options.sift.cross_check = config.cross_check
    return options


def verification_options(config: SfmConfig) -> pycolmap.TwoViewGeometryOptions:
    options = pycolmap.TwoViewGeometryOptions()
    options.min_num_inliers = config.min_num_inliers
    options.ransac.max_error = config.max_error
    options.ransac.min_inlier_ratio = config.min_inlier_ratio
    if config.random_seed >= 0:
        options.ransac.random_seed = config.random_seed
    return options


def pairing_options(config: SfmConfig):
    """(description, pycolmap match function, its pairing options)."""
    tree = Path(config.vocab_tree_path).expanduser() if config.vocab_tree_path else None
    if config.matcher == MATCHER_EXHAUSTIVE:
        return (
            f"exhaustive, block size {config.exhaustive_block_size}",
            pycolmap.match_exhaustive,
            pycolmap.ExhaustivePairingOptions(block_size=config.exhaustive_block_size),
        )
    if config.matcher == MATCHER_VOCAB_TREE:
        options = pycolmap.VocabTreePairingOptions()
        options.num_images = config.vocab_tree_num_images
        if tree is not None:
            options.vocab_tree_path = tree
        return f"vocabulary tree, {config.vocab_tree_num_images} candidates", pycolmap.match_vocabtree, options
    if config.matcher != MATCHER_SEQUENTIAL:
        raise SfmError(f"unknown matcher: {config.matcher!r}")
    options = pycolmap.SequentialPairingOptions(overlap=config.sequential_overlap)
    options.quadratic_overlap = config.quadratic_overlap
    options.loop_detection = config.loop_detection
    if config.loop_detection:
        options.loop_detection_period = config.loop_detection_period
        options.loop_detection_num_images = config.loop_detection_num_images
        if tree is not None:
            options.vocab_tree_path = tree
    loop = ", loop detection" if config.loop_detection else ""
    return f"sequential, overlap {config.sequential_overlap}{loop}", pycolmap.match_sequential, options


_PROCESSED_FILE = re.compile(r"Processed file \[(\d+)/(\d+)\]")
_PROCESSING_IMAGE = re.compile(r"Processing image \[(\d+)/(\d+)\]")
_PROCESSING_BLOCK = re.compile(r"Processing block \[(\d+)/(\d+), (\d+)/(\d+)\]")
_NATIVE_LOG_LOCK = threading.Lock()


@contextmanager
def native_log_lines(on_line):
    """Captures what native code (COLMAP's glog) writes to file descriptor 2
    while the block runs, handing each line to on_line -- and passing every
    line through to the real stderr unchanged, so nothing is lost from the
    terminal. A no-op when on_line is None. Process-wide by nature (fd 2 is
    shared), so one capture at a time (a lock); anything else that writes
    to stderr meanwhile is simply passed through."""
    if on_line is None:
        yield
        return
    with _NATIVE_LOG_LOCK:
        sys.stderr.flush()
        read_fd, write_fd = os.pipe()
        saved_fd = os.dup(2)
        os.dup2(write_fd, 2)
        os.close(write_fd)

        def pump() -> None:
            with os.fdopen(read_fd, "rb") as pipe:
                for raw in pipe:
                    os.write(saved_fd, raw)
                    try:
                        on_line(raw.decode("utf-8", "replace").rstrip("\n"))
                    except Exception:
                        pass  # progress reporting must never break the run

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)  # closes the pipe's last write end -> reader sees EOF
            reader.join(timeout=5)
            os.close(saved_fd)


def map_and_diagnose(
    database_path: Path,
    image_dir: Path,
    sparse_output_dir: Path,
    *,
    total_images: int | None = None,
    config: SfmConfig = SfmConfig(),
    progress_callback=None,
) -> tuple[pycolmap.Reconstruction, SfmDiagnostics, Path]:
    """Incremental mapping (or GLOMAP global mapping, config.mapper)
    against an already-populated database (from
    `extract_and_match`, or any other source of keypoints/matches -- e.g. a
    pre-populated database for testing). Returns the largest reconstruction
    (by registered image count), its diagnostics, and the directory it was
    written to -- `pycolmap.incremental_mapping` writes every candidate
    reconstruction it finds under `sparse_output_dir/<key>/`, not just the
    selected one, so callers that need to locate the actual selected model
    on disk (e.g. to hand it to external software) must use this returned
    path rather than guessing a subdirectory name.

    Note: 3D reconstruction requires real camera-position parallax between
    matched images. A set of views all rendered from a single panorama's
    optical center (e.g. six-face cubemap faces of one frame) has zero
    baseline and will correctly fail to register here regardless of match
    quality -- that's COLMAP's `init_min_tri_angle` requirement working as
    intended, not an adapter bug. Real parallax comes from distinct capture
    positions (consecutive video frames from a moving camera, or distinct
    photographs).

    Raises SfmRegistrationError if no reconstruction registers any images.
    """
    sparse_output_dir = Path(sparse_output_dir)
    sparse_output_dir.mkdir(parents=True, exist_ok=True)
    notify = progress_callback or (lambda *a: None)
    expected = total_images or 0
    registered = 0

    def report(what: str) -> None:
        if expected:
            notify(
                f"COLMAP step 3/3 (mapping): {registered}/{expected} images registered ({what})",
                min(registered, expected),
                expected,
            )
        else:
            notify(f"COLMAP step 3/3 (mapping): {registered} images registered ({what})", None, None)

    def on_initial_pair() -> None:
        nonlocal registered
        registered += 2
        report("initial pair")

    def on_next_image() -> None:
        nonlocal registered
        registered += 1
        report("registering")

    if config.mapper == MAPPER_GLOBAL:
        reconstructions = _global_mapping(database_path, image_dir, sparse_output_dir, config, notify, progress_callback)
    else:
        report("finding an initial pair")
        reconstructions = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=image_dir,
            output_path=sparse_output_dir,
            options=incremental_options(config),
            initial_image_pair_callback=on_initial_pair,
            next_image_callback=on_next_image,
        )

    if total_images is None:
        total_images = len(list(Path(image_dir).iterdir()))

    if not reconstructions:
        raise SfmRegistrationError(
            f"COLMAP registered no images from {image_dir} "
            "(0 usable reconstructions produced)"
        )

    best_key = max(reconstructions, key=lambda k: reconstructions[k].num_reg_images())
    selected = reconstructions[best_key]

    model_dir = sparse_output_dir / str(best_key)
    model_dir.mkdir(parents=True, exist_ok=True)
    selected.write(model_dir)

    registered = selected.num_reg_images()
    diagnostics = SfmDiagnostics(
        total_images=total_images,
        registered_images=registered,
        registered_ratio=registered / total_images if total_images else 0.0,
        num_connected_models=len(reconstructions),
        num_points3d=selected.num_points3D(),
        mean_reprojection_error=selected.compute_mean_reprojection_error(),
        mean_track_length=selected.compute_mean_track_length(),
        mean_observations_per_reg_image=selected.compute_mean_observations_per_reg_image(),
    )
    return selected, diagnostics, model_dir


def incremental_options(config: SfmConfig) -> pycolmap.IncrementalPipelineOptions:
    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = config.min_num_matches
    options.multiple_models = config.multiple_models
    if config.min_model_size > 0:
        options.min_model_size = config.min_model_size
    options.ba_refine_focal_length = config.refine_focal_length
    options.ba_refine_principal_point = config.refine_principal_point
    options.ba_refine_extra_params = config.refine_extra_params
    mapper = options.mapper
    mapper.init_min_num_inliers = config.init_min_num_inliers
    mapper.init_min_tri_angle = config.init_min_tri_angle
    mapper.init_max_forward_motion = config.init_max_forward_motion
    mapper.abs_pose_min_num_inliers = config.abs_pose_min_num_inliers
    mapper.abs_pose_min_inlier_ratio = config.abs_pose_min_inlier_ratio
    mapper.abs_pose_refine_focal_length = config.refine_focal_length
    mapper.abs_pose_refine_extra_params = config.refine_extra_params
    mapper.filter_max_reproj_error = config.filter_max_reproj_error
    mapper.filter_min_tri_angle = config.filter_min_tri_angle
    if config.random_seed >= 0:
        options.random_seed = mapper.random_seed = options.triangulation.random_seed = config.random_seed
    return options


def global_options(config: SfmConfig) -> pycolmap.GlobalPipelineOptions:
    options = pycolmap.GlobalPipelineOptions()
    options.min_num_matches = config.min_num_matches
    options.multiple_models = config.multiple_models
    if config.min_model_size > 0:
        options.min_model_size = config.min_model_size
    adjustment = options.mapper.bundle_adjustment
    adjustment.refine_focal_length = config.refine_focal_length
    adjustment.refine_principal_point = config.refine_principal_point
    adjustment.refine_extra_params = config.refine_extra_params
    if config.random_seed >= 0:
        options.random_seed = options.mapper.random_seed = config.random_seed
    return options


# GLOMAP's stages, in order, as its log announces them ("=== Running
# rotation averaging ===" ...) -- read off a real pycolmap 4.2 run.
_GLOBAL_STAGES = (
    "rotation averaging",
    "track establishment",
    "global positioning",
    "iterative bundle adjustment",
    "iterative retriangulation and refinement",
)
_GLOBAL_STAGE_LINE = re.compile(r"=== Running (.+?) ===")
_GLOBAL_COMPONENT_LINE = re.compile(r"=== Reconstructing component (\d+) / (\d+) with (\d+) images ===")


def _global_mapping(database_path, image_dir, sparse_output_dir, config, notify, progress_callback):
    """pycolmap.global_mapping has no Python callback, so progress comes
    from its log: one count per GLOMAP stage (of len(_GLOBAL_STAGES)) per
    connected component."""
    n = len(_GLOBAL_STAGES)
    component = ""

    def on_line(line: str) -> None:
        nonlocal component
        if match := _GLOBAL_COMPONENT_LINE.search(line):
            component = f"component {match.group(1)}/{match.group(2)}, {match.group(3)} images, " if match.group(2) != "1" else ""
        elif (match := _GLOBAL_STAGE_LINE.search(line)) and match.group(1) in _GLOBAL_STAGES:
            stage = _GLOBAL_STAGES.index(match.group(1)) + 1
            notify(f"COLMAP step 3/3 (global mapping): {component}stage {stage}/{n} ({match.group(1)})", stage - 1, n)

    notify(f"COLMAP step 3/3 (global mapping): starting ({n} stages)…", 0, n)
    with native_log_lines(on_line if progress_callback else None):
        reconstructions = pycolmap.global_mapping(
            database_path=database_path,
            image_path=image_dir,
            output_path=sparse_output_dir,
            options=global_options(config),
        )
    notify(f"COLMAP step 3/3 (global mapping): stage {n}/{n} done", n, n)
    return reconstructions


def run_sfm(
    image_dir: Path,
    database_path: Path,
    sparse_output_dir: Path,
    *,
    mask_dir: Path | None = None,
    config: SfmConfig = SfmConfig(),
    image_names: list[str] | None = None,
    progress_callback=None,
) -> tuple[pycolmap.Reconstruction, SfmDiagnostics, Path]:
    """extract_and_match + map_and_diagnose against real image files on disk
    (only image_names, relative to image_dir, if given)."""
    extract_and_match(
        image_dir, database_path, mask_dir=mask_dir, config=config, image_names=image_names,
        progress_callback=progress_callback,
    )
    total_images = len(image_names) if image_names else len(list(Path(image_dir).iterdir()))
    return map_and_diagnose(
        database_path, image_dir, sparse_output_dir, total_images=total_images, config=config,
        progress_callback=progress_callback,
    )


def evaluate_registration_quality(
    diagnostics: SfmDiagnostics,
    *,
    min_registered_ratio: float = 0.7,
    max_mean_reprojection_error: float = 2.0,
) -> list[str]:
    """Handover doc, section 4, step 6 "Validate": "Reject or warn on low
    registration ratio, fragmented models, ... poor reprojection error,
    sparse coverage." Returns human-readable warnings; empty if clean."""
    warnings = []
    if diagnostics.registered_ratio < min_registered_ratio:
        warnings.append(
            f"low registration ratio: {diagnostics.registered_ratio:.0%} "
            f"< {min_registered_ratio:.0%}"
        )
    if diagnostics.num_connected_models > 1:
        warnings.append(
            f"fragmented reconstruction: {diagnostics.num_connected_models} "
            "disconnected models produced"
        )
    if diagnostics.mean_reprojection_error > max_mean_reprojection_error:
        warnings.append(
            f"poor reprojection error: {diagnostics.mean_reprojection_error:.2f}px "
            f"> {max_mean_reprojection_error:.2f}px"
        )
    if diagnostics.num_points3d == 0:
        warnings.append("no triangulated 3D points (sparse coverage)")
    return warnings
