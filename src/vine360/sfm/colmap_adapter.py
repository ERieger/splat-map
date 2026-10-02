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


class SfmError(Exception):
    pass


class SfmRegistrationError(SfmError):
    """No reconstruction was produced at all (handover doc, section 4, step
    6 "Validate": "Reject or warn on ... disconnected models")."""


@dataclass(frozen=True)
class SfmConfig:
    camera_model: str = "SIMPLE_RADIAL"
    sequential_overlap: int = 10  # images matched forward/back in sequence


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
    """Feature extraction (masked, if mask_dir is given) plus sequential
    matching ("match temporally near frames first" -- section 4, step 5).
    Separated from mapping so it's independently testable: extraction and
    matching are meaningful (and were tested against real overlapping
    renders, including the masking integration) even for an image set that
    cannot be 3D-reconstructed -- see `map_and_diagnose`'s docstring.

    image_names, if given, restricts extraction to those paths (relative
    to image_dir) instead of every image under it -- pycolmap's own
    `extract_features(image_names=...)`, confirmed against the installed
    pycolmap 4.x signature. Mapping then only sees what was extracted."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    reader_options = pycolmap.ImageReaderOptions(camera_model=config.camera_model)
    if mask_dir is not None:
        reader_options.mask_path = Path(mask_dir)

    notify = progress_callback or (lambda *a: None)
    device = "GPU" if pycolmap.has_cuda else "CPU"
    total = len(image_names) if image_names else None

    # Progress: pycolmap's extractor/matcher have no Python callback, but
    # COLMAP logs "Processed file [i/N]" / "Processing image [i/N]" to the
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
            notify(f"COLMAP step 1/3 (features, {device}): frame {i}/{n}", i, n)

    def on_match_line(line: str) -> None:
        if match := _PROCESSING_IMAGE.search(line):
            i, n = int(match.group(1)), int(match.group(2))
            notify(f"COLMAP step 2/3 (matching, {device}): frame {i}/{n}", i, n)

    notify(f"COLMAP step 1/3 (features, {device}): starting on {total or 'all'} frames…", 0 if total else None, total)
    with native_log_lines(on_extract_line if progress_callback else None):
        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_dir,
            image_names=list(image_names or []),
            reader_options=reader_options,
        )
    notify(f"COLMAP step 2/3 (matching, {device}): sequential, overlap {config.sequential_overlap}…", None, None)
    with native_log_lines(on_match_line if progress_callback else None):
        pycolmap.match_sequential(
            database_path=database_path,
            pairing_options=pycolmap.SequentialPairingOptions(overlap=config.sequential_overlap),
        )


_PROCESSED_FILE = re.compile(r"Processed file \[(\d+)/(\d+)\]")
_PROCESSING_IMAGE = re.compile(r"Processing image \[(\d+)/(\d+)\]")
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
    progress_callback=None,
) -> tuple[pycolmap.Reconstruction, SfmDiagnostics, Path]:
    """Incremental mapping against an already-populated database (from
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

    report("finding an initial pair")
    reconstructions = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_output_dir,
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
        database_path, image_dir, sparse_output_dir, total_images=total_images, progress_callback=progress_callback
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
