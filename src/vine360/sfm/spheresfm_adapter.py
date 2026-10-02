"""SphereSfM adapter (https://github.com/json87/SphereSfM) -- docs/adr/0036.

**Verified against a real build** (COLMAP 3.8 fork, commit 6b40b2d,
CPU-only, built from source on this machine): every subcommand and flag
used below was checked against that binary's own `<command> -h` output
(tests/test_spheresfm_adapter.py re-checks them whenever the binary is
present), and the full extract -> match -> map -> TXT pipeline registers
a real synthetic 360 sequence (tests/test_spheresfm_run.py).

SphereSfM is a separate fork of COLMAP with its own `SPHERE` camera model
and sphere-aware two-view geometry / bundle adjustment -- different from
pycolmap's own EQUIRECTANGULAR model (the "native equirectangular" engine
in vine360.sfm.project_run). Things that matter when calling it:

- It must be compiled from C++ source; there's no wheel or prebuilt
  binary. find_binary() looks in a few known places and only accepts a
  `colmap` whose `help` lists `sphere_cubic_reprojecer` -- a stock COLMAP
  can't be mistaken for it.
- **pycolmap must never load its models.** SPHERE is camera model id 11
  in SphereSfM (src/base/camera_models.h), but pycolmap reads id 11 as
  RAD_TAN_THIN_PRISM_FISHEYE. Models are converted to TXT with
  SphereSfM's own `model_converter` and parsed by read_text_model.
- SPHERE params are "f,cx,cy" with f unused by its projection (README
  example: "1,3520,1760" for a 7040x3520 image) -- pixel = normalized *
  max(2cx, 2cy) + (cx, cy) (camera_models.h).
- Sphere-aware geometric verification lives in
  estimators/two_view_geometry.cc and is chosen per camera pair, so it
  applies to every matcher, sequential_matcher included -- not just the
  README's spatial_matcher (which needs pose priors vine360 doesn't have
  yet).
- The cube-face exporter's subcommand really is spelled
  `sphere_cubic_reprojecer` (sic).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from vine360.runners.base import Runner
from vine360.runners.local import LocalRunner
from vine360.sfm.options import MATCHER_EXHAUSTIVE, MATCHER_SEQUENTIAL, MATCHER_VOCAB_TREE, SfmConfig

SPHERESFM_REPO_URL = "https://github.com/json87/SphereSfM"
BINARY_ENV_VAR = "VINE360_SPHERESFM_COLMAP"
_SIGNATURE_COMMAND = "sphere_cubic_reprojecer"


class SphereSfmError(Exception):
    pass


def candidate_binaries() -> list[Path]:
    """Where find_binary looks, in order: an explicit env var, the
    `ninja install` prefix used on this machine, then the build tree."""
    candidates = []
    if os.environ.get(BINARY_ENV_VAR):
        candidates.append(Path(os.environ[BINARY_ENV_VAR]).expanduser())
    home = Path.home()
    candidates += [home / ".local" / "spheresfm" / "bin" / "colmap", home / "build" / "spheresfm" / "src" / "exe" / "colmap"]
    return candidates


def _env() -> dict[str, str]:
    # No display needed for CPU SIFT/matching/mapping; offscreen keeps Qt
    # from looking for one. subprocess's env= replaces the whole
    # environment, so start from the current one.
    return {**os.environ, "QT_QPA_PLATFORM": "offscreen"}


@lru_cache(maxsize=8)
def _probe(binary: str) -> str | None:
    """The binary's version line if it's a SphereSfM build, else None.
    Cached per path -- it's a subprocess call."""
    runner = LocalRunner()
    try:
        help_result = runner.run([binary, "help"], env=_env(), timeout=30)
    except Exception:
        return None
    if _SIGNATURE_COMMAND not in help_result.stdout + help_result.stderr:
        return None
    version = runner.run([binary, "-h"], env=_env(), timeout=30)
    lines = [line.strip() for line in (version.stdout or "").splitlines() if line.strip()]
    return " ".join(lines[:2]) if lines else "SphereSfM (version unknown)"


def find_binary() -> Path | None:
    for candidate in candidate_binaries():
        if candidate.is_file() and os.access(candidate, os.X_OK) and _probe(str(candidate)):
            return candidate
    return None


def has_cuda(version: str | None) -> bool:
    """COLMAP's own version banner says "with CUDA" or "without CUDA"."""
    return bool(version) and "with CUDA" in version and "without CUDA" not in version


def validate_installation() -> dict:
    binary = find_binary()
    version = _probe(str(binary)) if binary else None
    return {
        "found": binary is not None,
        "path": str(binary) if binary else None,
        "version": version,
        "cuda": has_cuda(version),
        "searched": [str(p) for p in candidate_binaries()],
        "env_var": BINARY_ENV_VAR,
    }


# -- command construction (pure; unit-testable without the binary) ---------


def _flag(value) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return f"{value:g}" if isinstance(value, float) else str(value)


def _seed(config: SfmConfig) -> list[str]:
    return ["--random_seed", str(config.random_seed)] if config.random_seed >= 0 else []


def build_feature_extraction_command(
    binary: Path,
    database_path: Path,
    image_path: Path,
    image_list_path: Path,
    *,
    width: int,
    height: int,
    use_gpu: bool = False,
    config: SfmConfig | None = None,
) -> list[str]:
    """use_gpu only for a CUDA build (has_cuda): a CPU-only build's GPU SIFT
    would fall back to OpenGL SiftGPU, which needs a display. config's
    feature options (docs/adr/0040) map onto `--SiftExtraction.*`; camera
    mode "auto" keeps the single shared SPHERE camera vine360 always used."""
    config = config or SfmConfig()
    # Affine-shape / DSP SIFT are CPU-only (SiftGPU has neither).
    gpu = use_gpu and config.use_gpu and not (config.estimate_affine_shape or config.domain_size_pooling)
    command = [
        str(binary),
        "feature_extractor",
        "--database_path", str(database_path),
        "--image_path", str(image_path),
        "--image_list_path", str(image_list_path),
        "--ImageReader.camera_model", "SPHERE",
        "--ImageReader.camera_params", f"1,{width / 2:g},{height / 2:g}",
        "--ImageReader.single_camera", _flag(config.camera_mode in ("auto", "single")),
    ]
    if config.camera_mode == "per_folder":
        command += ["--ImageReader.single_camera_per_folder", "1"]
    elif config.camera_mode == "per_image":
        command += ["--ImageReader.single_camera_per_image", "1"]
    command += ["--SiftExtraction.use_gpu", _flag(gpu)]
    if config.max_image_size > 0:
        command += ["--SiftExtraction.max_image_size", str(config.max_image_size)]
    command += [
        "--SiftExtraction.max_num_features", str(config.max_num_features),
        "--SiftExtraction.peak_threshold", _flag(config.peak_threshold),
        "--SiftExtraction.edge_threshold", _flag(config.edge_threshold),
        "--SiftExtraction.upright", _flag(config.upright),
        "--SiftExtraction.estimate_affine_shape", _flag(config.estimate_affine_shape),
        "--SiftExtraction.domain_size_pooling", _flag(config.domain_size_pooling),
    ]
    return command + _seed(config)


MATCHER_COMMANDS = {
    MATCHER_SEQUENTIAL: "sequential_matcher",
    MATCHER_EXHAUSTIVE: "exhaustive_matcher",
    MATCHER_VOCAB_TREE: "vocab_tree_matcher",
}


def build_matcher_command(
    binary: Path, database_path: Path, *, overlap: int | None = None, use_gpu: bool = False, config: SfmConfig | None = None
) -> list[str]:
    """config.matcher picks the subcommand (sequential by default);
    `overlap`, if given, overrides config.sequential_overlap."""
    config = config or SfmConfig()
    if overlap is not None:
        config = config.replace(sequential_overlap=overlap)
    if config.matcher not in MATCHER_COMMANDS:
        raise SphereSfmError(f"unknown matcher: {config.matcher!r}")
    tree = str(Path(config.vocab_tree_path).expanduser()) if config.vocab_tree_path else ""
    command = [
        str(binary),
        MATCHER_COMMANDS[config.matcher],
        "--database_path", str(database_path),
        "--SiftMatching.use_gpu", _flag(use_gpu and config.use_gpu),
        "--SiftMatching.max_ratio", _flag(config.max_ratio),
        "--SiftMatching.max_distance", _flag(config.max_distance),
        "--SiftMatching.cross_check", _flag(config.cross_check),
        "--SiftMatching.guided_matching", _flag(config.guided_matching),
        "--SiftMatching.min_num_inliers", str(config.min_num_inliers),
        "--SiftMatching.max_error", _flag(config.max_error),
        "--SiftMatching.min_inlier_ratio", _flag(config.min_inlier_ratio),
    ]
    if config.max_num_matches > 0:
        command += ["--SiftMatching.max_num_matches", str(config.max_num_matches)]
    if config.matcher == MATCHER_SEQUENTIAL:
        command += [
            "--SequentialMatching.overlap", str(config.sequential_overlap),
            "--SequentialMatching.quadratic_overlap", _flag(config.quadratic_overlap),
            "--SequentialMatching.loop_detection", _flag(config.loop_detection),
        ]
        if config.loop_detection:
            command += [
                "--SequentialMatching.loop_detection_period", str(config.loop_detection_period),
                "--SequentialMatching.loop_detection_num_images", str(config.loop_detection_num_images),
                "--SequentialMatching.vocab_tree_path", tree,
            ]
    elif config.matcher == MATCHER_EXHAUSTIVE:
        command += ["--ExhaustiveMatching.block_size", str(config.exhaustive_block_size)]
    else:
        command += [
            "--VocabTreeMatching.num_images", str(config.vocab_tree_num_images),
            "--VocabTreeMatching.vocab_tree_path", tree,
        ]
    return command + _seed(config)


def build_mapper_command(
    binary: Path, database_path: Path, image_path: Path, output_path: Path, *, config: SfmConfig | None = None
) -> list[str]:
    """Camera intrinsics stay fixed (README step 4) -- SPHERE has nothing
    meaningful to refine, so the refine options don't apply here. config's
    mapping thresholds map onto `--Mapper.*`."""
    config = config or SfmConfig()
    return [
        str(binary),
        "mapper",
        "--database_path", str(database_path),
        "--image_path", str(image_path),
        "--output_path", str(output_path),
        "--Mapper.sphere_camera", "1",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.min_num_matches", str(config.min_num_matches),
        "--Mapper.multiple_models", _flag(config.multiple_models),
        "--Mapper.init_min_num_inliers", str(config.init_min_num_inliers),
        "--Mapper.init_min_tri_angle", _flag(config.init_min_tri_angle),
        "--Mapper.init_max_forward_motion", _flag(config.init_max_forward_motion),
        "--Mapper.abs_pose_min_num_inliers", str(config.abs_pose_min_num_inliers),
        "--Mapper.abs_pose_min_inlier_ratio", _flag(config.abs_pose_min_inlier_ratio),
        "--Mapper.filter_max_reproj_error", _flag(config.filter_max_reproj_error),
        "--Mapper.filter_min_tri_angle", _flag(config.filter_min_tri_angle),
    ] + (["--Mapper.min_model_size", str(config.min_model_size)] if config.min_model_size > 0 else []) + _seed(config)


def build_model_converter_command(binary: Path, input_path: Path, output_path: Path) -> list[str]:
    return [
        str(binary),
        "model_converter",
        "--input_path", str(input_path),
        "--output_path", str(output_path),
        "--output_type", "TXT",
    ]


def build_cubic_reprojection_command(
    binary: Path, input_path: Path, image_path: Path, output_path: Path, *, image_size: int = 0, field_of_view: float = 45.0
) -> list[str]:
    """SphereSfM's own sphere -> pinhole cube-face exporter. vine360 uses it
    only in tests, as an independent check of vine360.sfm.frame_poses."""
    return [
        str(binary),
        _SIGNATURE_COMMAND,
        "--input_path", str(input_path),
        "--image_path", str(image_path),
        "--output_path", str(output_path),
        "--image_size", str(image_size),
        "--field_of_view", str(field_of_view),
    ]


# -- running ---------------------------------------------------------------


_PROCESSED_FILE = re.compile(r"Processed file \[(\d+)/(\d+)\]")
_MATCHING_IMAGE = re.compile(r"Matching image \[(\d+)/(\d+)\]")
_MATCHING_BLOCK = re.compile(r"(?:Matching|Processing) block \[(\d+)/(\d+), (\d+)/(\d+)\]")
_INITIAL_PAIR = re.compile(r"Initializing with image pair #(\d+) and #(\d+)")
_REGISTERING = re.compile(r"Registering image #(\d+) \((\d+)\)")


class ProgressParser:
    """Turns SphereSfM's (COLMAP 3.8's) own log lines into
    (message, current, total) progress -- the exact strings come from its
    source: feature/extraction.cc "Processed file [i/N]",
    feature/matching.cc "Matching image [i/N]", and
    controllers/incremental_mapper.cc "Initializing with image pair #a
    and #b" / "Registering image #id (n)" (n = images registered so far).
    `total_images` is what the mapper's count is shown against."""

    def __init__(self, prefix: str, total_images: int, notify):
        self.prefix = prefix
        self.total_images = total_images
        self.notify = notify
        self.registered = 0
        self.phase = ""

    def feed(self, line: str) -> None:
        if match := _PROCESSED_FILE.search(line):
            i, n = int(match.group(1)), int(match.group(2))
            self.notify(f"{self.prefix}frame {i}/{n}", i, n)
        elif match := _MATCHING_IMAGE.search(line):
            i, n = int(match.group(1)), int(match.group(2))
            self.notify(f"{self.prefix}frame {i}/{n}", i, n)
        elif match := _MATCHING_BLOCK.search(line):
            a, rows, b, cols = (int(g) for g in match.groups())
            self.notify(f"{self.prefix}block {(a - 1) * cols + b}/{rows * cols}", (a - 1) * cols + b, rows * cols)
        elif match := _INITIAL_PAIR.search(line):
            self.registered = 2
            self._mapper(f"initial pair #{match.group(1)} + #{match.group(2)}")
        elif match := _REGISTERING.search(line):
            self.registered = int(match.group(2))
            self._mapper(f"registering image #{match.group(1)}")
        elif "Finding good initial image pair" in line:
            self._mapper("finding a good initial image pair")
        elif "Global bundle adjustment" in line:
            self._mapper("global bundle adjustment")
        elif "Retriangulation" in line:
            self._mapper("retriangulation")

    def _mapper(self, what: str) -> None:
        self.notify(
            f"{self.prefix}{self.registered}/{self.total_images} frames registered ({what})",
            min(self.registered, self.total_images),
            self.total_images,
        )


def run_command(
    runner: Runner, command: list[str], what: str, *, on_line=None, log_path: Path | None = None
) -> None:
    """Runs one SphereSfM subcommand, streaming its output to on_line (for
    live progress) and to log_path (kept with the run, for diagnosis)."""
    log = open(log_path, "w", encoding="utf-8") if log_path else None

    def handle(line: str) -> None:
        if log:
            log.write(line + "\n")
        if on_line:
            on_line(line)

    try:
        result = runner.run(command, env=_env(), on_output=handle if (on_line or log) else None)
    finally:
        if log:
            log.close()
    if not result.ok:
        tail = "\n".join((result.stderr or result.stdout or "").strip().splitlines()[-15:])
        where = f"\n(full log: {log_path})" if log_path else ""
        raise SphereSfmError(f"SphereSfM {what} failed (exit {result.returncode}):\n{tail}{where}")


# -- TXT model reading -----------------------------------------------------


@dataclass
class TextCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]


@dataclass
class TextImage:
    image_id: int
    qvec: tuple[float, float, float, float]  # w, x, y, z -- cam_from_world rotation
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    points2d: list[tuple[float, float, int]] = field(default_factory=list)  # x, y, point3D_id (-1 = none)


@dataclass
class TextPoint:
    point_id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    track: list[tuple[int, int]]  # (image_id, point2D_idx)


@dataclass
class TextModel:
    cameras: dict[int, TextCamera]
    images: dict[int, TextImage]
    points: dict[int, TextPoint]


def _data_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def read_text_model(directory: Path) -> TextModel:
    """Parses COLMAP's documented TXT model format (cameras.txt,
    images.txt -- two lines per image -- and points3D.txt)."""
    directory = Path(directory)
    cameras = {}
    for line in _data_lines(directory / "cameras.txt"):
        parts = line.split()
        cameras[int(parts[0])] = TextCamera(int(parts[0]), parts[1], int(parts[2]), int(parts[3]), [float(p) for p in parts[4:]])

    images = {}
    lines = [line for line in (directory / "images.txt").read_text().splitlines() if not line.startswith("#")]
    for header, points_line in zip(lines[0::2], lines[1::2]):
        parts = header.split()
        if not parts:
            continue
        # The name is the last field and may contain spaces.
        match = re.match(r"^(\S+\s+){9}", header)
        name = header[match.end():] if match else parts[9]
        values = points_line.split()
        points2d = [
            (float(values[i]), float(values[i + 1]), int(values[i + 2])) for i in range(0, len(values) - 2, 3)
        ]
        images[int(parts[0])] = TextImage(
            image_id=int(parts[0]),
            qvec=tuple(float(v) for v in parts[1:5]),
            tvec=tuple(float(v) for v in parts[5:8]),
            camera_id=int(parts[8]),
            name=name,
            points2d=points2d,
        )

    points = {}
    for line in _data_lines(directory / "points3D.txt"):
        parts = line.split()
        track_values = [int(v) for v in parts[8:]]
        points[int(parts[0])] = TextPoint(
            point_id=int(parts[0]),
            xyz=tuple(float(v) for v in parts[1:4]),
            rgb=tuple(int(v) for v in parts[4:7]),
            error=float(parts[7]),
            track=list(zip(track_values[0::2], track_values[1::2])),
        )
    return TextModel(cameras, images, points)
