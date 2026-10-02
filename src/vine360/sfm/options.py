"""User-tunable SfM options (docs/adr/0040): one flat, frozen SfmConfig
shared by both engines, plus a declarative OPTION_SPECS table that the GUI
builds its "Advanced options" form from.

Stdlib-only on purpose -- the GUI, the queue and the activity log build,
validate and serialize these without importing pycolmap. Each adapter maps
the fields onto its own engine: pycolmap's option objects
(vine360.sfm.colmap_adapter) or SphereSfM's `--Section.option` flags
(vine360.sfm.spheresfm_adapter). Every option's default is the engine
behaviour vine360 had before these options existed, so an SfmConfig()
(and an older sfm_runs row or queued job with no "options") runs exactly
as before.

Not every option applies to every engine variant (`OptionSpec.variants`):
SphereSfM always uses its fixed SPHERE camera and has no global mapper,
the camera model is only a choice for the six-face projections, and
COLMAP-native equirectangular frames use the EQUIRECTANGULAR model.
Options that only make sense alongside another choice (e.g. the sequential
overlap with the sequential matcher) declare it in `OptionSpec.enabled_when`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from pathlib import Path

# The GUI's engine choices -- an engine plus the images it runs on.
VARIANT_PROJECTIONS = "colmap_projections"
VARIANT_EQUIRECT = "colmap_equirectangular"
VARIANT_SPHERESFM = "spheresfm"
ALL_VARIANTS = (VARIANT_PROJECTIONS, VARIANT_EQUIRECT, VARIANT_SPHERESFM)
COLMAP_VARIANTS = (VARIANT_PROJECTIONS, VARIANT_EQUIRECT)

MATCHER_SEQUENTIAL = "sequential"
MATCHER_EXHAUSTIVE = "exhaustive"
MATCHER_VOCAB_TREE = "vocab_tree"
MAPPER_INCREMENTAL = "incremental"
MAPPER_GLOBAL = "global"


@dataclass(frozen=True)
class SfmConfig:
    # Camera
    camera_model: str = "SIMPLE_RADIAL"
    camera_mode: str = "auto"  # auto | single | per_folder | per_image
    refine_focal_length: bool = True
    refine_principal_point: bool = False
    refine_extra_params: bool = True
    # Features
    use_gpu: bool = True  # only takes effect on a CUDA build
    max_image_size: int = 0  # 0 = engine default
    max_num_features: int = 8192
    peak_threshold: float = 0.0066666666666666671
    edge_threshold: float = 10.0
    upright: bool = False
    estimate_affine_shape: bool = False
    domain_size_pooling: bool = False
    # Matching
    matcher: str = MATCHER_SEQUENTIAL
    sequential_overlap: int = 10  # images matched forward/back in sequence
    quadratic_overlap: bool = True
    loop_detection: bool = False
    loop_detection_period: int = 10
    loop_detection_num_images: int = 50
    exhaustive_block_size: int = 50
    vocab_tree_num_images: int = 100
    vocab_tree_path: str = ""
    max_ratio: float = 0.8
    max_distance: float = 0.7
    cross_check: bool = True
    max_num_matches: int = 0  # 0 = engine default
    guided_matching: bool = False
    min_num_inliers: int = 15
    max_error: float = 4.0
    min_inlier_ratio: float = 0.25
    # Mapping
    mapper: str = MAPPER_INCREMENTAL
    min_num_matches: int = 15
    multiple_models: bool = True
    min_model_size: int = 0  # 0 = engine default (incremental 10, global 3)
    init_min_num_inliers: int = 100
    init_min_tri_angle: float = 16.0
    init_max_forward_motion: float = 0.95
    abs_pose_min_num_inliers: int = 30
    abs_pose_min_inlier_ratio: float = 0.25
    filter_max_reproj_error: float = 4.0
    filter_min_tri_angle: float = 1.5
    random_seed: int = -1  # -1 = not fixed
    # Quality checks (warnings after the run, not engine settings)
    min_registered_ratio: float = 0.7
    max_mean_reprojection_error: float = 2.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def non_default(self) -> dict:
        """Only the fields that differ from SfmConfig() -- what a summary
        or a queue job label needs to show."""
        default = SfmConfig()
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) != getattr(default, f.name)}

    @classmethod
    def from_dict(cls, data: dict | None) -> SfmConfig:
        """Tolerant of anything a stored row or queued job may hold: missing
        keys take their defaults, unknown keys (from a newer/older version)
        are ignored, and values are coerced to the field's type."""
        if not data:
            return cls()
        kwargs = {}
        for f in fields(cls):
            if f.name in data and data[f.name] is not None:
                kwargs[f.name] = _coerce(f.type, data[f.name])
        return cls(**kwargs)

    def replace(self, **changes) -> SfmConfig:
        return dataclasses.replace(self, **changes)

    def for_variant(self, variant: str) -> SfmConfig:
        """Returns a copy with the camera model the variant dictates (the
        six-face projections keep the chosen one) and every option the
        variant can't use reset to its default -- so a run's recorded
        options never claim something that wasn't applied."""
        default = SfmConfig()
        changes = {}
        for spec in OPTION_SPECS:
            if variant not in spec.variants and getattr(self, spec.key) != getattr(default, spec.key):
                changes[spec.key] = getattr(default, spec.key)
        if variant == VARIANT_EQUIRECT:
            changes["camera_model"] = "EQUIRECTANGULAR"
        elif variant == VARIANT_SPHERESFM:
            changes["camera_model"] = "SPHERE"
        return dataclasses.replace(self, **changes)

    def validate(self, variant: str) -> list[str]:
        """Problems that would make the run fail or silently ignore an
        option; empty if it's runnable as configured."""
        problems = []
        if variant == VARIANT_SPHERESFM and self.mapper == MAPPER_GLOBAL:
            problems.append("SphereSfM has no global mapper -- choose the incremental mapper")
        needs_tree = self.matcher == MATCHER_VOCAB_TREE or (self.matcher == MATCHER_SEQUENTIAL and self.loop_detection)
        if needs_tree:
            what = "vocabulary-tree matching" if self.matcher == MATCHER_VOCAB_TREE else "loop detection"
            if not self.vocab_tree_path:
                problems.append(f"{what} needs a vocabulary tree file")
            elif not Path(self.vocab_tree_path).expanduser().is_file():
                problems.append(f"vocabulary tree file not found: {self.vocab_tree_path}")
        if self.min_inlier_ratio > 1 or self.abs_pose_min_inlier_ratio > 1:
            problems.append("inlier ratios must be between 0 and 1")
        return problems

    def is_enabled(self, key: str) -> bool:
        """Whether the option `key` takes effect given the other choices
        (OptionSpec.enabled_when) -- e.g. the sequential overlap only with
        the sequential matcher."""
        spec = SPECS_BY_KEY[key]
        return all(getattr(self, dep) in values for dep, values in spec.enabled_when)


def _coerce(type_name, value):
    type_name = type_name if isinstance(type_name, str) else type_name.__name__
    if type_name == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if type_name == "int":
        return int(value)
    if type_name == "float":
        return float(value)
    return str(value)


@dataclass(frozen=True)
class OptionSpec:
    """One field of SfmConfig as the GUI shows it. kind is int/float/bool/
    choice/path; choices are (value, label) pairs; special_value_text is
    shown for the minimum value of a number (e.g. "Engine default" for 0)."""

    key: str
    group: str
    label: str
    kind: str
    help: str
    variants: tuple[str, ...] = ALL_VARIANTS
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    decimals: int = 2
    choices: tuple[tuple[str, str], ...] = ()
    special_value_text: str = ""
    enabled_when: tuple[tuple[str, tuple], ...] = ()


GROUP_CAMERA = "Camera"
GROUP_FEATURES = "Features"
GROUP_MATCHING = "Matching"
GROUP_MAPPING = "Mapping"
GROUP_QUALITY = "Quality checks"
GROUPS = (GROUP_CAMERA, GROUP_FEATURES, GROUP_MATCHING, GROUP_MAPPING, GROUP_QUALITY)

_INCREMENTAL = (("mapper", (MAPPER_INCREMENTAL,)),)
_SEQUENTIAL = (("matcher", (MATCHER_SEQUENTIAL,)),)

OPTION_SPECS: tuple[OptionSpec, ...] = (
    # -- Camera -----------------------------------------------------------
    OptionSpec(
        "camera_model", GROUP_CAMERA, "Camera model", "choice",
        "COLMAP camera model for the six-face projections. They're rendered as ideal pinhole views, so "
        "SIMPLE_PINHOLE/PINHOLE fit them exactly; SIMPLE_RADIAL (the default) also refines a little distortion. "
        "Raw 360 frames always use EQUIRECTANGULAR (COLMAP) or SPHERE (SphereSfM).",
        variants=(VARIANT_PROJECTIONS,),
        choices=(
            ("SIMPLE_PINHOLE", "SIMPLE_PINHOLE (f, cx, cy)"),
            ("PINHOLE", "PINHOLE (fx, fy, cx, cy)"),
            ("SIMPLE_RADIAL", "SIMPLE_RADIAL (f, cx, cy, k)"),
            ("RADIAL", "RADIAL (f, cx, cy, k1, k2)"),
            ("OPENCV", "OPENCV (fx, fy, cx, cy, k1, k2, p1, p2)"),
        ),
    ),
    OptionSpec(
        "camera_mode", GROUP_CAMERA, "Shared intrinsics", "choice",
        "Which images share one set of camera intrinsics. \"Single camera\" (every image identical) suits "
        "frames from one 360 camera and the six projected faces (all rendered at the same size and field of "
        "view). \"Per folder\" gives each folder its own camera. Automatic is COLMAP's default; for SphereSfM it "
        "means single camera, which is what vine360 has always used there.",
        choices=(
            ("auto", "Automatic (engine default)"),
            ("single", "Single camera for all images"),
            ("per_folder", "One camera per folder"),
            ("per_image", "One camera per image"),
        ),
    ),
    OptionSpec(
        "refine_focal_length", GROUP_CAMERA, "Refine focal length", "bool",
        "Let bundle adjustment refine the focal length. SphereSfM's SPHERE camera has nothing to refine.",
        variants=COLMAP_VARIANTS,
    ),
    OptionSpec(
        "refine_principal_point", GROUP_CAMERA, "Refine principal point", "bool",
        "Let bundle adjustment move the principal point (rarely helps; COLMAP leaves it fixed by default).",
        variants=COLMAP_VARIANTS,
    ),
    OptionSpec(
        "refine_extra_params", GROUP_CAMERA, "Refine distortion", "bool",
        "Let bundle adjustment refine the camera model's extra (distortion) parameters.",
        variants=COLMAP_VARIANTS,
    ),
    # -- Features ---------------------------------------------------------
    OptionSpec(
        "use_gpu", GROUP_FEATURES, "Use the GPU", "bool",
        "GPU SIFT feature extraction and matching. Only takes effect on a CUDA build (pycolmap-cuda12 / a "
        "CUDA SphereSfM build); otherwise the CPU is used regardless.",
    ),
    OptionSpec(
        "max_image_size", GROUP_FEATURES, "Max image size (px)", "int",
        "Images larger than this (longest side) are downscaled before feature extraction. The engines' default "
        "is 3200 px -- a 5.7K equirectangular frame is halved. Raise it to keep fine detail (slower, more memory).",
        minimum=0, maximum=16384, step=256, special_value_text="Engine default (3200)",
    ),
    OptionSpec(
        "max_num_features", GROUP_FEATURES, "Max features per image", "int",
        "Upper limit on SIFT keypoints per image. More helps low-texture scenes at the cost of time.",
        minimum=256, maximum=65536, step=1024,
    ),
    OptionSpec(
        "peak_threshold", GROUP_FEATURES, "Peak threshold", "float",
        "SIFT detection threshold -- lower finds more (weaker) features in low-contrast areas such as sky, "
        "shade or uniform canopy.",
        minimum=0.0005, maximum=0.05, step=0.0005, decimals=4,
    ),
    OptionSpec(
        "edge_threshold", GROUP_FEATURES, "Edge threshold", "float",
        "SIFT edge-response rejection -- higher keeps more edge-like features (wires, posts).",
        minimum=1.0, maximum=50.0, step=1.0, decimals=1,
    ),
    OptionSpec(
        "upright", GROUP_FEATURES, "Upright features", "bool",
        "Skip orientation estimation (one upright descriptor per keypoint). Faster and can match better when "
        "the camera never rolls -- not suitable for handheld 360 capture that tilts.",
    ),
    OptionSpec(
        "estimate_affine_shape", GROUP_FEATURES, "Estimate affine shape", "bool",
        "Affine-covariant SIFT -- more robust to strong viewpoint change. CPU only: COLMAP falls back to CPU "
        "extraction when this is on (much slower).",
    ),
    OptionSpec(
        "domain_size_pooling", GROUP_FEATURES, "Domain-size pooling", "bool",
        "DSP-SIFT descriptors -- often better matching. CPU only: COLMAP falls back to CPU extraction when this "
        "is on (much slower).",
    ),
    # -- Matching ---------------------------------------------------------
    OptionSpec(
        "matcher", GROUP_MATCHING, "Matching strategy", "choice",
        "Which image pairs are matched. Sequential (default) matches each frame to its neighbours in capture "
        "order -- right for video. Exhaustive matches every pair (quadratic cost: fine for a few hundred "
        "images). Vocabulary tree finds visually similar pairs anywhere in the set (needs a tree file).",
        choices=(
            (MATCHER_SEQUENTIAL, "Sequential (video order)"),
            (MATCHER_EXHAUSTIVE, "Exhaustive (every pair)"),
            (MATCHER_VOCAB_TREE, "Vocabulary tree (similar images)"),
        ),
    ),
    OptionSpec(
        "sequential_overlap", GROUP_MATCHING, "Sequential overlap", "int",
        "How many neighbouring images (in order) each image is matched against.",
        minimum=1, maximum=200, enabled_when=_SEQUENTIAL,
    ),
    OptionSpec(
        "quadratic_overlap", GROUP_MATCHING, "Quadratic overlap", "bool",
        "Also match images at quadratically growing distances (i+1, i+2, i+4, i+8...) for longer-range links.",
        enabled_when=_SEQUENTIAL,
    ),
    OptionSpec(
        "loop_detection", GROUP_MATCHING, "Loop detection", "bool",
        "Periodically match against visually similar images anywhere in the sequence, closing loops when a "
        "row is walked up and back. Needs a vocabulary tree file.",
        enabled_when=_SEQUENTIAL,
    ),
    OptionSpec(
        "loop_detection_period", GROUP_MATCHING, "Loop detection period", "int",
        "Run loop detection every N images.",
        minimum=1, maximum=1000, enabled_when=(("matcher", (MATCHER_SEQUENTIAL,)), ("loop_detection", (True,))),
    ),
    OptionSpec(
        "loop_detection_num_images", GROUP_MATCHING, "Loop detection candidates", "int",
        "How many similar images each loop-detection query is matched against.",
        minimum=1, maximum=1000, enabled_when=(("matcher", (MATCHER_SEQUENTIAL,)), ("loop_detection", (True,))),
    ),
    OptionSpec(
        "exhaustive_block_size", GROUP_MATCHING, "Exhaustive block size", "int",
        "Images per block for exhaustive matching (memory/throughput trade-off only).",
        minimum=1, maximum=1000, enabled_when=(("matcher", (MATCHER_EXHAUSTIVE,)),),
    ),
    OptionSpec(
        "vocab_tree_num_images", GROUP_MATCHING, "Vocab tree candidates", "int",
        "How many visually similar images each image is matched against.",
        minimum=1, maximum=1000, enabled_when=(("matcher", (MATCHER_VOCAB_TREE,)),),
    ),
    OptionSpec(
        "vocab_tree_path", GROUP_MATCHING, "Vocabulary tree file", "path",
        "Pre-trained vocabulary tree, from https://demuc.de/colmap/ or COLMAP's GitHub releases. The engines need "
        "different formats: pycolmap 4.x reads the FAISS trees (vocab_tree_faiss_*.bin); SphereSfM (COLMAP "
        "3.8) reads the older FLANN ones (vocab_tree_flickr100K_words*.bin).",
        enabled_when=(("matcher", (MATCHER_VOCAB_TREE, MATCHER_SEQUENTIAL)),),
    ),
    OptionSpec(
        "max_ratio", GROUP_MATCHING, "Ratio test", "float",
        "Lowe's ratio test -- lower is stricter (fewer, more reliable matches).",
        minimum=0.5, maximum=1.0, step=0.05,
    ),
    OptionSpec(
        "max_distance", GROUP_MATCHING, "Max descriptor distance", "float",
        "Largest descriptor distance accepted as a match -- lower is stricter.",
        minimum=0.1, maximum=1.0, step=0.05,
    ),
    OptionSpec(
        "cross_check", GROUP_MATCHING, "Cross-check", "bool",
        "Keep only matches that are each other's best match in both directions.",
    ),
    OptionSpec(
        "max_num_matches", GROUP_MATCHING, "Max matches per pair", "int",
        "Cap on raw matches per image pair (pycolmap's default is 32768, SphereSfM's 8192). GPU matching "
        "memory grows with this.",
        minimum=0, maximum=131072, step=1024, special_value_text="Engine default",
    ),
    OptionSpec(
        "guided_matching", GROUP_MATCHING, "Guided matching", "bool",
        "After geometric verification, re-match using the estimated geometry -- more matches, slower.",
    ),
    OptionSpec(
        "min_num_inliers", GROUP_MATCHING, "Min inliers per pair", "int",
        "Geometric verification: an image pair needs at least this many inlier matches to count.",
        minimum=4, maximum=1000,
    ),
    OptionSpec(
        "max_error", GROUP_MATCHING, "Verification max error (px)", "float",
        "RANSAC inlier threshold for two-view geometric verification.",
        minimum=0.5, maximum=32.0, step=0.5, decimals=1,
    ),
    OptionSpec(
        "min_inlier_ratio", GROUP_MATCHING, "Verification min inlier ratio", "float",
        "RANSAC minimum inlier ratio for two-view geometric verification.",
        minimum=0.0, maximum=1.0, step=0.05,
    ),
    # -- Mapping ----------------------------------------------------------
    OptionSpec(
        "mapper", GROUP_MAPPING, "Mapper", "choice",
        "Incremental (default) registers images one at a time -- robust, slower on large sets. Global (GLOMAP, "
        "built into pycolmap 4.x) solves all rotations and then positions at once -- much faster on large "
        "sets, but leans on good focal-length priors and reports progress per stage only. Not available for "
        "SphereSfM.",
        choices=((MAPPER_INCREMENTAL, "Incremental"), (MAPPER_GLOBAL, "Global (GLOMAP)")),
        variants=COLMAP_VARIANTS,
    ),
    OptionSpec(
        "min_num_matches", GROUP_MAPPING, "Min matches per pair", "int",
        "Image pairs with fewer verified matches are ignored by the mapper.",
        minimum=4, maximum=1000,
    ),
    OptionSpec(
        "multiple_models", GROUP_MAPPING, "Allow multiple models", "bool",
        "Keep reconstructing after the first model stops growing, producing separate models for disconnected "
        "parts. vine360 keeps the largest one either way.",
    ),
    OptionSpec(
        "min_model_size", GROUP_MAPPING, "Min model size (images)", "int",
        "Discard reconstructions with fewer registered images. The engine default is 10 for the incremental "
        "mapper and 3 for the global one.",
        minimum=0, maximum=10000, special_value_text="Engine default",
    ),
    OptionSpec(
        "init_min_num_inliers", GROUP_MAPPING, "Initial pair min inliers", "int",
        "The first image pair needs at least this many inliers.",
        minimum=10, maximum=10000, enabled_when=_INCREMENTAL,
    ),
    OptionSpec(
        "init_min_tri_angle", GROUP_MAPPING, "Initial pair min angle (°)", "float",
        "Minimum triangulation angle for the first image pair. Lower it if a slow-moving capture fails to find "
        "an initial pair; too low risks a poor start.",
        minimum=1.0, maximum=60.0, step=1.0, decimals=1, enabled_when=_INCREMENTAL,
    ),
    OptionSpec(
        "init_max_forward_motion", GROUP_MAPPING, "Initial pair max forward motion", "float",
        "Rejects initial pairs whose motion is mostly straight ahead (poorly conditioned). Walking down a row "
        "is mostly forward motion; raising this towards 1 accepts more such pairs.",
        minimum=0.5, maximum=1.0, step=0.01, enabled_when=_INCREMENTAL,
    ),
    OptionSpec(
        "abs_pose_min_num_inliers", GROUP_MAPPING, "Registration min inliers", "int",
        "An image needs at least this many 2D-3D inliers to be registered.",
        minimum=6, maximum=1000, enabled_when=_INCREMENTAL,
    ),
    OptionSpec(
        "abs_pose_min_inlier_ratio", GROUP_MAPPING, "Registration min inlier ratio", "float",
        "An image needs at least this inlier ratio to be registered.",
        minimum=0.0, maximum=1.0, step=0.05, enabled_when=_INCREMENTAL,
    ),
    OptionSpec(
        "filter_max_reproj_error", GROUP_MAPPING, "Filter max reprojection error (px)", "float",
        "Observations with a larger reprojection error are removed during mapping.",
        minimum=0.5, maximum=32.0, step=0.5, decimals=1, enabled_when=_INCREMENTAL,
    ),
    OptionSpec(
        "filter_min_tri_angle", GROUP_MAPPING, "Filter min triangulation angle (°)", "float",
        "3D points seen under a smaller angle are removed during mapping.",
        minimum=0.1, maximum=20.0, step=0.1, decimals=1, enabled_when=_INCREMENTAL,
    ),
    OptionSpec(
        "random_seed", GROUP_MAPPING, "Random seed", "int",
        "Fix RANSAC/mapping randomness so a run can be repeated exactly. -1 leaves it unfixed.",
        minimum=-1, maximum=2_147_483_647, special_value_text="Not fixed",
    ),
    # -- Quality checks ---------------------------------------------------
    OptionSpec(
        "min_registered_ratio", GROUP_QUALITY, "Warn below registered ratio", "float",
        "Warn after the run if fewer than this fraction of images registered.",
        minimum=0.0, maximum=1.0, step=0.05,
    ),
    OptionSpec(
        "max_mean_reprojection_error", GROUP_QUALITY, "Warn above mean reprojection error (px)", "float",
        "Warn after the run if the selected model's mean reprojection error is higher than this.",
        minimum=0.1, maximum=20.0, step=0.1, decimals=1,
    ),
)

SPECS_BY_KEY = {spec.key: spec for spec in OPTION_SPECS}

# Starting points, applied on top of the defaults. Only speed/thoroughness
# trade-offs that follow directly from what each option does -- nothing here
# is tuned on real vineyard footage yet.
PRESETS: dict[str, dict] = {
    "Default": {},
    "Fast preview": {
        "max_image_size": 1600,
        "max_num_features": 4096,
        "sequential_overlap": 5,
        "quadratic_overlap": False,
    },
    "Thorough": {
        "max_image_size": 6400,
        "max_num_features": 16384,
        "sequential_overlap": 20,
        "guided_matching": True,
    },
}


def preset_config(name: str) -> SfmConfig:
    return SfmConfig(**PRESETS[name])


def describe(config: SfmConfig) -> str:
    """Short human-readable list of the non-default options, for queue job
    labels and run summaries ("" when everything is default)."""
    parts = []
    for key, value in config.non_default().items():
        spec = SPECS_BY_KEY.get(key)
        if spec is None:
            continue
        if spec.kind == "choice":
            value = dict(spec.choices).get(value, value)
        elif spec.kind == "path":
            value = Path(value).name or value
        elif spec.kind == "bool":
            value = "on" if value else "off"
        parts.append(f"{spec.label.lower()} {value}")
    return ", ".join(parts)
