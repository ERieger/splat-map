"""User-tunable SfM options (vine360.sfm.options, docs/adr/0040): the pure
SfmConfig/OPTION_SPECS logic -- serialization, per-engine filtering,
validation. Stdlib only; the engine side is test_sfm_options_engines.py."""

import json

import pytest

from vine360.sfm.options import (
    ALL_VARIANTS,
    GROUPS,
    OPTION_SPECS,
    PRESETS,
    SPECS_BY_KEY,
    VARIANT_EQUIRECT,
    VARIANT_PROJECTIONS,
    VARIANT_SPHERESFM,
    SfmConfig,
    describe,
    preset_config,
)


# -- pure ---------------------------------------------------------------------


def test_every_field_has_exactly_one_spec_and_every_spec_default_is_valid():
    fields = set(SfmConfig().to_dict())
    assert fields == set(SPECS_BY_KEY)
    assert len(OPTION_SPECS) == len(SPECS_BY_KEY)
    default = SfmConfig()
    for spec in OPTION_SPECS:
        value = getattr(default, spec.key)
        assert spec.group in GROUPS
        assert spec.help and spec.label
        assert set(spec.variants) <= set(ALL_VARIANTS)
        if spec.kind == "choice":
            assert value in dict(spec.choices), spec.key
        if spec.kind in ("int", "float"):
            assert spec.minimum <= value <= spec.maximum, spec.key
        for dependency, _values in spec.enabled_when:
            assert dependency in SPECS_BY_KEY


def test_defaults_are_the_behaviour_before_options_existed():
    config = SfmConfig()
    assert config.camera_model == "SIMPLE_RADIAL"
    assert config.sequential_overlap == 10
    assert config.matcher == "sequential" and config.mapper == "incremental"
    assert config.non_default() == {}
    assert describe(config) == ""


def test_from_dict_round_trips_and_tolerates_old_and_odd_input():
    config = SfmConfig(matcher="exhaustive", max_num_features=4096, upright=True, init_min_tri_angle=8.0)
    assert SfmConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    assert SfmConfig.from_dict(None) == SfmConfig()
    assert SfmConfig.from_dict({}) == SfmConfig()
    odd = SfmConfig.from_dict({"max_num_features": "2048", "upright": "true", "no_such_option": 1, "max_error": None})
    assert odd.max_num_features == 2048 and odd.upright is True and odd.max_error == SfmConfig().max_error


def test_for_variant_dictates_the_camera_and_drops_options_the_engine_cant_use():
    config = SfmConfig(camera_model="PINHOLE", mapper="global", refine_focal_length=False, max_num_features=2048)
    assert config.for_variant(VARIANT_PROJECTIONS).camera_model == "PINHOLE"
    equirect = config.for_variant(VARIANT_EQUIRECT)
    assert equirect.camera_model == "EQUIRECTANGULAR" and equirect.mapper == "global"
    sphere = config.for_variant(VARIANT_SPHERESFM)
    assert sphere.camera_model == "SPHERE"
    assert sphere.mapper == "incremental"  # SphereSfM has no global mapper
    assert sphere.refine_focal_length is True  # SPHERE intrinsics are never refined
    assert sphere.max_num_features == 2048  # applies everywhere


def test_validate_catches_what_would_fail_or_be_silently_ignored(tmp_path):
    assert SfmConfig().validate(VARIANT_PROJECTIONS) == []
    assert "global mapper" in SfmConfig(mapper="global").validate(VARIANT_SPHERESFM)[0]
    assert SfmConfig(mapper="global").validate(VARIANT_EQUIRECT) == []
    assert "needs a vocabulary tree" in SfmConfig(matcher="vocab_tree").validate(VARIANT_EQUIRECT)[0]
    assert "loop detection" in SfmConfig(loop_detection=True).validate(VARIANT_EQUIRECT)[0]
    assert "not found" in SfmConfig(matcher="vocab_tree", vocab_tree_path=str(tmp_path / "x.bin")).validate(VARIANT_EQUIRECT)[0]
    tree = tmp_path / "tree.bin"
    tree.write_bytes(b"\0")
    assert SfmConfig(matcher="vocab_tree", vocab_tree_path=str(tree)).validate(VARIANT_EQUIRECT) == []
    # Loop detection only matters with the sequential matcher.
    assert SfmConfig(matcher="exhaustive", loop_detection=True).validate(VARIANT_EQUIRECT) == []


def test_dependent_options_are_enabled_only_alongside_their_choice():
    assert SfmConfig().is_enabled("sequential_overlap")
    assert not SfmConfig(matcher="exhaustive").is_enabled("sequential_overlap")
    assert SfmConfig(matcher="exhaustive").is_enabled("exhaustive_block_size")
    assert not SfmConfig().is_enabled("loop_detection_period")
    assert SfmConfig(loop_detection=True).is_enabled("loop_detection_period")
    assert not SfmConfig(mapper="global").is_enabled("init_min_tri_angle")


def test_presets_and_describe():
    assert preset_config("Default") == SfmConfig()
    for name, changes in PRESETS.items():
        assert preset_config(name).non_default() == changes
    text = describe(SfmConfig(matcher="exhaustive", upright=True, max_image_size=1600))
    assert "matching strategy Exhaustive (every pair)" in text
    assert "upright features on" in text
    assert "max image size (px) 1600" in text
