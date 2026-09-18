import numpy as np

from vine360.masking.semantics import (
    MaskBuildConfig,
    build_keep_and_exclude_masks,
    colmap_mask_path,
    combine_class_masks,
    dilate_exclude_mask,
    is_keep_fraction_anomalous,
    keep_fraction,
    keep_mask_from_exclude,
    load_mask,
    remove_small_components,
    save_mask,
)


def _mask(shape, true_slice):
    m = np.zeros(shape, dtype=np.uint8)
    m[true_slice] = 255
    return m


def test_combine_class_masks_ors_regions():
    person = _mask((10, 10), np.s_[0:3, 0:3])
    sky = _mask((10, 10), np.s_[7:10, 7:10])
    combined = combine_class_masks(person, sky)
    assert combined[1, 1] == 255
    assert combined[8, 8] == 255
    assert combined[5, 5] == 0


def test_combine_class_masks_requires_at_least_one():
    import pytest

    with pytest.raises(ValueError):
        combine_class_masks()


def test_remove_small_components_drops_small_keeps_large():
    mask = _mask((20, 20), np.s_[0:1, 0:1])  # 1px speck
    mask[10:18, 10:18] = 255  # 8x8 = 64px blob
    cleaned = remove_small_components(mask, min_area_px=10)
    assert cleaned[0, 0] == 0
    assert cleaned[14, 14] == 255


def test_dilate_exclude_mask_grows_region():
    mask = _mask((20, 20), np.s_[10:11, 10:11])
    dilated = dilate_exclude_mask(mask, dilation_px=2)
    assert dilated[10, 10] == 255
    assert dilated[8, 10] == 255  # grew by dilation_px
    assert dilated[0, 0] == 0


def test_dilate_zero_is_noop():
    mask = _mask((10, 10), np.s_[5:6, 5:6])
    assert np.array_equal(dilate_exclude_mask(mask, 0), mask)


def test_keep_mask_is_complement_of_exclude():
    exclude = _mask((5, 5), np.s_[0:2, 0:2])
    keep = keep_mask_from_exclude(exclude)
    assert keep[0, 0] == 0
    assert keep[3, 3] == 255


def test_keep_fraction_computes_ratio():
    keep = np.zeros((10, 10), dtype=np.uint8)
    keep[0:5, :] = 255  # half the image
    assert keep_fraction(keep) == 0.5


def test_build_keep_and_exclude_masks_pipeline():
    person = np.zeros((50, 50), dtype=np.uint8)
    person[40:42, 40:42] = 255  # tiny, isolated speck far from the sky region
    sky = np.zeros((50, 50), dtype=np.uint8)
    sky[0:20, :] = 255  # real region

    exclude, keep = build_keep_and_exclude_masks([person, sky], MaskBuildConfig(min_component_px=10, dilation_px=1))

    assert exclude[41, 41] == 0  # isolated speck removed before it could be dilated
    assert exclude[10, 10] == 255  # sky region present
    assert keep[45, 45] == 255  # well below sky and away from the speck, kept
    assert keep[10, 10] == 0


def test_is_keep_fraction_anomalous():
    assert is_keep_fraction_anomalous(0.1) is not None
    assert is_keep_fraction_anomalous(0.99) is not None
    assert is_keep_fraction_anomalous(0.7) is None


def test_colmap_mask_path_appends_png_to_full_name():
    from pathlib import Path

    path = colmap_mask_path("subdir/frame.jpg", Path("/project/sfm/masks"))
    assert path == Path("/project/sfm/masks/subdir/frame.jpg.png")


def test_save_and_load_mask_round_trip(tmp_path):
    mask = _mask((16, 16), np.s_[4:8, 4:8])
    path = tmp_path / "masks" / "keep.png"
    save_mask(mask, path)
    loaded = load_mask(path)
    assert np.array_equal(loaded, mask)
