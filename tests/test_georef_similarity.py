"""vine360.georef.similarity -- Umeyama with scale, residuals, leave-one-out
(docs/adr/0047). Exact synthetic transforms, 1e-9 tolerances."""

import numpy as np
import pytest

from georef_synthetic import random_rotation
from vine360.georef.similarity import Similarity, leave_one_out, residuals, suspect_flags, umeyama


def _known(seed: int):
    rng = np.random.default_rng(seed)
    return Similarity(float(rng.uniform(0.05, 20.0)), random_rotation(rng), rng.normal(scale=100.0, size=3)), rng


@pytest.mark.parametrize("seed", range(5))
def test_recovers_a_random_similarity(seed):
    truth, rng = _known(seed)
    src = rng.normal(scale=10.0, size=(12, 3))
    fit = umeyama(src, truth.apply(src))
    assert fit.scale == pytest.approx(truth.scale, rel=1e-9)
    np.testing.assert_allclose(fit.rotation, truth.rotation, atol=1e-9)
    np.testing.assert_allclose(fit.translation, truth.translation, atol=1e-7)
    assert np.linalg.det(fit.rotation) == pytest.approx(1.0, abs=1e-12)


def test_minimum_three_points():
    truth, rng = _known(7)
    src = rng.normal(size=(3, 3))
    fit = umeyama(src, truth.apply(src))
    np.testing.assert_allclose(fit.apply(src), truth.apply(src), atol=1e-9)
    with pytest.raises(ValueError):
        umeyama(src[:2], truth.apply(src[:2]))


def test_reflected_target_still_gives_a_proper_rotation():
    rng = np.random.default_rng(3)
    src = rng.normal(size=(8, 3))
    dst = src * np.array([1.0, 1.0, -1.0])  # a mirror: no rotation reproduces it
    fit = umeyama(src, dst)
    assert np.linalg.det(fit.rotation) == pytest.approx(1.0, abs=1e-12)


def test_collinear_points_are_refused():
    src = np.outer(np.arange(5.0), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="collinear"):
        umeyama(src, src)


def test_residuals_zero_on_exact_data_and_split_horizontal_vertical():
    truth, rng = _known(11)
    src = rng.normal(size=(6, 3))
    dst = truth.apply(src)
    exact = residuals(truth, src, dst)
    assert exact.rmse_3d < 1e-9
    shifted = dst.copy()
    shifted[0] += [3.0, 4.0, -12.0]
    r = residuals(truth, src, shifted)
    assert r.horizontal[0] == pytest.approx(5.0)
    assert r.vertical[0] == pytest.approx(-12.0)
    assert r.d3[0] == pytest.approx(13.0)


def test_leave_one_out_flags_an_injected_outlier():
    truth, rng = _known(5)
    src = rng.normal(scale=20.0, size=(10, 3))
    dst = truth.apply(src) + rng.normal(scale=0.01, size=(10, 3))
    dst[4] += [0.0, 0.0, 2.0]
    loo = leave_one_out(src, dst)
    flags = suspect_flags(loo)
    assert flags[4] and flags.sum() == 1


def test_round_trips_through_dict():
    truth, _ = _known(2)
    again = Similarity.from_dict(truth.to_dict())
    np.testing.assert_allclose(again.rotation, truth.rotation)
    assert again.scale == truth.scale
