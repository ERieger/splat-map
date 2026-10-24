"""vine360.georef.triangulate -- DLT + RANSAC over synthetic pinhole
cameras with known ground truth (docs/adr/0047)."""

import numpy as np
import pytest

from georef_synthetic import intrinsics, project, ring_cameras
from vine360.georef.triangulate import TagObservation, TagPosition, triangulate_tag

TAG_CENTER = np.array([1.0, -2.0, 0.5])
HALF = 0.2  # a 400 mm tag, lying flat


def _corners_world(center):
    return [center + np.array(d) for d in ([-HALF, -HALF, 0], [HALF, -HALF, 0], [HALF, HALF, 0], [-HALF, HALF, 0])]


def _observations(n=8, noise=0.0, seed=0, radius=8.0, height=6.0):
    rng = np.random.default_rng(seed)
    K = intrinsics()
    out = []
    for k, (R, t) in enumerate(ring_cameras(n, radius, height, TAG_CENTER)):
        center = project(K, R, t, TAG_CENTER) + rng.normal(scale=noise, size=2)
        corners = np.array([project(K, R, t, c) for c in _corners_world(TAG_CENTER)]) + rng.normal(scale=noise, size=(4, 2))
        out.append(TagObservation(f"v{k}", K, R, t, center, corners))
    return out


def test_exact_observations_recover_the_point():
    pos = triangulate_tag(20, _observations())
    assert pos.rejected is None
    np.testing.assert_allclose(pos.xyz, TAG_CENTER, atol=1e-9)
    assert pos.n_inliers == 8 and pos.rms_px < 1e-6
    assert pos.mean_edge_model == pytest.approx(2 * HALF, rel=1e-9)


def test_noisy_observations_recover_within_tolerance():
    pos = triangulate_tag(20, _observations(noise=0.5, seed=1))
    assert pos.rejected is None
    assert np.linalg.norm(pos.xyz - TAG_CENTER) < 0.02
    assert pos.rms_px < 1.5
    assert pos.mean_edge_model == pytest.approx(2 * HALF, rel=0.05)


def test_a_bad_observation_is_rejected_by_ransac():
    obs = _observations(noise=0.2, seed=2)
    obs[3].center = obs[3].center + np.array([60.0, -40.0])  # a misdetection
    pos = triangulate_tag(20, obs)
    assert pos.rejected is None
    assert "v3" not in pos.inlier_view_ids and pos.n_inliers == 7
    assert np.linalg.norm(pos.xyz - TAG_CENTER) < 0.01


def test_near_parallel_rays_are_rejected():
    # Every camera within ~1 m of each other, 60 m away: rays under 2 degrees apart.
    K = intrinsics()
    obs = []
    for k, (R, t) in enumerate(ring_cameras(6, 0.5, 60.0, TAG_CENTER)):
        obs.append(TagObservation(f"v{k}", K, R, t, project(K, R, t, TAG_CENTER)))
    pos = triangulate_tag(20, obs)
    assert pos.xyz is None and "ray angle" in pos.rejected


def test_too_few_views_are_rejected():
    pos = triangulate_tag(20, _observations(n=2))
    assert pos.xyz is None and "need 3" in pos.rejected
    one = triangulate_tag(20, _observations(n=1))
    assert one.xyz is None and "1 view" in one.rejected


def test_round_trips_through_dict():
    pos = triangulate_tag(20, _observations())
    again = TagPosition.from_dict(20, pos.to_dict())
    np.testing.assert_allclose(again.xyz, pos.xyz)
    assert again.inlier_view_ids == pos.inlier_view_ids
