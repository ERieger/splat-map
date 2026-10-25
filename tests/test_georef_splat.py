"""vine360.georef.splat: georeferencing an already-trained splat
(docs/adr/0047) -- the frame check (right frame / already local /
unrelated frame), non-destructive output with its .georef.json sidecar, and
the override. Uses the real solution of tests/georef_project.py's scene."""

import json

import numpy as np
import pytest

pytest.importorskip("pycolmap")

from georef_project import RUN_ID, build_scene
from test_georef_apply import FIELDS, _read_ply, _write_ply
from vine360.georef.apply import GeorefApplyError
from vine360.georef.similarity import Similarity
from vine360.georef.solve import load_solution, solve_georef
from vine360.georef.splat import (
    CHECK_ALREADY_LOCAL,
    CHECK_MISMATCH,
    CHECK_OK,
    CHECK_SKIPPED,
    SplatFrameMismatch,
    check_splat_frame,
    georeference_splat,
    reference_points,
    sample_splat_positions,
)


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    scene = build_scene(tmp_path_factory.mktemp("splat"))
    solution_id = solve_georef(scene.conn, scene.root, RUN_ID, scene.set_id)
    solution = load_solution(scene.conn, solution_id)
    yield scene, solution
    scene.conn.close()


def _splat_around(points: np.ndarray, rng, n=600, jitter=0.5) -> np.ndarray:
    """A fake splat: Gaussians scattered around points, plus a little far background."""
    picks = points[rng.integers(0, len(points), n)] + rng.normal(scale=jitter, size=(n, 3))
    background = points.mean(axis=0) + rng.normal(scale=200.0, size=(n // 50, 3))
    data = np.zeros((len(picks) + len(background), len(FIELDS)))
    data[:, :3] = np.vstack([picks, background])
    data[:, FIELDS.index("rot_0")] = 1.0
    return data


def _model_frame(solution, local_points):
    sim = Similarity.from_dict(solution["transform"])
    return (local_points - sim.translation) @ sim.rotation / sim.scale


def test_frame_check_tells_the_three_cases_apart(solved):
    _scene, solution = solved
    sim = Similarity.from_dict(solution["transform"])
    reference = reference_points(_scene.root, solution)
    assert reference is not None and len(reference) >= 14
    rng = np.random.default_rng(1)
    in_model_frame = _model_frame(solution, reference)
    assert check_splat_frame(in_model_frame, sim, reference).status == CHECK_OK
    assert check_splat_frame(reference, sim, reference).status == CHECK_ALREADY_LOCAL
    elsewhere = rng.normal(scale=5.0, size=(500, 3)) + 1e4
    assert check_splat_frame(elsewhere, sim, reference).status == CHECK_MISMATCH
    assert check_splat_frame(in_model_frame, sim, None).status == CHECK_SKIPPED


def test_georeference_splat_writes_a_copy_and_sidecars(solved, tmp_path):
    scene, solution = solved
    rng = np.random.default_rng(2)
    reference = reference_points(scene.root, solution)
    src = tmp_path / "trained.ply"
    _write_ply(src, _splat_around(_model_frame(solution, reference), rng, jitter=0.5 / 0.42))
    before = src.read_bytes()
    out = tmp_path / "trained.georef.ply"
    messages = []
    result = georeference_splat(scene.conn, scene.root, src, out, solution["solution_id"],
                                progress_callback=lambda m, c, t: messages.append(m))
    assert result.check.status == CHECK_OK and result.count == len(_read_ply(src))
    assert src.read_bytes() == before  # the original is only read
    moved = sample_splat_positions(out)
    assert np.allclose(moved, Similarity.from_dict(solution["transform"]).apply(sample_splat_positions(src)), atol=1e-2)
    meta = json.loads((tmp_path / "trained.georef.georef.json").read_text())
    assert meta["solution_id"] == solution["solution_id"] and meta["epsg"] == 7854
    assert meta["source_splat"] == str(src.resolve()) and meta["frame_check"]["status"] == CHECK_OK
    assert (tmp_path / "trained.georef.georef.prj").exists()
    assert any("step 1/2" in m for m in messages) and any("step 2/2" in m for m in messages)


def test_mismatch_refuses_unless_forced_and_never_overwrites_input(solved, tmp_path):
    scene, solution = solved
    rng = np.random.default_rng(3)
    src = tmp_path / "local.ply"
    _write_ply(src, _splat_around(reference_points(scene.root, solution), rng))  # already local
    out = tmp_path / "out.ply"
    with pytest.raises(SplatFrameMismatch) as excinfo:
        georeference_splat(scene.conn, scene.root, src, out, solution["solution_id"])
    assert excinfo.value.check.status == CHECK_ALREADY_LOCAL and not out.exists()
    result = georeference_splat(scene.conn, scene.root, src, out, solution["solution_id"], force=True)
    assert out.exists() and json.loads((tmp_path / "out.georef.json").read_text())["frame_check"]["overridden"]
    assert result.check.status == CHECK_ALREADY_LOCAL
    with pytest.raises(GeorefApplyError, match="overwrite the input"):
        georeference_splat(scene.conn, scene.root, src, src, solution["solution_id"], force=True)
