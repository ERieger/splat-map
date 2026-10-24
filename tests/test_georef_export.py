"""Export with and without a georeference solution (docs/adr/0047 and its
addendum): export only copies -- the "None" export is byte-identical before
and after a solve, and a georeferenced one is the materialised model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")

from georef_project import RUN_ID, build_scene
from vine360.export.postshot import PostshotExportError, _flatten_to_unique_filename, export_for_postshot, export_for_realityscan
from vine360.georef.solve import load_solution, solve_georef


def _tree(directory):
    return {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


@pytest.fixture
def scene(tmp_path):
    s = build_scene(tmp_path)
    yield s
    s.conn.close()


def test_none_export_is_byte_identical_before_and_after_a_solve(scene, tmp_path):
    before = tmp_path / "before"
    export_for_postshot(scene.conn, scene.root, before, run_id=RUN_ID)
    solve_georef(scene.conn, scene.root, RUN_ID, scene.set_id)
    after = tmp_path / "after"
    export_for_postshot(scene.conn, scene.root, after, run_id=RUN_ID)
    assert _tree(before) == _tree(after)
    assert json.loads((after / "export_info.json").read_text()) == {"run_id": RUN_ID, "georef_solution_id": "none"}
    assert not (after / "georef.json").exists()


def test_georeferenced_export_copies_the_materialised_model(scene, tmp_path):
    sid = solve_georef(scene.conn, scene.root, RUN_ID, scene.set_id)
    solution = load_solution(scene.conn, sid)
    out = tmp_path / "geo"
    export_for_postshot(scene.conn, scene.root, out, georef_solution_id=sid)
    info = json.loads((out / "export_info.json").read_text())
    assert info["georef_solution_id"] == sid and info["run_id"] == RUN_ID
    assert json.loads((out / "georef.json").read_text())["solution_id"] == sid
    assert (out / "georef.prj").exists()
    exported = pycolmap.Reconstruction(out / "sparse")
    materialised = pycolmap.Reconstruction(scene.root / solution["model_dir"])
    centers = {_flatten_to_unique_filename(Path(im.name)): im.projection_center() for im in materialised.images.values()}
    for image in exported.images.values():
        np.testing.assert_allclose(image.projection_center(), centers[image.name], atol=1e-9)
        assert np.all(np.abs(image.projection_center()) < 1000)  # local metres, never raw MGA

    # Re-exporting to the same folder with "None" drops the georef sidecars.
    export_for_postshot(scene.conn, scene.root, out, run_id=RUN_ID)
    assert not (out / "georef.json").exists() and not (out / "georef.prj").exists()


def test_solution_must_match_the_run(scene, tmp_path):
    sid = solve_georef(scene.conn, scene.root, RUN_ID, scene.set_id)
    with pytest.raises(PostshotExportError, match="belongs to"):
        export_for_postshot(scene.conn, scene.root, tmp_path / "x", run_id="sfm-other", georef_solution_id=sid)


def test_realityscan_export_in_the_local_frame(scene, tmp_path):
    sid = solve_georef(scene.conn, scene.root, RUN_ID, scene.set_id)
    out = tmp_path / "rs"
    result = export_for_realityscan(scene.conn, scene.root, out, include_camera_priors=True, georef_solution_id=sid)
    assert result.priors_path == out / "CameraPriors_local.csv" and result.num_priors == 24
    gcp = (out / "GroundControl_local.csv").read_text().splitlines()
    assert gcp[0] == "#name,x,y,alt" and len(gcp) == 15  # 14 points, base station left out
    name, x, y, alt = gcp[1].split(",")
    assert name == "020" and float(x) == pytest.approx(80.109) and float(y) == pytest.approx(117.946)
    assert json.loads((out / "export_info.json").read_text())["georef_solution_id"] == sid
