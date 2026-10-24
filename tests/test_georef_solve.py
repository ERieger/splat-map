"""vine360.georef.solve end to end on a synthetic scene with known ground
truth (tests/georef_project.py), including the non-destructive rules of the
ADR 0047 addendum: every solve is a new row and model folder, the run's own
model is never touched, deleting a solution touches nothing else."""

import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("pycolmap")

from georef_project import RUN_ID, build_scene
from vine360.georef.control_points import USE_CONTROL, USE_IGNORE, update_control_point
from vine360.georef.solve import (
    GeorefSolveError,
    active_solution,
    delete_solution,
    list_solutions,
    load_solution,
    set_active_solution,
    solve_georef,
    write_report_csv,
)
from vine360.georef.similarity import Similarity


def _checksums(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.rglob("*")) if p.is_file()}


@pytest.fixture
def scene(tmp_path):
    s = build_scene(tmp_path)
    yield s
    s.conn.close()


def test_recovers_the_known_transform(scene):
    progress = []
    solution_id = solve_georef(scene.conn, scene.root, RUN_ID, scene.set_id, progress_callback=lambda *a: progress.append(a))
    solution = load_solution(scene.conn, solution_id)
    fit = Similarity.from_dict(solution["transform"])
    assert fit.scale == pytest.approx(scene.truth.scale, rel=2e-3)
    np.testing.assert_allclose(fit.rotation, scene.truth.rotation, atol=2e-3)
    report = solution["report"]
    assert report["n_control"] == 13 and report["n_check"] == 1  # 14 tags, 005 is a check point
    assert report["rmse"]["d3"] < 0.05 and report["check_rmse"]["d3"] < 0.05
    assert [i["name"] for i in report["ignored"]] == ["PRS778941614462"]
    assert report["corner_scale_n"] == 14 and abs(report["corner_scale_diff"]) < 0.03  # per-point 250/400 mm sizes
    assert not report["seen_not_in_table"] and not report["not_triangulated"]
    assert {r["name"] for r in report["points"]} >= {"000", "005", "020"}
    assert any("step 2/4" in m for m, *_ in progress) and progress[-1][0].startswith("Georeference done")


def test_diagnostics_when_too_few_control_points(scene):
    for name in ("000", "001", "002", "003", "004", "020", "021", "022", "023", "024", "025"):
        update_control_point(scene.conn, scene.set_id, name, use=USE_IGNORE)
    update_control_point(scene.conn, scene.set_id, "026", tag_id=None)
    with pytest.raises(GeorefSolveError) as info:
        solve_georef(scene.conn, scene.root, RUN_ID, scene.set_id)
    message = str(info.value)
    assert "only 1 control point" in message
    assert "not in the control table" in message and "26" in message  # seen but now untagged
    assert "no tag id: 026" in message


def test_solving_twice_is_non_destructive(scene):
    model_dir = scene.root / "sfm" / "sparse" / RUN_ID
    before = _checksums(model_dir)
    first = solve_georef(scene.conn, scene.root, RUN_ID, scene.set_id)
    second = solve_georef(scene.conn, scene.root, RUN_ID, scene.set_id)
    assert first != second
    assert [s["solution_id"] for s in list_solutions(scene.conn, RUN_ID)] == [second, first]
    for sid in (first, second):
        out = scene.root / load_solution(scene.conn, sid)["model_dir"]
        assert out == scene.root / "sfm" / "georef" / RUN_ID / sid
        assert {"cameras.bin", "images.bin", "points3D.bin", "georef.json", "georef.prj"} <= {p.name for p in out.iterdir()}
        meta = json.loads((out / "georef.json").read_text())
        assert meta["solution_id"] == sid and meta["origin"] == [298900.0, 6126100.0, 0.0] and meta["epsg"] == 7854
    assert _checksums(model_dir) == before

    set_active_solution(scene.root, RUN_ID, first)
    delete_solution(scene.conn, scene.root, first)
    assert [s["solution_id"] for s in list_solutions(scene.conn, RUN_ID)] == [second]
    assert not (scene.root / "sfm" / "georef" / RUN_ID / first).exists()
    assert (scene.root / "sfm" / "georef" / RUN_ID / second / "images.bin").exists()
    assert active_solution(scene.root, RUN_ID) is None
    assert scene.conn.execute("SELECT COUNT(*) FROM sfm_runs").fetchone()[0] == 1
    assert _checksums(model_dir) == before


def test_materialised_model_is_in_local_metres(scene):
    import pycolmap

    sid = solve_georef(scene.conn, scene.root, RUN_ID, scene.set_id)
    solution = load_solution(scene.conn, sid)
    original = pycolmap.Reconstruction(scene.root / "sfm" / "sparse" / RUN_ID / "0")
    moved = pycolmap.Reconstruction(scene.root / solution["model_dir"])
    fit = Similarity.from_dict(solution["transform"])
    for image_id, image in original.images.items():
        np.testing.assert_allclose(
            moved.images[image_id].projection_center(), fit.apply(image.projection_center()[None])[0], atol=1e-6
        )
    # The tag points land on the surveyed coordinates minus the origin.
    for point in moved.points3D.values():
        assert np.all(np.abs(point.xyz) < 1000)  # local metres, not raw MGA


def test_active_solution_lives_in_project_yaml(scene):
    sid = solve_georef(scene.conn, scene.root, RUN_ID, scene.set_id)
    set_active_solution(scene.root, RUN_ID, sid)
    assert active_solution(scene.root, RUN_ID) == sid
    assert "active_solution" in (scene.root / "project.yaml").read_text()
    set_active_solution(scene.root, RUN_ID, None)
    assert active_solution(scene.root, RUN_ID) is None
    assert "active_solution" not in (scene.root / "project.yaml").read_text()


def test_report_csv(scene, tmp_path):
    sid = solve_georef(scene.conn, scene.root, RUN_ID, scene.set_id)
    path = tmp_path / "report.csv"
    write_report_csv(load_solution(scene.conn, sid), path)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("name,tag_id,use,views") and len(lines) == 15


def test_deleting_the_run_removes_its_solutions(scene):
    from vine360.data_manager import delete_sfm_run

    sid = solve_georef(scene.conn, scene.root, RUN_ID, scene.set_id)
    delete_sfm_run(scene.conn, scene.root, RUN_ID)
    assert scene.conn.execute("SELECT COUNT(*) FROM georef_solutions").fetchone()[0] == 0
    assert not (scene.root / "sfm" / "georef" / RUN_ID).exists()
    assert sid
