"""vine360.mcp_server (docs/adr/0048): the tools, called in-process through
MCPServer.call_tool against real projects -- a fresh one, and the real
georeferencing scene of tests/georef_project.py, whose solve runs as a real
background job through vine360.operations."""

import json

import anyio
import pytest

pytest.importorskip("mcp")

from mcp.server.mcpserver.exceptions import ToolError

from vine360.activity_log import list_entries, start_entry
from vine360.mcp_server import ORIGIN, Vine360Session, build_server
from vine360.project import open_index_db


@pytest.fixture(autouse=True)
def _config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VINE360_CONFIG_DIR", str(tmp_path / "config"))  # recent projects, tag-size choice


def call(server, tool_name, /, **arguments):
    result = anyio.run(server.call_tool, tool_name, arguments)
    assert not result.is_error, result.content
    content = result.structured_content
    if content is None:  # a dict return comes back as JSON text only
        return json.loads(result.content[0].text)
    return content["result"] if set(content) == {"result"} else content


def test_every_tool_is_described_and_annotated():
    tools = {t.name: t for t in anyio.run(build_server().list_tools)}
    assert {"open_project", "project_status", "run_sfm", "solve_georef", "export", "wait_for_job"} <= set(tools)
    assert not {"delete_data", "remove_source"} & set(tools)  # deletion stays in the app's Data manager
    for tool in tools.values():
        assert tool.description and tool.annotations is not None, tool.name
    assert tools["project_status"].annotations.read_only_hint
    assert tools["extract_frames"].annotations.destructive_hint  # re-extracting replaces downstream data
    assert not tools["solve_georef"].annotations.destructive_hint


def test_tools_need_an_open_project():
    with pytest.raises(ToolError, match="No project is open"):
        anyio.run(build_server().call_tool, "project_status", {})


def test_create_project_opens_it_and_reports_status(tmp_path):
    server = build_server()
    status = call(server, "create_project", path=str(tmp_path / "p"), name="Block 7", capture_mode="360")
    assert status["name"] == "Block 7" and status["capture_mode"] == "360"
    assert status["stages"]["import"] == "active" and status["stages"]["frames"] == "pending"
    assert status["counts"]["sources"] == 0 and status["runs_in_progress"] == [] and status["current_job"] is None
    assert call(server, "list_sources") == [] and call(server, "list_sfm_runs") == []
    assert [p["name"] for p in call(server, "list_recent_projects")] == ["Block 7"]
    assert {o["key"] for o in call(server, "sfm_options")} >= {"matcher", "max_num_features"}


def test_argument_errors_come_back_as_tool_errors(tmp_path):
    server = build_server()
    call(server, "create_project", path=str(tmp_path / "p"), name="p")
    with pytest.raises(ToolError, match="poses mode needs run_id"):
        anyio.run(server.call_tool, "export", {"mode": "poses"})
    with pytest.raises(ToolError, match="no coordinates to georeference"):
        anyio.run(server.call_tool, "export", {"mode": "images", "georef_solution_id": "geo-1"})
    with pytest.raises(ToolError, match="give frame_set_ids"):
        anyio.run(server.call_tool, "run_sfm", {"engine": "spheresfm"})
    with pytest.raises(ToolError, match="No job"):
        anyio.run(server.call_tool, "get_job", {"job_id": "nope"})


@pytest.fixture
def scene(tmp_path):
    pytest.importorskip("pycolmap")
    from georef_project import build_scene

    scene = build_scene(tmp_path)
    scene.conn.close()  # the server opens its own connections
    return scene


def test_edits_and_a_solve_job_through_the_server(scene):
    server = build_server(Vine360Session(scene.root))
    [cp_set] = call(server, "list_control_point_sets")
    assert cp_set["set_id"] == scene.set_id and cp_set["epsg"] == 7854
    points = {p["name"]: p for p in call(server, "get_control_points", set_id=scene.set_id)}
    assert points["005"]["use"] == "check"
    call(server, "set_control_point_use", set_id=scene.set_id, name="024", use="check")

    job = call(server, "solve_georef", run_id=scene.run_id, set_id=scene.set_id)
    assert job["status"] == "running"
    done = call(server, "wait_for_job", job_id=job["job_id"], timeout_seconds=120)
    assert done["status"] == "done", done.get("error")
    solution_id = done["result"]["solution_id"]
    assert done["result"]["n_check"] == 2 and done["result"]["rmse_3d_m"] < 0.05
    assert done["summary"].startswith(solution_id) and done["recent_progress"]

    [run] = call(server, "list_sfm_runs")
    assert [s["solution_id"] for s in run["georef_solutions"]] == [solution_id]
    call(server, "set_active_solution", run_id=scene.run_id, solution_id=solution_id)
    assert call(server, "list_sfm_runs")[0]["active_solution"] == solution_id
    report = call(server, "get_georef_solution", solution_id=solution_id)
    assert "tag_positions" not in report and report["report"]["n_control"] >= 3

    conn = open_index_db(scene.root)
    try:
        logged = [(e.operation, e.origin, e.status) for e in list_entries(conn)]
    finally:
        conn.close()
    assert ("georef.edit_point", ORIGIN, "done") in logged and ("georef.solve", ORIGIN, "done") in logged
    assert call(server, "activity_log", limit=1, operation="georef.active")[0]["origin"] == ORIGIN


def test_a_failed_job_reports_its_error(scene):
    server = build_server(Vine360Session(scene.root))
    job = call(server, "solve_georef", run_id=scene.run_id, set_id="gcp-missing")
    done = call(server, "wait_for_job", job_id=job["job_id"], timeout_seconds=60)
    assert done["status"] == "failed" and "gcp-missing" in done["error"]
    assert [j["status"] for j in call(server, "list_jobs")] == ["failed"]


def test_refuses_to_start_while_another_run_is_in_progress(scene):
    conn = open_index_db(scene.root)
    try:
        start_entry(conn, "sfm.run", "Pose estimation", target="src1~i1", origin="manual")  # e.g. the app, mid-run
    finally:
        conn.close()
    server = build_server(Vine360Session(scene.root))
    assert call(server, "project_status")["runs_in_progress"][0]["operation"] == "sfm.run"
    with pytest.raises(ToolError, match="allow_concurrent"):
        anyio.run(server.call_tool, "solve_georef", {"run_id": scene.run_id, "set_id": scene.set_id})
    job = call(server, "solve_georef", run_id=scene.run_id, set_id=scene.set_id, allow_concurrent=True)
    assert call(server, "wait_for_job", job_id=job["job_id"], timeout_seconds=120)["status"] == "done"
