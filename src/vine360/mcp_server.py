"""An MCP server for vine360 (docs/adr/0048): lets an MCP client (Claude
Code, Claude Desktop, ...) inspect a project and drive the whole pipeline --
add sources, extract frames, project, mask, run SfM, georeference, export --
through the same worker functions the desktop app and its queue run
(vine360.operations). Nothing here re-implements a stage.

Run over stdio: ``python -m vine360.mcp_server [--project PATH]`` (or
``./run-mcp.sh``; the repo's .mcp.json registers it with Claude Code).

Verified against the installed SDK: ``mcp`` 2.3.0 (``mcp.server.mcpserver.
MCPServer``, the 2.x rename of FastMCP), whose synchronous tools run on a
worker thread (``anyio.to_thread``) and whose ``ToolError`` reaches the
model as the tool's error text, and whose stdio transport points fd 1 at
stderr while serving -- so a library, a native extension (pycolmap's
logging) or a subprocess writing to stdout can't corrupt the protocol
stream. Exercised in-process by tests/
test_mcp_server.py through ``MCPServer.call_tool``; not yet through a real
client session.

Jobs. A long operation starts as a background job and the tool returns at
once with its job id; ``get_job`` / ``wait_for_job`` report its counted
progress (the same progress_callback messages the GUI's progress bar
shows). One job runs at a time per server, and a job can't be cancelled
once started -- the workers have no cancellation hook, same as in the app.
Each job is logged in the project's Activity log with origin "mcp".

Sharing a project with the app. The server opens the project's
index.sqlite on its own connections, like the app's workers do, so it can
run while the app is open. What it does refuse, unless the caller passes
``allow_concurrent=True``, is to start a job while the Activity log shows
another run still in progress (the app's, its queue's or another server's):
two workers writing one project at once is the race the app's own queue
lock exists to prevent (docs/adr/0023). An entry left "running" by an app
session that crashed shows the same way until the app next opens the
project (vine360.activity_log.mark_interrupted). The app doesn't watch for
outside changes, so after a job here its panels show the new data once
they're refreshed (switch tabs or reopen the project).

Not exposed: deleting data (the Data manager's job, and not undoable) and
removing a source. Re-running a stage on the same frame set does replace
that frame set's downstream data, exactly as in the app (the cascade-delete
invariant), so those tools say so.

When the stdio session ends (the client exits), a running job is allowed
to finish before the process exits: job threads aren't daemons.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import sqlite3
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from vine360 import operations
from vine360.activity_log import ORIGIN_KWARG, STATUS_RUNNING, list_entries, log_action
from vine360.config import CaptureMode
from vine360.project import load_project, open_index_db
from vine360.recent_projects import load_recent_projects, record_recent_project
from vine360.stage_status import STAGE_NAMES, compute_stage_statuses

ORIGIN = "mcp"
JOB_LOG_LINES = 200
MAX_WAIT_SECONDS = 600

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
EDIT = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
JOB = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
# A stage re-run replaces that frame set's downstream data (cascade delete).
REPLACING_JOB = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False)

SFM_ENGINES = ("colmap_projections", "colmap_equirectangular", "spheresfm")
EXPORT_MODES = ("poses", "images", "realityscan")
_EXPORT_FORMAT_DIRS = {"poses": "colmap", "images": "postshot", "realityscan": "realityscan"}

INSTRUCTIONS = """\
vine360 turns 360-degree/conventional capture into masked, posed, trainable 3D Gaussian Splatting
projects. Pipeline: add source -> extract frames -> project views -> masks -> pose estimation (SfM)
-> georeference (optional: import control points, detect AprilTags, solve) -> export.

Start with open_project (or create_project), then project_status. Long operations start a
background job and return its job_id at once: follow up with wait_for_job (blocks up to its
timeout) or get_job. One job runs at a time; jobs can't be cancelled once started. Before
starting a job the server refuses if the project's Activity log shows another run in progress
(e.g. in the desktop app); pass allow_concurrent=true only if the user confirms that run is
not really active. Re-running a stage on a frame set replaces that frame set's downstream data.
Paths: file paths are absolute; output paths may be relative to the project root.
"""


class _Job:
    def __init__(self, operation: str, label: str, params: dict):
        self.job_id = uuid.uuid4().hex[:8]
        self.operation = operation
        self.label = label
        self.params = params
        self.status = "running"
        self.started_at = _utcnow_iso()
        self.started = time.monotonic()
        self.finished_at: str | None = None
        self.duration_seconds: float | None = None
        self.message = ""
        self.current = 0
        self.total = 0
        self.log: deque[str] = deque(maxlen=JOB_LOG_LINES)
        self.result: Any = None
        self.summary: str | None = None
        self.error: str | None = None
        self.done = threading.Event()

    def progress(self, message: str, current: int, total: int) -> None:
        if message != self.message:
            self.log.append(f"{time.monotonic() - self.started:7.1f}s  {message}")
        self.message, self.current, self.total = message, current, total

    def to_dict(self, *, log_lines: int = 20) -> dict:
        out = {
            "job_id": self.job_id,
            "operation": self.operation,
            "label": self.label,
            "status": self.status,
            "params": self.params,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round(self.duration_seconds if self.duration_seconds is not None
                                     else time.monotonic() - self.started, 1),
            "progress": {"message": self.message, "current": self.current, "total": self.total},
        }
        if log_lines:
            out["recent_progress"] = list(self.log)[-log_lines:]
        if self.status == "done":
            out["summary"] = self.summary
            out["result"] = self.result
        if self.error:
            out["error"] = self.error
        return out


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    """JSON-friendly copy of a worker's result: dataclasses, paths, enums,
    numpy scalars/arrays and tuples."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if hasattr(value, "tolist"):  # numpy
        return value.tolist()
    return value


class Vine360Session:
    """The server's state: the open project and the jobs it started."""

    def __init__(self, project_root: Path | None = None):
        self.project_root: Path | None = None
        self.jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        if project_root is not None:
            self.open(project_root)

    # -- project ------------------------------------------------------------

    def open(self, path: Path) -> dict:
        root = Path(path).expanduser().resolve()
        project = load_project(root)
        open_index_db(root).close()  # migrates an older project's schema, like the app does on open
        self.project_root = root
        record_recent_project(root, project.name)
        return self.summary()

    def root(self) -> Path:
        if self.project_root is None:
            raise ToolError("No project is open -- call open_project (or create_project) first.")
        return self.project_root

    def conn(self) -> sqlite3.Connection:
        return open_index_db(self.root())

    def summary(self) -> dict:
        root = self.root()
        project = load_project(root)
        conn = self.conn()
        try:
            statuses = compute_stage_statuses(conn)
            counts = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("sources", "frame_sets", "frames", "views", "masks", "sfm_runs",
                              "control_point_sets", "tag_observations", "georef_solutions")
            }
            in_progress = self.runs_in_progress(conn)
        finally:
            conn.close()
        return {
            "project_root": str(root),
            "name": project.name,
            "capture_mode": project.capture_mode.value,
            "project_id": project.project_id,
            "stages": {STAGE_NAMES[k]: v for k, v in sorted(statuses.items())},
            "counts": counts,
            "runs_in_progress": in_progress,
            "current_job": self.running_job().to_dict(log_lines=0) if self.running_job() else None,
        }

    @staticmethod
    def runs_in_progress(conn: sqlite3.Connection) -> list[dict]:
        return [
            {"entry_id": e.entry_id, "operation": e.operation, "target": e.target, "origin": e.origin,
             "started_at": e.started_at}
            for e in list_entries(conn)
            if e.status == STATUS_RUNNING
        ]

    # -- jobs ---------------------------------------------------------------

    def running_job(self) -> _Job | None:
        return next((j for j in self.jobs.values() if j.status == "running"), None)

    def job(self, job_id: str) -> _Job:
        if job_id not in self.jobs:
            raise ToolError(f"No job {job_id!r} in this session (jobs: {', '.join(self.jobs) or 'none'}).")
        return self.jobs[job_id]

    def start_job(
        self,
        operation: str,
        label: str,
        worker: Callable,
        kwargs: dict,
        *,
        allow_concurrent: bool = False,
        describe: Callable[[Any, sqlite3.Connection], Any] | None = None,
    ) -> dict:
        """Runs worker(project_root, **kwargs, progress_callback=...) on a
        background thread. worker is a vine360.operations function, so it
        logs itself in the Activity log (origin "mcp")."""
        root = self.root()
        with self._lock:
            running = self.running_job()
            if running is not None:
                raise ToolError(
                    f"Job {running.job_id} ({running.label}) is still running -- one job at a time. "
                    "Use wait_for_job, then start the next one."
                )
            if not allow_concurrent:
                conn = self.conn()
                try:
                    others = self.runs_in_progress(conn)
                finally:
                    conn.close()
                if others:
                    listing = "; ".join(
                        f"{o['operation']} on {o['target'] or '-'} (origin {o['origin']}, started {o['started_at']})"
                        for o in others
                    )
                    raise ToolError(
                        f"The Activity log shows a run still in progress: {listing}. Starting another could race it "
                        "on the same project. Wait for it to finish; if it is left over from an app session that "
                        "crashed (the app marks those interrupted when it next opens the project), ask the user, "
                        "then retry with allow_concurrent=true."
                    )
            job = _Job(operation, label, _jsonable(kwargs))
            self.jobs[job.job_id] = job

        def run():
            try:
                result = worker(root, **kwargs, progress_callback=job.progress, **{ORIGIN_KWARG: ORIGIN})
                conn = open_index_db(root)
                try:
                    job.result = _jsonable(describe(result, conn) if describe else result)
                    job.summary = _logged_result(conn, operation)
                finally:
                    conn.close()
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 -- reported through the job
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "failed"
            finally:
                job.finished_at = _utcnow_iso()
                job.duration_seconds = time.monotonic() - job.started
                job.done.set()

        threading.Thread(target=run, name=f"vine360-job-{job.job_id}", daemon=False).start()
        return job.to_dict(log_lines=0)


def _logged_result(conn: sqlite3.Connection, operation: str) -> str | None:
    row = conn.execute(
        "SELECT result FROM activity_log WHERE operation = ? AND origin = ? ORDER BY entry_id DESC LIMIT 1",
        (operation, ORIGIN),
    ).fetchone()
    return row[0] if row else None


def _resolve_output(root: Path, path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else root / p


def _sfm_run_ids(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT run_id FROM sfm_runs")}


def _tool_errors(fn: Callable) -> Callable:
    """Library errors reach the model as their message (a ToolError) rather
    than the SDK's generic "Error executing tool"."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def build_server(session: Vine360Session | None = None) -> MCPServer:
    session = session or Vine360Session()
    server = MCPServer("vine360", instructions=INSTRUCTIONS)

    def tool(annotations: ToolAnnotations):
        def decorate(fn):
            return server.tool(annotations=annotations)(_tool_errors(fn))
        return decorate

    # -- projects -------------------------------------------------------------

    @tool(READ_ONLY)
    def list_recent_projects() -> list[dict]:
        """Projects recently opened in the desktop app or this server, newest first."""
        return [{"path": p.path, "name": p.name, "opened_at": p.opened_at} for p in load_recent_projects()]

    @tool(EDIT)
    def open_project(path: str) -> dict:
        """Open an existing vine360 project directory (the one holding project.yaml). Every other
        tool works on the open project. Returns the project status."""
        return session.open(Path(path))

    @tool(EDIT)
    def create_project(path: str, name: str, capture_mode: str = "360") -> dict:
        """Create a new project directory and open it. capture_mode: "360" or "conventional"."""
        from vine360.project import create_project as _create

        root = Path(path).expanduser().resolve()
        _create(root, name, CaptureMode(capture_mode))
        return session.open(root)

    @tool(READ_ONLY)
    def project_status() -> dict:
        """The open project's stage statuses (done/active/pending, pose "active" when stale),
        row counts, any run in progress in its Activity log, and this server's running job."""
        return session.summary()

    # -- reading ----------------------------------------------------------------

    @tool(READ_ONLY)
    def list_sources() -> list[dict]:
        """Registered media sources (recorded by reference, never copied)."""
        from vine360.data_manager import list_sources as _list

        conn = session.conn()
        try:
            return [_jsonable(s) for s in _list(conn)]
        finally:
            conn.close()

    @tool(READ_ONLY)
    def list_frame_sets(source_id: str | None = None) -> list[dict]:
        """Frame sets (one per extraction config of a source) with frame/view/mask counts and
        AprilTag detection progress. Every stage after Frames is scoped by frame_set_id."""
        from vine360.data_manager import list_frame_sets as _list
        from vine360.georef.observe import tag_counts_by_frame_set

        conn = session.conn()
        try:
            tags = tag_counts_by_frame_set(conn)
            return [
                _jsonable(info) | {"name": info.display_name, "tags": tags.get(info.frame_set_id)}
                for info in _list(conn, source_id=source_id)
            ]
        finally:
            conn.close()

    @tool(READ_ONLY)
    def list_sfm_runs() -> list[dict]:
        """Pose-estimation runs, newest last, each with its registration counts, its
        georeference solutions (RMSE) and which solution is active for export."""
        from vine360.data_manager import list_sfm_runs as _list
        from vine360.georef.solve import active_solution, list_solutions

        root = session.root()
        conn = session.conn()
        try:
            out = []
            for run in _list(conn):
                solutions = [
                    {"solution_id": s["solution_id"], "set_id": s["set_id"], "created_at": s["created_at"],
                     "rmse_3d_m": s["report"]["rmse"]["d3"],
                     "check_rmse_3d_m": (s["report"].get("check_rmse") or {}).get("d3"),
                     "n_control": s["report"]["n_control"], "n_check": s["report"].get("n_check")}
                    for s in list_solutions(conn, run.run_id)
                ]
                out.append(_jsonable(run) | {"engine_label": run.engine_label, "georef_solutions": solutions,
                                             "active_solution": active_solution(root, run.run_id)})
            return out
        finally:
            conn.close()

    @tool(READ_ONLY)
    def list_control_point_sets() -> list[dict]:
        """Imported surveyed control-point (GCP) sets: CRS (EPSG), height datum, origin."""
        from vine360.georef.control_points import list_control_point_sets as _list

        conn = session.conn()
        try:
            return [_jsonable(s) for s in _list(conn)]
        finally:
            conn.close()

    @tool(READ_ONLY)
    def get_control_points(set_id: str) -> list[dict]:
        """One set's points: name, CRS coordinates, tag id/size, use (control/check/ignore)."""
        from vine360.georef.control_points import get_control_point_set, load_control_points

        conn = session.conn()
        try:
            get_control_point_set(conn, set_id)
            return [_jsonable(p) for p in load_control_points(conn, set_id)]
        finally:
            conn.close()

    @tool(READ_ONLY)
    def get_georef_solution(solution_id: str) -> dict:
        """A georeference solution's report: RMSE (control and check), per-point residuals,
        leave-one-out errors, flags and the similarity transform."""
        from vine360.georef.solve import load_solution

        conn = session.conn()
        try:
            solution = load_solution(conn, solution_id)
        finally:
            conn.close()
        solution.pop("tag_positions", None)  # bulky; the report's per-point rows carry what matters
        return _jsonable(solution)

    @tool(READ_ONLY)
    def list_exports() -> list[dict]:
        """Export folders under exports/ (exports/<capture>/<format>/)."""
        from vine360.data_manager import list_export_dirs

        root = session.root()
        out = []
        for info in list_export_dirs(root):
            entry = _jsonable(info)
            meta = root / info.relative_path / "export_info.json"
            if meta.is_file():
                try:
                    entry["export_info"] = json.loads(meta.read_text(encoding="utf-8"))
                except ValueError:
                    pass
            out.append(entry)
        return out

    @tool(READ_ONLY)
    def activity_log(limit: int = 20, operation: str | None = None) -> list[dict]:
        """The project's Activity log, newest first: every run (app, queue or this server) with its
        parameters, timing, outcome and origin. operation filters by prefix, e.g. "sfm" or "georef"."""
        conn = session.conn()
        try:
            entries = [e for e in list_entries(conn) if not operation or e.operation.startswith(operation)]
        finally:
            conn.close()
        return [_jsonable(e) for e in reversed(entries[-max(1, limit):])]

    @tool(READ_ONLY)
    def sfm_options() -> list[dict]:
        """The advanced SfM options run_sfm accepts in its options argument: key, default, which
        engines it applies to, and a description. Unlisted options keep their defaults."""
        from vine360.sfm.options import OPTION_SPECS, SfmConfig

        defaults = SfmConfig().to_dict()
        return [
            {"key": spec.key, "group": spec.group, "label": spec.label, "kind": spec.kind,
             "default": defaults.get(spec.key), "choices": [value for value, _label in spec.choices],
             "engines": list(spec.variants), "help": spec.help}
            for spec in OPTION_SPECS
        ]

    # -- quick edits --------------------------------------------------------------

    @tool(EDIT)
    def set_control_point_use(set_id: str, name: str, use: str) -> dict:
        """Set one control point's use: "control" (fits the transform), "check" (independent
        accuracy test) or "ignore". Takes effect at the next solve."""
        from vine360.georef.control_points import update_control_point

        conn = session.conn()
        try:
            update_control_point(conn, set_id, name, use=use)
            log_action(conn, "georef.edit_point", "Georeference: control point use", target=name,
                       params={"set_id": set_id, "use": use}, origin=ORIGIN)
        finally:
            conn.close()
        return {"set_id": set_id, "name": name, "use": use}

    @tool(EDIT)
    def set_active_solution(run_id: str, solution_id: str | None = None) -> dict:
        """Choose which georeference solution is the default for a run's exports (None clears it)."""
        from vine360.georef.solve import load_solution
        from vine360.georef.solve import set_active_solution as _set

        root = session.root()
        if solution_id is not None:
            conn = session.conn()
            try:
                if load_solution(conn, solution_id)["run_id"] != run_id:
                    raise ToolError(f"Solution {solution_id} belongs to another run, not {run_id}.")
            finally:
                conn.close()
        _set(root, run_id, solution_id)
        conn = session.conn()
        try:
            log_action(conn, "georef.active_solution", "Georeference: active solution", target=run_id,
                       params={"solution_id": solution_id}, origin=ORIGIN)
        finally:
            conn.close()
        return {"run_id": run_id, "active_solution": solution_id}

    @tool(EDIT)
    def apply_tag_size_table(set_id: str, table_path: str | None = None) -> dict:
        """Set each tagged point's printed tag size from a tag-size table (this device's default
        table when table_path is omitted), replacing earlier sizes. Unlisted tags are left alone."""
        from vine360.georef.control_points import apply_tag_sizes, get_control_point_set
        from vine360.georef.tag_sizes import default_table_path, read_tag_size_table

        path = Path(table_path) if table_path else default_table_path()
        conn = session.conn()
        try:
            cp_set = get_control_point_set(conn, set_id)
            changed, unlisted = apply_tag_sizes(conn, set_id, read_tag_size_table(path, cp_set.tag_family))
            log_action(conn, "georef.tag_sizes", "Georeference: apply tag-size table", target=set_id,
                       params={"table": str(path), "changed": changed, "unlisted": unlisted}, origin=ORIGIN)
        finally:
            conn.close()
        return {"table": str(path), "changed": changed, "unlisted": unlisted}

    @tool(EDIT)
    def import_control_points(
        path: str,
        epsg: int,
        height_datum: str,
        tag_family: str = "tag36h11",
        default_tag_size_m: float | None = None,
        use_tag_size_table: bool = True,
    ) -> dict:
        """Import a surveyed GCP CSV as a new control-point set. epsg and height_datum describe the
        survey and are never inferred -- ask the user (e.g. 7854 = GDA2020 / MGA zone 54;
        height_datum e.g. "AHD" or "ellipsoidal"). Tag sizes come from this device's tag-size table
        (use_tag_size_table), with default_tag_size_m for unlisted tags."""
        from vine360.georef.tag_sizes import default_table_path

        root = session.root()
        table = str(default_table_path()) if use_tag_size_table else None
        set_id, warnings = operations.import_control_points_worker(
            root, Path(path), epsg, height_datum, tag_family, default_tag_size_m, table,
            **{ORIGIN_KWARG: ORIGIN},
        )
        return {"set_id": set_id, "warnings": warnings, "tag_size_table": table}

    # -- pipeline jobs --------------------------------------------------------------

    @tool(JOB)
    def add_source(
        file_path: str,
        capture_group: str | None = None,
        confirm_equirectangular: bool = False,
        allow_concurrent: bool = False,
    ) -> dict:
        """Register a video or image as a source (by reference; hashed, not copied). A 2:1 video
        whose metadata doesn't say it's 360 fails asking for confirm_equirectangular=true -- ask
        the user. capture_group e.g. "Insta360 (ground)" or "Antigravity A1 (aerial)". Job."""
        return session.start_job(
            "source.add", "Add source", operations.add_source_worker,
            {"file_path": Path(file_path), "capture_group": capture_group,
             "confirm_equirectangular": confirm_equirectangular, "added_at": _utcnow_iso()},
            allow_concurrent=allow_concurrent,
            describe=lambda source, _c: source.to_dict(),
        )

    @tool(REPLACING_JOB)
    def extract_frames(
        source_id: str,
        interval_seconds: float = 1.0,
        start_time: float | None = None,
        end_time: float | None = None,
        generate_thumbnails: bool = True,
        allow_concurrent: bool = False,
    ) -> dict:
        """Extract frames from a video source every interval_seconds (optionally between
        start_time and end_time). Each config is its own frame set; re-extracting the same config
        replaces that frame set's frames and everything derived from them. Job."""
        from vine360.ingest.frames import frame_set_id_for

        conn = session.conn()
        try:
            frame_set_id = frame_set_id_for(conn, source_id, interval_seconds=interval_seconds,
                                            start_time=start_time, end_time=end_time)
        finally:
            conn.close()
        return session.start_job(
            "frames.extract", "Frame extraction", operations.extract_frames_worker,
            {"source_id": source_id, "interval_seconds": interval_seconds, "start_time": start_time,
             "end_time": end_time, "generate_thumbnails": generate_thumbnails, "frame_set_id": frame_set_id},
            allow_concurrent=allow_concurrent,
            describe=lambda frames, _c: {"frame_set_id": frame_set_id, "frames": len(frames)},
        )

    @tool(REPLACING_JOB)
    def generate_views(
        frame_set_id: str,
        face_names: list[str] | None = None,
        face_size: int = 1024,
        fov_degrees: float = 90.0,
        ring_tilt_degrees: float = 45.0,
        max_workers: int | None = None,
        allow_concurrent: bool = False,
    ) -> dict:
        """Project a 360 frame set's equirectangular frames into pinhole views. face_names default
        to the four horizon cardinals (front, right, back, left); the catalogue is front,
        front-right, right, ..., front-left on the horizon, the same names + "-down"/"-up" for the
        lower/upper rings (tilted by ring_tilt_degrees), plus "up" and "down". Replaces the frame
        set's existing views and their masks. Job."""
        from vine360.projection.cubemap import CARDINAL_NAMES

        return session.start_job(
            "projection.generate", "Projection", operations.generate_views_worker,
            {"frame_set_id": frame_set_id, "face_size": face_size, "fov_degrees": fov_degrees,
             "face_names": list(face_names or CARDINAL_NAMES),
             "max_workers": max_workers or operations.default_parallel_worker_count(),
             "ring_tilt_degrees": ring_tilt_degrees},
            allow_concurrent=allow_concurrent,
            describe=lambda views, _c: {"frame_set_id": frame_set_id, "views": len(views)},
        )

    @tool(REPLACING_JOB)
    def build_masks(
        frame_set_id: str,
        mask_sky: bool = True,
        use_sam3_sky: bool = False,
        use_sam3_person: bool = False,
        mask_overexposure: bool = False,
        overexposure_clip: int = 250,
        overexposure_bloom_radius: int | None = None,
        allow_concurrent: bool = False,
    ) -> dict:
        """Build keep-masks for a frame set's views: sky (heuristic, or SAM 3 with use_sam3_sky),
        people (SAM 3 only; needs the gated Hugging Face weights), overexposure. Rebuilds the
        frame set's mask layers; hand-edited layers are kept. Job."""
        from vine360.masking.layers import LAYER_OVEREXPOSED, LAYER_SPECS

        bloom = (overexposure_bloom_radius if overexposure_bloom_radius is not None
                 else LAYER_SPECS[LAYER_OVEREXPOSED].default_params["bloom_radius_px"])
        return session.start_job(
            "masks.build", "Masks", operations.build_masks_worker,
            {"frame_set_id": frame_set_id, "use_sam3_person": use_sam3_person, "use_sam3_sky": use_sam3_sky,
             "mask_sky": mask_sky, "mask_overexposure": mask_overexposure,
             "overexposure_clip": overexposure_clip, "overexposure_bloom_radius": bloom},
            allow_concurrent=allow_concurrent,
        )

    @tool(JOB)
    def run_sfm(
        engine: str = "colmap_projections",
        frame_set_ids: list[str] | None = None,
        camera_model: str = "SIMPLE_RADIAL",
        options: dict | None = None,
        allow_concurrent: bool = False,
    ) -> dict:
        """Pose estimation. engine: "colmap_projections" (pycolmap on projected pinhole views; the
        default), "colmap_equirectangular" (pycolmap on raw 360 frames) or "spheresfm" (SphereSfM on
        raw 360 frames). frame_set_ids: one or more frame sets (omit for every frame set with
        views, projections engine only). camera_model applies to colmap_projections only. options:
        advanced options by key (see sfm_options). Every run is kept as a new run. Job."""
        from vine360.sfm.options import SfmConfig

        if engine not in SFM_ENGINES:
            raise ToolError(f"engine must be one of {SFM_ENGINES}")
        ids = list(frame_set_ids or [])
        if engine != "colmap_projections" and not ids:
            raise ToolError(f"{engine} runs on raw frames: give frame_set_ids.")
        config = SfmConfig.from_dict(options).replace(camera_model=camera_model).for_variant(engine)
        problems = config.validate(engine)
        if problems:
            raise ToolError("; ".join(problems))
        opts = {k: v for k, v in config.non_default().items() if k != "camera_model"}
        image_source, model, backend = {
            "spheresfm": ("frames", "SPHERE", "spheresfm"),
            "colmap_equirectangular": ("frames", "EQUIRECTANGULAR", "pycolmap"),
        }.get(engine, ("projections", config.camera_model, "pycolmap"))
        conn = session.conn()
        try:
            before = _sfm_run_ids(conn)
        finally:
            conn.close()

        def describe(result, conn):
            diagnostics, warnings = result
            new = sorted(_sfm_run_ids(conn) - before)
            return {"run_id": new[-1] if new else None, "diagnostics": _jsonable(diagnostics), "warnings": warnings}

        return session.start_job(
            "sfm.run", "Pose estimation", operations.run_sfm_worker,
            {"image_source": image_source, "frame_set_id": ids[0] if len(ids) == 1 else None,
             "camera_model": model, "engine": backend, "options": opts,
             "frame_set_ids": ids if len(ids) > 1 else None},
            allow_concurrent=allow_concurrent, describe=describe,
        )

    @tool(JOB)
    def detect_tags(
        frame_set_ids: list[str],
        tag_family: str = "tag36h11",
        detector_params: dict | None = None,
        allow_concurrent: bool = False,
    ) -> dict:
        """Detect AprilTags on the frame sets' projected pinhole views (never equirectangular
        frames). Detections are saved as they go; views already detected are skipped, so an
        interrupted run resumes. Job."""
        return session.start_job(
            "georef.detect_tags", "Georeference: detect tags", operations.detect_tags_worker,
            {"frame_set_ids": list(frame_set_ids), "tag_family": tag_family, "detector_params": detector_params or {}},
            allow_concurrent=allow_concurrent,
            describe=lambda summary, _c: _jsonable(summary) | {"description": summary.describe()},
        )

    @tool(JOB)
    def solve_georef(
        run_id: str,
        set_id: str,
        solve_params: dict | None = None,
        allow_concurrent: bool = False,
    ) -> dict:
        """Fit an SfM run to a control-point set: triangulate each tag from its detections, then a
        similarity into the set's local frame (CRS minus its origin). Non-destructive: a new
        solution every time, and the run's own model is never changed. solve_params: thresh_px,
        min_inlier_views, min_ray_angle_deg. Job."""
        def describe(solution_id, conn):
            from vine360.georef.solve import load_solution

            report = load_solution(conn, solution_id)["report"]
            return {"solution_id": solution_id, "rmse_3d_m": report["rmse"]["d3"],
                    "check_rmse_3d_m": (report.get("check_rmse") or {}).get("d3"),
                    "n_control": report["n_control"], "n_check": report.get("n_check")}

        return session.start_job(
            "georef.solve", "Georeference: solve", operations.solve_georef_worker,
            {"run_id": run_id, "set_id": set_id, "solve_params": solve_params or {}},
            allow_concurrent=allow_concurrent, describe=describe,
        )

    @tool(JOB)
    def georeference_splat(
        in_path: str,
        solution_id: str,
        out_path: str | None = None,
        force: bool = False,
        allow_concurrent: bool = False,
    ) -> dict:
        """Write a georeferenced copy of a trained splat .ply (plus .georef.json/.prj sidecars); the
        input is only read. Only for a splat trained from a non-georeferenced export of the
        solution's own run. A frame check refuses a splat that is already local or in an unrelated
        frame (the job fails with SplatFrameMismatch); force=true overrides it -- ask the user
        first. The default out_path never replaces an existing file. Job."""
        from vine360.georef.apply import default_splat_output

        src = Path(in_path).expanduser()
        out = _resolve_output(session.root(), out_path) if out_path else operations.unused_path(
            default_splat_output(src, solution_id))
        return session.start_job(
            "georef.splat", "Georeference a trained splat", operations.georef_splat_worker,
            {"in_path": src, "out_path": out, "solution_id": solution_id, "force": force},
            allow_concurrent=allow_concurrent,
        )

    @tool(JOB)
    def export(
        mode: str,
        output_dir: str | None = None,
        run_id: str | None = None,
        frame_set_id: str | None = None,
        georef_solution_id: str | None = None,
        include_camera_priors: bool = False,
        allow_concurrent: bool = False,
    ) -> dict:
        """Export for an external trainer. mode: "poses" (a run's sparse model + images + masks for
        Postshot/COLMAP trainers; needs run_id), "images" (images + masks only, the trainer runs its
        own SfM; optional frame_set_id) or "realityscan" (optional frame_set_id,
        include_camera_priors). georef_solution_id puts poses/RealityScan output in that solution's
        local frame (copied, never re-transformed); images mode has no coordinates to transform.
        output_dir defaults to exports/<capture>/<format>/. Job."""
        from vine360.data_manager import frame_set_folder_name

        if mode not in EXPORT_MODES:
            raise ToolError(f"mode must be one of {EXPORT_MODES}")
        root = session.root()
        if mode == "poses" and not run_id:
            raise ToolError("poses mode needs run_id (see list_sfm_runs).")
        if mode == "images" and georef_solution_id:
            raise ToolError("images mode has no coordinates to georeference -- drop georef_solution_id.")
        if output_dir:
            out = _resolve_output(root, output_dir)
        else:
            conn = session.conn()
            try:
                capture = frame_set_folder_name(conn, frame_set_id) if frame_set_id else "all"
            finally:
                conn.close()
            out = root / "exports" / capture / _EXPORT_FORMAT_DIRS[mode]
        if mode == "poses":
            op, label, worker = "export.postshot_poses", "Export: poses + images + masks", operations.export_postshot_worker
            kwargs = {"output_dir": out, "run_id": run_id, "georef_solution_id": georef_solution_id}
        elif mode == "images":
            op, label, worker = "export.postshot_images", "Export: images + masks", operations.export_frames_and_masks_worker
            kwargs = {"output_dir": out, "frame_set_id": frame_set_id}
        else:
            op, label, worker = "export.realityscan", "Export: RealityScan", operations.export_realityscan_worker
            kwargs = {"output_dir": out, "frame_set_id": frame_set_id, "include_camera_priors": include_camera_priors,
                      "georef_solution_id": georef_solution_id}
        return session.start_job(op, label, worker, kwargs, allow_concurrent=allow_concurrent)

    # -- job control ------------------------------------------------------------------

    @tool(READ_ONLY)
    def get_job(job_id: str, log_lines: int = 20) -> dict:
        """A job's status (running/done/failed), counted progress, recent progress messages, and
        its result once done."""
        return session.job(job_id).to_dict(log_lines=log_lines)

    @tool(READ_ONLY)
    def wait_for_job(job_id: str, timeout_seconds: float = 120.0) -> dict:
        """Block until the job finishes or timeout_seconds pass (at most 600), then report it as
        get_job does. Call again if it's still running -- a long SfM run can take hours."""
        job = session.job(job_id)
        job.done.wait(max(0.0, min(float(timeout_seconds), MAX_WAIT_SECONDS)))
        return job.to_dict()

    @tool(READ_ONLY)
    def list_jobs() -> list[dict]:
        """Every job this server session started, oldest first."""
        return [j.to_dict(log_lines=0) for j in session.jobs.values()]

    server.session_state = session  # for tests
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vine360-mcp", description="vine360 MCP server (stdio)")
    parser.add_argument("--project", type=Path, help="open this project at startup")
    args = parser.parse_args(argv)
    build_server(Vine360Session(args.project)).run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
