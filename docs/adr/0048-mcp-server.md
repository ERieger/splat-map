# 0048. An MCP server that drives the pipeline

## Status

Accepted. Builds on the activity log (ADR 0037) and the queue's one-writer rule (ADR 0023).

## Context

The pipeline could only be driven from the desktop app. An assistant that can talk to vine360
over the Model Context Protocol (MCP) could also drive it: Claude Code in this repo, or Claude
Desktop. It could inspect a project, run stages, read georeferencing residuals and choose check
points. The requirement was **full pipeline control**, not read-only inspection.

The app's long-running work lived in `vine360/gui/main_window.py` as worker functions. The panels
and the queue called them; each opened its own sqlite connection and was wrapped in
`logged_operation`. Those workers have no Qt in them, but importing them pulled in PySide6 and
a 6,000-line GUI module.

## Decision

**`vine360/operations.py`** now holds every worker. They were moved unchanged and given public
names. `main_window` imports each one under its old private name, so the panels, the queue
(`mw._run_sfm_worker`, ...) and the tests are unaffected. **`vine360/stage_status.py`** likewise
holds `compute_stage_statuses` and the stage constants. Neither module imports Qt.

**`vine360/mcp_server.py`** is a stdio MCP server on the official `mcp` SDK, written against the
2.x API (`MCPServer`, which replaced 1.x's FastMCP), new `mcp` extra. Its 29 tools fall into
four groups:

- **Projects:** `open_project`, `create_project`, `list_recent_projects`, `project_status`.
- **Read-only:** sources, frame sets (with tag-detection counts), SfM runs (with their
  solutions and the active one), control-point sets and points, a solution's report, exports,
  the Activity log, and the advanced SfM options.
- **Quick edits:**
  - a point's use (control/check/ignore);
  - the active solution;
  - applying a tag-size table;
  - importing a GCP CSV. EPSG and height datum are required arguments and never inferred
    (ADR 0047).
- **Jobs:**
  - `add_source`;
  - `extract_frames`, `generate_views` and `build_masks`;
  - `run_sfm`, whose engine is a variant id, so it takes the same choices as the Pose panel;
  - `detect_tags`, `solve_georef` and `georeference_splat`;
  - `export`, in poses, images or RealityScan mode;
  - with `get_job`, `wait_for_job` and `list_jobs` to follow them.

How jobs run:

- **Background thread.** A job starts on a background thread, and the tool returns its id at
  once. `wait_for_job` blocks for up to 600 s per call. A job reports the same counted
  `progress_callback` messages the GUI's progress bar shows.
- **One at a time.** One job runs at a time per server, and a job can't be cancelled once
  started, because the workers have no cancellation hook (as in the app).
- **Survives the client exiting.** Job threads aren't daemons, so when the client closes the
  stdio session, a running job finishes before the process exits.
- **Logged.** Every job, and every quick edit, is logged in the project's Activity log with
  origin `mcp`.
- **No duplicate pipeline logic.** Every job calls the `operations` worker the app calls.
  The only logic the server adds is turning tool arguments into worker arguments: an engine
  variant becomes image source, camera model and engine, as in the Pose panel; and the default
  export folder is `exports/<capture>/<format>/` (ADR 0028), via the new shared
  `data_manager.frame_set_folder_name`.

**Sharing a project with a running app.** Both use their own short-lived sqlite connections, as
the app's workers always have. A job is **refused while the Activity log shows another run in
progress**, whether it's the app's, its queue's or another server's. The caller can override this
with `allow_concurrent=true`, which the server's instructions tell the model to use only after
the user confirms.

Two workers writing one project at once is exactly what the queue's lock prevents inside the
app (ADR 0023). The Activity log is the only cross-process signal that exists, so that's what is
checked. An entry left "running" by a crashed app session also blocks, until the app next opens
the project and marks it interrupted. That's why the override exists. The app doesn't watch for
outside changes, so its panels show a job's results once they're refreshed.

**Not exposed:** deleting data (the Data manager's job, and not undoable) and removing a source.
Re-running a stage on the same frame set still replaces that frame set's downstream data, which
is the cascade-delete invariant. Those tools carry MCP's `destructive_hint` annotation and say so
in their descriptions.

**Errors:** library exceptions are re-raised as the SDK's `ToolError`, so the model reads
`ControlPointError: no point '006' in ...` rather than the SDK's generic "Error executing tool".
A job's failure is reported in its `error` field.

**Registration:** `.mcp.json` at the repo root registers `./run-mcp.sh`. It sets up the venv,
PYTHONPATH and bundled ffmpeg, as `run-gui.sh` does. Claude Code asks once to approve it. There
is also a console script, `vine360-mcp --project PATH`.

## Verification

- **Checked against the installed SDK source** (`mcp` 2.3.0):
  - synchronous tools run on a worker thread (`anyio.to_thread`), so a blocking `wait_for_job`
    doesn't stall the server;
  - `ToolError` text reaches the model;
  - the stdio transport points fd 1 at stderr while serving, so stray stdout from a library,
    native code or a subprocess can't corrupt the protocol stream.
- **`tests/test_mcp_server.py`** calls the tools in-process through `MCPServer.call_tool`:
  - on a fresh project;
  - on the real georeferencing scene (`tests/georef_project.py`). There a solve runs as a real
    background job through `operations`, and its result, its summary and both Activity log
    entries are checked. A failed job and the refusal while another run is in progress (and its
    override) are also tested.
- **One manual run over real stdio**, with the SDK's own client launching `./run-mcp.sh`:
  initialise, list tools, create a project, a tool error, and a job that failed as expected.
- **`claude mcp list`** shows the server from `.mcp.json`, pending approval.
- **Not run through the real pipeline yet.** No real capture has been driven end to end over MCP
  so far.

## Consequences

- A new pipeline operation gets an MCP tool by adding its worker to `operations.py`, which the
  app imports anyway, and a few lines in `mcp_server.py`.
- The app and the server can't see each other's jobs except through the Activity log. A server
  job won't show in the app's progress bar or queue.
- Deletion stays a human action in the app.
