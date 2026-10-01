# Critical review — 2026-09-22

A one-time review snapshot, not a living doc (unlike `docs/status.md`) — reflects the codebase as
of this date, specifically including the persisted-queue feature, the `models/` directory, and the
Training-panel removal added this session. Don't expect it to be kept in sync with later changes.

Reviewed against this project's own stated intentions (all 25 ADRs, `CLAUDE.md`), not generic best
practices — a deliberate documented decision (adapter-only training, one big GUI file, sources
recorded by reference) is not flagged as an oversight here.

## Executive summary

- **Correctness: 3 findings** (0 high, 2 moderate, 1 low). The library and GUI code are solid
  overall — most of the review time went into confirming things were *fine* (SQLite locking,
  frozen-dataclass defaults, cascade-delete symmetry) rather than finding new bugs.
- **UX/Design: 4 findings** (0 high, 2 moderate, 2 low).
- **Architecture/Maintainability: 4 findings** (0 high, 2 moderate, 2 low).
- No finding here is a crash-on-the-golden-path bug. The most important one (Correctness #1) is a
  real gap in a documented guarantee — the queue's dependency tracking doesn't survive a manual
  action taken outside the queue — but it fails loudly (a clear error message), not silently.

---

## Correctness

### 1. [Moderate] Queue dependency tracking doesn't survive a manual action taken outside the queue

`vine360/gui/queue_manager.py`, `QueueManager._resolve_prerequisites` (lines 300–355) only runs
**once, when a job is added** to the queue. It checks the project's current DB state at that
moment and either finds the prerequisite already satisfied (no `depends_on` edge needed) or
auto-inserts an upstream job. Once a job is `QUEUED` with `depends_on == []` because its data
already existed, nothing re-checks that the data *still* exists later.

**Concrete scenario**: a source already has views. User adds a Masks job for it — no auto-insert,
added as `QUEUED` with no dependency (this is correct at that moment). Before starting the queue,
the user manually clicks **Run now** on the Frames panel for the same source (allowed — the manual
button is only disabled while the queue is actively `is_running`, not merely non-empty). This
cascade-deletes that source's views and masks (`clear_frames_for_source`,
`vine360/ingest/frames.py:77-116`). The queue still has the Masks job sitting there as `QUEUED`
with no dependency pointing at anything that would flag it. Starting the queue now runs it, and
`build_masks_for_source` correctly raises `MaskBuildError: no views found for source ...` —
**it fails loudly, not silently** (caught by `_on_job_error`, surfaced as a `FAILED` row with the
real message) — but this is a confusing, avoidable failure the dependency graph was specifically
built to prevent for queue-internal reordering, and doesn't cover manual actions.

*Why moderate, not high*: no silent corruption or wrong output — the failure mode is a correct,
actionable error message. It's a gap in a documented guarantee (ADR 0023's dependency tracking),
not a data-integrity bug.

### 2. [Moderate] Export has no partial-failure protection

`vine360/export/postshot.py`, both `export_for_postshot` (139-264) and
`export_frames_and_masks_for_postshot` (267-338): `_reset_managed_subdirs` clears the destination
up front, then images/masks are copied one at a time in a loop with no transactional wrapper. If
`shutil.copy2` raises partway through (disk full, permission error, a network drive disconnecting
mid-copy — `output_dir` has no format constraint preventing a UNC/mapped-drive path), the exception
propagates uncaught out of the function. The GUI/queue path correctly surfaces this as an error
(`_on_export_error`/`_on_job_error`), so **the user is told it failed** — but the partially-written
`images/`/`masks/`/`sparse/` on disk is left with no marker distinguishing "complete" from "stopped
at image 743 of 2000." A user who dismisses the error and points Postshot at the folder anyway (or
a queue retry that succeeds later, leaving stale files from the first attempt mixed with the
second — `_reset_managed_subdirs` does clear before retrying, so this specific case is actually
fine) gets a confusing partial dataset.

*Concrete failure scenario*: export to a drive with just enough free space for 1500 of a project's
2000 images. Copy loop raises `OSError` at image 1501. Error dialog shown, user closes it without
reading closely, later manually opens `exports/postshot/images/` in Explorer, sees ~1500 files,
assumes it's the full set.

### 3. [Low] `IN (?,?,?...)` cascade-delete clauses are SQLite-version-dependent at very large scale

`clear_frames_for_source` (`vine360/ingest/frames.py:98-105`) and `clear_views_for_frame`
(`vine360/projection/generate.py:35-38`, single-frame so not at risk in practice) build a
`DELETE ... WHERE frame_id IN (?,?,...)` with one bound parameter per row being deleted. Tested for
real in this environment (SQLite 3.45.1, bundled with this venv's Python): 50,000 parameters in a
single `IN` clause work fine, so this is **not reproducible here**. Older SQLite builds (pre-3.32,
2020) default `SQLITE_MAX_VARIABLE_NUMBER` to 999, which a single source's frame count could
plausibly exceed (a "quality" 0.5s-interval preset on an 8+ minute clip → 1000+ frames) — would
raise `sqlite3.OperationalError: too many SQL variables`. Worth a defensive chunking note if this
is ever deployed against a different SQLite build; not an active bug in this environment.

**Checked and found fine, worth recording so this doesn't get re-litigated**: no bare `except:`
anywhere in `src/vine360/` (grepped); no mutable default arguments (the one dataclass-instance
default that looked suspicious, `MaskBuildConfig()` in `masking/semantics.py:94` and
`masking/build.py:50`, is `@dataclass(frozen=True)` — safe, correctly immutable, not the classic
trap); no unclosed file handles (every `open()` call found uses a `with` block); no direct
`subprocess` calls outside `runners/` (the Runner/Adapter boundary from ADR 0002 is upheld
everywhere); SQLite write transactions are all short-lived (per-row/per-view commits, not one
giant transaction held across a whole extraction/projection/masking run), so the lack of an
explicit `busy_timeout` never becomes a real problem — Python's `sqlite3.connect` default 5s
timeout comfortably covers it, confirmed by reading every write site's commit pattern.

---

## UX/Design

### 1. [Moderate] Export never tells the user it includes every source, unfiltered

Both export modes pull from every source in the project with no filter
(`vine360/export/postshot.py` — `export_frames_and_masks_for_postshot`'s
`SELECT image_path FROM views` has no `WHERE` clause at all; poses mode is scoped only by whatever
the chosen SfM run happened to cover). This is already tracked as a follow-up (task: "Select
source(s) to export, with an All option"), but independent of building that feature, **the Export
panel's own UI text never discloses this** — `ExportPanel`'s mode notes
(`vine360/gui/main_window.py` around the `_on_mode_changed` method) describe *what* gets exported
but not *which sources*.

*Concrete scenario*: a project with sources "vineyard-north" and "vineyard-south". User only wants
to hand off "vineyard-south" to Postshot, clicks Export — gets both sources' images mixed into one
folder with no warning, discovers the mistake only after opening the folder or after a confused
Postshot import.

### 2. [Moderate] QueuePanel resets scroll position on every status change

`QueuePanel.refresh()` (`vine360/gui/main_window.py`) does `self.list_widget.clear()` then
rebuilds every row from scratch, on every `queue_changed` signal — which fires on every single job
transition (start, finish, fail, add, remove, retry, skip, reorder), not just the ones relevant to
what's currently visible.

*Concrete scenario*: a 20-job overnight queue. User scrolls down to check job #15's status. Job #3
(elsewhere in the list) finishes, `queue_changed` fires, the whole list is rebuilt, scroll position
resets to the top — the user loses their place and has to scroll back down, repeatedly, to watch
any job past the first screenful.

### 3. [Low] No overall queue progress indicator

The Queue dock shows the *currently running* job's progress bar/message
(`QueuePanel.bind_manager`, wiring `job_progress`/`job_started`/`job_finished` to
`self.progress_area`), but nothing summarizes "3 of 12 done" for the queue as a whole. A user
glancing at the dock mid-run has to count status dots themselves.

### 4. [Low] No in-app hint that the Queue dock exists

The right-hand dock is unlabeled beyond its own "Queue" title and has no `View` menu toggle or
first-run callout. A first-time user who doesn't notice a narrow dock on the right edge (especially
if the window isn't maximized) may never discover "Add to Queue" has anywhere useful to go. Minor
discoverability gap, not a blocker — the dock is always visible by default, just easy to overlook.

---

## Architecture/Maintainability

### 1. [Moderate] `queue_manager.py`'s only real coupling to the GUI layer is a runtime import back into `main_window.py`

`QueueManager._start_job` (`vine360/gui/queue_manager.py:501-552`) does
`from vine360.gui import main_window as mw` to reach the five `_*_worker` functions
(`_extract_frames_worker`, `_generate_views_worker`, etc.) — thin wrappers that just open a
connection, call the real library function, and close it. Conceptually, `queue_manager.py`
shouldn't need to import GUI code at all to run a pipeline stage; it's reaching back into
`main_window.py` purely because that's where those wrapper functions happen to live. This makes
`queue_manager.py`'s own claimed boundary ("chains the same worker functions... never duplicates
library-calling logic," per its own module docstring) slightly leaky: a future non-GUI queue
consumer (a CLI-driven queue runner, say) would have to import PySide6-flavored `main_window.py`
just to reach those wrappers. Moving the five `_*_worker` functions into `queue_manager.py` (or a
shared, GUI-independent module) and having `main_window.py`'s panels import them from there instead
would close this gap and match the two-layer dependency-weight split ADR 0001/0006 already
establishes elsewhere in this codebase.

*(The deferred-import comment itself — "avoids a module-load-time cycle with main_window" — is
also slightly overstated: `queue_manager.py` has no top-level import of `main_window` at all, so
there's no actual circular-import deadlock being avoided here; a top-level import would work
identically, since `mw._extract_frames_worker` is only accessed at call time, by which point
`main_window` is always fully loaded. Harmless either way, just worth knowing it's not solving the
problem its comment describes.)*

### 2. [Moderate] The queue's actual dispatch path has zero automated test coverage

`tests/test_gui_queue_manager.py`'s failure-isolation and retry tests (`test_job_failure_blocks_...`,
`test_independent_chain_keeps_running_after_a_failure`, `test_retry_unblocks_dependent_job`) all
patch `QueueManager._start_job` directly with a fake that immediately calls `_on_job_success`/
`_on_job_error` — which is the right seam for testing the *state machine*, but it means the actual
`_start_job` method (the if/elif chain mapping `job.stage` → worker function → positional args,
lines 519-541, plus the deferred `mw` import, plus the real `run_in_background`/`QThread`
dispatch) is exercised by **zero automated tests**. The only thing that ever ran it for real was
one ad hoc offscreen smoke script written during this session
(`/home/erieger/.claude/jobs/.../queue_smoke.py`) — not part of the pytest suite, doesn't run in
CI, and won't catch a future regression. Concretely: if a future change renames
`_extract_frames_worker`'s `interval_seconds` parameter, or reorders its positional args, no test
in the suite would fail — `_start_job` would either raise `TypeError` at runtime (caught, surfaced
as a job failure, so not silently wrong) or, in a worse case, silently pass a value to the wrong
positional slot if two adjacent params happened to share a type.

### 3. [Low] Same root cause as Correctness #1, listed here as a design gap too

The queue validates prerequisites at add-time and revalidates its own graph on
add/remove/reorder/retry/skip, but has no hook to revalidate when the underlying project data
changes via a path outside the queue entirely (a manual "Run now" click). A `notify_change`-style
callback the queue could subscribe to (parallel to how sidebar status already refreshes via
`AppState.notify_change`) would close this without much new machinery.

### 4. [Low] The two `STAGE_*` constant vocabularies' name collision is a live footgun, not just history

`main_window.py`'s sidebar-index integers (`STAGE_FRAMES = 2`, etc.) and `queue_manager.py`'s job-
stage strings (`STAGE_FRAMES = "frames"`, etc.) share names by coincidence. This session's own
`QUEUE_STAGE_*` import-aliasing (`main_window.py`'s import block) exists specifically because the
unaliased import silently shadowed the later integer definition — a real bug caught mid-session,
not a hypothetical. Nothing prevents a future contributor from reintroducing it: `from
vine360.gui.queue_manager import STAGE_FRAMES` typed without the alias in `main_window.py` would
again silently break every queue job's stage identification, with no `ImportError`, no linter
warning (none is configured — see `CLAUDE.md`), and no test that would catch it before a `queue
add_job` call started routing to the wrong worker. Worth a one-line comment at both constant
blocks cross-referencing the other, or renaming one vocabulary (e.g. `JobStage.FRAMES` as a proper
enum instead of module-level string constants) to make the collision structurally impossible
rather than relying on every future import remembering to alias.

---

## What's solid

- **The Runner/Adapter boundary (ADR 0002) is upheld everywhere**, not just where it's convenient
  — confirmed by grep, zero direct `subprocess` calls outside `runners/` in the whole library.
- **The cascade-delete invariant is implemented consistently and carefully.** `clear_frames_for_source`
  and `clear_views_for_frame` mirror each other exactly (masks → views → frames, DB rows and files
  both), and both carry real regression-test references in their own docstrings, not just claims.
- **Error handling is genuinely consistent across every wired GUI panel**: a domain-specific
  exception (`FrameExtractionError`, `MaskBuildError`, `SfmRegistrationError`,
  `PostshotExportError`, ...) maps to `QMessageBox.warning` with the real message; anything else
  maps to `QMessageBox.critical` with the exception type name. This pattern holds across all six
  wired stage panels, including the new queue dispatch path.
- **The queue's core dependency-graph algorithm is sound** — hand-traced `_run_next`,
  `_revalidate_dependencies`, and `_on_job_error` against concurrent-independent-chains,
  retry-unblocking, and removed-dependency edge cases; found no logic errors in the algorithm
  itself (only the test-coverage and cross-boundary gaps noted above).
- **"Never guess under ambiguity" is a real, followed precedent, not just stated once.**
  `vine360.sfm.repair_selected_model` set it first; the new queue's prerequisite auto-insertion
  (`QueueManager._resolve_prerequisites`) explicitly follows the same rule rather than reinventing
  a looser one under time pressure.
- **The test suite's "real fixtures over mocks" convention (CLAUDE.md's testing-conventions
  section) is genuinely followed**, not just claimed — real `pycolmap.synthesize_dataset`
  reconstructions, real `ffmpeg` calls gated behind availability checks, mocks used narrowly and
  each one's comment says which real test already covers the mocked call.
