"""Persisted job queue for unattended, multi-stage pipeline runs -- see
docs/adr/0023-persisted-processing-queue.md.

Chains the same worker functions main_window.py's panels already call
manually via `run_in_background` (`_extract_frames_worker`,
`_generate_views_worker`, `_build_masks_worker`, `_run_sfm_worker`,
`_export_postshot_worker`/`_export_frames_and_masks_worker`) -- this
module never duplicates their library-calling logic, it only decides
*which one to call next* and persists the plan.

Two things the rest of the app doesn't need: a dependency graph (each job
records the job_id(s) it needs DONE first, not just its position in the
list) and prerequisite auto-insertion (adding a job whose input data
doesn't exist yet, e.g. Masks on a source with no projections, queues the
missing upstream job(s) ahead of it automatically). The dependency graph
is what lets one chain's failure block only its own dependents while an
unrelated source's queued chain keeps running -- see `_run_next`.

Every job's params are captured in full when it's added to the queue,
never re-prompted at execution time, since the queue may run unattended.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Signal

from vine360.config import FRAME_PRESET_INTERVALS, FramePreset
from vine360.projection.cubemap import ALL_FACE_NAMES

# Job status vocabulary. Deliberately separate from main_window's
# DONE/ACTIVE/PENDING stage-status vocabulary (a job's lifecycle -- has it
# run yet -- is a different question from a pipeline stage's data-derived
# status), even though QueuePanel renders both with the same dot idiom.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
BLOCKED = "blocked"
SKIPPED = "skipped"

TERMINAL_NOT_DONE = (FAILED, SKIPPED)  # statuses that can never satisfy a dependency

STAGE_FRAMES = "frames"
STAGE_PROJECTION = "projection"
STAGE_MASKS = "masks"
STAGE_POSE = "pose"
STAGE_EXPORT = "export"

_COLUMNS = (
    "job_id, order_index, stage, label, target_source_id, params, depends_on, "
    "status, auto_added, error_message, created_at, started_at, finished_at, target_frame_set_id"
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class QueueJob:
    job_id: str
    order_index: int
    stage: str
    label: str
    target_source_id: str | None
    params: dict
    depends_on: list[str] = field(default_factory=list)
    status: str = QUEUED
    auto_added: bool = False
    error_message: str | None = None
    created_at: str = field(default_factory=_utcnow_iso)
    started_at: str | None = None
    finished_at: str | None = None
    # The frame set (docs/adr/0034) a Frames/Projection/Masks job writes or
    # reads -- what prerequisite resolution is keyed on. A Frames job's is
    # computed up front (frame-set ids are deterministic per config), so
    # its dependents can target a frame set that doesn't exist yet.
    target_frame_set_id: str | None = None

    @classmethod
    def from_row(cls, row: tuple) -> "QueueJob":
        (
            job_id, order_index, stage, label, target_source_id, params_json, depends_on_json,
            status, auto_added, error_message, created_at, started_at, finished_at, target_frame_set_id,
        ) = row
        if target_frame_set_id is None and stage in (STAGE_PROJECTION, STAGE_MASKS):
            # Persisted before frame sets existed: the source's frames were
            # migrated into a frame set whose id is the source_id itself.
            target_frame_set_id = target_source_id
        return cls(
            job_id=job_id,
            order_index=order_index,
            stage=stage,
            label=label,
            target_source_id=target_source_id,
            params=json.loads(params_json),
            depends_on=json.loads(depends_on_json),
            status=status,
            auto_added=bool(auto_added),
            error_message=error_message,
            created_at=created_at,
            started_at=started_at,
            finished_at=finished_at,
            target_frame_set_id=target_frame_set_id,
        )

    def to_row(self) -> tuple:
        return (
            self.job_id, self.order_index, self.stage, self.label, self.target_source_id,
            json.dumps(self.params), json.dumps(self.depends_on), self.status,
            int(self.auto_added), self.error_message, self.created_at, self.started_at,
            self.finished_at, self.target_frame_set_id,
        )


def _load_jobs(conn: sqlite3.Connection) -> list[QueueJob]:
    rows = conn.execute(f"SELECT {_COLUMNS} FROM queue_jobs ORDER BY order_index").fetchall()
    jobs = [QueueJob.from_row(row) for row in rows]
    for job in jobs:
        if job.status == RUNNING:
            # Whatever worker (ffmpeg/pycolmap/SAM3) was mid-call when the
            # app last closed was killed, not finished -- it can't be
            # trusted. Safe to mark retryable rather than resuming it as
            # QUEUED: every worker is idempotent on a fresh call (cascade-
            # delete on frame/view regeneration, INSERT OR REPLACE for
            # masks, a fresh run_id/dir per SfM run, managed-subdir reset
            # on export -- see docs/adr/0017/0019), but none of them
            # support resuming a half-written call.
            job.status = FAILED
            job.error_message = "interrupted by app restart"
            _save_job(conn, job)
    return jobs


def _save_job(conn: sqlite3.Connection, job: QueueJob) -> None:
    conn.execute(
        f"INSERT OR REPLACE INTO queue_jobs ({_COLUMNS}) VALUES ({','.join('?' * 14)})", job.to_row()
    )
    conn.commit()


def _delete_job(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute("DELETE FROM queue_jobs WHERE job_id = ?", (job_id,))
    conn.commit()


def _has_frames(conn: sqlite3.Connection, frame_set_id: str) -> bool:
    return conn.execute("SELECT COUNT(*) FROM frames WHERE frame_set_id = ?", (frame_set_id,)).fetchone()[0] > 0


def _has_views(conn: sqlite3.Connection, frame_set_id: str) -> bool:
    return (
        conn.execute(
            "SELECT COUNT(*) FROM views v JOIN frames f ON v.frame_id = f.frame_id WHERE f.frame_set_id = ?",
            (frame_set_id,),
        ).fetchone()[0]
        > 0
    )


def _has_any_views(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT COUNT(*) FROM views").fetchone()[0] > 0


def _has_any_sfm_run(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT COUNT(*) FROM sfm_runs").fetchone()[0] > 0


def _frame_sets_with_frames_but_no_views(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """(frame_set_id, source_id) pairs."""
    rows = conn.execute(
        "SELECT DISTINCT f.frame_set_id, f.source_id FROM frames f "
        "LEFT JOIN views v ON v.frame_id = f.frame_id WHERE v.view_id IS NULL AND f.frame_set_id IS NOT NULL"
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _source_display_name(conn: sqlite3.Connection, source_id: str) -> str:
    row = conn.execute("SELECT path FROM sources WHERE source_id = ?", (source_id,)).fetchone()
    return Path(row[0]).name if row else source_id[:8]


def _default_params_for_prerequisite(stage: str) -> dict:
    """Matches the defaults each stage's manual panel shows -- see
    docs/adr/0023 and the queue's own module docstring."""
    if stage == STAGE_FRAMES:
        return {
            "interval_seconds": FRAME_PRESET_INTERVALS[FramePreset.BALANCED],
            "start_time": None,
            "end_time": None,
            "generate_thumbnails": True,
        }
    if stage == STAGE_PROJECTION:
        return {
            "face_size": 1024,
            "fov_degrees": 90.0,
            "face_names": [name for name in ALL_FACE_NAMES if name not in ("up", "down")],
        }
    raise ValueError(f"no default prerequisite params for stage {stage!r}")


def _default_label(stage: str, source_name: str) -> str:
    if stage == STAGE_FRAMES:
        return f"Frame extraction — {source_name} (balanced preset, auto)"
    if stage == STAGE_PROJECTION:
        return f"Projection — {source_name} (default faces, auto)"
    raise ValueError(stage)


def _params_frame_set_id(params: dict) -> str | None:
    """Pose/Export jobs persisted before frame sets carry "source_id",
    which is also that source's migrated legacy frame_set_id."""
    return params.get("frame_set_id", params.get("source_id"))


class QueueManager(QObject):
    """Owns the ordered queue of jobs for the currently open project,
    persisted to that project's queue_jobs table. Runs at most one job at
    a time, through the same `run_in_background` every manual panel
    action already uses, but a job's failure only blocks jobs that
    actually depend on it (`depends_on`) -- an independent source's
    queued chain keeps running past an unrelated failure. See
    docs/adr/0023-persisted-processing-queue.md."""

    job_started = Signal(str)  # job_id
    job_finished = Signal(str)  # job_id
    job_failed = Signal(str, str)  # job_id, error message
    job_progress = Signal(str, object, object)  # message, current, total
    queue_changed = Signal()  # any add/remove/reorder/status change
    queue_idle = Signal()  # nothing left runnable (queue finished or stalled)

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.jobs: list[QueueJob] = []
        self._running_job_id: str | None = None
        self._active = False
        self._owner = None
        self._cancel_requested = False

    # -- persistence ---------------------------------------------------

    def load(self) -> None:
        """Reloads the queue from the (possibly newly-opened) project's
        DB. Call after AppState.conn changes."""
        self.jobs = _load_jobs(self.state.conn) if self.state.conn is not None else []
        self._running_job_id = None
        self._active = False
        self._cancel_requested = False
        self.queue_changed.emit()

    def _persist(self, job: QueueJob) -> None:
        if self.state.conn is not None:
            _save_job(self.state.conn, job)

    @property
    def is_running(self) -> bool:
        return self._running_job_id is not None

    def _get(self, job_id: str) -> QueueJob | None:
        return next((j for j in self.jobs if j.job_id == job_id), None)

    def _renumber(self) -> None:
        for index, job in enumerate(self.jobs):
            job.order_index = index
            self._persist(job)

    # -- adding jobs, with prerequisite auto-insertion ------------------

    def add_job(
        self,
        stage: str,
        target_source_id: str | None,
        params: dict,
        label: str,
        *,
        target_frame_set_id: str | None = None,
    ) -> QueueJob:
        """target_frame_set_id: the frame set a Frames job will write (its
        deterministic id, vine360.ingest.frames.frame_set_id_for) or a
        Projection/Masks job will read. Pose/Export carry theirs in params
        ("frame_set_id"), since for them it's an optional filter.

        Left as None, it's derived from target_source_id where that's
        unambiguous (see _frame_set_for_source) -- a convenience for
        callers that think in sources; the GUI always passes it."""
        ambiguity = None
        if target_frame_set_id is None and target_source_id is not None:
            if stage == STAGE_FRAMES:
                from vine360.ingest.frames import frame_set_id_for

                target_frame_set_id = frame_set_id_for(
                    self.state.conn,
                    target_source_id,
                    interval_seconds=params["interval_seconds"],
                    start_time=params.get("start_time"),
                    end_time=params.get("end_time"),
                )
            elif stage in (STAGE_PROJECTION, STAGE_MASKS):
                target_frame_set_id, ambiguity = self._frame_set_for_source(target_source_id)
        if ambiguity:
            depends_on, blocked_reason = [], ambiguity
        else:
            depends_on, blocked_reason = self._resolve_prerequisites(stage, target_frame_set_id, params)
        job = QueueJob(
            job_id=f"job-{uuid.uuid4().hex[:8]}",
            order_index=len(self.jobs),
            stage=stage,
            label=label,
            target_source_id=target_source_id,
            params=params,
            depends_on=depends_on,
            status=BLOCKED if blocked_reason else QUEUED,
            error_message=blocked_reason,
            target_frame_set_id=target_frame_set_id,
        )
        self.jobs.append(job)
        self._persist(job)
        self._renumber()
        self.queue_changed.emit()
        return job

    def _find_existing(self, stage: str, frame_set_id: str | None) -> QueueJob | None:
        return next(
            (
                j
                for j in self.jobs
                if j.stage == stage
                and j.target_frame_set_id == frame_set_id
                and j.status not in TERMINAL_NOT_DONE
            ),
            None,
        )

    def _queue_prerequisite(self, stage: str, frame_set_id: str, source_id: str) -> QueueJob:
        conn = self.state.conn
        params = _default_params_for_prerequisite(stage)
        deps, blocked_reason = self._resolve_prerequisites(stage, frame_set_id, params)
        job = QueueJob(
            job_id=f"job-{uuid.uuid4().hex[:8]}",
            order_index=len(self.jobs),
            stage=stage,
            label=_default_label(stage, _source_display_name(conn, source_id)),
            target_source_id=source_id,
            params=params,
            depends_on=deps,
            status=BLOCKED if blocked_reason else QUEUED,
            error_message=blocked_reason,
            auto_added=True,
            target_frame_set_id=frame_set_id,
        )
        self.jobs.append(job)
        self._persist(job)
        return job

    def _frame_set_for_source(self, source_id: str) -> tuple[str | None, str | None]:
        """(frame_set_id, ambiguity_reason) for a source-level request: its
        only frame set (extracted or queued), the balanced-preset one if it
        has none yet, or -- never guessing between several -- (None,
        reason)."""
        from vine360.ingest.frames import frame_set_id_for

        candidates = {
            row[0]
            for row in self.state.conn.execute("SELECT frame_set_id FROM frame_sets WHERE source_id = ?", (source_id,))
        }
        candidates |= {
            j.target_frame_set_id
            for j in self.jobs
            if j.stage == STAGE_FRAMES
            and j.target_source_id == source_id
            and j.target_frame_set_id
            and j.status not in TERMINAL_NOT_DONE
        }
        if not candidates:
            # Pre-frame-set data: frames with no frame_sets row yet.
            row = self.state.conn.execute(
                "SELECT frame_set_id FROM frames WHERE source_id = ? AND frame_set_id IS NOT NULL LIMIT 1", (source_id,)
            ).fetchone()
            if row:
                return row[0], None
            defaults = _default_params_for_prerequisite(STAGE_FRAMES)
            return frame_set_id_for(self.state.conn, source_id, interval_seconds=defaults["interval_seconds"]), None
        if len(candidates) == 1:
            return candidates.pop(), None
        return None, f"source has {len(candidates)} frame sets -- choose which one this job is for"

    def _source_of_frame_set(self, frame_set_id: str) -> str | None:
        row = self.state.conn.execute(
            "SELECT source_id FROM frame_sets WHERE frame_set_id = ?", (frame_set_id,)
        ).fetchone()
        if row:
            return row[0]
        # Not extracted yet: a queued Frames job may already be targeting it...
        pending = self._find_existing(STAGE_FRAMES, frame_set_id)
        if pending:
            return pending.target_source_id
        # ...or it's an id computed for a not-yet-extracted source, which
        # is always "<source_id>~<tag>" (vine360.ingest.frames.frame_set_id_for).
        source_id = frame_set_id.split("~", 1)[0]
        exists = self.state.conn.execute("SELECT 1 FROM sources WHERE source_id = ?", (source_id,)).fetchone()
        return source_id if exists else None

    def _require_frames(self, frame_set_id: str) -> tuple[list[str], str | None]:
        """Dependency on frame_set_id's frames existing: already extracted,
        already queued, or auto-queued -- but only when the default
        (balanced-preset) extraction would produce exactly this frame set.
        Any other config would be a guess at settings nobody chose, so the
        job is BLOCKED instead (the "never guess" rule below)."""
        from vine360.ingest.frames import frame_set_id_for

        conn = self.state.conn
        existing = self._find_existing(STAGE_FRAMES, frame_set_id)
        if existing:
            return [existing.job_id], None
        if _has_frames(conn, frame_set_id):
            return [], None
        source_id = self._source_of_frame_set(frame_set_id)
        if source_id is None:
            return [], f"unknown frame set {frame_set_id!r} -- extract its frames first"
        defaults = _default_params_for_prerequisite(STAGE_FRAMES)
        if frame_set_id_for(conn, source_id, interval_seconds=defaults["interval_seconds"]) != frame_set_id:
            return [], f"frame set {frame_set_id!r} has no frames and no Frames job is queued for it -- add one first"
        return [self._queue_prerequisite(STAGE_FRAMES, frame_set_id, source_id).job_id], None

    def _require_views(self, frame_set_id: str) -> tuple[list[str], str | None]:
        existing = self._find_existing(STAGE_PROJECTION, frame_set_id)
        if existing:
            return [existing.job_id], None
        if _has_views(self.state.conn, frame_set_id):
            return [], None
        source_id = self._source_of_frame_set(frame_set_id)
        if source_id is None:
            return [], f"unknown frame set {frame_set_id!r} -- extract its frames first"
        return [self._queue_prerequisite(STAGE_PROJECTION, frame_set_id, source_id).job_id], None

    def _resolve_prerequisites(
        self, stage: str, target_frame_set_id: str | None, params: dict
    ) -> tuple[list[str], str | None]:
        """Returns (depends_on job_ids, blocked_reason). Auto-inserts a
        missing upstream job only when there's exactly one unambiguous
        choice of what to insert (e.g. a single frame set with extracted
        frames but no views). When the choice would be a guess -- no
        candidate, or more than one -- it never guesses: the job is added
        as BLOCKED with an explanatory error_message instead, matching
        the project's existing "never guess under ambiguity" precedent
        (vine360.sfm.repair_selected_model)."""
        conn = self.state.conn

        if stage == STAGE_PROJECTION and target_frame_set_id:
            return self._require_frames(target_frame_set_id)

        if stage == STAGE_MASKS and target_frame_set_id:
            return self._require_views(target_frame_set_id)

        if stage == STAGE_POSE:
            frame_set_id = _params_frame_set_id(params)
            if params.get("image_source") == "frames":
                if not frame_set_id:
                    return [], "no frame set selected for the native-equirectangular engine"
                return self._require_frames(frame_set_id)
            if frame_set_id:
                return self._require_views(frame_set_id)
            # Six-face-projections engine over the whole project/
            # projections/ -- not scoped to one frame set -- so its
            # prerequisite is "some" Projection output existing.
            return self._resolve_project_wide_projection_prerequisite(conn)

        if stage == STAGE_EXPORT:
            if params.get("mode") == "poses":
                run_frame_set_id = self._frame_run_frame_set(params.get("run_id"))
                if run_frame_set_id:
                    # A raw-360-frame run exports its frame set's projected
                    # views (vine360.sfm.frame_poses, docs/adr/0036).
                    return self._require_views(run_frame_set_id)
                if _has_any_sfm_run(conn):
                    return [], None
                existing = next(
                    (j for j in self.jobs if j.stage == STAGE_POSE and j.status not in TERMINAL_NOT_DONE), None
                )
                if existing:
                    return [existing.job_id], None
                return [], "no SfM run exists and no Pose estimation job is queued -- add one first"
            # Any other mode (frames_masks, realityscan -- both frame-set-
            # filterable, ADR 0026/0028/0034): a specific frame set makes
            # this unambiguous (same as Masks' own prerequisite check), so
            # it's handled precisely instead of falling back to the
            # project-wide "guess only if there's a single candidate" path
            # used when the mode's own filter is "All".
            export_frame_set_id = _params_frame_set_id(params)
            if export_frame_set_id:
                return self._require_views(export_frame_set_id)
            return self._resolve_project_wide_projection_prerequisite(conn)

        return [], None

    def _frame_run_frame_set(self, run_id: str | None) -> str | None:
        """The frame set of a raw-frame (360) SfM run, else None."""
        if not run_id:
            return None
        row = self.state.conn.execute("SELECT config FROM sfm_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        config = json.loads(row[0])
        if config.get("image_source") != "frames":
            return None
        return config.get("frame_set_id") or config.get("source_id")

    def _resolve_project_wide_projection_prerequisite(self, conn: sqlite3.Connection) -> tuple[list[str], str | None]:
        if _has_any_views(conn):
            return [], None
        existing = next(
            (j for j in self.jobs if j.stage == STAGE_PROJECTION and j.status not in TERMINAL_NOT_DONE), None
        )
        if existing:
            return [existing.job_id], None
        candidates = _frame_sets_with_frames_but_no_views(conn)
        if len(candidates) == 1:
            frame_set_id, source_id = candidates[0]
            return [self._queue_prerequisite(STAGE_PROJECTION, frame_set_id, source_id).job_id], None
        return (
            [],
            "no projected views exist and no single frame set to auto-project "
            f"({len(candidates)} candidate frame set(s)) -- add a Projection job first",
        )

    # -- reordering / removing / retrying --------------------------------

    def remove_job(self, job_id: str) -> None:
        job = self._get(job_id)
        if job is None or job.status == RUNNING:
            return
        self.jobs.remove(job)
        if self.state.conn is not None:
            _delete_job(self.state.conn, job_id)
        self._renumber()
        self._revalidate_dependencies()
        self.queue_changed.emit()

    def move_job(self, job_id: str, new_index: int) -> None:
        job = self._get(job_id)
        if job is None:
            return
        self.jobs.remove(job)
        self.jobs.insert(max(0, min(new_index, len(self.jobs))), job)
        self._renumber()
        self._revalidate_dependencies()
        self.queue_changed.emit()

    def reorder(self, job_ids: list[str]) -> None:
        """Replaces the queue's order to match job_ids, e.g. after a
        drag-and-drop reorder in QueuePanel's list widget. Any job_id not
        currently in the queue is ignored; any queued job not named in
        job_ids keeps its relative position, appended at the end."""
        by_id = {j.job_id: j for j in self.jobs}
        new_order = [by_id[jid] for jid in job_ids if jid in by_id]
        remaining = [j for j in self.jobs if j.job_id not in job_ids]
        self.jobs = new_order + remaining
        self._renumber()
        self._revalidate_dependencies()
        self.queue_changed.emit()

    def retry_job(self, job_id: str) -> None:
        job = self._get(job_id)
        if job is None or job.status != FAILED:
            return
        job.status = QUEUED
        job.error_message = None
        self._persist(job)
        self._revalidate_dependencies()
        self.queue_changed.emit()
        if self._active:
            self._run_next()

    def skip_job(self, job_id: str) -> None:
        job = self._get(job_id)
        if job is None or job.status not in (QUEUED, FAILED, BLOCKED):
            return
        job.status = SKIPPED
        self._persist(job)
        self._revalidate_dependencies()
        self.queue_changed.emit()
        if self._active:
            self._run_next()

    def _revalidate_dependencies(self) -> None:
        """A QUEUED job whose depends_on includes a job that's FAILED,
        BLOCKED, SKIPPED, or no longer in the queue flips to BLOCKED; a
        BLOCKED job whose dependencies are now all satisfied flips back
        to QUEUED. Runs to a fixed point so a change propagates through a
        whole dependent chain, not just one hop."""
        by_id = {j.job_id: j for j in self.jobs}
        changed = False
        for job in self.jobs:
            if job.status not in (QUEUED, BLOCKED):
                continue
            unmet = [
                dep
                for dep in job.depends_on
                if by_id.get(dep) is None or by_id[dep].status in (FAILED, BLOCKED, SKIPPED)
            ]
            new_status = BLOCKED if unmet else QUEUED
            if new_status != job.status:
                job.status = new_status
                if new_status == QUEUED:
                    job.error_message = None
                self._persist(job)
                changed = True
        if changed:
            self._revalidate_dependencies()

    # -- execution --------------------------------------------------------

    def start(self, owner) -> None:
        """owner: a QWidget (the QueuePanel) that outlives the queue's
        run_in_background calls."""
        self._owner = owner
        self._active = True
        self._run_next()

    def pause(self) -> None:
        """Stops advancing to new jobs once the current one finishes.
        Doesn't interrupt a job already running -- run_in_background's
        workers have no cancellation hook."""
        self._active = False

    def cancel_all(self) -> None:
        """Actually empties the pending backlog -- QUEUED/BLOCKED jobs are
        removed outright (via remove_job, which also persists the
        deletion), not just marked SKIPPED-and-left-visible. A job
        already RUNNING can't be interrupted -- run_in_background's
        workers have no cancellation hook (ffmpeg is a blocking
        subprocess.run; pycolmap/SAM3 are in-process Python calls with no
        process handle to kill) -- so it's left to finish for real;
        _cancel_requested records that it should be removed too the
        moment its own success/error handler fires (_on_job_success/
        _on_job_error), so the queue ends up genuinely empty once that
        job naturally completes. Already-terminal jobs (DONE/FAILED/
        SKIPPED) are left alone, matching retry_job/skip_job's existing
        habit of only ever touching non-terminal jobs."""
        self.pause()
        if self._running_job_id is not None:
            self._cancel_requested = True
        for job in list(self.jobs):
            if job.status in (QUEUED, BLOCKED):
                self.remove_job(job.job_id)
        self.queue_changed.emit()

    def _run_next(self) -> None:
        if not self._active or self._running_job_id is not None:
            return
        self._revalidate_dependencies()
        by_id = {j.job_id: j for j in self.jobs}
        runnable = next(
            (
                j
                for j in self.jobs
                if j.status == QUEUED and all(by_id.get(d) is not None and by_id[d].status == DONE for d in j.depends_on)
            ),
            None,
        )
        if runnable is None:
            self._active = False
            self.queue_idle.emit()
            return
        self._start_job(runnable)

    def _start_job(self, job: QueueJob) -> None:
        # Local import: avoids a module-load-time cycle with main_window
        # (which imports QueueManager), while still reusing its worker
        # functions unmodified -- by the time a job actually runs, the app
        # is fully constructed and main_window's module namespace is
        # complete.
        from vine360.gui import main_window as mw

        self._running_job_id = job.job_id
        job.status = RUNNING
        job.started_at = _utcnow_iso()
        job.error_message = None
        self._persist(job)
        self.queue_changed.emit()
        self.job_started.emit(job.job_id)

        project_root = self.state.project_root
        params = job.params
        if job.stage == STAGE_FRAMES:
            fn, args = mw._extract_frames_worker, (
                project_root, job.target_source_id, params["interval_seconds"],
                params.get("start_time"), params.get("end_time"), params.get("generate_thumbnails", True),
                job.target_frame_set_id,
            )
        elif job.stage == STAGE_PROJECTION:
            fn, args = mw._generate_views_worker, (
                project_root, job.target_frame_set_id, params["face_size"], params["fov_degrees"],
                params["face_names"], params.get("max_workers"),
            )
        elif job.stage == STAGE_MASKS:
            # .get(): jobs queued before mask layers (docs/adr/0038) only
            # carry the two SAM 3 flags -- they keep their old meaning.
            fn, args = mw._build_masks_worker, (
                project_root, job.target_frame_set_id, params["use_sam3_person"], params["use_sam3_sky"],
                params.get("mask_sky", True), params.get("mask_overexposure", False),
                params.get("overexposure_clip", 250), params.get("overexposure_bloom_radius", 40),
            )
        elif job.stage == STAGE_POSE:
            fn, args = mw._run_sfm_worker, (
                project_root, params["image_source"], _params_frame_set_id(params), params["camera_model"],
                params.get("engine", "pycolmap"),
            )
        elif job.stage == STAGE_EXPORT:
            output_dir = Path(params["output_dir"])
            export_mode = params.get("mode")
            if export_mode == "poses":
                run_id = params.get("run_id") or self._latest_run_id()
                fn, args = mw._export_postshot_worker, (project_root, output_dir, run_id)
            elif export_mode == "realityscan":
                fn, args = mw._export_realityscan_worker, (
                    project_root, output_dir, _params_frame_set_id(params),
                    params.get("include_camera_priors", False),
                )
            else:
                fn, args = mw._export_frames_and_masks_worker, (project_root, output_dir, _params_frame_set_id(params))
        else:
            raise ValueError(f"unknown queue job stage {job.stage!r}")

        from vine360.gui.main_window import run_in_background

        # Connected as bound methods of self (a QObject), not lambdas --
        # real bug, caught against a live run: a lambda closure has no
        # QObject thread affinity of its own, so Qt can't tell it needs
        # queuing onto the main thread and just calls it synchronously on
        # this background QThread instead. _handle_job_success/_error
        # then touch self.state.conn (created on the main thread), which
        # raises sqlite3.ProgrammingError ("SQLite objects created in a
        # thread can only be used in that same thread") -- silently
        # aborting the handler right after job.status was set in memory
        # but before it was ever persisted or the UI told about it, so
        # the job stays stuck showing RUNNING forever even though the
        # underlying work already finished. Bound methods of self *do*
        # carry real thread affinity, so Qt queues them onto the main
        # thread correctly, same as every manual panel's on_success (e.g.
        # FramesPanel's `on_success=self._on_extract_success`) already
        # relies on.
        run_in_background(
            self._owner,
            fn,
            *args,
            on_success=self._handle_job_success,
            on_error=self._handle_job_error,
            on_progress=self._handle_job_progress,
            # Recorded in the activity log (vine360.activity_log): which
            # queue job started this run.
            log_origin=f"queue: {job.label}",
        )

    def _handle_job_success(self, result) -> None:
        self._on_job_success(self._running_job_id, result)

    def _handle_job_error(self, exc: Exception) -> None:
        self._on_job_error(self._running_job_id, exc)

    def _handle_job_progress(self, message, current, total) -> None:
        self.job_progress.emit(message, current, total)

    def _latest_run_id(self) -> str | None:
        if self.state.conn is None:
            return None
        row = self.state.conn.execute("SELECT run_id FROM sfm_runs ORDER BY created_at DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def _on_job_success(self, job_id: str, result) -> None:
        job = self._get(job_id)
        if job is None:
            return
        job.status = DONE
        job.finished_at = _utcnow_iso()
        self._persist(job)
        self._running_job_id = None
        self.state.notify_change()
        self.job_finished.emit(job_id)
        if self._cancel_requested:
            self._cancel_requested = False
            self.remove_job(job_id)  # status is DONE now, not RUNNING -- remove_job's guard passes
        self.queue_changed.emit()
        self._run_next()

    def _on_job_error(self, job_id: str, exc: Exception) -> None:
        job = self._get(job_id)
        if job is None:
            return
        job.status = FAILED
        job.error_message = str(exc)
        job.finished_at = _utcnow_iso()
        self._persist(job)
        self._running_job_id = None
        self.state.notify_change()
        self.job_failed.emit(job_id, str(exc))
        if self._cancel_requested:
            self._cancel_requested = False
            self.remove_job(job_id)  # status is FAILED now, not RUNNING -- remove_job's guard passes
        self._revalidate_dependencies()  # blocks this chain's dependents only
        self.queue_changed.emit()
        self._run_next()  # keep going -- other, independent chains may still be runnable
