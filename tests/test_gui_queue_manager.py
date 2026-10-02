"""QueueManager's own bookkeeping: dependency-graph resolution/auto-
insertion, persistence, restart recovery, and failure isolation (a
failed job blocks only its own dependents, not an independent chain).

Runs under the offscreen Qt platform, same as every other GUI test here
(QueueManager subclasses QObject for its signals). Most tests here patch
QueueManager._start_job (the seam between "which job runs next" and "how
a job actually runs") rather than spin a real QThread, since the state
machine is what's being tested. `test_real_dispatch_...` below is the
exception -- it drives the real _start_job -> run_in_background ->
QThread path end to end, specifically because a bug lived exactly there:
_start_job originally wired on_success/on_error as bare lambdas, which
don't carry QObject thread affinity, so Qt ran them synchronously on the
background QThread instead of queuing them onto the main thread. Since
they touch state.conn (created on the main thread), that raised
sqlite3.ProgrammingError and silently aborted the handler -- a job's
status stayed RUNNING forever even after its real work had finished.
Caught against a real running app, not by any test (this file's other
tests all patch past the exact code path that broke). See docs/adr/0023
and the fix comment in queue_manager.py's _start_job."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from unittest.mock import patch

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QWidget

from vine360.config import CaptureMode
from vine360.gui.main_window import AppState
from vine360.gui.queue_manager import (
    BLOCKED,
    DONE,
    FAILED,
    QUEUED,
    RUNNING,
    STAGE_EXPORT,
    STAGE_MASKS,
    STAGE_POSE,
    STAGE_PROJECTION,
    QueueManager,
    _save_job,
)
from vine360.project import create_project, open_index_db


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proj"
    create_project(root, "Test", CaptureMode.THREE_SIXTY)
    conn = open_index_db(root)
    yield root, conn
    conn.close()


def _state(root, conn) -> AppState:
    state = AppState()
    state.project_root = root
    state.conn = conn
    state.notify_change = lambda: None
    return state


def _insert_source(conn, source_id: str) -> None:
    conn.execute(
        "INSERT INTO sources (source_id, path, checksum, media_type, projection, width, height, "
        "timestamps, capture_group) VALUES (?, ?, 'deadbeef', 'video', 'equirectangular', 64, 32, '{}', NULL)",
        (source_id, f"/{source_id}.mp4"),
    )
    conn.commit()


def _insert_frames_only(conn, source_id: str, frame_id: str) -> None:
    _insert_source(conn, source_id)
    conn.execute(
        "INSERT INTO frames (frame_id, source_id, source_time, extraction_settings, path, checksum, frame_set_id) "
        "VALUES (?, ?, 0.0, '{}', 'x', 'y', ?)",
        (frame_id, source_id, source_id),
    )
    conn.commit()


def _insert_frames_and_views(conn, source_id: str, frame_id: str, view_id: str) -> None:
    _insert_frames_only(conn, source_id, frame_id)
    conn.execute(
        "INSERT INTO views (view_id, frame_id, projection_id, width, height, intrinsics, "
        "fixed_rotation, image_path) VALUES (?, ?, 'p', 64, 64, '{}', '{}', 'x')",
        (view_id, frame_id),
    )
    conn.commit()


# -- prerequisite auto-insertion -----------------------------------------


def test_masks_job_without_views_auto_inserts_frames_and_projection(project, qapp):
    root, conn = project
    _insert_source(conn, "s1")
    manager = QueueManager(_state(root, conn))

    job = manager.add_job(STAGE_MASKS, "s1", {"use_sam3_person": False, "use_sam3_sky": False}, "masks s1")

    assert job.status == QUEUED
    assert len(manager.jobs) == 3  # Frames + Projection auto-inserted ahead of Masks
    stages = [j.stage for j in manager.jobs]
    assert stages == ["frames", STAGE_PROJECTION, STAGE_MASKS]

    frames_job, projection_job, masks_job = manager.jobs
    assert frames_job.auto_added and projection_job.auto_added
    assert not masks_job.auto_added
    assert projection_job.depends_on == [frames_job.job_id]
    assert masks_job.depends_on == [projection_job.job_id]


def test_masks_job_with_existing_views_needs_no_auto_insert(project, qapp):
    root, conn = project
    _insert_frames_and_views(conn, "s1", "frame-1", "frame-1:front")
    manager = QueueManager(_state(root, conn))

    job = manager.add_job(STAGE_MASKS, "s1", {"use_sam3_person": False, "use_sam3_sky": False}, "masks s1")

    assert job.status == QUEUED
    assert job.depends_on == []
    assert len(manager.jobs) == 1


def test_pose_six_face_blocked_when_prerequisite_is_ambiguous(project, qapp):
    """Two sources both have frames but no views -- there's no single
    unambiguous source to auto-project, so the job is blocked with an
    explanation rather than guessed (matches repair_selected_model's
    "never guess under ambiguity" precedent)."""
    root, conn = project
    _insert_frames_only(conn, "s1", "frame-1")
    _insert_frames_only(conn, "s2", "frame-2")
    manager = QueueManager(_state(root, conn))

    job = manager.add_job(
        STAGE_POSE, None, {"image_source": "projections", "source_id": None, "camera_model": "SIMPLE_RADIAL"}, "pose"
    )

    assert job.status == BLOCKED
    assert "single frame set" in job.error_message


def test_pose_six_face_auto_inserts_projection_when_unambiguous(project, qapp):
    root, conn = project
    _insert_frames_only(conn, "s1", "frame-1")
    manager = QueueManager(_state(root, conn))

    job = manager.add_job(
        STAGE_POSE, None, {"image_source": "projections", "source_id": None, "camera_model": "SIMPLE_RADIAL"}, "pose"
    )

    assert job.status == QUEUED
    assert len(job.depends_on) == 1
    projection_job = manager._get(job.depends_on[0])
    assert projection_job.stage == STAGE_PROJECTION
    assert projection_job.auto_added
    assert projection_job.depends_on == []  # s1 already has frames


def test_export_frames_masks_job_for_a_specific_source_auto_inserts_that_sources_projection(project, qapp):
    """A specific export source_id makes the prerequisite unambiguous --
    same as Masks' own check -- so it's resolved precisely instead of
    falling back to the "guess only if there's a single candidate"
    project-wide path used when source_id is None (\"All\")."""
    root, conn = project
    _insert_frames_only(conn, "s1", "frame-1")
    manager = QueueManager(_state(root, conn))

    job = manager.add_job(
        STAGE_EXPORT, None, {"mode": "frames_masks", "output_dir": str(root / "out"), "source_id": "s1"}, "export"
    )

    assert job.status == QUEUED
    assert len(job.depends_on) == 1
    projection_job = manager._get(job.depends_on[0])
    assert projection_job.stage == STAGE_PROJECTION
    assert projection_job.target_source_id == "s1"
    assert projection_job.auto_added


def test_export_realityscan_job_for_a_specific_source_auto_inserts_that_sources_projection(project, qapp):
    """realityscan mode falls through the same generic source-scoped
    prerequisite path as frames_masks -- nothing in _resolve_prerequisites
    special-cases either by name, only "poses" is special-cased, so this
    confirms the fallthrough actually covers the newer mode too."""
    root, conn = project
    _insert_frames_only(conn, "s1", "frame-1")
    manager = QueueManager(_state(root, conn))

    job = manager.add_job(
        STAGE_EXPORT, None, {"mode": "realityscan", "output_dir": str(root / "out"), "source_id": "s1"}, "export"
    )

    assert job.status == QUEUED
    assert len(job.depends_on) == 1
    projection_job = manager._get(job.depends_on[0])
    assert projection_job.stage == STAGE_PROJECTION
    assert projection_job.target_source_id == "s1"
    assert projection_job.auto_added


def test_export_frames_masks_job_for_all_sources_uses_ambiguity_check(project, qapp):
    """source_id=None ("All") keeps the old project-wide behavior: two
    candidate sources with frames but no views is ambiguous, so this
    blocks with an explanation rather than guessing which one to
    project."""
    root, conn = project
    _insert_frames_only(conn, "s1", "frame-1")
    _insert_frames_only(conn, "s2", "frame-2")
    manager = QueueManager(_state(root, conn))

    job = manager.add_job(
        STAGE_EXPORT, None, {"mode": "frames_masks", "output_dir": str(root / "out"), "source_id": None}, "export"
    )

    assert job.status == BLOCKED
    assert "single frame set" in job.error_message


# -- persistence -----------------------------------------------------------


def test_queue_persists_and_reloads_across_manager_instances(project, qapp):
    root, conn = project
    _insert_frames_and_views(conn, "s1", "frame-1", "frame-1:front")
    manager = QueueManager(_state(root, conn))
    job = manager.add_job(STAGE_MASKS, "s1", {"use_sam3_person": True, "use_sam3_sky": False}, "masks s1")

    reloaded = QueueManager(_state(root, conn))
    reloaded.load()

    assert len(reloaded.jobs) == 1
    reloaded_job = reloaded.jobs[0]
    assert reloaded_job.job_id == job.job_id
    assert reloaded_job.params == {"use_sam3_person": True, "use_sam3_sky": False}
    assert reloaded_job.status == QUEUED


def test_running_job_resets_to_failed_on_reload(project, qapp):
    """A job that was RUNNING when the app last closed can't be trusted --
    the worker was killed mid-call, not finished. Loading should mark it
    retryable rather than silently resuming it."""
    root, conn = project
    _insert_frames_and_views(conn, "s1", "frame-1", "frame-1:front")
    manager = QueueManager(_state(root, conn))
    job = manager.add_job(STAGE_MASKS, "s1", {"use_sam3_person": False, "use_sam3_sky": False}, "masks s1")
    job.status = RUNNING
    _save_job(conn, job)

    reloaded = QueueManager(_state(root, conn))
    reloaded.load()

    reloaded_job = reloaded._get(job.job_id)
    assert reloaded_job.status == FAILED
    assert reloaded_job.error_message == "interrupted by app restart"


# -- failure isolation -------------------------------------------------


def test_job_failure_blocks_only_its_own_dependents(project, qapp):
    root, conn = project
    _insert_frames_and_views(conn, "s1", "frame-1", "frame-1:front")
    manager = QueueManager(_state(root, conn))

    masks_job = manager.add_job(STAGE_MASKS, "s1", {"use_sam3_person": False, "use_sam3_sky": False}, "masks s1")
    export_job = manager.add_job(
        "export", None, {"mode": "poses", "output_dir": str(root / "out"), "run_id": None}, "export"
    )
    # Force a manual dependency, as if the export job had been queued
    # right after this masks job with masking as an (unrealistic but
    # sufficient for this test) stand-in dependency edge.
    export_job.depends_on = [masks_job.job_id]

    manager._running_job_id = masks_job.job_id
    masks_job.status = RUNNING
    manager._on_job_error(masks_job.job_id, RuntimeError("boom"))

    assert manager._get(masks_job.job_id).status == FAILED
    assert manager._get(export_job.job_id).status == BLOCKED


def test_independent_chain_keeps_running_after_a_failure(project, qapp):
    root, conn = project
    _insert_frames_and_views(conn, "s1", "frame-1", "frame-1:front")
    _insert_frames_and_views(conn, "s2", "frame-2", "frame-2:front")
    manager = QueueManager(_state(root, conn))

    job1 = manager.add_job(STAGE_MASKS, "s1", {"use_sam3_person": False, "use_sam3_sky": False}, "masks s1")
    job2 = manager.add_job(STAGE_MASKS, "s2", {"use_sam3_person": False, "use_sam3_sky": False}, "masks s2")
    outcomes = {job1.job_id: "fail", job2.job_id: "succeed"}

    def fake_start_job(self, job):
        self._running_job_id = job.job_id
        job.status = RUNNING
        self._persist(job)
        if outcomes[job.job_id] == "fail":
            self._on_job_error(job.job_id, RuntimeError("boom"))
        else:
            self._on_job_success(job.job_id, None)

    with patch.object(QueueManager, "_start_job", fake_start_job):
        manager.start(owner=None)

    assert manager._get(job1.job_id).status == FAILED
    assert manager._get(job2.job_id).status == DONE  # unaffected by job1's failure


def test_retry_unblocks_dependent_job(project, qapp):
    root, conn = project
    _insert_frames_and_views(conn, "s1", "frame-1", "frame-1:front")
    manager = QueueManager(_state(root, conn))

    masks_job = manager.add_job(STAGE_MASKS, "s1", {"use_sam3_person": False, "use_sam3_sky": False}, "masks s1")
    export_job = manager.add_job(
        "export", None, {"mode": "poses", "output_dir": str(root / "out"), "run_id": None}, "export"
    )
    export_job.depends_on = [masks_job.job_id]

    manager._running_job_id = masks_job.job_id
    masks_job.status = RUNNING
    manager._on_job_error(masks_job.job_id, RuntimeError("boom"))
    assert manager._get(export_job.job_id).status == BLOCKED

    manager.retry_job(masks_job.job_id)
    assert manager._get(masks_job.job_id).status == QUEUED
    assert manager._get(export_job.job_id).status == QUEUED  # unblocked once its dependency is retryable again


# -- real dispatch (no patched seam) -------------------------------------


def _run_queue_until_idle(manager: QueueManager, owner: QWidget, timeout_ms: int = 10000) -> None:
    """Pumps a real Qt event loop until the queue goes idle or fails --
    the point is letting real cross-thread Qt signal delivery actually
    happen (QThread -> main thread), which a patched _start_job never
    exercises."""
    loop = QEventLoop()
    manager.queue_idle.connect(loop.quit)
    manager.job_failed.connect(lambda *_args: loop.quit())
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    manager.start(owner)
    loop.exec()
    timer.stop()


def test_real_dispatch_persists_done_status_without_a_cross_thread_error(project, qapp, tmp_path):
    root, conn = project
    _insert_frames_and_views(conn, "s1", "frame-1", "frame-1:front")
    manager = QueueManager(_state(root, conn))
    job = manager.add_job(
        STAGE_EXPORT,
        None,
        {"mode": "frames_masks", "output_dir": str(tmp_path / "out"), "source_id": None},
        "export",
    )

    _run_queue_until_idle(manager, QWidget())

    assert job.status == DONE, f"job stuck at {job.status!r} (error: {job.error_message})"
    row = conn.execute("SELECT status FROM queue_jobs WHERE job_id = ?", (job.job_id,)).fetchone()
    assert row is not None and row[0] == "done"  # actually persisted, not just updated in memory

    # ...and the run was recorded in the activity log, attributed to the queue.
    from vine360.activity_log import list_entries

    (entry,) = list_entries(conn)
    assert entry.operation == "export.postshot_images" and entry.status == "done"
    assert entry.origin == "queue: export"
    assert entry.params["output_dir"] == str(tmp_path / "out")


# -- cancel / pause -------------------------------------------------------


def test_cancel_all_is_a_safe_noop_when_queue_is_already_empty(project, qapp):
    root, conn = project
    manager = QueueManager(_state(root, conn))

    manager.cancel_all()

    assert manager.jobs == []
    assert manager._active is False


def test_cancel_all_with_nothing_running_empties_the_queue(project, qapp):
    root, conn = project
    _insert_frames_and_views(conn, "s1", "frame-1", "frame-1:front")
    _insert_source(conn, "s2")
    manager = QueueManager(_state(root, conn))

    manager.add_job(STAGE_MASKS, "s1", {"use_sam3_person": False, "use_sam3_sky": False}, "masks s1")
    manager.add_job(STAGE_PROJECTION, "s2", {"face_size": 512, "fov_degrees": 90.0, "face_names": ["front"]}, "proj s2")

    manager.cancel_all()

    assert manager.jobs == []
    assert manager._active is False
    row = conn.execute("SELECT COUNT(*) FROM queue_jobs").fetchone()
    assert row[0] == 0  # actually removed from the DB, not just marked skipped in memory


def test_cancel_all_while_a_job_is_running_leaves_only_that_job_then_clears_once_it_succeeds(project, qapp):
    root, conn = project
    _insert_frames_and_views(conn, "s1", "frame-1", "frame-1:front")
    _insert_source(conn, "s2")
    manager = QueueManager(_state(root, conn))

    running_job = manager.add_job(STAGE_MASKS, "s1", {"use_sam3_person": False, "use_sam3_sky": False}, "masks s1")
    queued_job = manager.add_job(
        STAGE_PROJECTION, "s2", {"face_size": 512, "fov_degrees": 90.0, "face_names": ["front"]}, "proj s2"
    )
    manager._running_job_id = running_job.job_id
    running_job.status = RUNNING

    manager.cancel_all()

    assert manager._active is False
    assert [j.job_id for j in manager.jobs] == [running_job.job_id]
    assert manager._get(queued_job.job_id) is None  # removed immediately

    # The in-flight worker finishes for real (nothing was actually
    # interrupted) -- the deferred cancel should now remove it too.
    manager._on_job_success(running_job.job_id, None)

    assert manager.jobs == []
    row = conn.execute("SELECT COUNT(*) FROM queue_jobs").fetchone()
    assert row[0] == 0


def test_cancel_all_while_a_job_is_running_clears_it_once_it_fails(project, qapp):
    root, conn = project
    _insert_frames_and_views(conn, "s1", "frame-1", "frame-1:front")
    manager = QueueManager(_state(root, conn))

    running_job = manager.add_job(STAGE_MASKS, "s1", {"use_sam3_person": False, "use_sam3_sky": False}, "masks s1")
    manager._running_job_id = running_job.job_id
    running_job.status = RUNNING

    manager.cancel_all()
    manager._on_job_error(running_job.job_id, RuntimeError("boom"))

    assert manager.jobs == []
    row = conn.execute("SELECT COUNT(*) FROM queue_jobs").fetchone()
    assert row[0] == 0


def test_cancel_all_leaves_already_terminal_jobs_alone(project, qapp):
    root, conn = project
    _insert_frames_and_views(conn, "s1", "frame-1", "frame-1:front")
    manager = QueueManager(_state(root, conn))

    done_job = manager.add_job(STAGE_MASKS, "s1", {"use_sam3_person": False, "use_sam3_sky": False}, "masks s1")
    done_job.status = DONE
    manager._persist(done_job)

    manager.cancel_all()

    assert manager._get(done_job.job_id) is not None
    assert manager._get(done_job.job_id).status == DONE


# -- frame sets (docs/adr/0034) -------------------------------------------


def test_projection_job_for_a_queued_frame_set_depends_on_that_frames_job(project, qapp):
    root, conn = project
    _insert_source(conn, "s1")
    manager = QueueManager(_state(root, conn))

    frames_job = manager.add_job(
        "frames", "s1", {"interval_seconds": 0.5, "start_time": None, "end_time": None}, "frames s1 0.5s"
    )
    assert frames_job.target_frame_set_id == "s1~i0.5"

    projection_job = manager.add_job(
        STAGE_PROJECTION,
        "s1",
        {"face_size": 64, "fov_degrees": 90.0, "face_names": ["front"]},
        "proj",
        target_frame_set_id="s1~i0.5",
    )
    assert projection_job.depends_on == [frames_job.job_id]
    assert len(manager.jobs) == 2  # nothing auto-inserted -- the queued Frames job already covers it


def test_projection_job_for_an_unknown_non_default_frame_set_is_blocked_not_guessed(project, qapp):
    """Only the balanced-preset frame set can be auto-extracted -- anything
    else would mean inventing extraction settings nobody chose."""
    root, conn = project
    _insert_source(conn, "s1")
    manager = QueueManager(_state(root, conn))

    job = manager.add_job(
        STAGE_PROJECTION,
        "s1",
        {"face_size": 64, "fov_degrees": 90.0, "face_names": ["front"]},
        "proj",
        target_frame_set_id="s1~i0.25",
    )
    assert job.status == BLOCKED
    assert len(manager.jobs) == 1


def test_source_level_masks_job_is_blocked_when_the_source_has_several_frame_sets(project, qapp):
    root, conn = project
    _insert_source(conn, "s1")
    for frame_set_id in ("s1~i0.5", "s1~i1"):
        conn.execute(
            "INSERT INTO frame_sets (frame_set_id, source_id, label, extraction_settings, created_at) "
            "VALUES (?, 's1', 'x', '{}', '2026-01-01')",
            (frame_set_id,),
        )
    conn.commit()
    manager = QueueManager(_state(root, conn))

    job = manager.add_job(STAGE_MASKS, "s1", {"use_sam3_person": False, "use_sam3_sky": False}, "masks s1")

    assert job.status == BLOCKED
    assert "2 frame sets" in job.error_message


def test_jobs_persisted_before_frame_sets_reload_targeting_the_legacy_frame_set(project, qapp):
    """A Projection/Masks job saved before target_frame_set_id existed
    reloads targeting the source's migrated legacy frame set (id ==
    source_id)."""
    root, conn = project
    _insert_frames_only(conn, "s1", "frame-1")
    manager = QueueManager(_state(root, conn))
    job = manager.add_job(STAGE_MASKS, "s1", {"use_sam3_person": False, "use_sam3_sky": False}, "masks s1")
    conn.execute("UPDATE queue_jobs SET target_frame_set_id = NULL WHERE job_id = ?", (job.job_id,))
    conn.commit()

    reloaded = QueueManager(_state(root, conn))
    reloaded.load()
    masks_job = next(j for j in reloaded.jobs if j.stage == STAGE_MASKS)
    assert masks_job.target_frame_set_id == "s1"
