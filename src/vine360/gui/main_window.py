"""The real M6 desktop app -- built on the dashboard shape the user chose
after comparing wizard_preview.py and dashboard_preview.py (see
docs/status.md and memory).

Project creation, source ingest, frame extraction, projection generation,
mask building and SfM are all wired to real vine360 code, run on a
background QThread with a determinate progress bar + ETA wherever the
underlying function reports (current, total) counts. Training has no
sidebar stage here -- removed from the app's scope (docs/adr/0025); the
adapter-only library code (`vine360/training/`) still exists per ADR
0009, it's just not surfaced in this GUI.

Sidebar stage status (the colored dot next to each stage name) is derived
directly from the project's index database each time it changes, not
tracked ad hoc per panel -- see `compute_stage_statuses`.

Run with: python -m vine360.gui.main_window
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QPoint, Qt, QThread, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vine360.config import FRAME_PRESET_INTERVALS, CaptureMode, FramePreset
from vine360.data_manager import (
    DataManagerError,
    delete_export_dir,
    delete_frame_set,
    delete_masks_for_frame_set,
    delete_sfm_run,
    delete_views_for_frame_set,
    directory_size,
    export_dir_size,
    frame_set_disk_usage,
    list_export_dirs,
    list_frame_sets,
    list_sfm_runs,
    list_sources,
)
from vine360.ingest.frames import FrameExtractionError, extract_frames, frame_set_id_for
from vine360.ingest.media_probe import ProbeError
from vine360.ingest.sources import (
    EquirectangularConfirmationRequired,
    SourceError,
    add_source,
    remove_source,
)
from vine360.masking.build import MaskBuildError, build_masks_for_frame_set, set_view_flagged
from vine360.masking.sam3_adapter import validate_installation as sam3_validate_installation
from vine360.masking.semantics import is_keep_fraction_anomalous
from vine360.models import Project
from vine360.project import (
    ProjectAlreadyExistsError,
    ProjectNotFoundError,
    create_project,
    load_project,
    open_index_db,
)
from vine360.projection.cubemap import ALL_FACE_NAMES
from vine360.projection.generate import ProjectionGenerationError, generate_views_for_frame_set
from vine360.runners.local import LocalRunner
from vine360.runners.probe import probe_dependencies
from vine360.export.postshot import (
    PostshotExportError,
    export_for_postshot,
    export_for_realityscan,
    export_frames_and_masks_for_postshot,
    realityscan_priors_available,
)
from vine360.sfm.project_run import SfmRegistrationError, run_sfm_for_project
from vine360.gui.queue_manager import (
    BLOCKED,
    DONE as JOB_DONE,
    FAILED,
    QUEUED,
    RUNNING as JOB_RUNNING,
    SKIPPED,
    STAGE_EXPORT as QUEUE_STAGE_EXPORT,
    STAGE_FRAMES as QUEUE_STAGE_FRAMES,
    STAGE_MASKS as QUEUE_STAGE_MASKS,
    STAGE_POSE as QUEUE_STAGE_POSE,
    STAGE_PROJECTION as QUEUE_STAGE_PROJECTION,
    QueueManager,
)

# Real sample capture location (see memory: project_capture_equipment_and_sample_data) --
# used only as a file-dialog starting point, nothing here reads or processes it.
_SAMPLE_DATA_HINT_DIR = Path("/mnt/e/11-9-26_EstoWines_Capture1/Equi")

CAPTURE_GROUP_PRESETS = ["Insta360 (ground)", "Antigravity A1 (aerial)", "Other / custom…"]

DONE, ACTIVE, PENDING = "done", "active", "pending"
STATUS_COLOR = {DONE: QColor(70, 150, 90), ACTIVE: QColor(200, 160, 50), PENDING: QColor(120, 120, 120)}
# The Data manager tab isn't a pipeline stage, so it never takes a status
# color -- a fixed blue, distinct from every stage and job-status color,
# keeps its sidebar row lined up and visually of a piece with the rest.
DATA_MANAGER_DOT_COLOR = QColor(70, 120, 200)

# Job-status vocabulary (queue_manager.QUEUED/RUNNING/DONE/FAILED/BLOCKED/
# SKIPPED) is deliberately separate from the stage-status vocabulary above
# -- a job's lifecycle isn't the same question as a pipeline stage's
# data-derived status -- but reuses the same dot-rendering idiom (see
# _dot_pixmap) so the queue panel reads consistently with the sidebar.
JOB_STATUS_COLOR = {
    QUEUED: QColor(120, 120, 120),
    JOB_RUNNING: QColor(200, 160, 50),
    JOB_DONE: QColor(70, 150, 90),
    FAILED: QColor(190, 60, 60),
    BLOCKED: QColor(150, 90, 170),
    SKIPPED: QColor(150, 150, 150),
}

# Sidebar row indices -- used both to build the QStackedWidget and to
# address which dot compute_stage_statuses' result applies to. Training
# was removed from the GUI (docs/adr/0025) -- the adapter-only library
# code (vine360/training/) still exists, per ADR 0009, just isn't
# surfaced as a sidebar stage.
STAGE_PROJECT = 0
STAGE_IMPORT = 1
STAGE_FRAMES = 2
STAGE_PROJECTION = 3
STAGE_MASKS = 4
STAGE_POSE = 5
STAGE_EXPORT = 6
STAGE_DATA_MANAGER = 7  # not a pipeline stage: fixed-color dot, not in compute_stage_statuses


@dataclass
class AppState:
    project_root: Path | None = None
    project: Project | None = None
    conn: sqlite3.Connection | None = None
    # Set once by Vine360MainWindow after every panel is constructed, so any
    # panel can trigger a sidebar status refresh after it changes project
    # state, without each panel needing to know sidebar indices itself.
    notify_change: Callable[[], None] = field(default=lambda: None)
    # Set once by Vine360MainWindow. Panels check queue_manager.is_running
    # to disable their manual "Run now" button while the queue is actively
    # advancing through jobs, so a manual run can't race a queued one on
    # the same sqlite connection / output directories -- see
    # docs/adr/0023-persisted-processing-queue.md.
    queue_manager: "QueueManager | None" = None


def compute_stage_statuses(conn: sqlite3.Connection | None) -> dict[int, str]:
    """Derives every stage's status directly from the database -- the
    single source of truth, rather than tracking status ad hoc as panels
    act. A stage is DONE once it has produced at least one real artifact,
    ACTIVE once its prerequisite exists but it hasn't yet (or its output
    is stale relative to upstream changes -- see below), else PENDING.

    Frames/Projection/Masks fully invalidate (hard-delete) their old
    output when an earlier stage re-runs (clear_frames_for_source and
    clear_views_for_frame cascade all the way down), so their status
    already falls back to ACTIVE/PENDING correctly from the counts alone.
    Pose estimation (SfM) is different on purpose: its runs are kept as
    provenance history (docs/status.md's data contract), not overwritten,
    so an old sfm_runs row can still exist after its inputs changed. That
    case is detected here by comparing timestamps, and reported as ACTIVE
    ("stale, should re-run") rather than DONE."""
    if conn is None:
        return {
            STAGE_IMPORT: PENDING,
            STAGE_FRAMES: PENDING,
            STAGE_PROJECTION: PENDING,
            STAGE_MASKS: PENDING,
            STAGE_POSE: PENDING,
        }
    n_sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    n_video_sources = conn.execute("SELECT COUNT(*) FROM sources WHERE media_type = 'video'").fetchone()[0]
    n_frames = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    n_views = conn.execute("SELECT COUNT(*) FROM views").fetchone()[0]
    n_masks = conn.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
    n_sfm_runs = conn.execute("SELECT COUNT(*) FROM sfm_runs").fetchone()[0]

    if n_sfm_runs > 0:
        latest_run_at = conn.execute("SELECT MAX(created_at) FROM sfm_runs").fetchone()[0]
        latest_input_at = conn.execute(
            "SELECT MAX(t) FROM (SELECT MAX(updated_at) AS t FROM views UNION ALL SELECT MAX(updated_at) FROM masks)"
        ).fetchone()[0]
        pose_status = ACTIVE if (latest_input_at and latest_input_at > latest_run_at) else DONE
    else:
        pose_status = ACTIVE if n_masks > 0 else PENDING

    return {
        STAGE_IMPORT: DONE if n_sources > 0 else ACTIVE,
        STAGE_FRAMES: DONE if n_frames > 0 else (ACTIVE if n_video_sources > 0 else PENDING),
        STAGE_PROJECTION: DONE if n_views > 0 else (ACTIVE if n_frames > 0 else PENDING),
        STAGE_MASKS: DONE if n_masks > 0 else (ACTIVE if n_views > 0 else PENDING),
        STAGE_POSE: pose_status,
    }


@dataclass
class FrameSetChoice:
    """One entry in a stage panel's frame-set dropdown (docs/adr/0034)."""

    frame_set_id: str
    source_id: str
    name: str  # "<clip file name> — <config label>"
    frame_count: int = 0
    view_count: int = 0
    mask_count: int = 0
    pending: str | None = None  # why it has no frames yet ("queued", "not extracted"), None once extracted


def frame_set_choices(state: AppState, *, equirect_only: bool = False) -> list[FrameSetChoice]:
    """What the Projection/Masks/Pose/Export dropdowns offer: every
    extracted frame set, plus the ones that don't exist *yet* but can be
    targeted through the queue -- a frame set a queued Frames job will
    write, and, for a video source with no frame sets at all, the
    balanced-preset frame set the queue knows how to auto-extract (the
    same "Add to Queue on a source with nothing upstream yet" path these
    panels had per-source before frame sets)."""
    conn = state.conn
    if conn is None:
        return []
    from vine360.ingest.frames import frame_set_label

    sources = {
        source_id: (path, projection)
        for source_id, path, projection in conn.execute(
            "SELECT source_id, path, projection FROM sources WHERE media_type = 'video' ORDER BY rowid"
        )
    }

    def allowed(source_id: str) -> bool:
        return source_id in sources and (not equirect_only or sources[source_id][1] == "equirectangular")

    choices = [
        FrameSetChoice(
            info.frame_set_id, info.source_id, info.display_name, info.frame_count, info.view_count, info.mask_count
        )
        for info in list_frame_sets(conn)
        if allowed(info.source_id)
    ]
    known = {c.frame_set_id for c in choices}
    queue = state.queue_manager
    for job in queue.jobs if queue is not None else []:
        if (
            job.stage != QUEUE_STAGE_FRAMES
            or job.status in ("done", "failed", "skipped")
            or not job.target_frame_set_id
            or job.target_frame_set_id in known
            or not allowed(job.target_source_id)
        ):
            continue
        label = frame_set_label(
            {
                "mode": "interval",
                "interval_seconds": job.params["interval_seconds"],
                "start_time_seconds": job.params.get("start_time"),
                "end_time_seconds": job.params.get("end_time"),
            }
        )
        choices.append(
            FrameSetChoice(
                job.target_frame_set_id,
                job.target_source_id,
                f"{Path(sources[job.target_source_id][0]).name} — {label}",
                pending="queued",
            )
        )
        known.add(job.target_frame_set_id)
    sources_with_sets = {c.source_id for c in choices}
    balanced = FRAME_PRESET_INTERVALS[FramePreset.BALANCED]
    for source_id, (path, _projection) in sources.items():
        if source_id in sources_with_sets or not allowed(source_id):
            continue
        choices.append(
            FrameSetChoice(
                frame_set_id_for(conn, source_id, interval_seconds=balanced),
                source_id,
                f"{Path(path).name} — every {balanced:g}s",
                pending="not extracted -- the queue will extract it with the balanced preset",
            )
        )
    return choices


def _restore_combo_selection(combo: QComboBox, data) -> None:
    """Re-selects the entry whose userData is `data` after a combo was
    repopulated (no-op if it's gone)."""
    index = combo.findData(data) if data is not None else -1
    if index >= 0:
        combo.setCurrentIndex(index)


class _BackgroundWorker(QObject):
    """Runs one callable off the GUI thread. Real bug this fixes: without
    it, add_source's checksum on a multi-GB file (~25s for 7.65GB, seen on
    real capture footage) freezes the whole window with zero feedback --
    exactly the "video doesn't open" symptom.

    The callable must not touch any sqlite3.Connection created on the main
    thread (sqlite3 connections aren't usable across threads by default);
    callers open their own connection by project_root inside `fn` instead.
    """

    finished = Signal(object)
    failed = Signal(Exception)
    progress = Signal(str, object, object)  # message, current, total (current/total may be None)

    def __init__(self, fn, args, kwargs, report_progress: bool):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._report_progress = report_progress

    def run(self) -> None:
        try:
            if self._report_progress:
                result = self._fn(*self._args, progress_callback=self.progress.emit, **self._kwargs)
            else:
                result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # surfaced via `failed`, not swallowed
            self.failed.emit(exc)
        else:
            self.finished.emit(result)


def run_in_background(
    owner: QWidget, fn, *args, on_success=None, on_error=None, on_progress=None, **kwargs
) -> None:
    """Starts fn(*args, **kwargs) on a QThread. `owner` must outlive the
    call. Every in-flight (thread, worker) pair is kept in a list on
    `owner` -- a single overwritable attribute is NOT enough: our own
    `worker.finished`/`failed` fire (and callers may immediately start a
    *second* background call, e.g. re-extracting frames right after adding
    a source) before the QThread has actually stopped running (`quit()`
    only requests a stop; `QThread.finished` confirms it). Overwriting the
    reference before that point can garbage-collect a still-running
    QThread and crash the process with "QThread: Destroyed while thread is
    still running" -- reproduced and fixed while wiring remove-source and
    re-extraction."""
    thread = QThread()
    worker = _BackgroundWorker(fn, args, kwargs, report_progress=on_progress is not None)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    if on_success:
        worker.finished.connect(on_success)
    if on_error:
        worker.failed.connect(on_error)
    if on_progress:
        worker.progress.connect(on_progress)

    if not hasattr(owner, "_bg_pairs"):
        owner._bg_pairs = []
    pair = (thread, worker)
    owner._bg_pairs.append(pair)
    thread.finished.connect(lambda: owner._bg_pairs.remove(pair) if pair in owner._bg_pairs else None)

    thread.start()


class ProgressArea(QWidget):
    """A progress bar + status label shared by every panel that runs a
    background operation. Switches to determinate + an elapsed-time-based
    ETA whenever the underlying function reports (current, total); stays
    an indeterminate spinner for phases without a natural count (e.g. a
    single ffmpeg/COLMAP subprocess call with no internal progress)."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        layout.addWidget(self.bar)
        self.label = QLabel("")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self._start_time: float | None = None

    def start(self, message: str = "Starting…") -> None:
        self._start_time = time.monotonic()
        self.bar.setRange(0, 0)
        self.bar.setVisible(True)
        self.label.setText(message)

    def update_progress(self, message: str, current: int | None, total: int | None) -> None:
        if current is not None and total:
            self.bar.setRange(0, total)
            self.bar.setValue(current)
            self.label.setText(message + self._eta_suffix(current, total))
        else:
            self.bar.setRange(0, 0)
            self.label.setText(message)

    def _eta_suffix(self, current: int, total: int) -> str:
        if not self._start_time or current <= 0:
            return ""
        elapsed = time.monotonic() - self._start_time
        rate = current / elapsed
        if rate <= 0:
            return ""
        remaining = (total - current) / rate
        if remaining < 1:
            return ""
        return f"  (~{self._format_duration(remaining)} remaining)"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        minutes, secs = divmod(seconds, 60)
        return f"{minutes}m{secs:02d}s"

    def finish(self, message: str) -> None:
        self.bar.setVisible(False)
        self.label.setText(message)
        self._start_time = None


def _dot_pixmap(color: QColor) -> QPixmap:
    pixmap = QPixmap(12, 12)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(color)
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(1, 1, 10, 10)
    painter.end()
    return pixmap


def _status_dot(status: str) -> QPixmap:
    return _dot_pixmap(STATUS_COLOR[status])


def _job_status_dot(status: str) -> QPixmap:
    return _dot_pixmap(JOB_STATUS_COLOR[status])


def _flag_icon(size: int = 14) -> QPixmap:
    """A small drawn flag glyph -- used instead of the Unicode flag emoji
    (U+1F6A9), which renders as an empty box here: this machine has no
    emoji font installed at all (confirmed: `fc-match "Noto Color Emoji"`
    falls back to plain DejaVu Sans). Drawing icons with QPainter, like
    the sidebar status dots already do, has no font dependency at all."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pole_color = QColor(90, 90, 90)
    painter.setPen(pole_color)
    painter.drawLine(2, 1, 2, size - 1)
    painter.setBrush(QColor(210, 60, 50))
    painter.setPen(Qt.NoPen)
    painter.drawPolygon([QPoint(3, 1), QPoint(size - 1, 4), QPoint(3, 7)])
    painter.end()
    return pixmap


class StageIndicator(QWidget):
    """A small horizontal step row for a panel that has more than one
    distinct action to move through in sequence (e.g. build a mask, then
    review it; configure an export, then run it) -- reuses the sidebar's
    own status-dot/color vocabulary (_status_dot/STATUS_COLOR) so
    done/current/not-yet reads the same way here as it does there,
    instead of inventing a second visual language for the same idea."""

    def __init__(self, stage_labels: list[str]):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        self._dots: list[QLabel] = []
        self._texts: list[QLabel] = []
        for i, text in enumerate(stage_labels):
            dot = QLabel()
            dot.setPixmap(_status_dot(PENDING))
            label = QLabel(text)
            layout.addWidget(dot)
            layout.addWidget(label)
            self._dots.append(dot)
            self._texts.append(label)
            if i < len(stage_labels) - 1:
                arrow = QLabel("→")
                arrow.setStyleSheet("color: palette(mid);")
                layout.addWidget(arrow)
        layout.addStretch()
        self.statuses: list[str] = []
        self.set_stages([PENDING] * len(stage_labels))

    def set_stages(self, statuses: list[str]) -> None:
        """statuses[i] is one of DONE/ACTIVE/PENDING for stage i."""
        self.statuses = list(statuses)
        for dot, label, status in zip(self._dots, self._texts, statuses):
            dot.setPixmap(_status_dot(status))
            weight = "600" if status == ACTIVE else "400"
            color = "palette(mid)" if status == PENDING else "palette(text)"
            label.setStyleSheet(f"font-weight: {weight}; color: {color};")


THUMBNAIL_SIZE = 160


def _thumbnail_widget(path: Path, caption: str) -> QWidget:
    """Loads a real image thumbnail from disk (a projected view or a mask
    file), or a placeholder if the file is missing/unreadable -- never
    raises, since a stale db row pointing at a since-moved/deleted file
    shouldn't crash a preview."""
    pixmap = QPixmap(str(path))
    if pixmap.isNull():
        pixmap = QPixmap(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        pixmap.fill(QColor(80, 80, 80))
    else:
        pixmap = pixmap.scaled(
            THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    image_label = QLabel()
    image_label.setPixmap(pixmap)
    image_label.setAlignment(Qt.AlignCenter)
    image_label.setToolTip(str(path))
    caption_label = QLabel(caption)
    caption_label.setAlignment(Qt.AlignCenter)
    caption_label.setWordWrap(True)
    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.addWidget(image_label)
    layout.addWidget(caption_label)
    return wrapper


class PreviewGallery(QWidget):
    """A horizontally scrollable row of image thumbnails, shared by the
    Projection and Masks panels so generated views/masks can actually be
    looked at rather than only counted."""

    def __init__(self, empty_text: str = "Nothing to preview yet."):
        super().__init__()
        self._empty_text = empty_text
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.empty_label = QLabel(empty_text)
        self.empty_label.setStyleSheet("color: palette(mid);")
        outer.addWidget(self.empty_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(THUMBNAIL_SIZE + 60)
        self.scroll_area.setVisible(False)
        self._content = QWidget()
        self._content_layout = QHBoxLayout(self._content)
        self._content_layout.setContentsMargins(4, 4, 4, 4)
        self.scroll_area.setWidget(self._content)
        outer.addWidget(self.scroll_area)

    def set_items(self, items: list[tuple[Path, str]]) -> None:
        """items: list of (image_path, caption)."""
        while self._content_layout.count():
            child = self._content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        if not items:
            self.empty_label.setText(self._empty_text)
            self.empty_label.setVisible(True)
            self.scroll_area.setVisible(False)
            return

        self.empty_label.setVisible(False)
        self.scroll_area.setVisible(True)
        for path, caption in items:
            self._content_layout.addWidget(_thumbnail_widget(path, caption))
        self._content_layout.addStretch()


class TemporalFrameSelector(QWidget):
    """A slider for scrubbing through a source's extracted frames by time
    -- scales much better than a long dropdown once there are dozens or
    hundreds of frames (a real capture easily has 100+). Also offers
    Previous/Next single-step buttons, a bigger +/- BIG_JUMP-frame jump,
    and a spin box to type a frame number directly. The slider stays the
    single source of truth for "current frame"; every other control just
    moves it (via `_step`/`_on_spin_changed`) or mirrors it (the spin box
    on `_on_slider_changed`), so they can never drift out of sync."""

    BIG_JUMP = 10

    frame_changed = Signal()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setPageStep(self.BIG_JUMP)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        nav_row = QHBoxLayout()
        self.big_back_btn = QPushButton(f"«{self.BIG_JUMP}")
        self.big_back_btn.setToolTip(f"Back {self.BIG_JUMP} frames")
        self.big_back_btn.clicked.connect(lambda: self._step(-self.BIG_JUMP))
        nav_row.addWidget(self.big_back_btn)
        self.prev_btn = QPushButton("‹ Prev")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        nav_row.addWidget(self.prev_btn)

        # 1-indexed to match the "Frame N/total" the label already showed
        # -- typing a value outside [1, total] is impossible, QSpinBox
        # clamps/refuses it natively, satisfying "constrain to valid
        # values" without any extra validation code here.
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(1)
        self.frame_spin.setMaximum(1)
        self.frame_spin.setEnabled(False)
        self.frame_spin.setToolTip("Type a frame number to jump to it.")
        self.frame_spin.valueChanged.connect(self._on_spin_changed)
        nav_row.addWidget(self.frame_spin)
        self.total_label = QLabel("/ 0")
        nav_row.addWidget(self.total_label)

        self.next_btn = QPushButton("Next ›")
        self.next_btn.clicked.connect(lambda: self._step(1))
        nav_row.addWidget(self.next_btn)
        self.big_forward_btn = QPushButton(f"{self.BIG_JUMP}»")
        self.big_forward_btn.setToolTip(f"Forward {self.BIG_JUMP} frames")
        self.big_forward_btn.clicked.connect(lambda: self._step(self.BIG_JUMP))
        nav_row.addWidget(self.big_forward_btn)
        nav_row.addStretch()
        layout.addLayout(nav_row)

        self.label = QLabel("No frames.")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        self._frames: list[tuple[str, float, int]] = []
        self._nav_buttons = (self.big_back_btn, self.prev_btn, self.next_btn, self.big_forward_btn)

    def set_frames(self, frames: list[tuple[str, float, int]]) -> None:
        """frames: (frame_id, source_time, view_count), ordered by time."""
        self._frames = frames
        has_frames = len(frames) > 0

        self.slider.blockSignals(True)
        self.slider.setMaximum(max(len(frames) - 1, 0))
        self.slider.setValue(0)
        self.slider.setEnabled(has_frames)
        self.slider.blockSignals(False)

        self.frame_spin.blockSignals(True)
        self.frame_spin.setMaximum(max(len(frames), 1))
        self.frame_spin.setValue(1)
        self.frame_spin.setEnabled(has_frames)
        self.frame_spin.blockSignals(False)

        self.total_label.setText(f"/ {len(frames)}")
        for btn in self._nav_buttons:
            btn.setEnabled(has_frames)

        self._update_label()
        self.frame_changed.emit()

    def _step(self, delta: int) -> None:
        if not self._frames:
            return
        new_index = max(0, min(len(self._frames) - 1, self.slider.value() + delta))
        self.slider.setValue(new_index)  # no-op (and no signal) if already at that end

    def _on_spin_changed(self, value: int) -> None:
        if not self._frames:
            return
        self.slider.setValue(value - 1)  # spin box is 1-indexed, slider is 0-indexed

    def _on_slider_changed(self, value: int) -> None:
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(value + 1)
        self.frame_spin.blockSignals(False)
        self._update_label()
        self.frame_changed.emit()

    def _update_label(self) -> None:
        if not self._frames:
            self.label.setText("No frames.")
            return
        index = self.slider.value()
        frame_id, source_time, view_count = self._frames[index]
        views_text = f"{view_count} views" if view_count else "no views"
        self.label.setText(f"t={source_time:.2f}s  —  {views_text}")

    def current_frame_id(self) -> str | None:
        if not self._frames:
            return None
        return self._frames[self.slider.value()][0]

    def jump_to_frame_id(self, frame_id: str) -> bool:
        for index, (fid, _source_time, _view_count) in enumerate(self._frames):
            if fid == frame_id:
                self.slider.setValue(index)
                return True
        return False


def _panel_header(title: str, subtitle: str, layout: QVBoxLayout) -> None:
    heading = QLabel(title)
    heading.setStyleSheet("font-size: 16px; font-weight: 600;")
    layout.addWidget(heading)
    sub = QLabel(subtitle)
    sub.setStyleSheet("color: palette(mid);")
    sub.setWordWrap(True)
    layout.addWidget(sub)
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    layout.addWidget(line)


class ProjectPanel(QWidget):
    """Split into two clearly separate sections -- creating a new project
    and opening an existing one are different actions with different
    relevant controls (capture mode only matters when creating), and
    showing them side by side with shared fields was a real point of
    confusion raised in review."""

    project_changed = Signal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        layout = QVBoxLayout(self)
        _panel_header("Project", "Create a new project or open an existing one.", layout)

        self.path_label = QLabel("No project open.")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("padding: 4px; background: palette(alternate-base);")
        layout.addWidget(self.path_label)

        create_box = QGroupBox("Create a new project")
        create_layout = QVBoxLayout(create_box)
        form = QFormLayout()
        self.name_field = QLineEdit("Block 7 — North Row")
        form.addRow("Name:", self.name_field)
        mode_row = QHBoxLayout()
        self.mode_radios: dict[CaptureMode, QRadioButton] = {}
        for mode, label in [
            (CaptureMode.THREE_SIXTY, "360° only"),
            (CaptureMode.CONVENTIONAL, "Conventional"),
            (CaptureMode.MIXED, "Mixed"),
        ]:
            radio = QRadioButton(label)
            radio.setChecked(mode == CaptureMode.THREE_SIXTY)
            self.mode_radios[mode] = radio
            mode_row.addWidget(radio)
        form.addRow("Capture mode:", mode_row)
        create_layout.addLayout(form)
        note = QLabel(
            "Both real capture devices for this project (Insta360, Antigravity A1) are "
            "360° platforms, so '360° only' is the right mode for combining them -- "
            "'Mixed' here would mean a true conventional (non-equirectangular) camera "
            "alongside them, not the two 360 rigs."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid); font-style: italic;")
        create_layout.addWidget(note)
        create_btn = QPushButton("Create New Project…")
        create_btn.setToolTip("Choose an empty folder; the project is created there using the name and mode above.")
        create_btn.clicked.connect(self._on_create)
        create_layout.addWidget(create_btn)
        layout.addWidget(create_box)

        open_box = QGroupBox("Open an existing project")
        open_layout = QVBoxLayout(open_box)
        open_layout.addWidget(QLabel("Name and capture mode above don't apply here -- they're read from the project you pick."))
        open_btn = QPushButton("Open Existing Project…")
        open_btn.setToolTip("Choose a folder that already contains a project.yaml.")
        open_btn.clicked.connect(self._on_open)
        open_layout.addWidget(open_btn)
        layout.addWidget(open_box)

        layout.addStretch()

    def _selected_mode(self) -> CaptureMode:
        for mode, radio in self.mode_radios.items():
            if radio.isChecked():
                return mode
        return CaptureMode.THREE_SIXTY

    def _on_create(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose an empty folder for the new project")
        if not directory:
            return
        try:
            project = create_project(Path(directory), self.name_field.text(), self._selected_mode())
        except ProjectAlreadyExistsError as exc:
            QMessageBox.warning(self, "Cannot create project", str(exc))
            return
        self._set_active_project(Path(directory), project)

    def _on_open(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose an existing project folder")
        if not directory:
            return
        try:
            project = load_project(Path(directory))
        except ProjectNotFoundError as exc:
            QMessageBox.warning(self, "Cannot open project", str(exc))
            return
        self._set_active_project(Path(directory), project)

    def _set_active_project(self, root: Path, project: Project) -> None:
        if self.state.conn is not None:
            self.state.conn.close()
        self.state.project_root = root
        self.state.project = project
        self.state.conn = open_index_db(root)
        self.path_label.setText(
            f"Open: {project.name!r} ({project.capture_mode.value}) at {root}\n"
            f"project_id={project.project_id}  created_at={project.created_at}"
        )
        self.project_changed.emit()
        self.state.notify_change()


def _add_source_worker(
    project_root: Path,
    file_path: Path,
    capture_group: str | None,
    confirm_equirectangular: bool,
    added_at: str,
    progress_callback=None,
):
    """Runs on a background thread -- opens its own sqlite connection
    rather than reusing one created on the GUI thread (sqlite3 connections
    aren't valid across threads)."""
    conn = open_index_db(project_root)
    try:
        return add_source(
            conn,
            file_path,
            LocalRunner(),
            capture_group=capture_group,
            confirm_equirectangular=confirm_equirectangular,
            added_at=added_at,
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


def _remove_source_worker(project_root: Path, source_id: str) -> str:
    conn = open_index_db(project_root)
    try:
        remove_source(conn, project_root, source_id)
        return source_id
    finally:
        conn.close()


class ImportPanel(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._pending_file_path: str | None = None
        layout = QVBoxLayout(self)
        _panel_header("Import", "Register real media sources into the open project.", layout)

        self.no_project_label = QLabel("Create or open a project first (see the Project stage).")
        layout.addWidget(self.no_project_label)

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        add_row = QHBoxLayout()
        self.add_btn = QPushButton("Add Source…")
        self.add_btn.clicked.connect(self._on_add_source)
        add_row.addWidget(self.add_btn)
        add_row.addWidget(QLabel("Capture group:"))
        self.capture_group_combo = QComboBox()
        self.capture_group_combo.setEditable(True)
        self.capture_group_combo.addItems(CAPTURE_GROUP_PRESETS)
        add_row.addWidget(self.capture_group_combo)
        controls_layout.addLayout(add_row)

        self.progress_area = ProgressArea()
        controls_layout.addWidget(self.progress_area)

        self.sources_table = QTableWidget(0, 5)
        self.sources_table.setHorizontalHeaderLabels(["Media type", "Projection", "Capture group", "Checksum", "Path"])
        self.sources_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.sources_table.setSelectionMode(QTableWidget.SingleSelection)
        controls_layout.addWidget(self.sources_table)

        self.remove_btn = QPushButton("Remove Selected Source")
        self.remove_btn.setToolTip("Removes this source and any frames extracted from it. Never deletes the original file.")
        self.remove_btn.setEnabled(False)
        self.remove_btn.clicked.connect(self._on_remove_source)
        self.sources_table.itemSelectionChanged.connect(
            lambda: self.remove_btn.setEnabled(bool(self.sources_table.selectedItems()))
        )
        controls_layout.addWidget(self.remove_btn)

        controls_layout.addWidget(QLabel("Dependency check (real, live probe):"))
        dep_table = QTableWidget(0, 3)
        dep_table.setHorizontalHeaderLabels(["Tool", "Found", "Version"])
        for status in probe_dependencies():
            row = dep_table.rowCount()
            dep_table.insertRow(row)
            dep_table.setItem(row, 0, QTableWidgetItem(status.name))
            dep_table.setItem(row, 1, QTableWidgetItem("yes" if status.found else "NO"))
            dep_table.setItem(row, 2, QTableWidgetItem(status.version or "—"))
        dep_table.resizeColumnsToContents()
        controls_layout.addWidget(dep_table)

        layout.addWidget(self.controls)
        self.controls.setVisible(False)

    def on_project_changed(self) -> None:
        have_project = self.state.conn is not None
        self.no_project_label.setVisible(not have_project)
        self.controls.setVisible(have_project)
        if have_project:
            self._refresh_sources_table()

    def on_shown(self) -> None:
        if self.state.conn is not None:
            self._refresh_sources_table()

    def _resolved_capture_group(self) -> str | None:
        text = self.capture_group_combo.currentText().strip()
        if not text or text == "Other / custom…":
            return None
        return {
            "Insta360 (ground)": "insta360-ground",
            "Antigravity A1 (aerial)": "antigravity-a1-aerial",
        }.get(text, text)

    def _on_add_source(self) -> None:
        start_dir = str(_SAMPLE_DATA_HINT_DIR) if _SAMPLE_DATA_HINT_DIR.exists() else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a media file to register",
            start_dir,
            "Media (*.mp4 *.mov *.mkv *.avi *.insv *.jpg *.jpeg *.png *.tif *.tiff)",
        )
        if not file_path:
            return
        self._pending_file_path = file_path
        self._start_add_source(confirm_equirectangular=False)

    def _start_add_source(self, *, confirm_equirectangular: bool) -> None:
        self.add_btn.setEnabled(False)
        name = Path(self._pending_file_path).name
        self.progress_area.start(f"Adding {name}…")
        run_in_background(
            self,
            _add_source_worker,
            self.state.project_root,
            Path(self._pending_file_path),
            self._resolved_capture_group(),
            confirm_equirectangular,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            on_success=self._on_add_source_success,
            on_error=self._on_add_source_error,
            on_progress=self.progress_area.update_progress,
        )

    def _on_add_source_success(self, source) -> None:
        self.add_btn.setEnabled(True)
        self.progress_area.finish(f"Added {Path(source.path).name} ({source.media_type.value}, {source.projection.value}).")
        self._refresh_sources_table()
        self.state.notify_change()

    def _on_add_source_error(self, exc: Exception) -> None:
        self.add_btn.setEnabled(True)
        if isinstance(exc, EquirectangularConfirmationRequired):
            self.progress_area.finish("")
            name = Path(self._pending_file_path).name
            answer = QMessageBox.question(
                self, "Confirm equirectangular", f"{name} has a 2:1 aspect ratio. Is it truly equirectangular?"
            )
            if answer == QMessageBox.Yes:
                self._start_add_source(confirm_equirectangular=True)
            return
        self.progress_area.finish("")
        if isinstance(exc, (SourceError, ProbeError)):
            QMessageBox.warning(self, "Could not add source", str(exc))
        else:
            QMessageBox.critical(self, "Unexpected error adding source", f"{type(exc).__name__}: {exc}")

    def _refresh_sources_table(self) -> None:
        rows = self.state.conn.execute(
            "SELECT source_id, media_type, projection, capture_group, checksum, path FROM sources ORDER BY rowid"
        ).fetchall()
        self.sources_table.setRowCount(len(rows))
        for r, (source_id, media_type, projection, capture_group, checksum, path) in enumerate(rows):
            item0 = QTableWidgetItem(media_type)
            item0.setData(Qt.UserRole, source_id)
            self.sources_table.setItem(r, 0, item0)
            self.sources_table.setItem(r, 1, QTableWidgetItem(projection))
            self.sources_table.setItem(r, 2, QTableWidgetItem(capture_group or "—"))
            self.sources_table.setItem(r, 3, QTableWidgetItem(checksum[:12] + "…"))
            self.sources_table.setItem(r, 4, QTableWidgetItem(path))
        self.sources_table.resizeColumnsToContents()
        self.remove_btn.setEnabled(False)

    def _selected_source_id(self) -> str | None:
        row = self.sources_table.currentRow()
        if row < 0:
            return None
        item = self.sources_table.item(row, 0)
        return item.data(Qt.UserRole) if item else None

    def _on_remove_source(self) -> None:
        source_id = self._selected_source_id()
        if source_id is None:
            return
        row = self.sources_table.currentRow()
        path = self.sources_table.item(row, 4).text()
        answer = QMessageBox.question(
            self,
            "Remove source",
            f"Remove {Path(path).name} from this project?\n\n"
            "This deletes any frames already extracted from it, but never the original file.",
        )
        if answer != QMessageBox.Yes:
            return
        self.remove_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.progress_area.start(f"Removing {Path(path).name}…")
        run_in_background(
            self,
            _remove_source_worker,
            self.state.project_root,
            source_id,
            on_success=self._on_remove_source_success,
            on_error=self._on_remove_source_error,
        )

    def _on_remove_source_success(self, source_id: str) -> None:
        self.add_btn.setEnabled(True)
        self.progress_area.finish("Removed.")
        self._refresh_sources_table()
        self.state.notify_change()

    def _on_remove_source_error(self, exc: Exception) -> None:
        self.add_btn.setEnabled(True)
        self.progress_area.finish("")
        if isinstance(exc, SourceError):
            QMessageBox.warning(self, "Could not remove source", str(exc))
        else:
            QMessageBox.critical(self, "Unexpected error removing source", f"{type(exc).__name__}: {exc}")


def _extract_frames_worker(
    project_root: Path,
    source_id: str,
    interval_seconds: float,
    start_time: float | None,
    end_time: float | None,
    generate_thumbnails: bool,
    frame_set_id: str | None = None,
    progress_callback=None,
):
    conn = open_index_db(project_root)
    try:
        return extract_frames(
            conn,
            project_root,
            source_id,
            LocalRunner(),
            interval_seconds=interval_seconds,
            start_time=start_time,
            end_time=end_time,
            generate_thumbnails=generate_thumbnails,
            frame_set_id=frame_set_id,
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


class FramesPanel(QWidget):
    """Real extraction, wired to vine360.ingest.frames.extract_frames, run
    off the GUI thread (a real capture clip's extraction can run for a
    while -- it should never freeze the window the way unthreaded
    add_source did on a 7.65GB file)."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._source_durations: dict[str, float] = {}
        layout = QVBoxLayout(self)
        _panel_header("Frames", "Extract deterministic frames from a registered video source.", layout)

        self.no_project_label = QLabel("Create or open a project first, then add a video source on Import.")
        layout.addWidget(self.no_project_label)

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        form.addRow("Video source:", self.source_combo)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(
            [FramePreset.PREVIEW.value, FramePreset.BALANCED.value, FramePreset.QUALITY.value, FramePreset.CUSTOM.value]
        )
        self.preset_combo.setCurrentText(FramePreset.BALANCED.value)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        form.addRow("Frame preset:", self.preset_combo)

        self.custom_interval_spin = QDoubleSpinBox()
        self.custom_interval_spin.setRange(0.01, 3600.0)
        self.custom_interval_spin.setDecimals(2)
        self.custom_interval_spin.setSuffix(" s")
        self.custom_interval_spin.setValue(FRAME_PRESET_INTERVALS[FramePreset.BALANCED])
        self.custom_interval_spin.setEnabled(False)
        # Last value typed while "custom" was selected -- restored on
        # switching back to it, since a preset overwrites the (disabled)
        # spin box to show that preset's own interval.
        self._custom_interval = self.custom_interval_spin.value()
        self.custom_interval_spin.valueChanged.connect(self._on_custom_interval_edited)
        form.addRow("Interval:", self.custom_interval_spin)
        controls_layout.addLayout(form)

        range_row = QHBoxLayout()
        self.time_range_check = QCheckBox("Limit to a time range")
        self.time_range_check.toggled.connect(self._on_time_range_toggled)
        range_row.addWidget(self.time_range_check)
        range_row.addWidget(QLabel("Start (s):"))
        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setRange(0.0, 0.0)  # rebound in _on_source_changed once duration is known
        self.start_time_spin.setDecimals(2)
        self.start_time_spin.setEnabled(False)
        range_row.addWidget(self.start_time_spin)
        range_row.addWidget(QLabel("End (s):"))
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(0.0, 0.0)
        self.end_time_spin.setDecimals(2)
        self.end_time_spin.setEnabled(False)
        range_row.addWidget(self.end_time_spin)
        range_row.addStretch()
        controls_layout.addLayout(range_row)

        self.thumbnails_check = QCheckBox("Generate thumbnails")
        self.thumbnails_check.setChecked(True)
        self.thumbnails_check.setToolTip(
            "Writes a small JPEG thumbnail per extracted frame into frames/<source>/thumbs/. Nothing "
            "in the app currently reads these back -- unchecking skips that pass to save time on a "
            "large or high-resolution source."
        )
        controls_layout.addWidget(self.thumbnails_check)

        intervals_note = QLabel(
            "Each interval/time-range combination is its own frame set, kept side by side with the "
            "source's others -- extracting a combination that already exists replaces just that set. "
            "A time range only limits which part of the source is sampled; the interval still applies "
            "within it. Remove frame sets on the Data manager tab."
        )
        intervals_note.setWordWrap(True)
        intervals_note.setStyleSheet("color: palette(mid);")
        controls_layout.addWidget(intervals_note)

        controls_layout.addWidget(QLabel("Frame sets already extracted from this source:"))
        self.frame_sets_list = QListWidget()
        self.frame_sets_list.setMaximumHeight(110)
        self.frame_sets_list.setSelectionMode(QAbstractItemView.NoSelection)
        controls_layout.addWidget(self.frame_sets_list)

        btn_row = QHBoxLayout()
        self.extract_btn = QPushButton("Extract Frames")
        self.extract_btn.clicked.connect(self._on_extract)
        btn_row.addWidget(self.extract_btn)
        self.queue_btn = QPushButton("Add to Queue")
        self.queue_btn.setToolTip("Adds this exact configuration as a queued job instead of running it now.")
        self.queue_btn.clicked.connect(self._on_add_to_queue)
        btn_row.addWidget(self.queue_btn)
        controls_layout.addLayout(btn_row)

        self.progress_area = ProgressArea()
        controls_layout.addWidget(self.progress_area)

        layout.addWidget(self.controls)
        self.controls.setVisible(False)
        layout.addStretch()

        if self.state.queue_manager is not None:
            self.state.queue_manager.queue_changed.connect(self._refresh_run_enabled)

    def on_project_changed(self) -> None:
        have_project = self.state.conn is not None
        self.no_project_label.setVisible(not have_project)
        self.controls.setVisible(have_project)
        if have_project:
            self.refresh_sources()

    def on_shown(self) -> None:
        if self.state.conn is not None:
            self.refresh_sources()

    def refresh_sources(self) -> None:
        self.source_combo.clear()
        if self.state.conn is None:
            return
        rows = self.state.conn.execute(
            "SELECT s.source_id, s.path, s.timestamps, "
            "(SELECT COUNT(*) FROM frame_sets WHERE frame_sets.source_id = s.source_id) AS set_count "
            "FROM sources s WHERE s.media_type = 'video' ORDER BY s.rowid"
        ).fetchall()
        self._source_durations: dict[str, float] = {}
        for source_id, path, timestamps_json, set_count in rows:
            duration = json.loads(timestamps_json).get("duration_seconds") if timestamps_json else None
            if duration:
                self._source_durations[source_id] = duration
            existing = f"{set_count} frame set{'s' if set_count != 1 else ''}" if set_count else "no frames yet"
            self.source_combo.addItem(f"{Path(path).name} ({source_id[:8]}) — {existing}", userData=source_id)
        has_sources = self.source_combo.count() > 0
        self._has_sources = has_sources
        self.queue_btn.setEnabled(has_sources)
        self._refresh_run_enabled()
        self._on_source_changed()
        self.progress_area.label.setText("" if has_sources else "No video sources registered yet -- add one on Import.")

    def _refresh_run_enabled(self) -> None:
        busy = bool(self.state.queue_manager and self.state.queue_manager.is_running)
        self.extract_btn.setEnabled(getattr(self, "_has_sources", False) and not busy)

    def _on_preset_changed(self, text: str) -> None:
        preset = FramePreset(text)
        is_custom = preset == FramePreset.CUSTOM
        self.custom_interval_spin.blockSignals(True)
        self.custom_interval_spin.setValue(self._custom_interval if is_custom else FRAME_PRESET_INTERVALS[preset])
        self.custom_interval_spin.blockSignals(False)
        self.custom_interval_spin.setEnabled(is_custom)

    def _on_custom_interval_edited(self, value: float) -> None:
        if FramePreset(self.preset_combo.currentText()) == FramePreset.CUSTOM:
            self._custom_interval = value

    def _current_interval(self) -> float:
        preset = FramePreset(self.preset_combo.currentText())
        if preset == FramePreset.CUSTOM:
            return self.custom_interval_spin.value()
        return FRAME_PRESET_INTERVALS[preset]

    def _on_source_changed(self, *_args) -> None:
        source_id = self.source_combo.currentData()
        duration = self._source_durations.get(source_id, 0.0) if source_id else 0.0
        for spin in (self.start_time_spin, self.end_time_spin):
            spin.setMaximum(duration)
        self.start_time_spin.setValue(0.0)
        self.end_time_spin.setValue(duration)
        self._refresh_frame_sets_list()

    def _refresh_frame_sets_list(self) -> None:
        self.frame_sets_list.clear()
        source_id = self.source_combo.currentData()
        if self.state.conn is None or source_id is None:
            return
        for info in list_frame_sets(self.state.conn, source_id=source_id):
            self.frame_sets_list.addItem(
                f"{info.label} — {info.frame_count} frames, {info.view_count} views, {info.mask_count} masks"
            )
        if self.frame_sets_list.count() == 0:
            self.frame_sets_list.addItem("(none yet)")

    def _on_time_range_toggled(self, checked: bool) -> None:
        self.start_time_spin.setEnabled(checked)
        self.end_time_spin.setEnabled(checked)

    def _current_time_range(self) -> tuple[float | None, float | None]:
        if not self.time_range_check.isChecked():
            return None, None
        return self.start_time_spin.value(), self.end_time_spin.value()

    def _on_add_to_queue(self) -> None:
        source_id = self.source_combo.currentData()
        if source_id is None:
            return
        interval = self._current_interval()
        start_time, end_time = self._current_time_range()
        if end_time is not None and end_time <= (start_time or 0.0):
            QMessageBox.warning(self, "Invalid time range", "End time must be greater than start time.")
            return
        name = self.source_combo.currentText().split(" (")[0]
        range_label = f", {start_time:.1f}s–{end_time:.1f}s" if end_time is not None else ""
        generate_thumbnails = self.thumbnails_check.isChecked()
        self.state.queue_manager.add_job(
            QUEUE_STAGE_FRAMES,
            source_id,
            {
                "interval_seconds": interval,
                "start_time": start_time,
                "end_time": end_time,
                "generate_thumbnails": generate_thumbnails,
            },
            f"Frame extraction — {name} (every {interval}s{range_label}"
            f"{'' if generate_thumbnails else ', no thumbnails'})",
            target_frame_set_id=frame_set_id_for(
                self.state.conn, source_id, interval_seconds=interval, start_time=start_time, end_time=end_time
            ),
        )

    def _on_extract(self) -> None:
        source_id = self.source_combo.currentData()
        if source_id is None:
            return
        interval = self._current_interval()
        start_time, end_time = self._current_time_range()
        if end_time is not None and end_time <= (start_time or 0.0):
            QMessageBox.warning(self, "Invalid time range", "End time must be greater than start time.")
            return

        self.extract_btn.setEnabled(False)
        self.progress_area.start(f"Extracting frames every {interval}s…")
        run_in_background(
            self,
            _extract_frames_worker,
            self.state.project_root,
            source_id,
            interval,
            start_time,
            end_time,
            self.thumbnails_check.isChecked(),
            on_success=self._on_extract_success,
            on_error=self._on_extract_error,
            on_progress=self.progress_area.update_progress,
        )

    def _on_extract_success(self, frames) -> None:
        current = self.source_combo.currentIndex()
        self.refresh_sources()  # frame-set counts changed
        self.source_combo.setCurrentIndex(current)
        self.progress_area.finish(f"Extracted {len(frames)} frames.")
        self.state.notify_change()

    def _on_extract_error(self, exc: Exception) -> None:
        self._refresh_run_enabled()
        self.progress_area.finish("")
        if isinstance(exc, FrameExtractionError):
            QMessageBox.warning(self, "Frame extraction failed", str(exc))
        else:
            QMessageBox.critical(self, "Unexpected error extracting frames", f"{type(exc).__name__}: {exc}")


def _default_parallel_worker_count() -> int:
    return min(os.cpu_count() or 1, 4)


def _generate_views_worker(
    project_root: Path,
    frame_set_id: str,
    face_size: int,
    fov_degrees: float,
    face_names: list[str],
    max_workers: int | None,
    progress_callback=None,
):
    conn = open_index_db(project_root)
    try:
        return generate_views_for_frame_set(
            conn,
            project_root,
            frame_set_id,
            face_size=face_size,
            fov_degrees=fov_degrees,
            face_names=face_names,
            max_workers=max_workers,
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


class ProjectionPanel(QWidget):
    """Real projection: vine360.projection.generate.generate_views_for_frame_set,
    run off the GUI thread. Previously this stage never left PENDING in the
    sidebar because it had no execution at all -- it does now."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._choices: dict[str, FrameSetChoice] = {}
        layout = QVBoxLayout(self)
        _panel_header(
            "Projection",
            "Renders perspective views from each frame already extracted on the Frames stage "
            "(this does not touch the raw video again).",
            layout,
        )

        self.no_project_label = QLabel("Extract frames from a video source first (see the Frames stage).")
        layout.addWidget(self.no_project_label)

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._refresh_frame_list)
        self.source_combo.currentIndexChanged.connect(self._refresh_run_enabled)
        form.addRow("Frame set:", self.source_combo)
        self.face_size_spin = QSpinBox()
        self.face_size_spin.setRange(128, 4096)
        self.face_size_spin.setSingleStep(128)
        self.face_size_spin.setValue(1024)
        form.addRow("Face size (px):", self.face_size_spin)
        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(50.0, 170.0)
        self.fov_spin.setValue(90.0)
        self.fov_spin.setSuffix("°")
        form.addRow("Field of view:", self.fov_spin)
        controls_layout.addLayout(form)

        controls_layout.addWidget(QLabel("Views to generate per frame:"))
        faces_row = QHBoxLayout()
        self.face_checks: dict[str, QCheckBox] = {}
        for name in ALL_FACE_NAMES:
            cb = QCheckBox(name)
            cb.setChecked(name not in ("up", "down"))  # polar faces off by default -- little use, terrestrial capture
            self.face_checks[name] = cb
            faces_row.addWidget(cb)
        controls_layout.addLayout(faces_row)
        polar_note = QLabel("up/down (polar) are off by default: pointed at sky/ground, little use for a terrestrial vineyard capture.")
        polar_note.setWordWrap(True)
        polar_note.setStyleSheet("color: palette(mid);")
        controls_layout.addWidget(polar_note)

        parallel_row = QHBoxLayout()
        self.parallel_check = QCheckBox("Speed up with parallel processing")
        self.parallel_check.setChecked(True)
        self.parallel_check.toggled.connect(self._on_parallel_toggled)
        parallel_row.addWidget(self.parallel_check)
        parallel_row.addWidget(QLabel("Workers:"))
        self.worker_count_spin = QSpinBox()
        self.worker_count_spin.setRange(1, max(os.cpu_count() or 1, 1))
        self.worker_count_spin.setValue(_default_parallel_worker_count())
        self.worker_count_spin.setToolTip(
            "Each worker holds roughly a decoded copy of the full equirectangular frame in memory "
            "(~1GB for an 8K source) -- more workers means proportionally more peak RAM, up to your "
            f"CPU's thread count ({os.cpu_count() or 1}). Uncheck the box to fall back to the "
            "original single-process behavior."
        )
        parallel_row.addWidget(self.worker_count_spin)
        parallel_row.addStretch()
        controls_layout.addLayout(parallel_row)

        btn_row = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Projections")
        self.generate_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self.generate_btn)
        self.queue_btn = QPushButton("Add to Queue")
        self.queue_btn.setToolTip("Adds this exact configuration as a queued job instead of running it now.")
        self.queue_btn.clicked.connect(self._on_add_to_queue)
        btn_row.addWidget(self.queue_btn)
        controls_layout.addLayout(btn_row)

        self.progress_area = ProgressArea()
        controls_layout.addWidget(self.progress_area)

        controls_layout.addWidget(QLabel("Preview frame:"))
        self.frame_selector = TemporalFrameSelector()
        self.frame_selector.frame_changed.connect(self._refresh_preview)
        controls_layout.addWidget(self.frame_selector)
        self.preview = PreviewGallery("No views generated yet for this frame.")
        controls_layout.addWidget(self.preview)

        layout.addWidget(self.controls)
        self.controls.setVisible(False)
        layout.addStretch()

        if self.state.queue_manager is not None:
            self.state.queue_manager.queue_changed.connect(self._refresh_run_enabled)

    def on_project_changed(self) -> None:
        self.on_shown()

    def on_shown(self) -> None:
        have_project = self.state.conn is not None
        self.no_project_label.setVisible(not have_project)
        self.controls.setVisible(have_project)
        if have_project:
            self.refresh_sources()

    def refresh_sources(self) -> None:
        previous = self.source_combo.currentData()
        self.source_combo.clear()
        if self.state.conn is None:
            return
        self._choices = {c.frame_set_id: c for c in frame_set_choices(self.state)}
        for choice in self._choices.values():
            if choice.pending:
                detail = f"no frames yet ({choice.pending})"
            else:
                views = f"{choice.view_count} views already generated" if choice.view_count else "no views yet"
                detail = f"{choice.frame_count} frames, {views}"
            self.source_combo.addItem(f"{choice.name} — {detail}", userData=choice.frame_set_id)
        _restore_combo_selection(self.source_combo, previous)
        has_sources = self.source_combo.count() > 0
        self._has_sources = has_sources
        self.queue_btn.setEnabled(has_sources)
        self._refresh_run_enabled()
        self.progress_area.label.setText(
            "" if has_sources else "No video sources registered yet -- add one on Import, then extract frames."
        )
        self._refresh_frame_list()

    def _refresh_run_enabled(self) -> None:
        busy = bool(self.state.queue_manager and self.state.queue_manager.is_running)
        choice = self._choices.get(self.source_combo.currentData())
        has_frames = choice is not None and choice.frame_count > 0
        self.generate_btn.setEnabled(has_frames and not busy)
        self.generate_btn.setToolTip(
            ""
            if has_frames
            else 'This frame set has no extracted frames yet -- use "Add to Queue" instead; it will queue '
            "frame extraction automatically ahead of this projection job (or wait for the one already queued)."
        )

    def _refresh_frame_list(self) -> None:
        frame_set_id = self.source_combo.currentData()
        frames: list[tuple[str, float, int]] = []
        if self.state.conn is not None and frame_set_id is not None:
            frames = self.state.conn.execute(
                "SELECT f.frame_id, f.source_time, COUNT(v.view_id) AS view_count "
                "FROM frames f LEFT JOIN views v ON v.frame_id = f.frame_id "
                "WHERE f.frame_set_id = ? GROUP BY f.frame_id ORDER BY f.source_time",
                (frame_set_id,),
            ).fetchall()
        self.frame_selector.set_frames(frames)  # emits frame_changed -> _refresh_preview

    def _refresh_preview(self) -> None:
        frame_id = self.frame_selector.current_frame_id()
        if self.state.conn is None or frame_id is None:
            self.preview.set_items([])
            return
        rows = self.state.conn.execute(
            "SELECT projection_id, image_path FROM views WHERE frame_id = ? ORDER BY view_id", (frame_id,)
        ).fetchall()
        items = [
            (self.state.project_root / image_path, Path(image_path).stem)
            for _projection_id, image_path in rows
        ]
        self.preview.set_items(items)

    def _selected_face_names(self) -> list[str]:
        return [name for name, cb in self.face_checks.items() if cb.isChecked()]

    def _on_parallel_toggled(self, checked: bool) -> None:
        self.worker_count_spin.setEnabled(checked)

    def _current_max_workers(self) -> int | None:
        return self.worker_count_spin.value() if self.parallel_check.isChecked() else None

    def _on_add_to_queue(self) -> None:
        choice = self._choices.get(self.source_combo.currentData())
        if choice is None:
            return
        face_names = self._selected_face_names()
        if not face_names:
            QMessageBox.warning(self, "No views selected", "Check at least one face to generate.")
            return
        name = choice.name
        max_workers = self._current_max_workers()
        self.state.queue_manager.add_job(
            QUEUE_STAGE_PROJECTION,
            choice.source_id,
            {
                "face_size": self.face_size_spin.value(),
                "fov_degrees": self.fov_spin.value(),
                "face_names": face_names,
                "max_workers": max_workers,
            },
            f"Projection — {name} ({', '.join(face_names)})",
            target_frame_set_id=choice.frame_set_id,
        )

    def _on_generate(self) -> None:
        frame_set_id = self.source_combo.currentData()
        if frame_set_id is None:
            return
        face_names = self._selected_face_names()
        if not face_names:
            QMessageBox.warning(self, "No views selected", "Check at least one face to generate.")
            return
        max_workers = self._current_max_workers()
        self.generate_btn.setEnabled(False)
        self.progress_area.start("Generating projections…")
        run_in_background(
            self,
            _generate_views_worker,
            self.state.project_root,
            frame_set_id,
            self.face_size_spin.value(),
            self.fov_spin.value(),
            face_names,
            max_workers,
            on_success=self._on_generate_success,
            on_error=self._on_generate_error,
            on_progress=self.progress_area.update_progress,
        )

    def _on_generate_success(self, views) -> None:
        self.progress_area.finish(f"Generated {len(views)} views.")
        current_frame_id = self.frame_selector.current_frame_id()
        self.refresh_sources()
        if current_frame_id:
            self.frame_selector.jump_to_frame_id(current_frame_id)
        self.state.notify_change()

    def _on_generate_error(self, exc: Exception) -> None:
        self._refresh_run_enabled()
        self.progress_area.finish("")
        if isinstance(exc, ProjectionGenerationError):
            QMessageBox.warning(self, "Projection failed", str(exc))
        else:
            QMessageBox.critical(self, "Unexpected error generating projections", f"{type(exc).__name__}: {exc}")


def _build_masks_worker(
    project_root: Path, frame_set_id: str, use_sam3_person: bool, use_sam3_sky: bool, progress_callback=None
):
    conn = open_index_db(project_root)
    try:
        return build_masks_for_frame_set(
            conn,
            project_root,
            frame_set_id,
            use_sam3_person=use_sam3_person,
            use_sam3_sky=use_sam3_sky,
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


class MasksPanel(QWidget):
    """Real masking: vine360.masking.build.build_masks_for_frame_set, run off
    the GUI thread. SAM 3, if requested, loads once for the whole batch
    (not once per view)."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._sam3_checked = False  # deferred to first on_shown(): importing
        # torch/transformers to check availability is expensive (roughly
        # 140MB -> 700MB+ RSS observed) and shouldn't cost anything at app
        # startup for users who never open this panel.
        self._choices: dict[str, FrameSetChoice] = {}
        layout = QVBoxLayout(self)
        _panel_header("Masks", "Segment person/sky and build the keep-mask for each view.", layout)

        self.no_project_label = QLabel("Generate projections first (see the Projection stage).")
        layout.addWidget(self.no_project_label)

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        self.stage_indicator = StageIndicator(["Build masks", "Review flagged"])
        controls_layout.addWidget(self.stage_indicator)

        form = QFormLayout()
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._refresh_frame_list)
        self.source_combo.currentIndexChanged.connect(self._refresh_run_enabled)
        form.addRow("Frame set:", self.source_combo)
        controls_layout.addLayout(form)

        sam3_row = QHBoxLayout()
        self.sam3_person_check = QCheckBox("SAM 3: exclude person")
        self.sam3_sky_check = QCheckBox("SAM 3: exclude sky (real segmentation)")
        self.sam3_person_check.setEnabled(False)
        self.sam3_sky_check.setEnabled(False)
        sam3_row.addWidget(self.sam3_person_check)
        sam3_row.addWidget(self.sam3_sky_check)
        controls_layout.addLayout(sam3_row)
        self.sam3_note = QLabel(
            "Sky is always masked -- via real SAM 3 if checked, a classical brightness/color heuristic "
            "otherwise (coarser; see vine360.masking.classical_sky). Person exclusion has no non-SAM-3 "
            "fallback: no color/brightness heuristic reliably separates a person from vineyard foliage. "
            "(Checking SAM 3 availability…)"
        )
        self.sam3_note.setWordWrap(True)
        self.sam3_note.setStyleSheet("color: palette(mid);")
        controls_layout.addWidget(self.sam3_note)

        build_btn_row = QHBoxLayout()
        self.build_btn = QPushButton("Build Masks")
        self.build_btn.clicked.connect(self._on_build)
        build_btn_row.addWidget(self.build_btn)
        self.queue_btn = QPushButton("Add to Queue")
        self.queue_btn.setToolTip("Adds this exact configuration as a queued job instead of running it now.")
        self.queue_btn.clicked.connect(self._on_add_to_queue)
        build_btn_row.addWidget(self.queue_btn)
        controls_layout.addLayout(build_btn_row)

        self.progress_area = ProgressArea()
        controls_layout.addWidget(self.progress_area)

        self.flags_summary_label = QLabel("")
        self.flags_summary_label.setWordWrap(True)
        controls_layout.addWidget(self.flags_summary_label)
        self.flags_list = QListWidget()
        self.flags_list.setMaximumHeight(120)
        self.flags_list.setVisible(False)
        self.flags_list.itemDoubleClicked.connect(self._on_flags_list_double_clicked)
        controls_layout.addWidget(self.flags_list)

        controls_layout.addWidget(QLabel("Preview frame:"))
        self.frame_selector = TemporalFrameSelector()
        self.frame_selector.frame_changed.connect(self._refresh_face_combo)
        controls_layout.addWidget(self.frame_selector)

        review_row = QHBoxLayout()
        review_row.addWidget(QLabel("View:"))
        self.face_combo = QComboBox()
        self.face_combo.currentIndexChanged.connect(self._refresh_preview)
        review_row.addWidget(self.face_combo)
        self.prev_flagged_btn = QPushButton("Previous Flagged")
        self.prev_flagged_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
        self.prev_flagged_btn.clicked.connect(lambda: self._jump_to_flagged(-1))
        review_row.addWidget(self.prev_flagged_btn)
        self.flag_btn = QPushButton("Flag for Review")
        self.flag_btn.setIcon(QIcon(_flag_icon()))
        self.flag_btn.setCheckable(True)
        self.flag_btn.toggled.connect(self._on_flag_toggled)
        review_row.addWidget(self.flag_btn)
        self.next_flagged_btn = QPushButton("Next Flagged")
        self.next_flagged_btn.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.next_flagged_btn.clicked.connect(lambda: self._jump_to_flagged(1))
        review_row.addWidget(self.next_flagged_btn)
        controls_layout.addLayout(review_row)

        self.preview = PreviewGallery("No views generated yet.")
        controls_layout.addWidget(self.preview)

        layout.addWidget(self.controls)
        self.controls.setVisible(False)
        layout.addStretch()
        # (frame_id, view_id, keep_fraction, flagged) across the whole
        # selected source, ordered by (frame time, view_id) -- backs the
        # previous/next-flagged navigation buttons.
        self._flat_views: list[tuple[str, str, float | None, bool]] = []

        if self.state.queue_manager is not None:
            self.state.queue_manager.queue_changed.connect(self._refresh_run_enabled)

    def on_project_changed(self) -> None:
        self.on_shown()

    def on_shown(self) -> None:
        if not self._sam3_checked:
            self._sam3_checked = True
            status = sam3_validate_installation()
            available = status.torch_installed and status.transformers_installed
            self.sam3_person_check.setEnabled(available)
            self.sam3_sky_check.setEnabled(available)
            if not available:
                self.sam3_note.setText(
                    self.sam3_note.text().replace(" (Checking SAM 3 availability…)", "")
                    + " (SAM 3 unavailable in this environment: torch/transformers not installed.)"
                )
            else:
                self.sam3_note.setText(self.sam3_note.text().replace(" (Checking SAM 3 availability…)", ""))

        have_project = self.state.conn is not None
        self.no_project_label.setVisible(not have_project)
        self.controls.setVisible(have_project)
        if have_project:
            self.refresh_sources()

    def refresh_sources(self) -> None:
        previous = self.source_combo.currentData()
        self.source_combo.clear()
        if self.state.conn is None:
            return
        self._choices = {c.frame_set_id: c for c in frame_set_choices(self.state)}
        for choice in self._choices.values():
            if choice.pending:
                detail = f"no frames yet ({choice.pending})"
            else:
                masked = f"{choice.mask_count} masked already" if choice.mask_count else "not masked yet"
                detail = f"{choice.view_count} views, {masked}"
            self.source_combo.addItem(f"{choice.name} — {detail}", userData=choice.frame_set_id)
        _restore_combo_selection(self.source_combo, previous)
        has_sources = self.source_combo.count() > 0
        self._has_sources = has_sources
        self.queue_btn.setEnabled(has_sources)
        self._refresh_run_enabled()
        self.progress_area.label.setText("" if has_sources else "No video sources registered yet -- add one on Import.")
        self._refresh_frame_list()

    def _refresh_run_enabled(self) -> None:
        busy = bool(self.state.queue_manager and self.state.queue_manager.is_running)
        choice = self._choices.get(self.source_combo.currentData())
        has_views = choice is not None and choice.view_count > 0
        self.build_btn.setEnabled(has_views and not busy)
        self.build_btn.setToolTip(
            ""
            if has_views
            else 'This frame set has no projected views yet -- use "Add to Queue" instead; it will queue '
            "projection automatically ahead of this masking job."
        )

    def _refresh_frame_list(self) -> None:
        frame_set_id = self.source_combo.currentData()
        self._flat_views = []
        frames: list[tuple[str, float, int]] = []
        if self.state.conn is not None and frame_set_id is not None:
            frame_rows = self.state.conn.execute(
                "SELECT f.frame_id, f.source_time, COUNT(v.view_id) AS view_count "
                "FROM frames f JOIN views v ON v.frame_id = f.frame_id "
                "WHERE f.frame_set_id = ? GROUP BY f.frame_id ORDER BY f.source_time",
                (frame_set_id,),
            ).fetchall()
            frames = [(fid, t, vc) for fid, t, vc in frame_rows]
            self._flat_views = self.state.conn.execute(
                "SELECT f.frame_id, v.view_id, m.keep_fraction, COALESCE(m.flagged_for_review, 0) "
                "FROM views v JOIN frames f ON f.frame_id = v.frame_id LEFT JOIN masks m ON m.view_id = v.view_id "
                "WHERE f.frame_set_id = ? ORDER BY f.source_time, v.view_id",
                (frame_set_id,),
            ).fetchall()
        self.frame_selector.set_frames(frames)  # emits frame_changed -> _refresh_face_combo
        self._update_stage_indicator()

    def _update_stage_indicator(self) -> None:
        has_masks = any(keep_fraction is not None for _, _, keep_fraction, _ in self._flat_views)
        if not has_masks:
            self.stage_indicator.set_stages([ACTIVE, PENDING])
        else:
            any_flagged = any(flagged for _, _, _, flagged in self._flat_views)
            self.stage_indicator.set_stages([DONE, ACTIVE if any_flagged else DONE])

    def _refresh_face_combo(self) -> None:
        self.face_combo.blockSignals(True)
        self.face_combo.clear()
        frame_id = self.frame_selector.current_frame_id()
        for f_id, view_id, keep_fraction, flagged in self._flat_views:
            if f_id != frame_id:
                continue
            face_name = view_id.split(":")[-1]
            status = f"keep {keep_fraction:.0%}" if keep_fraction is not None else "unmasked"
            icon = QIcon(_flag_icon()) if flagged else QIcon()
            self.face_combo.addItem(icon, f"{face_name} ({status})", userData=view_id)
        self.face_combo.blockSignals(False)
        self._refresh_preview()

    def _current_view_id(self) -> str | None:
        return self.face_combo.currentData()

    def _refresh_preview(self) -> None:
        view_id = self._current_view_id()
        self.flag_btn.blockSignals(True)
        if view_id is None:
            self.flag_btn.setChecked(False)
            self.flag_btn.setEnabled(False)
        else:
            flagged = any(v == view_id and f for _fr, v, _kf, f in self._flat_views)
            self.flag_btn.setChecked(bool(flagged))
            self.flag_btn.setEnabled(any(v == view_id for _fr, v, _kf, _f in self._flat_views))
        self.flag_btn.blockSignals(False)

        if self.state.conn is None or view_id is None:
            self.preview.set_items([])
            return
        row = self.state.conn.execute("SELECT image_path FROM views WHERE view_id = ?", (view_id,)).fetchone()
        items = []
        if row is not None:
            items.append((self.state.project_root / row[0], "original view"))
            keep_path = self.state.project_root / "masks" / "keep" / (Path(row[0]).relative_to("projections").as_posix() + ".png")
            if keep_path.exists():
                items.append((keep_path, "keep mask"))
        self.preview.set_items(items)

    def _on_flag_toggled(self, checked: bool) -> None:
        view_id = self._current_view_id()
        if self.state.conn is None or view_id is None:
            return
        try:
            set_view_flagged(self.state.conn, view_id, checked)
        except MaskBuildError as exc:
            QMessageBox.warning(self, "Could not update flag", str(exc))
            return
        for i, (frame_id, v, kf, _f) in enumerate(self._flat_views):
            if v == view_id:
                self._flat_views[i] = (frame_id, v, kf, checked)
                break
        self._refresh_face_combo_labels_only()
        self._update_stage_indicator()

    def _refresh_face_combo_labels_only(self) -> None:
        """Updates the face combo's item text/icon (flag marker) without
        emitting currentIndexChanged / disturbing the current preview."""
        self.face_combo.blockSignals(True)
        for i in range(self.face_combo.count()):
            view_id = self.face_combo.itemData(i)
            match = next((v for v in self._flat_views if v[1] == view_id), None)
            if match:
                _frame_id, _v, keep_fraction, flagged = match
                face_name = view_id.split(":")[-1]
                status = f"keep {keep_fraction:.0%}" if keep_fraction is not None else "unmasked"
                self.face_combo.setItemText(i, f"{face_name} ({status})")
                self.face_combo.setItemIcon(i, QIcon(_flag_icon()) if flagged else QIcon())
        self.face_combo.blockSignals(False)

    def _jump_to_flagged(self, direction: int) -> None:
        flagged_views = [(frame_id, view_id) for frame_id, view_id, _kf, flagged in self._flat_views if flagged]
        if not flagged_views:
            QMessageBox.information(self, "No flagged views", "No views are currently flagged for review.")
            return
        current_view_id = self._current_view_id()
        ids = [v for _f, v in flagged_views]
        if current_view_id in ids:
            start = ids.index(current_view_id)
            target_frame_id, target_view_id = flagged_views[(start + direction) % len(flagged_views)]
        else:
            target_frame_id, target_view_id = flagged_views[0]
        self.frame_selector.jump_to_frame_id(target_frame_id)  # triggers _refresh_face_combo
        index = self.face_combo.findData(target_view_id)
        if index >= 0:
            self.face_combo.setCurrentIndex(index)

    def _on_add_to_queue(self) -> None:
        choice = self._choices.get(self.source_combo.currentData())
        if choice is None:
            return
        use_person = self.sam3_person_check.isChecked()
        use_sky = self.sam3_sky_check.isChecked()
        self.state.queue_manager.add_job(
            QUEUE_STAGE_MASKS,
            choice.source_id,
            {"use_sam3_person": use_person, "use_sam3_sky": use_sky},
            f"Masks — {choice.name} (SAM 3 person={use_person}, sky={use_sky})",
            target_frame_set_id=choice.frame_set_id,
        )

    def _on_build(self) -> None:
        frame_set_id = self.source_combo.currentData()
        if frame_set_id is None:
            return
        self.build_btn.setEnabled(False)
        self.flags_summary_label.setText("")
        self.flags_list.setVisible(False)
        self.progress_area.start("Building masks…")
        run_in_background(
            self,
            _build_masks_worker,
            self.state.project_root,
            frame_set_id,
            self.sam3_person_check.isChecked(),
            self.sam3_sky_check.isChecked(),
            on_success=self._on_build_success,
            on_error=self._on_build_error,
            on_progress=self.progress_area.update_progress,
        )

    def _on_build_success(self, masks) -> None:
        self._refresh_run_enabled()
        self.progress_area.finish(f"Masked {len(masks)} views.")

        flagged = [
            (m.view_id, is_keep_fraction_anomalous(m.keep_fraction))
            for m in masks
            if m.keep_fraction is not None and is_keep_fraction_anomalous(m.keep_fraction)
        ]
        self.flags_list.clear()
        if flagged:
            self.flags_summary_label.setText(
                f"{len(flagged)} view(s) auto-flagged for review -- double-click one to jump to it "
                "(or use Previous/Next Flagged below):"
            )
            for view_id, reason in flagged:
                item = QListWidgetItem(f"{view_id}: {reason}")
                item.setData(Qt.UserRole, view_id)
                self.flags_list.addItem(item)
            self.flags_list.setVisible(True)
        else:
            self.flags_summary_label.setText("")
            self.flags_list.setVisible(False)

        current_frame_id = self.frame_selector.current_frame_id()
        current_view_id = self._current_view_id()
        self.refresh_sources()
        if current_frame_id and self.frame_selector.jump_to_frame_id(current_frame_id) and current_view_id:
            index = self.face_combo.findData(current_view_id)
            if index >= 0:
                self.face_combo.setCurrentIndex(index)
        self.state.notify_change()

    def _on_flags_list_double_clicked(self, item: QListWidgetItem) -> None:
        view_id = item.data(Qt.UserRole)
        match = next((v for v in self._flat_views if v[1] == view_id), None)
        if match is None:
            return
        frame_id, _v, _kf, _flagged = match
        self.frame_selector.jump_to_frame_id(frame_id)
        index = self.face_combo.findData(view_id)
        if index >= 0:
            self.face_combo.setCurrentIndex(index)

    def _on_build_error(self, exc: Exception) -> None:
        self._refresh_run_enabled()
        self.progress_area.finish("")
        if isinstance(exc, MaskBuildError):
            QMessageBox.warning(self, "Masking failed", str(exc))
        else:
            QMessageBox.critical(self, "Unexpected error building masks", f"{type(exc).__name__}: {exc}")


ENGINE_COLMAP_PROJECTIONS = "colmap_projections"
ENGINE_COLMAP_EQUIRECTANGULAR = "colmap_equirectangular"
ENGINE_SPHERESFM = "spheresfm"


def _run_sfm_worker(
    project_root: Path,
    image_source: str,
    frame_set_id: str | None,
    camera_model: str,
    progress_callback=None,
):
    from vine360.sfm.colmap_adapter import SfmConfig

    conn = open_index_db(project_root)
    try:
        return run_sfm_for_project(
            conn,
            project_root,
            config=SfmConfig(camera_model=camera_model),
            image_source=image_source,
            frame_set_id=frame_set_id,
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


class PoseEstimationPanel(QWidget):
    """Real SfM: vine360.sfm.project_run.run_sfm_for_project (pycolmap),
    run off the GUI thread. Two real engine choices (six-face projections,
    the default; or COLMAP's own native equirectangular camera model
    directly on raw frames) plus SphereSfM, a genuinely separate COLMAP
    fork that needs building from C++ source -- unverified here, its Run
    button stays disabled with an explanation (see docs/adr/0017).

    COLMAP needs real camera-position parallax across multiple frames to
    reconstruct anything -- a single frame's projected faces (or a single
    raw equirectangular frame) share one optical center and cannot be
    3D-reconstructed regardless of match quality (see docs/adr/0007)."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._choices: dict[str, FrameSetChoice] = {}
        layout = QVBoxLayout(self)
        _panel_header("Pose estimation", "Run COLMAP feature extraction, matching and mapping.", layout)

        self.no_project_label = QLabel("Generate projections first (see the Projection stage).")
        layout.addWidget(self.no_project_label)

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("COLMAP (six-face projections)", userData=ENGINE_COLMAP_PROJECTIONS)
        self.engine_combo.addItem(
            "COLMAP native equirectangular (raw frames)", userData=ENGINE_COLMAP_EQUIRECTANGULAR
        )
        self.engine_combo.addItem("SphereSfM (external -- unverified)", userData=ENGINE_SPHERESFM)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        form.addRow("Engine:", self.engine_combo)
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._refresh_enabled)
        form.addRow("Frame set:", self.source_combo)
        controls_layout.addLayout(form)

        self.engine_note = QLabel("")
        self.engine_note.setWordWrap(True)
        self.engine_note.setStyleSheet("color: palette(mid);")
        controls_layout.addWidget(self.engine_note)

        run_btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run SfM")
        self.run_btn.clicked.connect(self._on_run)
        run_btn_row.addWidget(self.run_btn)
        self.queue_btn = QPushButton("Add to Queue")
        self.queue_btn.setToolTip("Adds this exact configuration as a queued job instead of running it now.")
        self.queue_btn.clicked.connect(self._on_add_to_queue)
        run_btn_row.addWidget(self.queue_btn)
        controls_layout.addLayout(run_btn_row)

        self.progress_area = ProgressArea()
        controls_layout.addWidget(self.progress_area)

        self.stats_label = QLabel("No SfM run yet.")
        self.stats_label.setWordWrap(True)
        controls_layout.addWidget(self.stats_label)

        layout.addWidget(self.controls)
        self.controls.setVisible(False)
        layout.addStretch()
        self._on_engine_changed()  # set initial note/visibility

        if self.state.queue_manager is not None:
            self.state.queue_manager.queue_changed.connect(self._refresh_enabled)

    def on_project_changed(self) -> None:
        self.on_shown()

    def on_shown(self) -> None:
        have_project = self.state.conn is not None
        self.no_project_label.setVisible(not have_project)
        self.controls.setVisible(have_project)
        if have_project:
            self._refresh_sources()
            self._refresh_enabled()
            self._show_latest_run()

    def _refresh_sources(self) -> None:
        """The six-face engine can run project-wide ("All frame sets") or
        on one frame set's views; the native-equirectangular engine always
        needs one equirectangular frame set (docs/adr/0034)."""
        previous = self.source_combo.currentData()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self._choices = {}
        if self.state.conn is not None:
            engine = self.engine_combo.currentData()
            frames_engine = engine == ENGINE_COLMAP_EQUIRECTANGULAR
            if not frames_engine:
                self.source_combo.addItem("All frame sets (every projected view in the project)", userData=None)
            self._choices = {
                c.frame_set_id: c for c in frame_set_choices(self.state, equirect_only=frames_engine)
            }
            for choice in self._choices.values():
                if choice.pending:
                    detail = f"no frames yet ({choice.pending})"
                elif frames_engine:
                    detail = f"{choice.frame_count} frames"
                else:
                    detail = f"{choice.view_count} views" if choice.view_count else "no views yet"
                self.source_combo.addItem(f"{choice.name} — {detail}", userData=choice.frame_set_id)
            _restore_combo_selection(self.source_combo, previous)
        self.source_combo.blockSignals(False)

    def _on_engine_changed(self) -> None:
        engine = self.engine_combo.currentData()
        self._refresh_sources()
        self.source_combo.setEnabled(engine != ENGINE_SPHERESFM)
        if engine == ENGINE_COLMAP_PROJECTIONS:
            self.engine_note.setText(
                "Runs against the projected views of the chosen frame set -- or every projected view in "
                "project/projections/ for \"All frame sets\" -- using masks from project/masks/keep/ if any "
                "have been built. The handover doc's originally recommended pipeline. If a source has more "
                "than one frame set, pick one: \"All\" would feed COLMAP near-duplicate images of the same "
                "moments."
            )
        elif engine == ENGINE_COLMAP_EQUIRECTANGULAR:
            self.engine_note.setText(
                "Runs directly on the selected frame set's raw equirectangular frames, using COLMAP's own native "
                "EQUIRECTANGULAR camera model -- no projection step needed. Confirmed for real with synthetic "
                "ground truth (full registration, near-zero error); real-photo feature-matching quality near the "
                "poles and across the seam is unproven. No masking support yet on this path."
            )
        else:
            from vine360.sfm.spheresfm_adapter import SPHERESFM_REPO_URL, validate_installation as spheresfm_status

            status = spheresfm_status()
            self.engine_note.setText(
                f"SphereSfM ({SPHERESFM_REPO_URL}) is a separate COLMAP fork with its own spherical camera "
                "model and sphere-aware matching/mapping. It must be compiled from C++ source -- no pip package "
                "or prebuilt binary exists, and this environment cannot build it. "
                f"colmap binary on PATH: {'yes, but ' + status['note'] if status['colmap_binary_found'] else 'no'}."
            )
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        if self.state.conn is None:
            self.run_btn.setEnabled(False)
            return
        busy = bool(self.state.queue_manager and self.state.queue_manager.is_running)
        engine = self.engine_combo.currentData()
        if engine == ENGINE_SPHERESFM:
            self.run_btn.setEnabled(False)
            self.run_btn.setToolTip("Not runnable here -- SphereSfM needs a custom build; see the note above.")
            return
        self.run_btn.setToolTip("")
        choice = self._choices.get(self.source_combo.currentData())
        if engine == ENGINE_COLMAP_EQUIRECTANGULAR:
            has_frames = choice is not None and choice.frame_count > 0
            self.run_btn.setEnabled(has_frames and not busy)
            if not has_frames:
                self.run_btn.setToolTip(
                    'This frame set has no extracted frames yet -- use "Add to Queue" instead; it will '
                    "queue frame extraction automatically first."
                )
        elif choice is not None:
            self.run_btn.setEnabled(choice.view_count > 0 and not busy)
            if choice.view_count == 0:
                self.run_btn.setToolTip(
                    'This frame set has no projected views yet -- use "Add to Queue" instead; it will '
                    "queue projection automatically first."
                )
        else:
            n_views = self.state.conn.execute("SELECT COUNT(*) FROM views").fetchone()[0]
            self.run_btn.setEnabled(n_views > 0 and not busy)

    def _show_latest_run(self) -> None:
        row = self.state.conn.execute(
            "SELECT model_stats, created_at FROM sfm_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return
        stats = json.loads(row[0])
        self.stats_label.setText(
            f"Last run ({row[1]}): registered {stats['registered_images']}/{stats['total_images']} "
            f"({stats['registered_ratio']:.0%}), {stats['num_points3d']} points, "
            f"mean reprojection error {stats['mean_reprojection_error']:.2f}px, "
            f"{stats['num_connected_models']} connected model(s)."
        )

    def _current_sfm_args(self) -> tuple[str, str | None, str]:
        engine = self.engine_combo.currentData()
        if engine == ENGINE_COLMAP_EQUIRECTANGULAR:
            return "frames", self.source_combo.currentData(), "EQUIRECTANGULAR"
        return "projections", self.source_combo.currentData(), "SIMPLE_RADIAL"

    def _on_add_to_queue(self) -> None:
        if self.engine_combo.currentData() == ENGINE_SPHERESFM:
            QMessageBox.warning(self, "Not runnable", "SphereSfM needs a custom build; see the note above.")
            return
        image_source, frame_set_id, camera_model = self._current_sfm_args()
        if image_source == "frames" and frame_set_id is None:
            QMessageBox.warning(self, "No frame set selected", "Choose an equirectangular frame set first.")
            return
        engine_label = (
            "COLMAP (equirectangular)" if image_source == "frames" else "COLMAP (six-face projections)"
        )
        choice = self._choices.get(frame_set_id)
        scope = choice.name if choice else "all frame sets"
        self.state.queue_manager.add_job(
            QUEUE_STAGE_POSE,
            choice.source_id if choice else None,
            {"image_source": image_source, "frame_set_id": frame_set_id, "camera_model": camera_model},
            f"Pose estimation — {engine_label}, {scope}",
        )

    def _on_run(self) -> None:
        image_source, frame_set_id, camera_model = self._current_sfm_args()

        self.run_btn.setEnabled(False)
        self.progress_area.start("Running SfM…")
        run_in_background(
            self,
            _run_sfm_worker,
            self.state.project_root,
            image_source,
            frame_set_id,
            camera_model,
            on_success=self._on_run_success,
            on_error=self._on_run_error,
            on_progress=self.progress_area.update_progress,
        )

    def _on_run_success(self, result) -> None:
        diagnostics, warnings = result
        self._refresh_enabled()
        self.progress_area.finish("SfM complete.")
        text = (
            f"Registered {diagnostics.registered_images}/{diagnostics.total_images} "
            f"({diagnostics.registered_ratio:.0%}), {diagnostics.num_points3d} points, "
            f"mean reprojection error {diagnostics.mean_reprojection_error:.2f}px, "
            f"{diagnostics.num_connected_models} connected model(s)."
        )
        if warnings:
            text += "\n⚠ " + "; ".join(warnings)
        self.stats_label.setText(text)
        self.state.notify_change()

    def _on_run_error(self, exc: Exception) -> None:
        self._refresh_enabled()
        self.progress_area.finish("")
        if isinstance(exc, SfmRegistrationError):
            QMessageBox.warning(
                self,
                "No usable reconstruction",
                f"{exc}\n\nThis is a real, expected outcome for a view set with no camera-position "
                "parallax (e.g. views from only a single frame) -- not necessarily a bug.",
            )
        else:
            QMessageBox.critical(self, "Unexpected error running SfM", f"{type(exc).__name__}: {exc}")


class QueuePanel(QWidget):
    """The right-hand dock: an editable, persisted plan of queued jobs
    (see vine360.gui.queue_manager) that runs unattended, one job at a
    time, while every stage panel's own "Run now" button keeps working
    for manual single-step use. Jobs are added from each stage panel's
    "Add to Queue" button -- this panel only displays/reorders/starts the
    plan, it never builds job params itself."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        header = QLabel("Queue")
        header.setStyleSheet("font-weight: 600; font-size: 13px;")
        layout.addWidget(header)
        subtitle = QLabel("Runs queued jobs in order. Failures block only their own dependents.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: palette(mid);")
        layout.addWidget(subtitle)

        button_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Queue")
        self.start_btn.clicked.connect(self._on_start)
        button_row.addWidget(self.start_btn)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._on_pause)
        button_row.addWidget(self.pause_btn)
        self.cancel_btn = QPushButton("Cancel All")
        self.cancel_btn.clicked.connect(self._on_cancel_all)
        button_row.addWidget(self.cancel_btn)
        layout.addLayout(button_row)

        self.progress_area = ProgressArea()
        layout.addWidget(self.progress_area)

        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QAbstractItemView.InternalMove)
        self.list_widget.setDefaultDropAction(Qt.MoveAction)
        self.list_widget.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self.list_widget, stretch=1)

        self._manager: QueueManager | None = None

    def bind_manager(self, manager: QueueManager) -> None:
        self._manager = manager
        manager.queue_changed.connect(self.refresh)
        manager.job_started.connect(lambda _job_id: self.progress_area.start("Running…"))
        manager.job_progress.connect(self.progress_area.update_progress)
        manager.job_finished.connect(lambda _job_id: self.progress_area.finish(""))
        manager.job_failed.connect(lambda _job_id, msg: self.progress_area.finish(""))
        manager.queue_idle.connect(self._on_idle)
        self.refresh()

    def refresh(self) -> None:
        if self._manager is None:
            return
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for job in self._manager.jobs:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, job.job_id)
            widget = self._row_widget(job)
            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)
        self.list_widget.blockSignals(False)
        self.start_btn.setEnabled(not self._manager.is_running)
        self.pause_btn.setEnabled(self._manager.is_running)
        self.cancel_btn.setEnabled(any(j.status in (QUEUED, BLOCKED, JOB_RUNNING) for j in self._manager.jobs))

    def _row_widget(self, job) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(2, 2, 2, 2)

        dot = QLabel()
        dot.setPixmap(_job_status_dot(job.status))
        row_layout.addWidget(dot)

        # job.label is already self-describing (stage, source, and the
        # relevant settings -- see each panel's _on_add_to_queue), so it's
        # shown as-is rather than re-prefixed with the stage name and a
        # raw source_id fragment, which would just repeat it less clearly.
        text = job.label
        if job.auto_added:
            text += "  (auto)"
        label = QLabel(text)
        label.setWordWrap(True)
        if job.error_message:
            label.setToolTip(job.error_message)
        row_layout.addWidget(label, stretch=1)

        if job.status == FAILED:
            retry_btn = QPushButton("Retry")
            retry_btn.clicked.connect(lambda: self._manager.retry_job(job.job_id))
            row_layout.addWidget(retry_btn)
        if job.status in (QUEUED, FAILED, BLOCKED):
            skip_btn = QPushButton("Skip")
            skip_btn.clicked.connect(lambda: self._manager.skip_job(job.job_id))
            row_layout.addWidget(skip_btn)
        remove_btn = QPushButton("✕")
        remove_btn.setEnabled(job.status != JOB_RUNNING)
        remove_btn.setFixedWidth(24)
        remove_btn.clicked.connect(lambda: self._manager.remove_job(job.job_id))
        row_layout.addWidget(remove_btn)
        return row

    def _on_rows_moved(self, *_args) -> None:
        if self._manager is None:
            return
        job_ids = [
            self.list_widget.item(i).data(Qt.UserRole) for i in range(self.list_widget.count())
        ]
        self._manager.reorder(job_ids)

    def _on_start(self) -> None:
        if self._manager is not None:
            self._manager.start(self)

    def _on_pause(self) -> None:
        if self._manager is not None:
            self._manager.pause()
            self.refresh()

    def _on_cancel_all(self) -> None:
        if self._manager is None:
            return
        if self._manager.is_running:
            self.progress_area.label.setText(
                "Clearing queue — the job currently running can't be interrupted and will finish first."
            )
        self._manager.cancel_all()

    def _on_idle(self) -> None:
        self.progress_area.finish("")
        self.refresh()


def _export_postshot_worker(project_root: Path, output_dir: Path, run_id: str | None, progress_callback=None):
    conn = open_index_db(project_root)
    try:
        return export_for_postshot(
            conn, project_root, output_dir, run_id=run_id, progress_callback=progress_callback
        )
    finally:
        conn.close()


def _export_frames_and_masks_worker(
    project_root: Path, output_dir: Path, frame_set_id: str | None, progress_callback=None
):
    conn = open_index_db(project_root)
    try:
        return export_frames_and_masks_for_postshot(
            conn, project_root, output_dir, frame_set_id=frame_set_id, progress_callback=progress_callback
        )
    finally:
        conn.close()


def _export_realityscan_worker(
    project_root: Path,
    output_dir: Path,
    frame_set_id: str | None,
    include_camera_priors: bool = False,
    progress_callback=None,
):
    conn = open_index_db(project_root)
    try:
        return export_for_realityscan(
            conn,
            project_root,
            output_dir,
            frame_set_id=frame_set_id,
            include_camera_priors=include_camera_priors,
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


# exports/<capture>/<format>/ -- see docs/adr/0028. Keyed by mode_combo's
# userData; ExportPanel._default_output_dir is the only place this is used.
_EXPORT_FORMAT_TAGS = {"poses": "colmap", "frames_masks": "postshot", "realityscan": "realityscan"}
_EXPORT_BUTTON_LABELS = {
    "poses": "Export for Postshot",
    "frames_masks": "Export for Postshot",
    "realityscan": "Export for RealityScan",
}


class ExportPanel(QWidget):
    """Bundles a completed SfM run's poses/images/masks (or, for the two
    images-only modes, just images+masks) for import into Postshot,
    RealityScan, or any other COLMAP-based/images-only external trainer
    -- see vine360.export.postshot's module docstring and docs/adr/0018/
    0028 for exactly what's copied/renamed and why, and what's still
    unverified (no real Postshot or RealityScan install is available
    here)."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._choices: dict[str, FrameSetChoice] = {}
        layout = QVBoxLayout(self)
        _panel_header(
            "Export",
            "Bundle a Pose estimation run's poses, images and masks for Postshot, RealityScan, or "
            "another COLMAP-based/images-only external trainer. Verified against each tool's "
            "published docs, not against a real install -- see docs/adr/0018 and docs/adr/0028.",
            layout,
        )

        self.no_project_label = QLabel("Open a project first.")
        layout.addWidget(self.no_project_label)

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        self.stage_indicator = StageIndicator(["Configure", "Export"])
        controls_layout.addWidget(self.stage_indicator)
        self._exported = False  # whether the *current* configuration has been successfully exported

        form = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Poses + images + masks (import our SfM run)", userData="poses")
        self.mode_combo.addItem(
            "Images + masks only (let Postshot run its own pose estimation)", userData="frames_masks"
        )
        self.mode_combo.addItem(
            "RealityScan images + masks (let RealityScan run its own pose estimation)",
            userData="realityscan",
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Export:", self.mode_combo)
        self.run_combo = QComboBox()
        form.addRow("SfM run:", self.run_combo)
        self.source_combo = QComboBox()
        self.source_combo.setToolTip(
            "Only applies to the two images + masks modes -- the poses mode always exports whatever "
            "the chosen SfM run covers, with no per-frame-set filter."
        )
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        form.addRow("Frame set:", self.source_combo)
        self.priors_checkbox = QCheckBox("Include camera priors CSV (from vine360's own SfM run)")
        self.priors_checkbox.toggled.connect(self._on_priors_toggled)
        form.addRow("Priors:", self.priors_checkbox)
        controls_layout.addLayout(form)

        self.mode_note = QLabel("")
        self.mode_note.setWordWrap(True)
        self.mode_note.setStyleSheet("color: palette(mid);")
        controls_layout.addWidget(self.mode_note)

        self.output_dir: Path | None = None
        # Whether the output folder is a manually-chosen pin (stays put
        # across mode/source changes) or a live suggested default (which
        # _update_output_dir keeps recomputing as mode/source change) --
        # see _on_choose_output_dir.
        self._output_dir_user_chosen = False
        output_row = QHBoxLayout()
        self.output_label = QLabel("No output folder chosen.")
        self.output_label.setWordWrap(True)
        choose_btn = QPushButton("Choose output folder…")
        choose_btn.clicked.connect(self._on_choose_output_dir)
        output_row.addWidget(self.output_label, stretch=1)
        output_row.addWidget(choose_btn)
        controls_layout.addLayout(output_row)

        export_btn_row = QHBoxLayout()
        self.export_btn = QPushButton("Export for Postshot")
        self.export_btn.clicked.connect(self._on_export)
        export_btn_row.addWidget(self.export_btn)
        self.queue_btn = QPushButton("Add to Queue")
        self.queue_btn.setToolTip("Adds this exact configuration as a queued job instead of running it now.")
        self.queue_btn.clicked.connect(self._on_add_to_queue)
        export_btn_row.addWidget(self.queue_btn)
        controls_layout.addLayout(export_btn_row)

        self.progress_area = ProgressArea()
        controls_layout.addWidget(self.progress_area)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        controls_layout.addWidget(self.result_label)

        layout.addWidget(self.controls)
        self.controls.setVisible(False)
        layout.addStretch()
        self._on_mode_changed()  # set initial note text

        if self.state.queue_manager is not None:
            self.state.queue_manager.queue_changed.connect(self._refresh_enabled)

    def on_project_changed(self) -> None:
        # A folder pinned via "Choose output folder…" belongs to the
        # previous project -- drop it so the new project's own exports/
        # default takes over again.
        self._output_dir_user_chosen = False
        self.output_dir = None
        self.output_label.setText("No output folder chosen.")
        self._exported = False
        self.on_shown()

    def on_shown(self) -> None:
        have_project = self.state.conn is not None
        self.no_project_label.setVisible(not have_project)
        self.controls.setVisible(have_project)
        if have_project:
            self._refresh_runs()
            self._refresh_sources()
            self._on_mode_changed()  # also updates output_dir as a side effect

    def _refresh_runs(self) -> None:
        self.run_combo.clear()
        rows = self.state.conn.execute(
            "SELECT run_id, created_at, model_stats FROM sfm_runs ORDER BY created_at DESC"
        ).fetchall()
        for run_id, created_at, model_stats in rows:
            stats = json.loads(model_stats)
            label = (
                f"{created_at} -- {stats['registered_images']}/{stats['total_images']} registered "
                f"({run_id[:12]})"
            )
            self.run_combo.addItem(label, userData=run_id)
        self._refresh_enabled()

    def _refresh_sources(self) -> None:
        """"All frame sets" (the default, matching this panel's original
        all-sources behavior) plus every frame set, whether or not it has
        projected views yet -- a not-yet-projected one is still selectable
        so it can be added to the queue, which auto-inserts the missing
        Projection job ahead of it; _refresh_enabled, not this method, is
        what actually gates the Export button on the selected frame set's
        view count."""
        current = self.source_combo.currentData()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItem("All frame sets", userData=None)
        self._choices = {c.frame_set_id: c for c in frame_set_choices(self.state)}
        for choice in self._choices.values():
            if choice.pending:
                detail = f"no frames yet ({choice.pending})"
            else:
                detail = f"{choice.view_count} views" if choice.view_count else "no views yet"
            self.source_combo.addItem(f"{choice.name} — {detail}", userData=choice.frame_set_id)
        index = self.source_combo.findData(current)
        self.source_combo.setCurrentIndex(index if index >= 0 else 0)
        self.source_combo.blockSignals(False)

    def _on_source_changed(self) -> None:
        self._exported = False  # a source change means the last export no longer matches this configuration
        self._update_output_dir()
        self._refresh_enabled()

    def _on_priors_toggled(self) -> None:
        self._exported = False  # a priors toggle means the last export no longer matches this configuration
        self._refresh_enabled()

    def _on_mode_changed(self) -> None:
        mode = self.mode_combo.currentData()
        poses_mode = mode == "poses"
        self.run_combo.setEnabled(poses_mode)
        self.source_combo.setEnabled(not poses_mode)
        self.export_btn.setText(_EXPORT_BUTTON_LABELS[mode])
        if poses_mode:
            self.mode_note.setText(
                "Imports vine360's own already-computed COLMAP reconstruction into Postshot. Always "
                "exports whatever the chosen SfM run covers -- the Frame set selector above doesn't "
                "filter this mode (see docs/adr/0026)."
            )
        elif mode == "realityscan":
            self.mode_note.setText(
                "No pose data is exported -- RealityScan runs its own pose estimation on these "
                "images instead. Only the six-face projection images are used, with masks written "
                "next to each image using RealityScan's own naming convention (not Postshot's). "
                "Unverified against a real RealityScan install -- see docs/adr/0028. If vine360's own "
                "SfM has already run, the \"Include camera priors\" option below can also write a "
                "position-only CSV to help RealityScan's alignment on repetitive/self-similar scenes "
                "(e.g. vineyard rows) -- see docs/adr/0033."
            )
        else:
            self.mode_note.setText(
                "No pose data is exported -- Postshot runs its own COLMAP-based pose estimation on "
                "these images instead. Only the six-face projection images are used (masks aren't "
                "built against raw equirectangular frames yet); no SfM run is required in vine360 "
                "for this mode."
            )
        self._exported = False  # a mode change means the last export no longer matches this configuration
        self._update_output_dir()
        self._refresh_enabled()

    def _update_output_dir(self) -> None:
        """Keeps output_dir following the mode/source selection as a live
        suggested default, unless the user has manually pinned a folder
        via "Choose output folder…" (see _on_choose_output_dir). A no-op
        before a project is open -- _on_mode_changed (which calls this)
        runs once unconditionally from __init__, before state.project_root
        is set."""
        if self._output_dir_user_chosen or self.state.project_root is None:
            return
        self.output_dir = self._default_output_dir()
        self.output_label.setText(str(self.output_dir))

    def _default_output_dir(self) -> Path:
        mode = self.mode_combo.currentData()
        return self.state.project_root / "exports" / self._capture_folder_name(mode) / _EXPORT_FORMAT_TAGS[mode]

    def _capture_folder_name(self, mode: str) -> str:
        """The <capture> half of exports/<capture>/<format>/ (docs/adr/
        0028). The two images-only modes are frame-set-filterable (ADR
        0026/0034): a chosen frame set gets its own folder, "All frame
        sets" gets "all". Poses mode has no filter of its own, but if the
        selected run was scoped to one frame set anyway, default to that
        frame set's folder too -- a nicer default path only, not a new
        filter; still "all" for a project-wide run or when no run is
        selected yet."""
        if mode == "poses":
            run_id = self.run_combo.currentData()
            if run_id and self.state.conn is not None:
                row = self.state.conn.execute(
                    "SELECT config FROM sfm_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if row:
                    config = json.loads(row[0])
                    frame_set_id = config.get("frame_set_id") or (
                        config.get("source_id") if config.get("image_source") == "frames" else None
                    )
                    if frame_set_id:
                        return self._frame_set_folder_name(frame_set_id)
            return "all"
        frame_set_id = self.source_combo.currentData()
        return self._frame_set_folder_name(frame_set_id) if frame_set_id else "all"

    def _frame_set_folder_name(self, frame_set_id: str) -> str:
        """A filesystem-safe folder name for a per-frame-set export default
        -- the source's own file stem where available (more useful than a
        bare id at a glance) plus the frame set's config tag (e.g.
        "clip_i0.5"), falling back to the id if sanitizing leaves nothing
        usable. A legacy frame set (id == source_id, docs/adr/0034) keeps
        the bare stem, so its exports land where they always did."""
        source_id, _, tag = frame_set_id.partition("~")
        row = self.state.conn.execute("SELECT path FROM sources WHERE source_id = ?", (source_id,)).fetchone()
        stem = Path(row[0]).stem if row else ""
        name = f"{stem}_{tag}" if (stem and tag) else stem
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
        return safe or frame_set_id[:8]

    def _refresh_enabled(self) -> None:
        busy = bool(self.state.queue_manager and self.state.queue_manager.is_running)
        mode = self.mode_combo.currentData()
        if mode == "poses":
            configured = self.run_combo.count() > 0 and self.output_dir is not None
            has_prereq = True
        else:
            choice = self._choices.get(self.source_combo.currentData())
            has_prereq = choice is None or choice.view_count > 0
            configured = self.output_dir is not None and has_prereq
        self.export_btn.setEnabled(configured and not busy)
        self.export_btn.setToolTip(
            ""
            if has_prereq
            else 'This frame set has no projected views yet -- use "Add to Queue" instead; it will queue '
            "projection automatically ahead of this export job."
        )
        self.queue_btn.setEnabled(self.output_dir is not None)

        if mode == "realityscan":
            priors_available = self.state.conn is not None and realityscan_priors_available(
                self.state.conn, frame_set_id=self.source_combo.currentData()
            )
            self.priors_checkbox.setEnabled(priors_available and not busy)
            if not priors_available and self.priors_checkbox.isChecked():
                self.priors_checkbox.setChecked(False)
            self.priors_checkbox.setToolTip(
                ""
                if priors_available
                else "No usable vine360 SfM run found for this frame set selection -- run Pose estimation first."
            )
        else:
            self.priors_checkbox.setEnabled(False)
            if self.priors_checkbox.isChecked():
                self.priors_checkbox.setChecked(False)

        self.stage_indicator.set_stages(
            [
                DONE if configured else ACTIVE,
                DONE if self._exported else (ACTIVE if configured else PENDING),
            ]
        )

    def _on_choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Choose a folder to export into", str(self._dialog_start_dir())
        )
        if directory:
            self.output_dir = Path(directory)
            self._output_dir_user_chosen = True  # pin it -- stop following mode/source changes
            self.output_label.setText(str(self.output_dir))
            self._exported = False  # a new output folder means the last export no longer matches it
            self._refresh_enabled()

    def _dialog_start_dir(self) -> Path:
        """The suggested exports/<capture>/<format>/ default usually doesn't
        exist yet (it's only created by the export itself), and Qt's folder
        dialog silently falls back to the process's working directory for a
        missing path -- so start from the nearest folder that does exist,
        which is at worst the project's own exports/."""
        candidate = self.output_dir or (self.state.project_root / "exports")
        while not candidate.is_dir() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate

    def _on_add_to_queue(self) -> None:
        if self.output_dir is None:
            QMessageBox.warning(self, "No output folder", "Choose an output folder first.")
            return
        mode = self.mode_combo.currentData()
        frame_set_id = self.source_combo.currentData() if mode != "poses" else None
        params = {"mode": mode, "output_dir": str(self.output_dir), "frame_set_id": frame_set_id}
        if mode == "realityscan":
            params["include_camera_priors"] = self.priors_checkbox.isChecked()
        if mode == "poses":
            params["run_id"] = self.run_combo.currentData()  # None -> resolved lazily at run time
            label = f"Export — poses + images + masks → {self.output_dir.name}"
        else:
            choice = self._choices.get(frame_set_id)
            source_text = choice.name if choice else "all frame sets"
            mode_text = "RealityScan images + masks" if mode == "realityscan" else "images + masks only"
            label = f"Export — {mode_text} ({source_text}) → {self.output_dir.name}"
        self.state.queue_manager.add_job(QUEUE_STAGE_EXPORT, None, params, label)

    def _on_export(self) -> None:
        self.export_btn.setEnabled(False)
        self.progress_area.start("Exporting…")
        mode = self.mode_combo.currentData()
        if mode == "poses":
            run_in_background(
                self,
                _export_postshot_worker,
                self.state.project_root,
                self.output_dir,
                self.run_combo.currentData(),
                on_success=self._on_export_success,
                on_error=self._on_export_error,
                on_progress=self.progress_area.update_progress,
            )
        elif mode == "realityscan":
            run_in_background(
                self,
                _export_realityscan_worker,
                self.state.project_root,
                self.output_dir,
                self.source_combo.currentData(),
                self.priors_checkbox.isChecked(),
                on_success=self._on_export_success,
                on_error=self._on_export_error,
                on_progress=self.progress_area.update_progress,
            )
        else:
            run_in_background(
                self,
                _export_frames_and_masks_worker,
                self.state.project_root,
                self.output_dir,
                self.source_combo.currentData(),
                on_success=self._on_export_success,
                on_error=self._on_export_error,
                on_progress=self.progress_area.update_progress,
            )

    def _on_export_success(self, result) -> None:
        self._exported = True
        self._refresh_enabled()
        self.progress_area.finish("Export complete.")
        text = f"Exported to {result.output_dir}\n{result.num_images} image(s), {result.num_masks} mask(s).\n"
        if result.sparse_dir is not None:
            text += (
                f"In Postshot: import {result.sparse_dir} as a COLMAP dataset with images from "
                f"{result.images_dir}"
            )
        else:
            text += f"In Postshot: start a new project from the images under {result.images_dir}"
        if result.masks_dir is not None:
            text += f", then drop the files under {result.masks_dir} into the Image Masks list."
        else:
            text += " (no masks were built)."
        if result.priors_path is not None:
            text += (
                f"\nCamera priors: {result.num_priors} position(s) written to {result.priors_path} -- "
                "in RealityScan: WORKFLOW tab → Import Metadata → Trajectory."
            )
        if result.warnings:
            text += "\n⚠ " + "; ".join(result.warnings)
        self.result_label.setText(text)
        self.state.notify_change()

    def _on_export_error(self, exc: Exception) -> None:
        self._refresh_enabled()
        self.progress_area.finish("")
        if isinstance(exc, PostshotExportError):
            QMessageBox.warning(self, "Export failed", str(exc))
        else:
            QMessageBox.critical(self, "Unexpected error exporting", f"{type(exc).__name__}: {exc}")

def _format_bytes(n: int | None) -> str:
    if n is None:
        return "…"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _disk_usage_worker(project_root: Path, generation: int) -> tuple[int, dict]:
    """Walks every frame set's and export folder's files for the Data
    manager's size column -- off the GUI thread, since a real project can
    hold tens of thousands of images on a slow (e.g. Windows-mounted)
    drive."""
    conn = open_index_db(project_root)
    try:
        frame_sets = {
            info.frame_set_id: frame_set_disk_usage(conn, project_root, info.frame_set_id)
            for info in list_frame_sets(conn)
        }
        sfm_runs = {run.run_id: directory_size(project_root / "sfm" / "sparse" / run.run_id) for run in list_sfm_runs(conn)}
    finally:
        conn.close()
    exports = {info.relative_path: export_dir_size(project_root, info) for info in list_export_dirs(project_root)}
    return generation, {"frame_sets": frame_sets, "sfm_runs": sfm_runs, "exports": exports}


def _data_manager_delete_worker(project_root: Path, action: str, target: str) -> str:
    conn = open_index_db(project_root)
    try:
        if action == "frame_set":
            delete_frame_set(conn, project_root, target)
        elif action == "views":
            delete_views_for_frame_set(conn, project_root, target)
        elif action == "masks":
            delete_masks_for_frame_set(conn, project_root, target)
        elif action == "sfm_run":
            delete_sfm_run(conn, project_root, target)
        elif action == "export":
            delete_export_dir(project_root, target)
        else:
            raise ValueError(f"unknown delete action {action!r}")
        return action
    finally:
        conn.close()


class DataManagerPanel(QWidget):
    """Lists a project's sources (read-only -- removing one stays on
    Import) and everything derived from them, and deletes derived data:
    a whole frame set (cascading to its projections and masks), just a
    frame set's projections (and their masks) or just its masks, an SfM
    run, or an export folder. Every delete goes through
    vine360.data_manager, which keeps the cascade-delete invariant
    (docs/adr/0017, 0035). Not a pipeline stage, so its sidebar dot is a
    fixed color (DATA_MANAGER_DOT_COLOR), not a status."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._usage: dict = {}
        self._usage_generation = 0
        layout = QVBoxLayout(self)
        _panel_header(
            "Data manager",
            "Everything this project holds, by source and frame set, with what it costs on disk. "
            "Deleting never touches original source media.",
            layout,
        )

        self.no_project_label = QLabel("Open a project first.")
        layout.addWidget(self.no_project_label)

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        top_row = QHBoxLayout()
        self.summary_label = QLabel("")
        top_row.addWidget(self.summary_label, stretch=1)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        top_row.addWidget(refresh_btn)
        controls_layout.addLayout(top_row)

        sources_box = QGroupBox("Sources (read-only -- add or remove them on Import)")
        sources_layout = QVBoxLayout(sources_box)
        self.sources_table = QTableWidget(0, 5)
        self.sources_table.setHorizontalHeaderLabels(["Name", "Type", "Projection", "Duration", "Frame sets"])
        self.sources_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sources_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.sources_table.verticalHeader().setVisible(False)
        self.sources_table.setMaximumHeight(130)
        sources_layout.addWidget(self.sources_table)
        controls_layout.addWidget(sources_box)

        sets_box = QGroupBox("Frame sets and their projections / masks")
        sets_layout = QVBoxLayout(sets_box)
        self.sets_tree = QTreeWidget()
        self.sets_tree.setHeaderLabels(["Source / frame set", "Frames", "Views", "Masks", "Frames on disk",
                                        "Projections on disk", "Masks on disk"])
        self.sets_tree.itemSelectionChanged.connect(self._refresh_enabled)
        sets_layout.addWidget(self.sets_tree)
        sets_btn_row = QHBoxLayout()
        self.delete_set_btn = QPushButton("Delete frame set…")
        self.delete_set_btn.setToolTip("Deletes the frames and everything built from them (projections, masks).")
        self.delete_set_btn.clicked.connect(lambda: self._delete_frame_set_part("frame_set"))
        self.delete_views_btn = QPushButton("Delete projections…")
        self.delete_views_btn.setToolTip("Deletes this frame set's projected views and their masks; keeps the frames.")
        self.delete_views_btn.clicked.connect(lambda: self._delete_frame_set_part("views"))
        self.delete_masks_btn = QPushButton("Delete masks…")
        self.delete_masks_btn.setToolTip("Deletes this frame set's masks only; keeps frames and projections.")
        self.delete_masks_btn.clicked.connect(lambda: self._delete_frame_set_part("masks"))
        for btn in (self.delete_set_btn, self.delete_views_btn, self.delete_masks_btn):
            sets_btn_row.addWidget(btn)
        sets_btn_row.addStretch()
        sets_layout.addLayout(sets_btn_row)
        controls_layout.addWidget(sets_box, stretch=2)

        runs_box = QGroupBox("Pose estimation (SfM) runs")
        runs_layout = QVBoxLayout(runs_box)
        self.runs_table = QTableWidget(0, 5)
        self.runs_table.setHorizontalHeaderLabels(["Created", "Engine", "Frame set", "Registered", "On disk"])
        self.runs_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.runs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.runs_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.runs_table.verticalHeader().setVisible(False)
        self.runs_table.itemSelectionChanged.connect(self._refresh_enabled)
        runs_layout.addWidget(self.runs_table)
        self.delete_run_btn = QPushButton("Delete run…")
        self.delete_run_btn.clicked.connect(self._delete_sfm_run)
        runs_btn_row = QHBoxLayout()
        runs_btn_row.addWidget(self.delete_run_btn)
        runs_btn_row.addStretch()
        runs_layout.addLayout(runs_btn_row)
        controls_layout.addWidget(runs_box, stretch=1)

        exports_box = QGroupBox("Exports")
        exports_layout = QVBoxLayout(exports_box)
        self.exports_table = QTableWidget(0, 2)
        self.exports_table.setHorizontalHeaderLabels(["Folder", "On disk"])
        self.exports_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.exports_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.exports_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.exports_table.verticalHeader().setVisible(False)
        self.exports_table.itemSelectionChanged.connect(self._refresh_enabled)
        exports_layout.addWidget(self.exports_table)
        self.delete_export_btn = QPushButton("Delete export folder…")
        self.delete_export_btn.clicked.connect(self._delete_export)
        exports_btn_row = QHBoxLayout()
        exports_btn_row.addWidget(self.delete_export_btn)
        exports_btn_row.addStretch()
        exports_layout.addLayout(exports_btn_row)
        controls_layout.addWidget(exports_box, stretch=1)

        self.progress_area = ProgressArea()
        controls_layout.addWidget(self.progress_area)

        layout.addWidget(self.controls, stretch=1)
        self.controls.setVisible(False)
        self._busy = False
        self._refresh_enabled()

        if self.state.queue_manager is not None:
            self.state.queue_manager.queue_changed.connect(self._refresh_enabled)

    def on_project_changed(self) -> None:
        self.on_shown()

    def on_shown(self) -> None:
        have_project = self.state.conn is not None
        self.no_project_label.setVisible(not have_project)
        self.controls.setVisible(have_project)
        if have_project:
            self.refresh()

    # -- listing ----------------------------------------------------------

    def refresh(self) -> None:
        if self.state.conn is None:
            return
        conn = self.state.conn
        sources = list_sources(conn)
        self.sources_table.setRowCount(len(sources))
        for r, src in enumerate(sources):
            duration = f"{src.duration_seconds:.1f}s" if src.duration_seconds else "—"
            for c, text in enumerate(
                [src.name, src.media_type, src.projection, duration, str(src.frame_set_count)]
            ):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setToolTip(src.path)
                self.sources_table.setItem(r, c, item)
        self.sources_table.resizeColumnsToContents()

        selected = self._selected_frame_set_id()
        self.sets_tree.clear()
        frame_sets = list_frame_sets(conn)
        by_source: dict[str, QTreeWidgetItem] = {}
        for src in sources:
            if src.frame_set_count:
                parent = QTreeWidgetItem([src.name])
                parent.setToolTip(0, src.path)
                self.sets_tree.addTopLevelItem(parent)
                by_source[src.source_id] = parent
        for info in frame_sets:
            parent = by_source.get(info.source_id)
            if parent is None:
                continue
            item = QTreeWidgetItem(
                [info.label, str(info.frame_count), str(info.view_count), str(info.mask_count), "", "", ""]
            )
            item.setData(0, Qt.UserRole, info.frame_set_id)
            item.setData(1, Qt.UserRole, (info.frame_count, info.view_count, info.mask_count))
            item.setToolTip(0, f"{info.frame_set_id} (created {info.created_at})")
            parent.addChild(item)
            if info.frame_set_id == selected:
                item.setSelected(True)
        self.sets_tree.expandAll()
        for c in range(self.sets_tree.columnCount()):
            self.sets_tree.resizeColumnToContents(c)

        runs = list_sfm_runs(conn)
        set_names = {info.frame_set_id: info.display_name for info in frame_sets}
        self.runs_table.setRowCount(len(runs))
        for r, run in enumerate(runs):
            engine = "Equirectangular (raw frames)" if run.image_source == "frames" else "Six-face projections"
            scope = set_names.get(run.frame_set_id, run.frame_set_id or "all frame sets")
            cells = [run.created_at, engine, scope, f"{run.registered_images}/{run.total_images}", ""]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setData(Qt.UserRole, run.run_id)
                    item.setToolTip(run.run_id)
                self.runs_table.setItem(r, c, item)
        self.runs_table.resizeColumnsToContents()

        exports = list_export_dirs(self.state.project_root)
        self.exports_table.setRowCount(len(exports))
        for r, info in enumerate(exports):
            if info.in_exports_root:
                text = "exports  (written directly into exports/)"
            else:
                text = info.relative_path + ("  (older flat layout)" if info.legacy_layout else "")
            item = QTableWidgetItem(text)
            item.setData(Qt.UserRole, info.relative_path)
            self.exports_table.setItem(r, 0, item)
            self.exports_table.setItem(r, 1, QTableWidgetItem(""))
        self.exports_table.resizeColumnsToContents()

        self.summary_label.setText(
            f"{len(sources)} source(s), {len(frame_sets)} frame set(s), {len(runs)} SfM run(s), "
            f"{len(exports)} export folder(s)."
        )
        self._apply_usage()
        self._start_usage_scan()
        self._refresh_enabled()

    def _start_usage_scan(self) -> None:
        # Bound-method callbacks, not lambdas: a lambda has no QObject
        # thread affinity, so Qt would run it on the worker thread (see
        # QueueManager._start_job). The generation number rides along in
        # the result instead of a closure.
        self._usage_generation += 1
        run_in_background(
            self,
            _disk_usage_worker,
            self.state.project_root,
            self._usage_generation,
            on_success=self._on_usage,
            on_error=self._on_usage_error,
        )

    def _on_usage_error(self, _exc: Exception) -> None:
        pass  # sizes are informational; the lists themselves are already correct

    def _on_usage(self, result: tuple[int, dict]) -> None:
        generation, usage = result
        if generation != self._usage_generation:
            return  # a newer refresh superseded this scan
        self._usage = usage
        self._apply_usage()

    def _apply_usage(self) -> None:
        frame_sets = self._usage.get("frame_sets", {})
        for i in range(self.sets_tree.topLevelItemCount()):
            parent = self.sets_tree.topLevelItem(i)
            for k in range(parent.childCount()):
                item = parent.child(k)
                sizes = frame_sets.get(item.data(0, Qt.UserRole))
                counts = item.data(1, Qt.UserRole)
                for col, key, count in zip((4, 5, 6), ("frames", "projections", "masks"), counts):
                    size = sizes[key] if sizes else None
                    if size == 0 and count > 0:
                        # Rows in the index but nothing on disk where they
                        # belong -- say so rather than show a plausible "0 B".
                        item.setText(col, "files missing")
                        item.setToolTip(col, f"{count} {key} recorded, but no files found in the project's {key}/ folder.")
                    else:
                        item.setText(col, _format_bytes(size))
                        item.setToolTip(col, "")
        runs = self._usage.get("sfm_runs", {})
        for r in range(self.runs_table.rowCount()):
            run_id = self.runs_table.item(r, 0).data(Qt.UserRole)
            self.runs_table.item(r, 4).setText(_format_bytes(runs.get(run_id)))
        exports = self._usage.get("exports", {})
        for r in range(self.exports_table.rowCount()):
            rel = self.exports_table.item(r, 0).data(Qt.UserRole)
            self.exports_table.item(r, 1).setText(_format_bytes(exports.get(rel)))

    # -- selection / enablement -------------------------------------------

    def _selected_frame_set_item(self) -> QTreeWidgetItem | None:
        items = self.sets_tree.selectedItems()
        if not items or items[0].data(0, Qt.UserRole) is None:
            return None  # nothing, or a source row
        return items[0]

    def _selected_frame_set_id(self) -> str | None:
        item = self._selected_frame_set_item()
        return item.data(0, Qt.UserRole) if item else None

    @staticmethod
    def _selected_row_data(table: QTableWidget) -> str | None:
        rows = table.selectionModel().selectedRows() if table.selectionModel() else []
        if not rows:
            return None
        item = table.item(rows[0].row(), 0)
        return item.data(Qt.UserRole) if item else None

    def _refresh_enabled(self) -> None:
        queue_busy = bool(self.state.queue_manager and self.state.queue_manager.is_running)
        blocked = queue_busy or self._busy
        item = self._selected_frame_set_item()
        counts = item.data(1, Qt.UserRole) if item else (0, 0, 0)
        self.delete_set_btn.setEnabled(item is not None and not blocked)
        self.delete_views_btn.setEnabled(item is not None and counts[1] > 0 and not blocked)
        self.delete_masks_btn.setEnabled(item is not None and counts[2] > 0 and not blocked)
        self.delete_run_btn.setEnabled(self._selected_row_data(self.runs_table) is not None and not blocked)
        self.delete_export_btn.setEnabled(self._selected_row_data(self.exports_table) is not None and not blocked)

    # -- deleting ---------------------------------------------------------

    def _confirm(self, title: str, text: str) -> bool:
        return QMessageBox.question(self, title, text) == QMessageBox.Yes

    def _queued_jobs_touching(self, frame_set_id: str) -> int:
        queue = self.state.queue_manager
        if queue is None:
            return 0
        return sum(
            1
            for job in queue.jobs
            if job.status in ("queued", "blocked")
            and (job.target_frame_set_id == frame_set_id or job.params.get("frame_set_id") == frame_set_id)
        )

    def _delete_frame_set_part(self, action: str) -> None:
        item = self._selected_frame_set_item()
        if item is None:
            return
        frame_set_id = item.data(0, Qt.UserRole)
        frames, views, masks = item.data(1, Qt.UserRole)
        name = f"{item.parent().text(0)} — {item.text(0)}"
        what = {
            "frame_set": f"the whole frame set: {frames} frames, {views} projected views and {masks} masks",
            "views": f"{views} projected views and {masks} masks (the {frames} frames are kept)",
            "masks": f"{masks} masks (frames and projected views are kept)",
        }[action]
        text = f"Delete {what} from\n{name}?\n\nFiles are removed from disk. This can't be undone."
        if self._queued_jobs_touching(frame_set_id):
            text += "\n\nQueued jobs still target this frame set -- they will fail or re-create it when run."
        if self._confirm("Delete from frame set", text):
            self._run_delete(action, frame_set_id)

    def _delete_sfm_run(self) -> None:
        run_id = self._selected_row_data(self.runs_table)
        if run_id and self._confirm(
            "Delete SfM run",
            f"Delete SfM run {run_id} and its sfm/sparse/{run_id}/ folder?\n\n"
            "Exports already made from it are kept. This can't be undone.",
        ):
            self._run_delete("sfm_run", run_id)

    def _delete_export(self) -> None:
        rel = self._selected_row_data(self.exports_table)
        if rel == "exports":
            question = ("Delete the export written directly into exports/ (its images/, masks/, sparse/ and "
                        "CameraPriors.csv)?\n\nOther export folders are kept. This can't be undone.")
        else:
            question = f"Delete {rel}/ and everything in it?\n\nThis can't be undone."
        if rel and self._confirm("Delete export folder", question):
            self._run_delete("export", rel)

    def _run_delete(self, action: str, target: str) -> None:
        self._busy = True
        self._refresh_enabled()
        self.progress_area.start("Deleting…")
        run_in_background(
            self,
            _data_manager_delete_worker,
            self.state.project_root,
            action,
            target,
            on_success=self._on_delete_success,
            on_error=self._on_delete_error,
        )

    def _on_delete_success(self, _action: str) -> None:
        self._busy = False
        self.progress_area.finish("Deleted.")
        self.refresh()
        self.state.notify_change()

    def _on_delete_error(self, exc: Exception) -> None:
        self._busy = False
        self.progress_area.finish("")
        self.refresh()
        if isinstance(exc, DataManagerError):
            QMessageBox.warning(self, "Could not delete", str(exc))
        else:
            QMessageBox.critical(self, "Unexpected error deleting", f"{type(exc).__name__}: {exc}")


class Vine360MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vineyard 360 3DGS")
        self.resize(920, 660)
        self.state = AppState()
        self.queue_manager = QueueManager(self.state)
        self.state.queue_manager = self.queue_manager

        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.stack = QStackedWidget()

        project_panel = ProjectPanel(self.state)
        import_panel = ImportPanel(self.state)
        frames_panel = FramesPanel(self.state)
        projection_panel = ProjectionPanel(self.state)
        masks_panel = MasksPanel(self.state)
        pose_panel = PoseEstimationPanel(self.state)
        export_panel = ExportPanel(self.state)
        data_manager_panel = DataManagerPanel(self.state)

        project_panel.project_changed.connect(import_panel.on_project_changed)
        project_panel.project_changed.connect(frames_panel.on_project_changed)
        project_panel.project_changed.connect(projection_panel.on_project_changed)
        project_panel.project_changed.connect(masks_panel.on_project_changed)
        project_panel.project_changed.connect(pose_panel.on_project_changed)
        project_panel.project_changed.connect(export_panel.on_project_changed)
        project_panel.project_changed.connect(data_manager_panel.on_project_changed)
        project_panel.project_changed.connect(self._on_project_changed_for_queue)

        self._panels = [
            project_panel,
            import_panel,
            frames_panel,
            projection_panel,
            masks_panel,
            pose_panel,
            export_panel,
            data_manager_panel,
        ]
        labels = [
            "Project",
            "Import",
            "Frames",
            "Projection",
            "Masks",
            "Pose estimation",
            "Export",
            "Data manager",
        ]
        self._sidebar_items: list[QListWidgetItem] = []
        for index, (label, widget) in enumerate(zip(labels, self._panels)):
            item = QListWidgetItem(f"  {label}")
            if index <= STAGE_EXPORT:  # pipeline stages get a status dot
                item.setIcon(QIcon(_status_dot(PENDING)))
            else:  # Data manager isn't a stage: a fixed-color dot instead
                item.setIcon(QIcon(_dot_pixmap(DATA_MANAGER_DOT_COLOR)))
            self._sidebar_items.append(item)
            self.sidebar.addItem(item)
            self.stack.addWidget(widget)

        self.sidebar.currentRowChanged.connect(self._on_sidebar_row_changed)
        self.sidebar.setCurrentRow(0)

        splitter = QSplitter()
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.queue_panel = QueuePanel(self.state)
        self.queue_panel.bind_manager(self.queue_manager)
        queue_dock = QDockWidget("Queue", self)
        queue_dock.setWidget(self.queue_panel)
        queue_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, queue_dock)

        self.state.notify_change = self.refresh_stage_statuses
        self.refresh_stage_statuses()

    def _on_project_changed_for_queue(self) -> None:
        self.queue_manager.load()

    def refresh_stage_statuses(self) -> None:
        self._sidebar_items[STAGE_PROJECT].setIcon(
            QIcon(_status_dot(DONE if self.state.conn is not None else ACTIVE))
        )
        statuses = compute_stage_statuses(self.state.conn)
        for index, status in statuses.items():
            self._sidebar_items[index].setIcon(QIcon(_status_dot(status)))

    def _on_sidebar_row_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        widget = self.stack.widget(index)
        on_shown = getattr(widget, "on_shown", None)
        if callable(on_shown):
            on_shown()


def main() -> int:
    app = QApplication(sys.argv)
    window = Vine360MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
