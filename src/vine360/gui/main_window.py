"""The real M6 desktop app -- built on the dashboard shape the user chose
after comparing wizard_preview.py and dashboard_preview.py (see
docs/status.md and memory).

Project creation, source ingest, frame extraction, projection generation,
mask building and SfM are all wired to real vine360 code, run on a
background QThread with a determinate progress bar + ETA wherever the
underlying function reports (current, total) counts. Training stays
adapter-only (no backend installed, no run) -- an explicit standing
decision, not an oversight; see docs/adr/0009.

Sidebar stage status (the colored dot next to each stage name) is derived
directly from the project's index database each time it changes, not
tracked ad hoc per panel -- see `compute_stage_statuses`.

Run with: python -m vine360.gui.main_window
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vine360.config import FRAME_PRESET_INTERVALS, CaptureMode, FramePreset
from vine360.ingest.frames import FrameExtractionError, extract_frames
from vine360.ingest.media_probe import ProbeError
from vine360.ingest.sources import (
    EquirectangularConfirmationRequired,
    SourceError,
    add_source,
    remove_source,
)
from vine360.masking.build import MaskBuildError, build_masks_for_source
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
from vine360.projection.generate import ProjectionGenerationError, generate_views_for_source
from vine360.runners.local import LocalRunner
from vine360.runners.probe import probe_dependencies
from vine360.sfm.project_run import SfmRegistrationError, run_sfm_for_project

# Real sample capture location (see memory: project_capture_equipment_and_sample_data) --
# used only as a file-dialog starting point, nothing here reads or processes it.
_SAMPLE_DATA_HINT_DIR = Path("/mnt/e/11-9-26_EstoWines_Capture1/Equi")

CAPTURE_GROUP_PRESETS = ["Insta360 (ground)", "Antigravity A1 (aerial)", "Other / custom…"]

DONE, ACTIVE, PENDING = "done", "active", "pending"
STATUS_COLOR = {DONE: QColor(70, 150, 90), ACTIVE: QColor(200, 160, 50), PENDING: QColor(120, 120, 120)}

# Sidebar row indices -- used both to build the QStackedWidget and to
# address which dot compute_stage_statuses' result applies to.
STAGE_PROJECT = 0
STAGE_IMPORT = 1
STAGE_FRAMES = 2
STAGE_PROJECTION = 3
STAGE_MASKS = 4
STAGE_POSE = 5
STAGE_TRAINING_PRESET = 6
STAGE_TRAINING_MONITOR = 7
STAGE_EXPORT = 8


@dataclass
class AppState:
    project_root: Path | None = None
    project: Project | None = None
    conn: sqlite3.Connection | None = None
    # Set once by Vine360MainWindow after every panel is constructed, so any
    # panel can trigger a sidebar status refresh after it changes project
    # state, without each panel needing to know sidebar indices itself.
    notify_change: Callable[[], None] = field(default=lambda: None)


def compute_stage_statuses(conn: sqlite3.Connection | None) -> dict[int, str]:
    """Derives every stage's status directly from the database -- the
    single source of truth, rather than tracking status ad hoc as panels
    act. A stage is DONE once it has produced at least one real artifact,
    ACTIVE once its prerequisite exists but it hasn't yet, else PENDING."""
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
    return {
        STAGE_IMPORT: DONE if n_sources > 0 else ACTIVE,
        STAGE_FRAMES: DONE if n_frames > 0 else (ACTIVE if n_video_sources > 0 else PENDING),
        STAGE_PROJECTION: DONE if n_views > 0 else (ACTIVE if n_frames > 0 else PENDING),
        STAGE_MASKS: DONE if n_masks > 0 else (ACTIVE if n_views > 0 else PENDING),
        STAGE_POSE: DONE if n_sfm_runs > 0 else (ACTIVE if n_masks > 0 else PENDING),
    }


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


def _status_dot(status: str) -> QPixmap:
    pixmap = QPixmap(12, 12)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(STATUS_COLOR[status])
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(1, 1, 10, 10)
    painter.end()
    return pixmap


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


def _extract_frames_worker(project_root: Path, source_id: str, interval_seconds: float, progress_callback=None):
    conn = open_index_db(project_root)
    try:
        return extract_frames(
            conn,
            project_root,
            source_id,
            LocalRunner(),
            interval_seconds=interval_seconds,
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
        layout = QVBoxLayout(self)
        _panel_header("Frames", "Extract deterministic frames from a registered video source.", layout)

        self.no_project_label = QLabel("Create or open a project first, then add a video source on Import.")
        layout.addWidget(self.no_project_label)

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self.source_combo = QComboBox()
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
        form.addRow("Custom interval:", self.custom_interval_spin)
        controls_layout.addLayout(form)

        intervals_note = QLabel(
            "Preset intervals (vine360.config.FRAME_PRESET_INTERVALS): "
            + ", ".join(f"{p.value}={FRAME_PRESET_INTERVALS[p]}s" for p in FRAME_PRESET_INTERVALS)
            + ". Re-extracting a source replaces its previous frames."
        )
        intervals_note.setWordWrap(True)
        intervals_note.setStyleSheet("color: palette(mid);")
        controls_layout.addWidget(intervals_note)

        self.extract_btn = QPushButton("Extract Frames")
        self.extract_btn.clicked.connect(self._on_extract)
        controls_layout.addWidget(self.extract_btn)

        self.progress_area = ProgressArea()
        controls_layout.addWidget(self.progress_area)

        layout.addWidget(self.controls)
        self.controls.setVisible(False)
        layout.addStretch()

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
            "SELECT s.source_id, s.path, "
            "(SELECT COUNT(*) FROM frames WHERE frames.source_id = s.source_id) AS frame_count "
            "FROM sources s WHERE s.media_type = 'video' ORDER BY s.rowid"
        ).fetchall()
        for source_id, path, frame_count in rows:
            existing = f"{frame_count} frames already extracted" if frame_count else "no frames yet"
            self.source_combo.addItem(f"{Path(path).name} ({source_id[:8]}) — {existing}", userData=source_id)
        has_sources = self.source_combo.count() > 0
        self.extract_btn.setEnabled(has_sources)
        self.progress_area.label.setText("" if has_sources else "No video sources registered yet -- add one on Import.")

    def _on_preset_changed(self, text: str) -> None:
        self.custom_interval_spin.setEnabled(FramePreset(text) == FramePreset.CUSTOM)

    def _on_extract(self) -> None:
        source_id = self.source_combo.currentData()
        if source_id is None:
            return
        preset = FramePreset(self.preset_combo.currentText())
        if preset == FramePreset.CUSTOM:
            interval = self.custom_interval_spin.value()
        else:
            interval = FRAME_PRESET_INTERVALS[preset]

        self.extract_btn.setEnabled(False)
        self.progress_area.start(f"Extracting frames every {interval}s…")
        run_in_background(
            self,
            _extract_frames_worker,
            self.state.project_root,
            source_id,
            interval,
            on_success=self._on_extract_success,
            on_error=self._on_extract_error,
            on_progress=self.progress_area.update_progress,
        )

    def _on_extract_success(self, frames) -> None:
        self.extract_btn.setEnabled(True)
        self.progress_area.finish(f"Extracted {len(frames)} frames.")
        self.state.notify_change()

    def _on_extract_error(self, exc: Exception) -> None:
        self.extract_btn.setEnabled(True)
        self.progress_area.finish("")
        if isinstance(exc, FrameExtractionError):
            QMessageBox.warning(self, "Frame extraction failed", str(exc))
        else:
            QMessageBox.critical(self, "Unexpected error extracting frames", f"{type(exc).__name__}: {exc}")


def _generate_views_worker(
    project_root: Path, source_id: str, face_size: int, fov_degrees: float, face_names: list[str], progress_callback=None
):
    conn = open_index_db(project_root)
    try:
        return generate_views_for_source(
            conn,
            project_root,
            source_id,
            face_size=face_size,
            fov_degrees=fov_degrees,
            face_names=face_names,
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


class ProjectionPanel(QWidget):
    """Real projection: vine360.projection.generate.generate_views_for_source,
    run off the GUI thread. Previously this stage never left PENDING in the
    sidebar because it had no execution at all -- it does now."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
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
        self.source_combo.currentIndexChanged.connect(self._refresh_frame_combo)
        form.addRow("Source (its extracted frames):", self.source_combo)
        self.frame_combo = QComboBox()
        self.frame_combo.currentIndexChanged.connect(self._refresh_preview)
        form.addRow("Preview frame:", self.frame_combo)
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

        self.generate_btn = QPushButton("Generate Projections")
        self.generate_btn.clicked.connect(self._on_generate)
        controls_layout.addWidget(self.generate_btn)

        self.progress_area = ProgressArea()
        controls_layout.addWidget(self.progress_area)

        controls_layout.addWidget(QLabel("Preview:"))
        self.preview = PreviewGallery("No views generated yet for this frame.")
        controls_layout.addWidget(self.preview)

        layout.addWidget(self.controls)
        self.controls.setVisible(False)
        layout.addStretch()

    def on_project_changed(self) -> None:
        self.on_shown()

    def on_shown(self) -> None:
        have_project = self.state.conn is not None
        self.no_project_label.setVisible(not have_project)
        self.controls.setVisible(have_project)
        if have_project:
            self.refresh_sources()

    def refresh_sources(self) -> None:
        self.source_combo.clear()
        if self.state.conn is None:
            return
        rows = self.state.conn.execute(
            "SELECT s.source_id, s.path, "
            "COUNT(DISTINCT f.frame_id) AS frame_count, COUNT(v.view_id) AS view_count "
            "FROM sources s JOIN frames f ON f.source_id = s.source_id "
            "LEFT JOIN views v ON v.frame_id = f.frame_id "
            "GROUP BY s.source_id ORDER BY s.rowid"
        ).fetchall()
        for source_id, path, frame_count, view_count in rows:
            existing = f"{view_count} views already generated" if view_count else "no views yet"
            self.source_combo.addItem(
                f"{Path(path).name} ({source_id[:8]}) — {frame_count} frames, {existing}", userData=source_id
            )
        has_sources = self.source_combo.count() > 0
        self.generate_btn.setEnabled(has_sources)
        self.progress_area.label.setText("" if has_sources else "No extracted frames yet -- extract frames first.")
        self._refresh_frame_combo()

    def _refresh_frame_combo(self) -> None:
        self.frame_combo.blockSignals(True)
        self.frame_combo.clear()
        source_id = self.source_combo.currentData()
        if self.state.conn is not None and source_id is not None:
            rows = self.state.conn.execute(
                "SELECT f.frame_id, f.source_time, COUNT(v.view_id) AS view_count "
                "FROM frames f LEFT JOIN views v ON v.frame_id = f.frame_id "
                "WHERE f.source_id = ? GROUP BY f.frame_id ORDER BY f.source_time",
                (source_id,),
            ).fetchall()
            for frame_id, source_time, view_count in rows:
                label = f"t={source_time:.2f}s" + (f" ({view_count} views)" if view_count else " (no views)")
                self.frame_combo.addItem(label, userData=frame_id)
        self.frame_combo.blockSignals(False)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        frame_id = self.frame_combo.currentData()
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

    def _on_generate(self) -> None:
        source_id = self.source_combo.currentData()
        if source_id is None:
            return
        face_names = self._selected_face_names()
        if not face_names:
            QMessageBox.warning(self, "No views selected", "Check at least one face to generate.")
            return
        self.generate_btn.setEnabled(False)
        self.progress_area.start("Generating projections…")
        run_in_background(
            self,
            _generate_views_worker,
            self.state.project_root,
            source_id,
            self.face_size_spin.value(),
            self.fov_spin.value(),
            face_names,
            on_success=self._on_generate_success,
            on_error=self._on_generate_error,
            on_progress=self.progress_area.update_progress,
        )

    def _on_generate_success(self, views) -> None:
        self.generate_btn.setEnabled(True)
        self.progress_area.finish(f"Generated {len(views)} views.")
        self.refresh_sources()
        self.state.notify_change()

    def _on_generate_error(self, exc: Exception) -> None:
        self.generate_btn.setEnabled(True)
        self.progress_area.finish("")
        if isinstance(exc, ProjectionGenerationError):
            QMessageBox.warning(self, "Projection failed", str(exc))
        else:
            QMessageBox.critical(self, "Unexpected error generating projections", f"{type(exc).__name__}: {exc}")


def _build_masks_worker(
    project_root: Path, source_id: str, use_sam3_person: bool, use_sam3_sky: bool, progress_callback=None
):
    conn = open_index_db(project_root)
    try:
        return build_masks_for_source(
            conn,
            project_root,
            source_id,
            use_sam3_person=use_sam3_person,
            use_sam3_sky=use_sam3_sky,
            progress_callback=progress_callback,
        )
    finally:
        conn.close()


class MasksPanel(QWidget):
    """Real masking: vine360.masking.build.build_masks_for_source, run off
    the GUI thread. SAM 3, if requested, loads once for the whole batch
    (not once per view)."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._sam3_checked = False  # deferred to first on_shown(): importing
        # torch/transformers to check availability is expensive (roughly
        # 140MB -> 700MB+ RSS observed) and shouldn't cost anything at app
        # startup for users who never open this panel.
        layout = QVBoxLayout(self)
        _panel_header("Masks", "Segment person/sky and build the keep-mask for each view.", layout)

        self.no_project_label = QLabel("Generate projections first (see the Projection stage).")
        layout.addWidget(self.no_project_label)

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._refresh_view_combo)
        form.addRow("Source (its generated views):", self.source_combo)
        self.view_combo = QComboBox()
        self.view_combo.currentIndexChanged.connect(self._refresh_preview)
        form.addRow("Preview view:", self.view_combo)
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

        self.build_btn = QPushButton("Build Masks")
        self.build_btn.clicked.connect(self._on_build)
        controls_layout.addWidget(self.build_btn)

        self.progress_area = ProgressArea()
        controls_layout.addWidget(self.progress_area)

        self.flags_label = QLabel("")
        self.flags_label.setWordWrap(True)
        controls_layout.addWidget(self.flags_label)

        controls_layout.addWidget(QLabel("Preview (original view, then its keep-mask if built):"))
        self.preview = PreviewGallery("No views generated yet.")
        controls_layout.addWidget(self.preview)

        layout.addWidget(self.controls)
        self.controls.setVisible(False)
        layout.addStretch()

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
        self.source_combo.clear()
        if self.state.conn is None:
            return
        rows = self.state.conn.execute(
            "SELECT s.source_id, s.path, COUNT(DISTINCT v.view_id) AS view_count, COUNT(m.view_id) AS mask_count "
            "FROM sources s JOIN frames f ON f.source_id = s.source_id "
            "JOIN views v ON v.frame_id = f.frame_id LEFT JOIN masks m ON m.view_id = v.view_id "
            "GROUP BY s.source_id ORDER BY s.rowid"
        ).fetchall()
        for source_id, path, view_count, mask_count in rows:
            existing = f"{mask_count} masked already" if mask_count else "not masked yet"
            self.source_combo.addItem(
                f"{Path(path).name} ({source_id[:8]}) — {view_count} views, {existing}", userData=source_id
            )
        has_sources = self.source_combo.count() > 0
        self.build_btn.setEnabled(has_sources)
        self.progress_area.label.setText("" if has_sources else "No projected views yet -- generate projections first.")
        self._refresh_view_combo()

    def _refresh_view_combo(self) -> None:
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        source_id = self.source_combo.currentData()
        if self.state.conn is not None and source_id is not None:
            rows = self.state.conn.execute(
                "SELECT v.view_id, m.keep_fraction FROM views v "
                "JOIN frames f ON f.frame_id = v.frame_id LEFT JOIN masks m ON m.view_id = v.view_id "
                "WHERE f.source_id = ? ORDER BY v.view_id",
                (source_id,),
            ).fetchall()
            for view_id, keep_fraction in rows:
                label = view_id + (f" (keep {keep_fraction:.0%})" if keep_fraction is not None else " (unmasked)")
                self.view_combo.addItem(label, userData=view_id)
        self.view_combo.blockSignals(False)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        view_id = self.view_combo.currentData()
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

    def _on_build(self) -> None:
        source_id = self.source_combo.currentData()
        if source_id is None:
            return
        self.build_btn.setEnabled(False)
        self.flags_label.setText("")
        self.progress_area.start("Building masks…")
        run_in_background(
            self,
            _build_masks_worker,
            self.state.project_root,
            source_id,
            self.sam3_person_check.isChecked(),
            self.sam3_sky_check.isChecked(),
            on_success=self._on_build_success,
            on_error=self._on_build_error,
            on_progress=self.progress_area.update_progress,
        )

    def _on_build_success(self, masks) -> None:
        self.build_btn.setEnabled(True)
        self.progress_area.finish(f"Masked {len(masks)} views.")
        flagged = [
            (m.view_id, is_keep_fraction_anomalous(m.keep_fraction))
            for m in masks
            if m.keep_fraction is not None and is_keep_fraction_anomalous(m.keep_fraction)
        ]
        if flagged:
            lines = "\n".join(f"  {view_id}: {reason}" for view_id, reason in flagged[:8])
            more = f"\n  … and {len(flagged) - 8} more" if len(flagged) > 8 else ""
            self.flags_label.setText(f"⚠ {len(flagged)} view(s) flagged for review:\n{lines}{more}")
        self.refresh_sources()
        self.state.notify_change()

    def _on_build_error(self, exc: Exception) -> None:
        self.build_btn.setEnabled(True)
        self.progress_area.finish("")
        if isinstance(exc, MaskBuildError):
            QMessageBox.warning(self, "Masking failed", str(exc))
        else:
            QMessageBox.critical(self, "Unexpected error building masks", f"{type(exc).__name__}: {exc}")


def _run_sfm_worker(project_root: Path, progress_callback=None):
    conn = open_index_db(project_root)
    try:
        return run_sfm_for_project(conn, project_root, progress_callback=progress_callback)
    finally:
        conn.close()


class PoseEstimationPanel(QWidget):
    """Real SfM: vine360.sfm.project_run.run_sfm_for_project (pycolmap),
    run off the GUI thread. Operates on the whole project's projections/
    masks at once -- COLMAP needs the full multi-frame view set to have
    any real camera-position parallax to reconstruct from; a single
    frame's projected faces share one optical center and cannot be
    3D-reconstructed regardless of match quality (see docs/adr/0007)."""

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        layout = QVBoxLayout(self)
        _panel_header("Pose estimation", "Run COLMAP feature extraction, matching and mapping.", layout)

        self.no_project_label = QLabel("Generate projections first (see the Projection stage).")
        layout.addWidget(self.no_project_label)

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        controls_layout.addWidget(QLabel(
            "Runs against every projected view in the project (project/projections/), "
            "using masks from project/masks/keep/ if any have been built."
        ))

        self.run_btn = QPushButton("Run SfM")
        self.run_btn.clicked.connect(self._on_run)
        controls_layout.addWidget(self.run_btn)

        self.progress_area = ProgressArea()
        controls_layout.addWidget(self.progress_area)

        self.stats_label = QLabel("No SfM run yet.")
        self.stats_label.setWordWrap(True)
        controls_layout.addWidget(self.stats_label)

        layout.addWidget(self.controls)
        self.controls.setVisible(False)
        layout.addStretch()

    def on_project_changed(self) -> None:
        self.on_shown()

    def on_shown(self) -> None:
        have_project = self.state.conn is not None
        self.no_project_label.setVisible(not have_project)
        self.controls.setVisible(have_project)
        if have_project:
            self._refresh_enabled()
            self._show_latest_run()

    def _refresh_enabled(self) -> None:
        n_views = self.state.conn.execute("SELECT COUNT(*) FROM views").fetchone()[0]
        self.run_btn.setEnabled(n_views > 0)

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

    def _on_run(self) -> None:
        self.run_btn.setEnabled(False)
        self.progress_area.start("Running SfM…")
        run_in_background(
            self,
            _run_sfm_worker,
            self.state.project_root,
            on_success=self._on_run_success,
            on_error=self._on_run_error,
            on_progress=self.progress_area.update_progress,
        )

    def _on_run_success(self, result) -> None:
        diagnostics, warnings = result
        self.run_btn.setEnabled(True)
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
        self.run_btn.setEnabled(True)
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


def build_training_preset_panel() -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    _panel_header("Training preset", "Adapter exists (M5); no backend installed/run.", layout)
    combo = QComboBox()
    combo.addItems(["Preview", "Balanced", "Quality", "Custom"])
    layout.addWidget(combo)
    layout.addStretch()
    return widget


def build_training_monitor_panel() -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    _panel_header("Training monitor", "Not wired up yet -- see docs/adr/0009.", layout)
    progress = QProgressBar()
    progress.setValue(0)
    layout.addWidget(progress)
    button = QPushButton("Start Training")
    button.setEnabled(False)
    button.setToolTip("Adapter-only by design (docs/adr/0009): no training backend is installed or run here.")
    layout.addWidget(button)
    layout.addStretch()
    return widget


def build_export_panel() -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    _panel_header("Export", "Not wired up yet.", layout)
    button = QPushButton("Export")
    button.setEnabled(False)
    button.setToolTip("Execution isn't wired up yet.")
    layout.addWidget(button)
    layout.addStretch()
    return widget


class Vine360MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vineyard 360 3DGS")
        self.resize(920, 660)
        self.state = AppState()

        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.stack = QStackedWidget()

        project_panel = ProjectPanel(self.state)
        import_panel = ImportPanel(self.state)
        frames_panel = FramesPanel(self.state)
        projection_panel = ProjectionPanel(self.state)
        masks_panel = MasksPanel(self.state)
        pose_panel = PoseEstimationPanel(self.state)

        project_panel.project_changed.connect(import_panel.on_project_changed)
        project_panel.project_changed.connect(frames_panel.on_project_changed)
        project_panel.project_changed.connect(projection_panel.on_project_changed)
        project_panel.project_changed.connect(masks_panel.on_project_changed)
        project_panel.project_changed.connect(pose_panel.on_project_changed)

        self._panels = [
            project_panel,
            import_panel,
            frames_panel,
            projection_panel,
            masks_panel,
            pose_panel,
            build_training_preset_panel(),
            build_training_monitor_panel(),
            build_export_panel(),
        ]
        labels = [
            "Project",
            "Import",
            "Frames",
            "Projection",
            "Masks",
            "Pose estimation",
            "Training preset",
            "Training monitor",
            "Export",
        ]
        self._sidebar_items: list[QListWidgetItem] = []
        for label, widget in zip(labels, self._panels):
            item = QListWidgetItem(f"  {label}")
            item.setIcon(QIcon(_status_dot(PENDING)))
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

        self.state.notify_change = self.refresh_stage_statuses
        self.refresh_stage_statuses()

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
