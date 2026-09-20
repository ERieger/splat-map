"""The real M6 desktop app -- built on the dashboard shape the user chose
after comparing wizard_preview.py and dashboard_preview.py (see
docs/status.md and memory).

Project creation, source ingest, and frame extraction are wired to real
vine360 code (vine360.project, vine360.ingest.sources,
vine360.ingest.frames), run on a background QThread so a large real file
(e.g. a 7.65GB capture: ~25s to checksum, longer to extract frames from)
never freezes the window. Projection, masking and SfM stay preset-only for
now: their controls reflect real config/enums, but their action buttons
are disabled with an explanatory tooltip -- wiring those to actually
execute is the next step, not done here. See docs/status.md.

Run with: python -m vine360.gui.main_window
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
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
from vine360.masking.semantics import is_keep_fraction_anomalous
from vine360.models import Project
from vine360.project import (
    ProjectAlreadyExistsError,
    ProjectNotFoundError,
    create_project,
    load_project,
    open_index_db,
)
from vine360.projection.cubemap import six_face_preset
from vine360.runners.local import LocalRunner
from vine360.runners.probe import probe_dependencies

# Real sample capture location (see memory: project_capture_equipment_and_sample_data) --
# used only as a file-dialog starting point, nothing here reads or processes it.
_SAMPLE_DATA_HINT_DIR = Path("/mnt/e/11-9-26_EstoWines_Capture1/Equi")

CAPTURE_GROUP_PRESETS = ["Insta360 (ground)", "Antigravity A1 (aerial)", "Other / custom…"]

DONE, ACTIVE, PENDING = "done", "active", "pending"
STATUS_COLOR = {DONE: QColor(70, 150, 90), ACTIVE: QColor(200, 160, 50), PENDING: QColor(120, 120, 120)}


@dataclass
class AppState:
    project_root: Path | None = None
    project: Project | None = None
    conn: sqlite3.Connection | None = None


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

    def __init__(self, fn, args, kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # surfaced via `failed`, not swallowed
            self.failed.emit(exc)
        else:
            self.finished.emit(result)


def run_in_background(owner: QWidget, fn, *args, on_success=None, on_error=None, **kwargs) -> None:
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
    worker = _BackgroundWorker(fn, args, kwargs)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    if on_success:
        worker.finished.connect(on_success)
    if on_error:
        worker.failed.connect(on_error)

    if not hasattr(owner, "_bg_pairs"):
        owner._bg_pairs = []
    pair = (thread, worker)
    owner._bg_pairs.append(pair)
    thread.finished.connect(lambda: owner._bg_pairs.remove(pair) if pair in owner._bg_pairs else None)

    thread.start()


def _status_dot(status: str) -> QPixmap:
    pixmap = QPixmap(12, 12)
    pixmap.fill(Qt.transparent)
    from PySide6.QtGui import QPainter

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(STATUS_COLOR[status])
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(1, 1, 10, 10)
    painter.end()
    return pixmap


def _placeholder_thumb(width: int, height: int, color: QColor, text: str) -> QWidget:
    pixmap = QPixmap(width, height)
    pixmap.fill(color)
    label = QLabel()
    label.setPixmap(pixmap)
    label.setAlignment(Qt.AlignCenter)
    caption = QLabel(text)
    caption.setAlignment(Qt.AlignCenter)
    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.addWidget(label)
    layout.addWidget(caption)
    return wrapper


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


def _disabled_action_button(label: str) -> QPushButton:
    button = QPushButton(label)
    button.setEnabled(False)
    button.setToolTip("Execution isn't wired up yet -- this project stage is preset/preview only so far.")
    return button


class ProjectPanel(QWidget):
    project_changed = Signal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        layout = QVBoxLayout(self)
        _panel_header("Project", "Create a new project or open an existing one.", layout)

        self.path_label = QLabel("No project open.")
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        form = QFormLayout()
        self.name_field = QLineEdit("Block 7 — North Row")
        form.addRow("Name (for a new project):", self.name_field)

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
        layout.addLayout(form)

        note = QLabel(
            "Both real capture devices for this project (Insta360, Antigravity A1) are "
            "360° platforms, so '360° only' is the right mode for combining them -- "
            "'Mixed' here would mean a true conventional (non-equirectangular) camera "
            "alongside them, not the two 360 rigs."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid); font-style: italic;")
        layout.addWidget(note)

        button_row = QHBoxLayout()
        create_btn = QPushButton("Create New Project…")
        create_btn.clicked.connect(self._on_create)
        open_btn = QPushButton("Open Existing Project…")
        open_btn.clicked.connect(self._on_open)
        button_row.addWidget(create_btn)
        button_row.addWidget(open_btn)
        layout.addLayout(button_row)
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


def _add_source_worker(
    project_root: Path, file_path: Path, capture_group: str | None, confirm_equirectangular: bool, added_at: str
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

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        controls_layout.addWidget(self.progress)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)

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
        self.progress.setVisible(True)
        name = Path(self._pending_file_path).name
        self.status_label.setText(f"Adding {name}… (checksumming a large file can take tens of seconds)")
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
        )

    def _on_add_source_success(self, source) -> None:
        self.add_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText(f"Added {Path(source.path).name} ({source.media_type.value}, {source.projection.value}).")
        self._refresh_sources_table()

    def _on_add_source_error(self, exc: Exception) -> None:
        self.add_btn.setEnabled(True)
        self.progress.setVisible(False)
        if isinstance(exc, EquirectangularConfirmationRequired):
            name = Path(self._pending_file_path).name
            answer = QMessageBox.question(
                self, "Confirm equirectangular", f"{name} has a 2:1 aspect ratio. Is it truly equirectangular?"
            )
            if answer == QMessageBox.Yes:
                self._start_add_source(confirm_equirectangular=True)
            else:
                self.status_label.setText("")
            return
        self.status_label.setText("")
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
        self.progress.setVisible(True)
        self.status_label.setText(f"Removing {Path(path).name}…")
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
        self.progress.setVisible(False)
        self.status_label.setText("Removed.")
        self._refresh_sources_table()

    def _on_remove_source_error(self, exc: Exception) -> None:
        self.add_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("")
        if isinstance(exc, SourceError):
            QMessageBox.warning(self, "Could not remove source", str(exc))
        else:
            QMessageBox.critical(self, "Unexpected error removing source", f"{type(exc).__name__}: {exc}")


def _extract_frames_worker(project_root: Path, source_id: str, interval_seconds: float):
    conn = open_index_db(project_root)
    try:
        return extract_frames(conn, project_root, source_id, LocalRunner(), interval_seconds=interval_seconds)
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

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        controls_layout.addWidget(self.progress)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)

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
            "SELECT source_id, path FROM sources WHERE media_type = 'video' ORDER BY rowid"
        ).fetchall()
        for source_id, path in rows:
            self.source_combo.addItem(f"{Path(path).name} ({source_id[:8]})", userData=source_id)
        has_sources = self.source_combo.count() > 0
        self.extract_btn.setEnabled(has_sources)
        self.status_label.setText("" if has_sources else "No video sources registered yet -- add one on Import.")

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
        self.progress.setVisible(True)
        self.status_label.setText(
            f"Extracting frames every {interval}s -- this can take a while for long or high-resolution video…"
        )
        run_in_background(
            self,
            _extract_frames_worker,
            self.state.project_root,
            source_id,
            interval,
            on_success=self._on_extract_success,
            on_error=self._on_extract_error,
        )

    def _on_extract_success(self, frames) -> None:
        self.extract_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText(f"Extracted {len(frames)} frames.")

    def _on_extract_error(self, exc: Exception) -> None:
        self.extract_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status_label.setText("")
        if isinstance(exc, FrameExtractionError):
            QMessageBox.warning(self, "Frame extraction failed", str(exc))
        else:
            QMessageBox.critical(self, "Unexpected error extracting frames", f"{type(exc).__name__}: {exc}")


def build_projection_panel() -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    _panel_header("Projection", "Real face count from vine360.projection.cubemap.", layout)
    faces = six_face_preset(face_size=1024, fov_degrees=90.0)
    info = QLabel(f"Six-face preset produces {len(faces)} views per frame: {', '.join(f.name for f in faces)}.")
    info.setWordWrap(True)
    layout.addWidget(info)
    layout.addWidget(_disabled_action_button("Generate Projections"))
    layout.addStretch()
    return widget


def build_masks_panel() -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    _panel_header("Masks", "Illustrative keep-fractions, flagged for real via is_keep_fraction_anomalous.", layout)
    grid = QGridLayout()
    for i, fraction in enumerate([0.82, 0.79, 0.15, 0.91]):
        reason = is_keep_fraction_anomalous(fraction)
        color = QColor(150, 60, 60) if reason else QColor(60, 110, 70)
        label = f"view {i:02d}\nkeep {fraction:.0%}" + ("\n⚠ flagged" if reason else "")
        grid.addWidget(_placeholder_thumb(110, 80, color, label), 0, i)
    layout.addLayout(grid)
    layout.addWidget(_disabled_action_button("Build Masks"))
    layout.addStretch()
    return widget


def build_pose_panel() -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    _panel_header("Pose estimation", "Not wired up yet.", layout)
    layout.addWidget(_disabled_action_button("Run SfM"))
    layout.addStretch()
    return widget


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
    _panel_header("Training monitor", "Not wired up yet.", layout)
    progress = QProgressBar()
    progress.setValue(0)
    layout.addWidget(progress)
    layout.addWidget(_disabled_action_button("Start Training"))
    layout.addStretch()
    return widget


def build_export_panel() -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    _panel_header("Export", "Not wired up yet.", layout)
    layout.addWidget(_disabled_action_button("Export"))
    layout.addStretch()
    return widget


class Vine360MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vineyard 360 3DGS")
        self.resize(900, 620)
        self.state = AppState()

        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.stack = QStackedWidget()

        project_panel = ProjectPanel(self.state)
        import_panel = ImportPanel(self.state)
        frames_panel = FramesPanel(self.state)
        project_panel.project_changed.connect(import_panel.on_project_changed)
        project_panel.project_changed.connect(frames_panel.on_project_changed)
        project_panel.project_changed.connect(self._on_project_changed)

        stages = [
            ("Project", ACTIVE, project_panel),
            ("Import", PENDING, import_panel),
            ("Frames", PENDING, frames_panel),
            ("Projection", PENDING, build_projection_panel()),
            ("Masks", PENDING, build_masks_panel()),
            ("Pose estimation", PENDING, build_pose_panel()),
            ("Training preset", PENDING, build_training_preset_panel()),
            ("Training monitor", PENDING, build_training_monitor_panel()),
            ("Export", PENDING, build_export_panel()),
        ]
        self._sidebar_items: list[QListWidgetItem] = []
        for label, status, widget in stages:
            item = QListWidgetItem(f"  {label}")
            item.setIcon(QIcon(_status_dot(status)))
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

    def _on_project_changed(self) -> None:
        self._sidebar_items[0].setIcon(QIcon(_status_dot(DONE)))
        self._sidebar_items[1].setIcon(QIcon(_status_dot(ACTIVE)))

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
