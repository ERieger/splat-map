"""The real M6 desktop app -- built on the dashboard shape the user chose
after comparing wizard_preview.py and dashboard_preview.py (see
docs/status.md and memory).

Project creation/loading and source ingest are wired to real vine360 code
(vine360.project, vine360.ingest.sources) -- these are fast, local,
metadata-only operations (ffprobe + a checksum), not pipeline "runs".

Frame extraction, projection, masking and SfM stay preset-only for now:
their controls reflect real config/enums, but their action buttons are
disabled with an explanatory tooltip. Wiring those to actually execute is
a deliberate next step, not done here -- see docs/status.md.

Run with: python -m vine360.gui.main_window
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
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

from vine360.config import CaptureMode, FramePreset
from vine360.ingest.media_probe import ProbeError
from vine360.ingest.sources import (
    EquirectangularConfirmationRequired,
    SourceError,
    add_source,
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


class ImportPanel(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        layout = QVBoxLayout(self)
        _panel_header("Import", "Register real media sources into the open project.", layout)

        self.no_project_label = QLabel("Create or open a project first (see the Project stage).")
        layout.addWidget(self.no_project_label)

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        add_row = QHBoxLayout()
        add_btn = QPushButton("Add Source…")
        add_btn.clicked.connect(self._on_add_source)
        add_row.addWidget(add_btn)
        add_row.addWidget(QLabel("Capture group:"))
        self.capture_group_combo = QComboBox()
        self.capture_group_combo.setEditable(True)
        self.capture_group_combo.addItems(CAPTURE_GROUP_PRESETS)
        add_row.addWidget(self.capture_group_combo)
        controls_layout.addLayout(add_row)

        self.sources_table = QTableWidget(0, 5)
        self.sources_table.setHorizontalHeaderLabels(["Media type", "Projection", "Capture group", "Checksum", "Path"])
        controls_layout.addWidget(self.sources_table)

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

        confirm_equirect = False
        while True:
            try:
                add_source(
                    self.state.conn,
                    Path(file_path),
                    LocalRunner(),
                    capture_group=self._resolved_capture_group(),
                    confirm_equirectangular=confirm_equirect,
                    added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
                break
            except EquirectangularConfirmationRequired:
                answer = QMessageBox.question(
                    self,
                    "Confirm equirectangular",
                    f"{Path(file_path).name} has a 2:1 aspect ratio. Is it truly equirectangular?",
                )
                if answer != QMessageBox.Yes:
                    return
                confirm_equirect = True
                continue
            except (SourceError, ProbeError) as exc:
                QMessageBox.warning(self, "Could not add source", str(exc))
                return

        self._refresh_sources_table()

    def _refresh_sources_table(self) -> None:
        rows = self.state.conn.execute(
            "SELECT media_type, projection, capture_group, checksum, path FROM sources ORDER BY rowid"
        ).fetchall()
        self.sources_table.setRowCount(len(rows))
        for r, (media_type, projection, capture_group, checksum, path) in enumerate(rows):
            self.sources_table.setItem(r, 0, QTableWidgetItem(media_type))
            self.sources_table.setItem(r, 1, QTableWidgetItem(projection))
            self.sources_table.setItem(r, 2, QTableWidgetItem(capture_group or "—"))
            self.sources_table.setItem(r, 3, QTableWidgetItem(checksum[:12] + "…"))
            self.sources_table.setItem(r, 4, QTableWidgetItem(path))
        self.sources_table.resizeColumnsToContents()


def build_frames_panel() -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    _panel_header("Frames", "Frame preset (real enum; extraction not wired up yet).", layout)
    combo = QComboBox()
    combo.addItems([p.value for p in FramePreset])
    combo.setCurrentText(FramePreset.BALANCED.value)
    layout.addWidget(combo)
    layout.addWidget(_disabled_action_button("Extract Frames"))
    layout.addStretch()
    return widget


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
        project_panel.project_changed.connect(import_panel.on_project_changed)
        project_panel.project_changed.connect(self._on_project_changed)

        stages = [
            ("Project", ACTIVE, project_panel),
            ("Import", PENDING, import_panel),
            ("Frames", PENDING, build_frames_panel()),
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

        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.sidebar.setCurrentRow(0)

        splitter = QSplitter()
        splitter.addWidget(self.sidebar)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    def _on_project_changed(self) -> None:
        self._sidebar_items[0].setIcon(QIcon(_status_dot(DONE)))
        self._sidebar_items[1].setIcon(QIcon(_status_dot(ACTIVE)))


def main() -> int:
    app = QApplication(sys.argv)
    window = Vine360MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
