"""A dashboard-style alternative to wizard_preview.py: a persistent sidebar
of pipeline stages plus a main content area, instead of a linear
Back/Next flow. Same preview caveats as wizard_preview.py apply -- most
content is illustrative placeholder, with the dependency probe, cubemap
face count, and keep-fraction flagging wired to real backend calls.

Rationale for comparing this against the wizard: the handover doc frames
this as an "orchestration and project-management application" with
resumable, revisitable stages (inspect registration diagnostics, adjust
masks, monitor a running training job while reviewing something else) --
a dashboard supports jumping between stages and keeping project context
visible in a way a strict linear wizard doesn't.

Run with: python -m vine360.gui.dashboard_preview
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
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
from vine360.masking.semantics import is_keep_fraction_anomalous
from vine360.projection.cubemap import six_face_preset
from vine360.runners.probe import probe_dependencies

DONE = "done"
ACTIVE = "active"
PENDING = "pending"

STATUS_COLOR = {
    DONE: QColor(70, 150, 90),
    ACTIVE: QColor(200, 160, 50),
    PENDING: QColor(120, 120, 120),
}
STATUS_LABEL = {DONE: "done", ACTIVE: "in progress", PENDING: "not started"}


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


def _status_dot(status: str) -> QLabel:
    pixmap = QPixmap(12, 12)
    pixmap.fill(Qt.transparent)
    from PySide6.QtGui import QPainter

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(STATUS_COLOR[status])
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(1, 1, 10, 10)
    painter.end()
    label = QLabel()
    label.setPixmap(pixmap)
    return label


def _panel(title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
    widget = QWidget()
    layout = QVBoxLayout(widget)
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
    return widget, layout


def build_project_panel() -> QWidget:
    widget, layout = _panel("Project", "Project-wide settings, visible regardless of which stage you're working on.")
    form = QFormLayout()
    form.addRow("Name:", QLineEdit("Block 7 — North Row"))
    mode_row = QHBoxLayout()
    for mode, label in [
        (CaptureMode.THREE_SIXTY, "360° only"),
        (CaptureMode.CONVENTIONAL, "Conventional"),
        (CaptureMode.MIXED, "Mixed"),
    ]:
        radio = QRadioButton(label)
        radio.setChecked(mode == CaptureMode.THREE_SIXTY)
        mode_row.addWidget(radio)
    form.addRow("Capture mode:", mode_row)
    layout.addLayout(form)
    layout.addStretch()
    return widget


def build_import_panel() -> QWidget:
    widget, layout = _panel("Import", "Storage estimate, detected projection, GPU availability and dependencies.")
    file_list = QListWidget()
    file_list.addItems(
        [
            "GX010042.mp4 — 5760x2880, 2:1, 3m 12s (equirectangular, X6)",
            "GX010043.mp4 — 5760x2880, 2:1, 4m 05s (equirectangular, X6)",
            "DJI_0301.JPG — 5280x3956 (drone photo)",
            "DJI_0302.JPG — 5280x3956 (drone photo)",
        ]
    )
    layout.addWidget(QLabel("Detected sources (illustrative):"))
    layout.addWidget(file_list)
    layout.addWidget(QLabel("Estimated storage for full extraction: ~14.2 GB (illustrative)"))

    layout.addWidget(QLabel("Dependency check (real, live probe):"))
    table = QTableWidget(0, 3)
    table.setHorizontalHeaderLabels(["Tool", "Found", "Version"])
    for status in probe_dependencies():
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QTableWidgetItem(status.name))
        table.setItem(row, 1, QTableWidgetItem("yes" if status.found else "NO"))
        table.setItem(row, 2, QTableWidgetItem(status.version or "—"))
    table.resizeColumnsToContents()
    layout.addWidget(table)
    return widget


def build_frames_panel() -> QWidget:
    widget, layout = _panel("Frames", "Frame preset and contact sheet.")
    combo = QComboBox()
    combo.addItems([p.value for p in FramePreset])
    combo.setCurrentText(FramePreset.BALANCED.value)
    layout.addWidget(combo)
    layout.addWidget(QLabel("Contact sheet (illustrative placeholder frames):"))
    grid = QGridLayout()
    colors = [QColor(60, 90, 60), QColor(70, 100, 70), QColor(50, 80, 90), QColor(90, 80, 60)]
    for i in range(8):
        grid.addWidget(_placeholder_thumb(96, 54, colors[i % len(colors)], f"frame {i:03d}"), i // 4, i % 4)
    layout.addLayout(grid)
    layout.addStretch()
    return widget


def build_projection_panel() -> QWidget:
    widget, layout = _panel("Projection", "Seam placement and the number of generated views.")
    combo = QComboBox()
    combo.addItems(["Six-face cubemap (90° FOV)", "Six-face cubemap + poles", "Dense yaw/pitch"])
    layout.addWidget(combo)

    faces = six_face_preset(face_size=1024, fov_degrees=90.0)
    faces_with_poles = six_face_preset(face_size=1024, fov_degrees=90.0, include_polar_faces=True)
    info = QLabel(
        f"Real computation from vine360.projection.cubemap: {len(faces)} views per frame by "
        f"default ({', '.join(f.name for f in faces)}); {len(faces_with_poles)} if poles are included."
    )
    info.setWordWrap(True)
    layout.addWidget(info)

    grid = QGridLayout()
    for i, face in enumerate(faces):
        grid.addWidget(_placeholder_thumb(110, 110, QColor(80, 70, 100), face.name), 0, i)
    layout.addLayout(grid)
    layout.addStretch()
    return widget


def build_masks_panel() -> QWidget:
    widget, layout = _panel("Masks", "Sampled mask grid; flag unusually low/high retained area.")
    grid = QGridLayout()
    example_fractions = [0.82, 0.79, 0.15, 0.91, 0.99, 0.74]
    for i, fraction in enumerate(example_fractions):
        reason = is_keep_fraction_anomalous(fraction)
        color = QColor(150, 60, 60) if reason else QColor(60, 110, 70)
        label = f"view {i:02d}\nkeep {fraction:.0%}" + ("\n⚠ flagged" if reason else "")
        grid.addWidget(_placeholder_thumb(110, 80, color, label), i // 3, i % 3)
    layout.addLayout(grid)
    note = QLabel(
        "Colors/flags computed for real via is_keep_fraction_anomalous on illustrative "
        "keep-fractions; the thumbnails themselves are placeholders."
    )
    note.setWordWrap(True)
    layout.addWidget(note)
    layout.addStretch()
    return widget


def build_pose_panel() -> QWidget:
    widget, layout = _panel("Pose estimation", "Registration statistics and sparse model preview.")
    form = QFormLayout()
    for label, value in [
        ("Registered images", "142 / 150 (94.7%) — illustrative"),
        ("Connected models", "1 — illustrative"),
        ("Sparse points", "48,213 — illustrative"),
        ("Mean reprojection error", "0.61 px — illustrative"),
        ("Mean track length", "3.8 — illustrative"),
    ]:
        form.addRow(label + ":", QLabel(value))
    layout.addLayout(form)
    layout.addStretch()
    return widget


def build_training_preset_panel() -> QWidget:
    widget, layout = _panel("Training preset", "Choose based on GPU memory and desired quality.")
    combo = QComboBox()
    combo.addItems(["Preview", "Balanced", "Quality", "Custom"])
    combo.setCurrentText("Balanced")
    layout.addWidget(combo)
    layout.addWidget(QLabel("Estimated disk use: ~6.4 GB (illustrative)"))
    layout.addStretch()
    return widget


def build_training_monitor_panel() -> QWidget:
    widget, layout = _panel("Training monitor", "Stays reachable while you review other stages.")
    progress = QProgressBar()
    progress.setValue(37)
    layout.addWidget(progress)
    layout.addWidget(QLabel("Iteration 11,100 / 30,000 — illustrative"))
    row = QHBoxLayout()
    row.addWidget(QPushButton("Stop"))
    resume_btn = QPushButton("Resume from checkpoint")
    resume_btn.setEnabled(False)
    row.addWidget(resume_btn)
    layout.addLayout(row)
    layout.addWidget(_placeholder_thumb(320, 180, QColor(40, 55, 40), "intermediate render (illustrative)"))
    layout.addStretch()
    return widget


def build_export_panel() -> QWidget:
    widget, layout = _panel("Export", "PLY, poses, configuration, logs, thumbnails, mask summary, manifest.")
    file_list = QListWidget()
    file_list.addItems(
        ["splat.ply", "transforms.json", "config.yaml", "logs/train.log", "masks/summary.json", "manifest.json"]
    )
    layout.addWidget(file_list)
    layout.addWidget(QPushButton("Open in viewer (not wired up in this preview)"))
    layout.addStretch()
    return widget


# (sidebar label, status, panel builder) -- statuses are illustrative,
# showing what mid-project state would look like: earlier stages done,
# masking in progress, everything after not started yet.
STAGES = [
    ("Project", DONE, build_project_panel),
    ("Import", DONE, build_import_panel),
    ("Frames", DONE, build_frames_panel),
    ("Projection", DONE, build_projection_panel),
    ("Masks", ACTIVE, build_masks_panel),
    ("Pose estimation", PENDING, build_pose_panel),
    ("Training preset", PENDING, build_training_preset_panel),
    ("Training monitor", PENDING, build_training_monitor_panel),
    ("Export", PENDING, build_export_panel),
]


class Vine360DashboardPreview(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vineyard 360 3DGS — dashboard preview (M6 shape only)")
        self.resize(860, 600)

        sidebar = QListWidget()
        sidebar.setFixedWidth(200)
        stack = QStackedWidget()

        for label, status, builder in STAGES:
            item = QListWidgetItem(f"  {label}")
            item.setIcon(QIcon(_status_dot(status).pixmap()))
            item.setToolTip(STATUS_LABEL[status])
            sidebar.addItem(item)
            stack.addWidget(builder())

        sidebar.currentRowChanged.connect(stack.setCurrentIndex)
        sidebar.setCurrentRow(4)  # land on "Masks", the in-progress stage

        header = QLabel("Block 7 — North Row   ·   360° capture   ·   resumable stage dashboard")
        header.setStyleSheet("padding: 6px; font-weight: 500; background: palette(alternate-base);")

        splitter = QSplitter()
        splitter.addWidget(sidebar)
        splitter.addWidget(stack)
        splitter.setStretchFactor(1, 1)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(header)
        central_layout.addWidget(splitter)
        self.setCentralWidget(central)


def main() -> int:
    app = QApplication(sys.argv)
    window = Vine360DashboardPreview()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
