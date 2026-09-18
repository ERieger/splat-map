"""A navigation/shape preview of the M6 desktop wizard (handover doc,
section 6 "User workflow"). NOT a real implementation of M6: most pages
are static placeholders. Two things are wired to real backend code because
doing so cost nothing: the dependency probe (import step) and the cubemap
face count (projection preset step). Everything else -- the file list, the
contact sheet, mask thumbnails, registration stats, training progress --
is illustrative placeholder content, clearly labeled as such in the UI.

Run with: python -m vine360.gui.wizard_preview
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from vine360.config import CaptureMode, FramePreset
from vine360.masking.semantics import is_keep_fraction_anomalous
from vine360.projection.cubemap import six_face_preset
from vine360.runners.probe import probe_dependencies


def _placeholder_thumb(width: int, height: int, color: QColor, text: str) -> QLabel:
    pixmap = QPixmap(width, height)
    pixmap.fill(color)
    label = QLabel()
    label.setPixmap(pixmap)
    label.setAlignment(Qt.AlignCenter)
    label.setToolTip(text)
    caption = QLabel(text)
    caption.setAlignment(Qt.AlignCenter)
    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    layout.setContentsMargins(2, 2, 2, 2)
    layout.addWidget(label)
    layout.addWidget(caption)
    return wrapper


class WelcomePage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Vineyard 360 3DGS")
        self.setSubTitle("Preview build — shows the wizard's shape, not the real pipeline.")
        layout = QVBoxLayout(self)
        label = QLabel(
            "This wizard walks through the workflow from the handover doc, section 6:\n"
            "create a project, import captures, choose presets, review masks and\n"
            "registration, train, and export.\n\n"
            "Most screens here are placeholders — this build exists to show the shape\n"
            "of the eventual desktop UI (M6), not to run a real reconstruction."
        )
        label.setWordWrap(True)
        layout.addWidget(label)


class ProjectPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("1. Create a project")
        layout = QFormLayout(self)

        self.name_field = QLineEdit("Block 7 — North Row")
        layout.addRow("Project name:", self.name_field)

        mode_box = QGroupBox()
        mode_layout = QVBoxLayout(mode_box)
        for mode, label in [
            (CaptureMode.THREE_SIXTY, "360° only (Insta360 X6)"),
            (CaptureMode.CONVENTIONAL, "Conventional photos only (drone)"),
            (CaptureMode.MIXED, "Mixed capture (360° + drone)"),
        ]:
            radio = QRadioButton(label)
            radio.setChecked(mode == CaptureMode.THREE_SIXTY)
            mode_layout.addWidget(radio)
        layout.addRow("Capture mode:", mode_box)


class ImportPage(QWizardPage):
    """The one page with real backend content: the dependency probe."""

    def __init__(self):
        super().__init__()
        self.setTitle("2. Import media")
        self.setSubTitle("Storage estimate, detected projection, GPU availability and dependencies.")
        layout = QVBoxLayout(self)

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


class FramePresetPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("3. Frame preset")
        self.setSubTitle("Generate a contact sheet before committing to full extraction.")
        layout = QVBoxLayout(self)

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


class ProjectionPresetPage(QWizardPage):
    """Real backend content: the actual six-face preset face count."""

    def __init__(self):
        super().__init__()
        self.setTitle("4. Projection preset")
        self.setSubTitle("Preview seam placement and the number of generated views.")
        layout = QVBoxLayout(self)

        combo = QComboBox()
        combo.addItems(["Six-face cubemap (90° FOV)", "Six-face cubemap + poles", "Dense yaw/pitch"])
        layout.addWidget(combo)

        faces = six_face_preset(face_size=1024, fov_degrees=90.0)
        faces_with_poles = six_face_preset(face_size=1024, fov_degrees=90.0, include_polar_faces=True)
        info = QLabel(
            f"Real computation from vine360.projection.cubemap: "
            f"{len(faces)} views per frame by default "
            f"({', '.join(f.name for f in faces)}); "
            f"{len(faces_with_poles)} if poles are included."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addWidget(QLabel("Seam/coverage preview (illustrative):"))
        grid = QGridLayout()
        for i, face in enumerate(faces):
            grid.addWidget(_placeholder_thumb(110, 110, QColor(80, 70, 100), face.name), 0, i)
        layout.addLayout(grid)


class MaskingReviewPage(QWizardPage):
    """Real backend content: is_keep_fraction_anomalous applied to
    illustrative numbers, to show what a flagged mask would look like."""

    def __init__(self):
        super().__init__()
        self.setTitle("5. Review masks")
        self.setSubTitle("Sampled mask grid; flag unusually low/high retained area.")
        layout = QVBoxLayout(self)

        grid = QGridLayout()
        example_fractions = [0.82, 0.79, 0.15, 0.91, 0.99, 0.74]
        for i, fraction in enumerate(example_fractions):
            reason = is_keep_fraction_anomalous(fraction)
            color = QColor(150, 60, 60) if reason else QColor(60, 110, 70)
            label = f"view {i:02d}\nkeep {fraction:.0%}" + ("\n⚠ flagged" if reason else "")
            grid.addWidget(_placeholder_thumb(110, 80, color, label), i // 3, i % 3)
        layout.addLayout(grid)
        note = QLabel("Colors/flags computed for real via vine360.masking.semantics.is_keep_fraction_anomalous "
                       "on illustrative keep-fractions; the thumbnails themselves are placeholders.")
        note.setWordWrap(True)
        layout.addWidget(note)


class PoseEstimationPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("6. Estimate poses")
        self.setSubTitle("Registration statistics and sparse model preview before training.")
        layout = QFormLayout(self)
        for label, value in [
            ("Registered images", "142 / 150 (94.7%) — illustrative"),
            ("Connected models", "1 — illustrative"),
            ("Sparse points", "48,213 — illustrative"),
            ("Mean reprojection error", "0.61 px — illustrative"),
            ("Mean track length", "3.8 — illustrative"),
        ]:
            layout.addRow(label + ":", QLabel(value))


class TrainingPresetPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("7. Training preset")
        self.setSubTitle("Choose based on GPU memory and desired quality.")
        layout = QVBoxLayout(self)
        combo = QComboBox()
        combo.addItems(["Preview", "Balanced", "Quality", "Custom"])
        combo.setCurrentText("Balanced")
        layout.addWidget(combo)
        layout.addWidget(QLabel("Estimated disk use: ~6.4 GB (illustrative — not a fabricated duration estimate)"))


class TrainingMonitorPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("8. Monitor training")
        self.setSubTitle("Inspect intermediate renders, stop safely, resume from a checkpoint.")
        layout = QVBoxLayout(self)
        progress = QProgressBar()
        progress.setValue(37)
        layout.addWidget(progress)
        layout.addWidget(QLabel("Iteration 11,100 / 30,000 — illustrative"))
        row = QVBoxLayout()
        stop_btn = QPushButton("Stop")
        resume_btn = QPushButton("Resume from checkpoint")
        resume_btn.setEnabled(False)
        row.addWidget(stop_btn)
        row.addWidget(resume_btn)
        layout.addLayout(row)
        layout.addWidget(_placeholder_thumb(320, 180, QColor(40, 55, 40), "intermediate render (illustrative)"))


class ExportPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("9. Export")
        self.setSubTitle("PLY, poses, configuration, logs, thumbnails, mask summary, manifest.")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Export bundle (illustrative):"))
        file_list = QListWidget()
        file_list.addItems(
            [
                "splat.ply",
                "transforms.json",
                "config.yaml",
                "logs/train.log",
                "masks/summary.json",
                "manifest.json",
            ]
        )
        layout.addWidget(file_list)
        layout.addWidget(QPushButton("Open in viewer (not wired up in this preview)"))


class Vine360WizardPreview(QWizard):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vineyard 360 3DGS — wizard preview (M6 shape only)")
        self.setWizardStyle(QWizard.ModernStyle)
        for page in [
            WelcomePage(),
            ProjectPage(),
            ImportPage(),
            FramePresetPage(),
            ProjectionPresetPage(),
            MaskingReviewPage(),
            PoseEstimationPage(),
            TrainingPresetPage(),
            TrainingMonitorPage(),
            ExportPage(),
        ]:
            self.addPage(page)
        self.resize(720, 560)


def main() -> int:
    app = QApplication(sys.argv)
    wizard = Vine360WizardPreview()
    wizard.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
