"""Graphical picker for which perspective views to render from each 360
frame (docs/adr/0041). Replaces the Projection panel's row of six
checkboxes: every direction in vine360.projection.cubemap's catalog is a
clickable marker drawn over the current frame's own panorama, and every
selected direction's footprint -- what that view will actually see, at
the panel's current field of view and ring tilt -- is outlined on it, so
overlap and gaps are visible before anything is generated. Hovering a
marker renders a small live preview of that view from the panorama.

Pure presentation: geometry comes from cubemap.footprint_outline /
direction_center_pixel and rendering from render.project_equirect_to_face,
so nothing here re-derives projection math.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from vine360.projection.cubemap import (
    ALL_DIRECTION_NAMES,
    CARDINAL_NAMES,
    DEFAULT_FOV_DEGREES,
    DEFAULT_RING_TILT_DEGREES,
    DIRECTION_CATALOG,
    DIRECTION_PRESETS,
    DIRECTIONS_BY_NAME,
    MAX_RING_TILT_DEGREES,
    MIN_RING_TILT_DEGREES,
    RING_HORIZON,
    RING_LOWER,
    RING_POLE_DOWN,
    RING_POLE_UP,
    RING_UPPER,
    DirectionSpec,
    direction_center_pixel,
    direction_preset,
    footprint_outline,
)
from vine360.projection.render import project_equirect_to_face

RING_COLORS = {
    RING_HORIZON: QColor(76, 201, 120),
    RING_LOWER: QColor(242, 166, 46),
    RING_UPPER: QColor(84, 160, 255),
    RING_POLE_UP: QColor(190, 190, 190),
    RING_POLE_DOWN: QColor(190, 190, 190),
}
MARKER_RADIUS = 7.0
HIT_RADIUS = 13.0
POLE_MARKER_INSET = 0.07  # pole markers sit this fraction of the height in from the top/bottom edge
BACKGROUND_MAX_WIDTH = 1024
HOVER_PREVIEW_SIZE = 160


def _load_panorama(path: Path) -> np.ndarray | None:
    """RGB array of the panorama, downscaled to at most
    BACKGROUND_MAX_WIDTH wide -- the picker never needs full 8K detail.
    None (never raises) on a missing/unreadable file."""
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.width > BACKGROUND_MAX_WIDTH:
                img = img.resize((BACKGROUND_MAX_WIDTH, max(1, BACKGROUND_MAX_WIDTH * img.height // img.width)))
            return np.asarray(img).copy()
    except (OSError, ValueError):
        return None


class _PanoramaCanvas(QWidget):
    """The clickable map itself. Draws into the largest 2:1 rectangle that
    fits, so the equirect proportions (and therefore footprints) stay true."""

    toggled = Signal(str)
    hovered = Signal(object)  # str | None

    def __init__(self, picker: "DirectionPicker"):
        super().__init__()
        self._picker = picker
        self._hover: str | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(360, 180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def sizeHint(self) -> QSize:
        return QSize(720, 360)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(180, width // 2)

    def map_rect(self) -> QRectF:
        w, h = self.width(), self.height()
        if w >= 2 * h:
            mw, mh = 2.0 * h, float(h)
        else:
            mw, mh = float(w), w / 2.0
        return QRectF((w - mw) / 2.0, (h - mh) / 2.0, mw, mh)

    def marker_position(self, spec: DirectionSpec) -> QPointF:
        rect = self.map_rect()
        if spec.ring == RING_POLE_UP:
            return QPointF(rect.center().x(), rect.top() + rect.height() * POLE_MARKER_INSET)
        if spec.ring == RING_POLE_DOWN:
            return QPointF(rect.center().x(), rect.bottom() - rect.height() * POLE_MARKER_INSET)
        u, v = direction_center_pixel(spec, int(rect.width()), int(rect.height()), self._picker.ring_tilt())
        return QPointF(rect.left() + u + 0.5, rect.top() + v + 0.5)

    def name_at(self, pos: QPointF) -> str | None:
        best, best_dist = None, HIT_RADIUS
        for spec in DIRECTION_CATALOG:
            p = self.marker_position(spec)
            dist = ((p.x() - pos.x()) ** 2 + (p.y() - pos.y()) ** 2) ** 0.5
            if dist <= best_dist:
                best, best_dist = spec.name, dist
        return best

    # -- events -------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            name = self.name_at(event.position())
            if name is not None:
                self.toggled.emit(name)

    def mouseMoveEvent(self, event) -> None:
        name = self.name_at(event.position())
        if name != self._hover:
            self._hover = name
            self.setCursor(Qt.PointingHandCursor if name else Qt.ArrowCursor)
            self.hovered.emit(name)
            self.update()

    def leaveEvent(self, event) -> None:
        if self._hover is not None:
            self._hover = None
            self.hovered.emit(None)
            self.update()

    # -- painting -----------------------------------------------------------------

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.map_rect()
        background = self._picker.background_qimage()
        if background is not None:
            painter.drawImage(rect, background)
            painter.fillRect(rect, QColor(0, 0, 0, 70))  # dim so outlines read on any content
        else:
            painter.fillRect(rect, QColor(40, 44, 52))
        self._paint_grid(painter, rect)

        selected = set(self._picker.selected_names())
        for spec in DIRECTION_CATALOG:
            if spec.name in selected or spec.name == self._hover:
                self._paint_footprint(painter, rect, spec, hovered=spec.name == self._hover)
        for spec in DIRECTION_CATALOG:
            self._paint_marker(painter, spec, spec.name in selected, spec.name == self._hover)

        painter.setPen(QPen(QColor(255, 255, 255, 110), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        painter.end()

    def _paint_grid(self, painter: QPainter, rect: QRectF) -> None:
        tilt = self._picker.ring_tilt()
        painter.setPen(QPen(QColor(255, 255, 255, 45), 1, Qt.DashLine))
        for pitch in (tilt, 0.0, -tilt):
            y = rect.top() + (90.0 - pitch) / 180.0 * rect.height()
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.setPen(QColor(255, 255, 255, 170))
        font = painter.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1))
        painter.setFont(font)
        for label, lon in (("back", -180.0), ("left", -90.0), ("front", 0.0), ("right", 90.0), ("back", 180.0)):
            x = rect.left() + (lon + 180.0) / 360.0 * rect.width()
            align = Qt.AlignLeft if lon == -180.0 else Qt.AlignRight if lon == 180.0 else Qt.AlignHCenter
            text_rect = QRectF(x - 40 if align != Qt.AlignLeft else x + 3, rect.top() + 2, 80 if align == Qt.AlignHCenter else 37, 14)
            painter.drawText(text_rect, align | Qt.AlignTop, label)
        for label, pitch in ((f"+{tilt:g}°", tilt), ("0°", 0.0), (f"−{tilt:g}°", -tilt)):
            y = rect.top() + (90.0 - pitch) / 180.0 * rect.height()
            painter.drawText(QRectF(rect.left() + 3, y - 14, 60, 13), Qt.AlignLeft | Qt.AlignBottom, label)

    def _paint_footprint(self, painter: QPainter, rect: QRectF, spec: DirectionSpec, hovered: bool) -> None:
        color = QColor(RING_COLORS[spec.ring])
        polylines = footprint_outline(
            spec, self._picker.fov(), int(rect.width()), int(rect.height()), ring_tilt_degrees=self._picker.ring_tilt()
        )
        fill = QColor(color)
        fill.setAlpha(70 if hovered else 38)
        if len(polylines) == 1:
            points = [QPointF(rect.left() + u + 0.5, rect.top() + v + 0.5) for u, v in polylines[0]]
            if spec.is_polar:
                # The pole's outline runs the full width; close it along the
                # top/bottom edge so the cap itself is filled.
                edge_y = rect.top() if spec.ring == RING_POLE_UP else rect.bottom()
                points = points + [QPointF(points[-1].x(), edge_y), QPointF(points[0].x(), edge_y)]
            path = QPainterPath()
            path.addPolygon(QPolygonF(points))
            path.closeSubpath()
            painter.fillPath(path, fill)
        pen = QPen(color, 2.5 if hovered else 1.6)
        if hovered and spec.name not in self._picker.selected_names():
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        for line in polylines:
            painter.drawPolyline(QPolygonF([QPointF(rect.left() + u + 0.5, rect.top() + v + 0.5) for u, v in line]))

    def _paint_marker(self, painter: QPainter, spec: DirectionSpec, selected: bool, hovered: bool) -> None:
        center = self.marker_position(spec)
        color = RING_COLORS[spec.ring]
        radius = MARKER_RADIUS + (2 if hovered else 0)
        painter.setPen(QPen(QColor(0, 0, 0, 160), 3))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, radius, radius)
        painter.setPen(QPen(color, 2))
        painter.setBrush(color if selected else QColor(0, 0, 0, 90))
        painter.drawEllipse(center, radius, radius)
        if hovered:
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(QRectF(center.x() - 70, center.y() + radius + 1, 140, 14), Qt.AlignHCenter | Qt.AlignTop, spec.name)


class DirectionPicker(QWidget):
    """selection_changed fires with the selected names (catalog order)
    whenever the selection, ring tilt or field of view changes anything
    a caller would re-read."""

    selection_changed = Signal(list)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._selected: set[str] = set(CARDINAL_NAMES)
        self._fov = DEFAULT_FOV_DEGREES
        self._frame_count = 0
        self._panorama: np.ndarray | None = None
        self._background: QImage | None = None
        self._preview_cache: dict[tuple, QImage] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        presets_row = QHBoxLayout()
        presets_row.addWidget(QLabel("Presets:"))
        self.preset_buttons: dict[str, QPushButton] = {}
        for label, names in DIRECTION_PRESETS.items():
            btn = QPushButton(label)
            btn.setToolTip(", ".join(names))
            btn.clicked.connect(lambda _checked=False, n=names: self.set_selected(n))
            self.preset_buttons[label] = btn
            presets_row.addWidget(btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.set_selected([]))
        presets_row.addWidget(clear_btn)
        presets_row.addStretch()
        layout.addLayout(presets_row)

        map_row = QHBoxLayout()
        self.canvas = _PanoramaCanvas(self)
        self.canvas.toggled.connect(self.toggle)
        self.canvas.hovered.connect(self._on_hovered)
        map_row.addWidget(self.canvas, 1)

        side = QVBoxLayout()
        self.hover_preview = QLabel()
        self.hover_preview.setFixedSize(HOVER_PREVIEW_SIZE, HOVER_PREVIEW_SIZE)
        self.hover_preview.setAlignment(Qt.AlignCenter)
        self.hover_preview.setStyleSheet("border: 1px solid palette(mid); color: palette(mid);")
        self.hover_preview.setWordWrap(True)
        side.addWidget(self.hover_preview)
        self.hover_caption = QLabel()
        self.hover_caption.setAlignment(Qt.AlignHCenter)
        self.hover_caption.setFixedWidth(HOVER_PREVIEW_SIZE)
        side.addWidget(self.hover_caption)
        side.addStretch()
        map_row.addLayout(side)
        layout.addLayout(map_row)

        tilt_row = QHBoxLayout()
        tilt_row.addWidget(QLabel("Up/down ring tilt:"))
        self.tilt_spin = QDoubleSpinBox()
        self.tilt_spin.setRange(MIN_RING_TILT_DEGREES, MAX_RING_TILT_DEGREES)
        self.tilt_spin.setSingleStep(5.0)
        self.tilt_spin.setDecimals(0)
        self.tilt_spin.setSuffix("°")
        self.tilt_spin.setValue(DEFAULT_RING_TILT_DEGREES)
        self.tilt_spin.setToolTip(
            "How far the -down views look below the horizon (and the -up views above it). With a 90° field of "
            "view, 45° covers horizon-to-straight-down; a smaller tilt keeps more wall, a larger one more floor."
        )
        self.tilt_spin.valueChanged.connect(self._on_tilt_changed)
        tilt_row.addWidget(self.tilt_spin)
        legend = QLabel(
            '<span style="color:#4cc978">●</span> horizon &nbsp; '
            '<span style="color:#f2a62e">●</span> tilted down &nbsp; '
            '<span style="color:#54a0ff">●</span> tilted up &nbsp; '
            '<span style="color:#bebebe">●</span> straight up/down &nbsp; — click a marker to toggle it'
        )
        tilt_row.addWidget(legend)
        tilt_row.addStretch()
        layout.addLayout(tilt_row)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self._show_hover_placeholder()
        self._update_summary()

    # -- public API ---------------------------------------------------------------

    def selected_names(self) -> list[str]:
        return [name for name in ALL_DIRECTION_NAMES if name in self._selected]

    def set_selected(self, names) -> None:
        unknown = set(names) - set(ALL_DIRECTION_NAMES)
        if unknown:
            raise ValueError(f"unknown view direction(s): {sorted(unknown)}")
        new = set(names)
        if new != self._selected:
            self._selected = new
            self._changed()

    def toggle(self, name: str) -> None:
        if name in self._selected:
            self._selected.discard(name)
        else:
            self._selected.add(name)
        self._changed()

    def ring_tilt(self) -> float:
        return float(self.tilt_spin.value())

    def set_ring_tilt(self, degrees: float) -> None:
        self.tilt_spin.setValue(degrees)

    def fov(self) -> float:
        return self._fov

    def set_fov(self, degrees: float) -> None:
        if degrees != self._fov:
            self._fov = float(degrees)
            self._preview_cache.clear()
            self.canvas.update()

    def set_frame_count(self, count: int) -> None:
        self._frame_count = count
        self._update_summary()

    def set_background(self, path: Path | None) -> None:
        """Draws the given equirect image (a frame or its thumbnail) under
        the markers; None or an unreadable file falls back to a plain
        grid, never raises."""
        self._panorama = _load_panorama(Path(path)) if path is not None else None
        if self._panorama is None:
            self._background = None
        else:
            h, w, _ = self._panorama.shape
            self._background = QImage(self._panorama.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._preview_cache.clear()
        self.canvas.update()

    def background_qimage(self) -> QImage | None:
        return self._background

    # -- internals ----------------------------------------------------------------

    def _changed(self) -> None:
        self._update_summary()
        self.canvas.update()
        self.selection_changed.emit(self.selected_names())

    def _on_tilt_changed(self) -> None:
        self._preview_cache.clear()
        self.canvas.update()
        self.selection_changed.emit(self.selected_names())

    def _update_summary(self) -> None:
        count = len(self._selected)
        if count == 0:
            self.summary_label.setText("No views selected -- click a marker or pick a preset.")
            return
        text = f"{count} view{'s' if count != 1 else ''} per frame"
        if self._frame_count:
            text += f" × {self._frame_count} frames = {count * self._frame_count:,} images"
        if count > 8:
            text += " -- masking and pose-estimation time grow with the number of images."
        self.summary_label.setText(text)

    def _show_hover_placeholder(self) -> None:
        self.hover_preview.setPixmap(QPixmap())
        self.hover_preview.setText("Hover a marker to preview that view")
        self.hover_caption.setText("")

    def _on_hovered(self, name: str | None) -> None:
        if name is None:
            self._show_hover_placeholder()
            return
        spec = DIRECTIONS_BY_NAME[name]
        self.hover_caption.setText(
            f"{name}\nyaw {spec.yaw_degrees:g}°, pitch {spec.pitch_degrees(self.ring_tilt()):+g}°"
        )
        if self._panorama is None:
            self.hover_preview.setText("(no frame to preview)")
            return
        self.hover_preview.setPixmap(QPixmap.fromImage(self._hover_render(name)))

    def _hover_render(self, name: str) -> QImage:
        key = (name, self._fov, self.ring_tilt())
        if key not in self._preview_cache:
            face = direction_preset(HOVER_PREVIEW_SIZE, self._fov, [name], ring_tilt_degrees=self.ring_tilt())[0]
            rendered = np.ascontiguousarray(project_equirect_to_face(self._panorama, face))
            self._preview_cache[key] = QImage(
                rendered.data, HOVER_PREVIEW_SIZE, HOVER_PREVIEW_SIZE, 3 * HOVER_PREVIEW_SIZE, QImage.Format_RGB888
            ).copy()
        return self._preview_cache[key]
