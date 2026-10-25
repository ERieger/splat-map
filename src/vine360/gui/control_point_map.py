"""Plan-view map of a control-point set for the Georeference panel
(docs/adr/0047), so check points can be chosen by where they are rather
than by name. Points are drawn in the set's local frame (CRS - origin),
north up, at equal scale on both axes: control points as filled
triangles, check points as hollow circles, ignored points as grey
crosses. The dashed outline is the convex hull of the control points: a
check point inside it tests the fit where it interpolates, one outside it
tests extrapolation. After a solve, each point's horizontal residual can
be overlaid as an arrow, exaggerated by a stated factor.

Clicking a point switches it between control and check; right-clicking
offers all three uses. The map only *requests* the change
(use_change_requested) -- the panel writes it to the index and reloads.

Ignored points are left out of the zoom (the far-away base station would
otherwise shrink every other point into one spot) and counted in the
legend if they fall off the map. No basemap: nothing here needs the
network or a GIS library.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QMenu, QSizePolicy, QToolTip, QWidget

from vine360.georef.control_points import ControlPoint

USE_COLORS = {
    "control": QColor(46, 160, 90),
    "check": QColor(70, 130, 230),
    "ignore": QColor(150, 150, 150),
}
SELECTED_COLOR = QColor(240, 170, 40)
RESIDUAL_COLOR = QColor(220, 70, 60)
MARKER_RADIUS = 7.0
HIT_RADIUS = 12.0
MARGIN = 36.0
RESIDUAL_ARROW_PX = 40.0  # the largest horizontal residual is drawn about this long


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Counter-clockwise hull (monotone chain); collinear points dropped.
    Fewer than three distinct points come back as they are."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def nice_length(target: float) -> float:
    """The largest 1/2/5 x 10^k not above target (for the scale bar)."""
    if target <= 0:
        return 1.0
    power = 10 ** math.floor(math.log10(target))
    for step in (5, 2, 1):
        if step * power <= target:
            return step * power
    return power


def _format_metres(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:g} km"
    if value >= 1:
        return f"{value:g} m"
    return f"{value * 100:g} cm"


class ControlPointMap(QWidget):
    use_change_requested = Signal(str, str)  # (point name, new use)
    point_selected = Signal(str)  # point name

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._points: list[ControlPoint] = []
        self._origin = (0.0, 0.0)
        self._residuals: dict[str, tuple[float, float]] = {}
        self._residual_label = ""
        self._selected: str | None = None
        self._hover: str | None = None

    def sizeHint(self) -> QSize:
        return QSize(420, 340)

    # -- data ----------------------------------------------------------------

    def set_points(self, points: list[ControlPoint], origin: tuple[float, ...]) -> None:
        self._points = list(points)
        self._origin = (float(origin[0]), float(origin[1]))
        names = {p.name for p in self._points}
        if self._selected not in names:
            self._selected = None
        self.update()

    def set_residuals(self, rows: list[dict] | None, label: str = "") -> None:
        """rows: a solution report's "points" (name, dE, dN). None clears."""
        self._residuals = {
            r["name"]: (float(r["dE"]), float(r["dN"])) for r in rows or [] if r.get("dE") is not None
        }
        self._residual_label = label if self._residuals else ""
        self.update()

    def set_selected(self, name: str | None) -> None:
        if name != self._selected:
            self._selected = name
            self.update()

    def selected(self) -> str | None:
        return self._selected

    # -- geometry ------------------------------------------------------------

    def _local(self, p: ControlPoint) -> tuple[float, float]:
        return p.east - self._origin[0], p.north - self._origin[1]

    def _fit_points(self) -> list[ControlPoint]:
        shown = [p for p in self._points if p.use != "ignore"]
        return shown or list(self._points)

    def _transform(self):
        """(metres -> pixels scale, centre E, centre N), equal on both axes."""
        fit = [self._local(p) for p in self._fit_points()]
        if not fit:
            return None
        es = [e for e, _ in fit]
        ns = [n for _, n in fit]
        span_e = max(max(es) - min(es), 1.0)
        span_n = max(max(ns) - min(ns), 1.0)
        width = max(self.width() - 2 * MARGIN, 10.0)
        height = max(self.height() - 2 * MARGIN - 18.0, 10.0)  # room for the legend line
        scale = min(width / span_e, height / span_n)
        return scale, (max(es) + min(es)) / 2, (max(ns) + min(ns)) / 2

    def to_pixel(self, p: ControlPoint) -> QPointF | None:
        transform = self._transform()
        if transform is None:
            return None
        scale, ce, cn = transform
        e, n = self._local(p)
        cx = self.width() / 2
        cy = (self.height() - 18.0) / 2
        return QPointF(cx + (e - ce) * scale, cy - (n - cn) * scale)

    def point_at(self, pos: QPointF) -> ControlPoint | None:
        best, best_d = None, HIT_RADIUS
        for p in self._points:
            pixel = self.to_pixel(p)
            if pixel is None:
                continue
            d = math.hypot(pixel.x() - pos.x(), pixel.y() - pos.y())
            if d <= best_d:
                best, best_d = p, d
        return best

    # -- painting ------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        palette = self.palette()
        painter.fillRect(self.rect(), palette.base())
        text_color = palette.text().color()
        if not self._points:
            painter.setPen(palette.mid().color())
            painter.drawText(self.rect(), Qt.AlignCenter, "No control points imported yet.")
            return
        transform = self._transform()
        scale = transform[0]
        area = QRectF(0, 0, self.width(), self.height() - 18.0)

        control = [p for p in self._points if p.use == "control"]
        hull = convex_hull([self._local(p) for p in control])
        if len(hull) >= 3:
            by_local = {self._local(p): p for p in control}
            polygon = QPolygonF([self.to_pixel(by_local[h]) for h in hull])
            fill = QColor(USE_COLORS["control"])
            fill.setAlpha(28)
            painter.setBrush(fill)
            painter.setPen(QPen(USE_COLORS["control"], 1.2, Qt.DashLine))
            painter.drawPolygon(polygon)

        # Residual arrows under the markers, exaggerated to a readable length.
        exaggeration = 0.0
        if self._residuals:
            largest = max(math.hypot(*v) for v in self._residuals.values())
            if largest > 0:
                exaggeration = RESIDUAL_ARROW_PX / (largest * scale)
            painter.setPen(QPen(RESIDUAL_COLOR, 1.6))
            for p in self._points:
                vector = self._residuals.get(p.name)
                pixel = self.to_pixel(p)
                if vector is None or not exaggeration:
                    continue
                tip = QPointF(pixel.x() + vector[0] * exaggeration * scale, pixel.y() - vector[1] * exaggeration * scale)
                self._draw_arrow(painter, pixel, tip)

        off_map = 0
        for p in self._points:
            pixel = self.to_pixel(p)
            if not area.adjusted(-2, -2, 2, 2).contains(pixel):
                off_map += 1
                continue
            self._draw_marker(painter, p, pixel)
            painter.setPen(text_color if p.use != "ignore" else palette.mid().color())
            painter.drawText(QPointF(pixel.x() + MARKER_RADIUS + 3, pixel.y() - MARKER_RADIUS + 2), p.name)

        self._draw_scale_bar(painter, scale, text_color)
        self._draw_north_arrow(painter, text_color)
        self._draw_legend(painter, text_color, off_map, exaggeration)

    def _draw_marker(self, painter: QPainter, p: ControlPoint, c: QPointF) -> None:
        color = USE_COLORS.get(p.use, USE_COLORS["ignore"])
        r = MARKER_RADIUS
        if p.name in (self._selected, self._hover):
            painter.setPen(QPen(SELECTED_COLOR, 2.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(c, r + 5, r + 5)
        untagged = p.tag_id is None
        pen = QPen(color, 2.0, Qt.DotLine if untagged else Qt.SolidLine)
        if p.use == "control":
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush if untagged else color)
            painter.drawPolygon(QPolygonF([
                QPointF(c.x(), c.y() - r), QPointF(c.x() + r * 0.95, c.y() + r * 0.7), QPointF(c.x() - r * 0.95, c.y() + r * 0.7),
            ]))
        elif p.use == "check":
            painter.setPen(QPen(color, 2.5, pen.style()))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(c, r * 0.85, r * 0.85)
            painter.setBrush(color)
            painter.drawEllipse(c, 1.5, 1.5)
        else:
            painter.setPen(QPen(color, 2.0))
            d = r * 0.7
            painter.drawLine(QPointF(c.x() - d, c.y() - d), QPointF(c.x() + d, c.y() + d))
            painter.drawLine(QPointF(c.x() - d, c.y() + d), QPointF(c.x() + d, c.y() - d))

    @staticmethod
    def _draw_arrow(painter: QPainter, start: QPointF, tip: QPointF) -> None:
        painter.drawLine(start, tip)
        angle = math.atan2(tip.y() - start.y(), tip.x() - start.x())
        if math.hypot(tip.x() - start.x(), tip.y() - start.y()) < 4:
            return
        head = QPainterPath(tip)
        for side in (-1, 1):
            a = angle + math.pi - side * 0.45
            head.moveTo(tip)
            head.lineTo(tip.x() + 7 * math.cos(a), tip.y() + 7 * math.sin(a))
        painter.drawPath(head)

    def _draw_scale_bar(self, painter: QPainter, scale: float, color: QColor) -> None:
        metres = nice_length(self.width() * 0.25 / scale)
        length = metres * scale
        y = self.height() - 30.0
        x0 = 12.0
        painter.setPen(QPen(color, 2.0))
        painter.drawLine(QPointF(x0, y), QPointF(x0 + length, y))
        painter.drawLine(QPointF(x0, y - 4), QPointF(x0, y + 4))
        painter.drawLine(QPointF(x0 + length, y - 4), QPointF(x0 + length, y + 4))
        painter.drawText(QPointF(x0 + length + 6, y + 4), _format_metres(metres))

    def _draw_north_arrow(self, painter: QPainter, color: QColor) -> None:
        x, y = self.width() - 20.0, 14.0
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([QPointF(x, y), QPointF(x + 6, y + 16), QPointF(x - 6, y + 16)]))
        painter.drawText(QRectF(x - 10, y + 17, 20, 14), Qt.AlignCenter, "N")

    def _draw_legend(self, painter: QPainter, color: QColor, off_map: int, exaggeration: float) -> None:
        counts = {use: sum(p.use == use for p in self._points) for use in USE_COLORS}
        parts = [f"▲ control {counts['control']}", f"○ check {counts['check']}", f"✕ ignored {counts['ignore']}"]
        if off_map:
            parts.append(f"{off_map} off map")
        if exaggeration:
            parts.append(f"residuals ×{exaggeration:.3g}" + (f" ({self._residual_label})" if self._residual_label else ""))
        painter.setPen(color)
        painter.drawText(QRectF(8, self.height() - 18, self.width() - 16, 16), Qt.AlignLeft | Qt.AlignVCenter,
                         "   ".join(parts))

    # -- interaction ---------------------------------------------------------

    def mouseMoveEvent(self, event) -> None:
        p = self.point_at(event.position())
        name = p.name if p else None
        if name != self._hover:
            self._hover = name
            self.update()
        if p is None:
            QToolTip.hideText()
            return
        e, n = self._local(p)
        lines = [f"{p.name}  ({p.use})", f"E {p.east:.3f}  N {p.north:.3f}  H {p.height:.3f}",
                 f"local {e:.2f}, {n:.2f} m",
                 "untagged -- can't be used" if p.tag_id is None else f"tag {p.tag_id}"]
        if p.name in self._residuals:
            de, dn = self._residuals[p.name]
            lines.append(f"horizontal residual {math.hypot(de, dn):.3f} m")
        if p.note:
            lines.append(p.note)
        QToolTip.showText(event.globalPosition().toPoint(), "\n".join(lines), self)

    def leaveEvent(self, _event) -> None:
        if self._hover is not None:
            self._hover = None
            self.update()

    def mousePressEvent(self, event) -> None:
        p = self.point_at(event.position())
        if p is None:
            return
        self.set_selected(p.name)
        self.point_selected.emit(p.name)
        if event.button() == Qt.LeftButton and p.use in ("control", "check"):
            self.use_change_requested.emit(p.name, "check" if p.use == "control" else "control")
        elif event.button() == Qt.RightButton:
            menu = QMenu(self)
            for use in ("control", "check", "ignore"):
                action = menu.addAction(use)
                action.setCheckable(True)
                action.setChecked(use == p.use)
                action.setData(use)
            chosen = menu.exec(event.globalPosition().toPoint())
            if chosen is not None and chosen.data() != p.use:
                self.use_change_requested.emit(p.name, chosen.data())
