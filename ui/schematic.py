"""Side view of the bench: SMF and MMF beams drawn only as far as the SLM."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

_YELLOW = QColor(255, 220, 40)
_BLUE = QColor(80, 160, 255)
_AXIS = QColor(140, 140, 140)
_HARDWARE = QColor(230, 230, 230)


class OpticalBench(QWidget):
    """SMF to the SLM in yellow, MMF to the SLM in blue.

    Yellow half-height is ``fin * NA_SMF``. Blue half-height is the mode
    coverage radius, scaled by the current ``fout`` against the focal length
    the engine was built with. Both beams stop on the SLM.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fin_mm = 1.0
        self._fout_mm = 1.0
        self._na_smf = 0.1
        self._modes_radius_mm = None
        self._fout_ref_mm = None
        self.setFixedHeight(168)

    def set_beam(self, fin_mm: float, fout_mm: float, na_smf: float) -> None:
        self._fin_mm = max(float(fin_mm), 0.1)
        self._fout_mm = max(float(fout_mm), 0.1)
        self._na_smf = max(float(na_smf), 0.0)
        self.update()

    def set_modes_radius(self, radius_mm: float | None, fout_mm: float) -> None:
        """Mode footprint on the SLM, measured at ``fout_mm``."""
        self._modes_radius_mm = None if radius_mm is None else float(radius_mm)
        self._fout_ref_mm = float(fout_mm) if radius_mm is not None else None
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(8, 6, -8, -6)
        painter.fillRect(self.rect(), QColor(12, 12, 12))
        if rect.width() < 80 or rect.height() < 40:
            return

        cy = rect.center().y() + 4
        span = rect.height() * 0.30
        yellow_mm = self._fin_mm * self._na_smf
        blue_mm = self._blue_mm()
        scale = span / max(yellow_mm, blue_mm or 0.0, 1e-6)
        yellow = yellow_mm * scale
        blue = None if blue_mm is None else blue_mm * scale

        smf, lens1, slm, lens2, mmf = self._stations(rect)
        painter.setPen(QPen(_AXIS, 1))
        painter.drawLine(QPointF(smf, cy), QPointF(mmf, cy))

        self._beam(painter, _YELLOW, smf, lens1, slm, cy, yellow)
        if blue is not None:
            self._beam(painter, _BLUE, mmf, lens2, slm, cy, blue)

        self._fibre(painter, smf, cy, rect.bottom() - 16, 3.5, "SMF", -1)
        self._lens(painter, lens1, cy)
        self._slm(painter, slm, cy, rect.height() * 0.36)
        self._lens(painter, lens2, cy)
        self._fibre(painter, mmf, cy, rect.bottom() - 16, 6.0, "MMF", 1)
        self._gap_label(painter, lens1, slm, rect.bottom() - 16, f"fin {self._fin_mm:.1f} mm")
        self._gap_label(painter, slm, lens2, rect.bottom() - 16, f"fout {self._fout_mm:.1f} mm")

    def _blue_mm(self) -> float | None:
        if self._modes_radius_mm is None or not self._fout_ref_mm:
            return None
        return self._modes_radius_mm * (self._fout_mm / self._fout_ref_mm)

    def _stations(self, rect: QRectF) -> tuple[float, float, float, float, float]:
        """SMF, first lens, SLM, second lens, MMF. Gaps follow the focal lengths."""
        weights = (self._fin_mm, self._fin_mm, self._fout_mm, self._fout_mm)
        start = rect.left() + 24
        end = rect.right() - 24
        usable = max(end - start, 40.0)
        gaps = [usable * weight / sum(weights) for weight in weights]
        floor = 28.0
        if usable >= floor * len(gaps) and min(gaps) < floor:
            spare = usable - floor * len(gaps)
            extra = [max(0.0, gap - floor) for gap in gaps]
            extra_sum = sum(extra) or 1.0
            gaps = [floor + spare * piece / extra_sum for piece in extra]
        smf = start
        lens1 = smf + gaps[0]
        slm = lens1 + gaps[1]
        lens2 = slm + gaps[2]
        mmf = lens2 + gaps[3]
        return smf, lens1, slm, lens2, mmf

    def _beam(self, painter: QPainter, color: QColor, source: float, lens: float, slm: float, cy: float, half: float):
        fill = QColor(color)
        fill.setAlpha(55)
        path = QPainterPath()
        path.moveTo(source, cy)
        path.lineTo(lens, cy - half)
        path.lineTo(slm, cy - half)
        path.lineTo(slm, cy + half)
        path.lineTo(lens, cy + half)
        path.closeSubpath()
        painter.fillPath(path, fill)
        pen = QPen(color, 1.4)
        painter.setPen(pen)
        painter.drawLine(QPointF(source, cy), QPointF(lens, cy - half))
        painter.drawLine(QPointF(source, cy), QPointF(lens, cy + half))
        painter.drawLine(QPointF(lens, cy - half), QPointF(slm, cy - half))
        painter.drawLine(QPointF(lens, cy + half), QPointF(slm, cy + half))

    def _fibre(self, painter: QPainter, x: float, cy: float, label_top: float, radius: float, label: str, sign: int):
        painter.setPen(QPen(_HARDWARE, 1.5))
        painter.setBrush(_HARDWARE)
        painter.drawLine(QPointF(x, cy), QPointF(x + sign * 18, cy))
        painter.drawEllipse(QPointF(x, cy), radius, radius * 1.35)
        painter.setPen(_HARDWARE)
        painter.drawText(QRectF(x - 22, label_top, 44, 16), Qt.AlignmentFlag.AlignCenter, label)

    def _lens(self, painter: QPainter, x: float, cy: float):
        path = QPainterPath()
        path.moveTo(x, cy - 28)
        path.quadTo(x + 9, cy, x, cy + 28)
        path.quadTo(x - 9, cy, x, cy - 28)
        painter.setPen(QPen(_HARDWARE, 1.4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _slm(self, painter: QPainter, x: float, cy: float, half: float):
        rect = QRectF(x - 5, cy - half, 10, 2 * half)
        painter.setPen(QPen(_HARDWARE, 1.4))
        painter.setBrush(QColor(70, 70, 78))
        painter.drawRect(rect)
        painter.setPen(_HARDWARE)
        painter.drawText(QRectF(x - 24, cy - half - 16, 48, 14), Qt.AlignmentFlag.AlignCenter, "SLM")

    def _gap_label(self, painter: QPainter, left: float, right: float, top: float, text: str):
        painter.setPen(QColor(190, 190, 190))
        span = right - left
        if painter.fontMetrics().horizontalAdvance(text) > span - 4:
            text = text.split(" ", 1)[-1]
        painter.drawText(QRectF(left, top, span, 16), Qt.AlignmentFlag.AlignCenter, text)
