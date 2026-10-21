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
    """Both beams are traced back from their coverage radii on the SLM.

    The SLM bar is the aperture. A radius larger than the aperture sticks out
    of the bar. ``NA_SMF eq`` is the illumination radius over ``fin``, and
    ``NA_MMF eq`` is the mode radius over ``fout``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fin_mm = 1.0
        self._fout_mm = 1.0
        self._ill_radius_mm = None
        self._modes_radius_mm = None
        self._pupil_mm = None
        self._fin_ref_mm = None
        self._fout_ref_mm = None
        self.setFixedHeight(168)

    def set_beam(self, fin_mm: float, fout_mm: float) -> None:
        self._fin_mm = max(float(fin_mm), 0.1)
        self._fout_mm = max(float(fout_mm), 0.1)
        self.update()

    def set_footprints(
        self,
        ill_mm: float | None,
        modes_mm: float | None,
        pupil_mm: float | None,
        fin_mm: float,
        fout_mm: float,
    ) -> None:
        """Coverage radii and the aperture, measured at these focal lengths."""
        self._ill_radius_mm = None if ill_mm is None else float(ill_mm)
        self._modes_radius_mm = None if modes_mm is None else float(modes_mm)
        self._pupil_mm = None if not pupil_mm else float(pupil_mm)
        self._fin_ref_mm = float(fin_mm) if ill_mm is not None else None
        self._fout_ref_mm = float(fout_mm) if modes_mm is not None else None
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(12, 12, 12))
        painter.setPen(QColor(210, 210, 210))
        painter.drawText(QRectF(0, 2, self.width(), 16), Qt.AlignmentFlag.AlignCenter, "Optical Schematic")
        rect = self.rect().adjusted(8, 20, -8, -6)
        if rect.width() < 80 or rect.height() < 40:
            return

        cy = rect.center().y() + 2
        smf, lens1, slm, lens2, mmf = self._stations(rect)
        bar = rect.height() * 0.30
        yellow = self._half_px(self._scaled(self._ill_radius_mm, self._fin_mm, self._fin_ref_mm), bar, rect)
        blue = self._half_px(self._scaled(self._modes_radius_mm, self._fout_mm, self._fout_ref_mm), bar, rect)
        na_smf = self._equivalent_na(self._ill_radius_mm, self._fin_ref_mm)
        na_mmf = self._equivalent_na(self._modes_radius_mm, self._fout_ref_mm)

        painter.setPen(QPen(_AXIS, 1))
        painter.drawLine(QPointF(smf, cy), QPointF(mmf, cy))
        if yellow is not None:
            self._beam(painter, _YELLOW, smf, lens1, slm, cy, yellow)
        if blue is not None:
            self._beam(painter, _BLUE, mmf, lens2, slm, cy, blue)
        self._lens(painter, lens1, cy, bar)
        self._lens(painter, lens2, cy, bar)
        self._bar(painter, slm, cy, bar, "SLM")

        self._fibre(painter, smf, cy, rect.bottom() - 16, 3.5, "SMF", -1)
        self._fibre(painter, mmf, cy, rect.bottom() - 16, 6.0, "MMF", 1)
        self._apex_label(painter, lens1, rect.bottom() - 16, f"fin {self._fin_mm:.1f} mm")
        self._apex_label(painter, lens2, rect.bottom() - 16, f"fout {self._fout_mm:.1f} mm")
        if na_smf is not None:
            self._na_label(painter, (smf + lens1) / 2, rect.top(), f"NA_SMF eq {na_smf:.3f}", _YELLOW)
        if na_mmf is not None:
            self._na_label(painter, (lens2 + mmf) / 2, rect.top(), f"NA_MMF eq {na_mmf:.3f}", _BLUE)

    def _scaled(self, radius_mm: float | None, focal_mm: float, focal_ref_mm: float | None) -> float | None:
        if radius_mm is None or not focal_ref_mm:
            return None
        return radius_mm * (focal_mm / focal_ref_mm)

    def _equivalent_na(self, radius_mm: float | None, focal_mm: float | None) -> float | None:
        if radius_mm is None or not focal_mm:
            return None
        return radius_mm / focal_mm

    def _half_px(self, radius_mm: float | None, bar: float, rect: QRectF) -> float | None:
        """Map a radius onto the aperture bar. Larger than the pupil sticks out."""
        if radius_mm is None or not self._pupil_mm:
            return None
        return min(bar * (radius_mm / self._pupil_mm), rect.height() * 0.44)

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

    def _lens(self, painter: QPainter, x: float, cy: float, half: float):
        bulge = max(6.0, half * 9.0 / 28.0)
        path = QPainterPath()
        path.moveTo(x, cy - half)
        path.quadTo(x + bulge, cy, x, cy + half)
        path.quadTo(x - bulge, cy, x, cy - half)
        painter.setPen(QPen(_HARDWARE, 1.6))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _bar(self, painter: QPainter, x: float, cy: float, half: float, label: str | None = None):
        rect = QRectF(x - 4, cy - half, 8, 2 * half)
        fill = QColor(70, 70, 78, 90)
        painter.setPen(QPen(_HARDWARE, 1.6))
        painter.setBrush(fill)
        painter.drawRect(rect)
        if label is None:
            return
        painter.setPen(_HARDWARE)
        painter.drawText(QRectF(x - 24, cy - half - 16, 48, 14), Qt.AlignmentFlag.AlignCenter, label)

    def _apex_label(self, painter: QPainter, x: float, top: float, text: str):
        painter.setPen(QColor(190, 190, 190))
        painter.drawText(QRectF(x - 52, top, 104, 16), Qt.AlignmentFlag.AlignCenter, text)

    def _na_label(self, painter: QPainter, x: float, top: float, text: str, color: QColor):
        painter.setPen(color)
        painter.drawText(QRectF(x - 58, top, 116, 16), Qt.AlignmentFlag.AlignCenter, text)
