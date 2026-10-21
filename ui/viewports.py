"""Pyqtgraph views. These widgets do not build or run the engine."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGraphicsProxyWidget,
    QHBoxLayout,
    QLabel,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from visualization import complexToRGB

from .utils import (
    PASS_COLORS,
    PASS_LABELS,
    circle_xy,
    fibre_plane,
    mask_histories,
    pass_label,
    slm_complex_field,
    summary_series,
)

pg.setConfigOptions(useOpenGL=False, antialias=False)


class TwinPlot(pg.PlotWidget):
    """Left axis plus a right-hand viewbox that shares the x axis."""

    def __init__(self, title="", parent=None):
        super().__init__(parent=parent, title=title)
        self.showAxis("right")
        self.right = pg.ViewBox()
        self.scene().addItem(self.right)
        self.getAxis("right").linkToView(self.right)
        self.right.setXLink(self)
        self.getViewBox().sigResized.connect(self._sync_right)
        self._sync_right()

    def _sync_right(self):
        self.right.setGeometry(self.getViewBox().sceneBoundingRect())
        self.right.linkedViewChanged(self.getViewBox(), self.right.XAxis)

    def add_right(self, item):
        self.right.addItem(item)
        self._sync_right()


class _SquarePlotHost(QWidget):
    """Centers a square plot so a locked range stays put when the window is resized."""

    def __init__(self, plot: pg.PlotWidget, parent=None):
        super().__init__(parent)
        self._plot = plot
        self._limit = None
        plot.setParent(self)
        plot.setMinimumSize(0, 0)
        self.setMinimumSize(80, 80)

    def set_limit(self, limit):
        self._limit = None if limit is None else float(limit)
        self._apply_range()

    def _apply_range(self):
        if self._limit is None:
            return
        limit = self._limit
        self._plot.setRange(xRange=(-limit, limit), yRange=(-limit, limit), padding=0.0)

    def resizeEvent(self, event):
        side = max(1, min(self.width(), self.height()))
        self._plot.setGeometry((self.width() - side) // 2, (self.height() - side) // 2, side, side)
        self._apply_range()
        super().resizeEvent(event)


class FibreSketch(QWidget):
    """Core and forbidden-region circles in micrometres, with no field."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot = pg.PlotWidget(title="Fibre plane")
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "x", units="um")
        self._plot.setLabel("left", "y", units="um")
        self._plot.showGrid(x=False, y=False)
        self._plot.enableAutoRange(False)
        self._core = pg.PlotCurveItem(pen=pg.mkPen((80, 200, 120), width=1.5))
        self._forbidden = pg.PlotCurveItem(pen=pg.mkPen((220, 60, 60), width=1.5))
        self._spectral = pg.PlotCurveItem(pen=pg.mkPen((180, 180, 180), width=1))
        self._core_text = pg.TextItem(color=(80, 200, 120), anchor=(1.0, 0.5))
        self._forbidden_text = pg.TextItem(color=(220, 60, 60), anchor=(0.5, 0.0))
        self._pixel_text = pg.TextItem(color="w", anchor=(0.5, 1.0))
        self._spectral_text = pg.TextItem(color=(180, 180, 180), anchor=(0.0, 1.0))
        for item in (
            self._spectral,
            self._spectral_text,
            self._core,
            self._forbidden,
            self._core_text,
            self._forbidden_text,
            self._pixel_text,
        ):
            self._plot.addItem(item)
        self._host = _SquarePlotHost(self._plot)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._host)

    def set_properties(self, properties: dict | None):
        if not properties:
            for curve in (self._core, self._forbidden, self._spectral):
                curve.setData([], [])
            for text in (self._core_text, self._forbidden_text, self._pixel_text, self._spectral_text):
                text.setText("")
            self._host.set_limit(None)
            return
        core = float(properties["core_radius_um"])
        forbidden = float(properties["forbidden_radius_um"])
        spectral = float(properties["max_spectral_distance_um"])
        pixel = float(properties["pixel_size_fft_um"])
        _draw_apertures(
            self._core,
            self._forbidden,
            self._core_text,
            self._forbidden_text,
            core,
            forbidden,
            properties["core_diameter_um"],
        )
        self._pixel_text.setText(f"pixel {pixel:.4g} um")
        self._pixel_text.setPos(0.0, forbidden)
        self._spectral.setData(*_square(spectral))
        self._spectral_text.setText(f"max spectral ± {spectral:.3g} um")
        self._spectral_text.setPos(spectral, spectral)
        self._host.set_limit(max(forbidden, core, pixel) * 1.7)


class SLMSketch(QWidget):
    """Illumination on the SLM, in millimetres, with the waist and coverage circles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._plot = pg.PlotWidget(title="SLM plane")
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "x", units="mm")
        self._plot.setLabel("left", "y", units="mm")
        self._plot.showGrid(x=False, y=False)
        self._plot.enableAutoRange(False)
        self._image = pg.ImageItem(axisOrder="row-major")
        self._waist = pg.PlotCurveItem(pen=pg.mkPen((255, 255, 255), width=1.5))
        self._illumination = pg.PlotCurveItem(pen=pg.mkPen((255, 220, 40), width=1.5))
        self._modes = pg.PlotCurveItem(pen=pg.mkPen((80, 160, 255), width=1.5))
        self._aperture = pg.PlotCurveItem(pen=pg.mkPen((190, 190, 190), width=1.5))
        self._waist_text = pg.TextItem(color="w", anchor=(0.5, 0.5))
        self._pixel_text = pg.TextItem(color="w", anchor=(0.5, 1.0))
        self._illumination_text = pg.TextItem(color=(255, 220, 40), anchor=(1.0, 0.5))
        self._modes_text = pg.TextItem(color=(80, 160, 255), anchor=(1.0, 0.5))
        self._aperture_text = pg.TextItem(color=(190, 190, 190), anchor=(0.5, 0.0))
        for item in (
            self._image,
            self._aperture,
            self._illumination,
            self._modes,
            self._waist,
            self._waist_text,
            self._pixel_text,
            self._illumination_text,
            self._modes_text,
            self._aperture_text,
        ):
            self._plot.addItem(item)
        self._host = _SquarePlotHost(self._plot)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._host)

    def set_properties(self, properties: dict | None, illumination=None):
        self._image.clear()
        for curve in (self._waist, self._illumination, self._modes, self._aperture):
            curve.setData([], [])
        for text in (
            self._waist_text,
            self._pixel_text,
            self._illumination_text,
            self._modes_text,
            self._aperture_text,
        ):
            text.setText("")
        if not properties:
            self._host.set_limit(None)
            return
        limits = []
        waist_um = properties.get("beam_waist_slm_um")
        if waist_um is not None:
            waist_mm = float(waist_um) * 1e-3
            self._waist.setData(*circle_xy(waist_mm))
            self._waist_text.setText(f"w0 {waist_mm:.3g} mm")
            self._waist_text.setPos(0.0, 0.0)
            limits.append(abs(waist_mm))
        pixel = properties.get("slm_pixel_size_um")
        if pixel is not None and waist_um is not None:
            self._pixel_text.setText(f"pixel {float(pixel):.4g} um")
            self._pixel_text.setPos(0.0, float(waist_um) * 1e-3)
        length_um = properties.get("artificial_slm_size_um")
        samples = properties.get("image_plane_samples")
        if illumination is not None and length_um is not None:
            width_mm = float(length_um) * 1e-3
            self._image.setImage(_rgb(illumination), autoLevels=False, levels=(0, 255))
            self._image.setRect(QRectF(-0.5 * width_mm, -0.5 * width_mm, width_mm, width_mm))
        ill_mm = _coverage_mm(properties.get("ill_coverage_rad_px_approx"), length_um, samples)
        modes_mm = _coverage_mm(properties.get("modes_coverage_rad_px_approx"), length_um, samples)
        _place_circle(self._illumination, self._illumination_text, ill_mm, "", 0.0, 0.0)
        _place_circle(self._modes, self._modes_text, modes_mm, "", 0.0, 0.0)
        pupil_um = properties.get("pupil_radius_um")
        pupil_mm = None if pupil_um is None else float(pupil_um) * 1e-3
        _place_circle(self._aperture, self._aperture_text, pupil_mm, "", 0.0, 0.0)
        for radius in (ill_mm, modes_mm, pupil_mm):
            if radius is not None:
                limits.append(abs(radius))
        limit = None if not limits else max(limits) * 1.45
        self._host.set_limit(limit)
        labels = []
        if ill_mm is not None:
            labels.append((self._illumination_text, "illumination"))
        if modes_mm is not None:
            labels.append((self._modes_text, "modes"))
        if pupil_mm is not None:
            labels.append((self._aperture_text, "aperture"))
        if limit is not None:
            x = -0.92 * limit
            for index, (text, label) in enumerate(labels):
                text.setAnchor((0.0, 0.5))
                text.setText(label)
                text.setPos(x, (0.90 - 0.16 * index) * limit)


class SummaryPlots(QWidget):
    """The two run-wide plots, each in its own tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._marks = []
        self._show_failures = True
        self._min_efficiency = None

        self._goal = pg.PlotWidget(title="Efficiency and overlap")
        self._goal.setLabel("bottom", "mask")
        self._goal.setLabel("left", "efficiency / overlap")
        self._goal.addLegend(offset=(-12, 12))
        self._efficiency = self._goal.plot(**_sample_style((255, 255, 255), "efficiency"))
        self._overlap = self._goal.plot(**_sample_style((80, 180, 255), "overlap"))
        self._error_legend = pg.LegendItem(offset=(12, 12))
        self._error_legend.setParentItem(self._goal.getViewBox())
        for code in range(1, 8):
            sample = pg.PlotDataItem(pen=pg.mkPen(PASS_COLORS[code], width=2))
            self._error_legend.addItem(sample, f"{code}  {PASS_LABELS[code]}")
        self._min_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen((255, 255, 255, 170), width=1.5),
            movable=False,
        )
        self._min_label = pg.TextItem(color=(255, 255, 255, 210), anchor=(0.5, 1.0))
        self._goal.addItem(self._min_line)
        # Sitting on the line, not in the data bounds. Auto must ignore it so a
        # zoom does not stretch the axis up to the text.
        self._goal.addItem(self._min_label, ignoreBounds=True)
        self._min_line.setVisible(False)
        self._goal.getViewBox().sigRangeChanged.connect(self._place_min_label)
        self._failures_box = QCheckBox("failures")
        self._failures_box.setChecked(True)
        self._failures_box.setStyleSheet(
            "QCheckBox { color: rgb(230, 230, 230); background-color: rgba(0, 0, 0, 150);"
            " padding: 1px 4px; }"
        )
        self._failures_proxy = QGraphicsProxyWidget()
        self._failures_proxy.setWidget(self._failures_box)
        self._failures_proxy.setZValue(1000)
        self._failures_proxy.setParentItem(self._goal.plotItem)
        self._failures_box.toggled.connect(self._toggle_failures)
        self._goal.getViewBox().sigResized.connect(self._place_failures_box)

        self._power = TwinPlot(title="Power and SNR")
        self._power.setLabel("bottom", "mask")
        self._power.setLabel("left", "percent")
        self._power.setLabel("right", "SNR", units="dB")
        self._power.addLegend()
        self._dumping = self._power.plot(**_sample_style((255, 160, 40), "dumping %"))
        self._core = self._power.plot(**_sample_style((80, 200, 120), "core %"))
        self._leak = self._power.plot(**_sample_style((220, 80, 80), "core leakage %"))
        self._snr = pg.PlotCurveItem(**_sample_style((255, 220, 60), "SNR (dB)"))
        self._power.add_right(self._snr)
        self._power.plotItem.legend.addItem(self._snr, "SNR (dB)")

        self._plots = (self._goal, self._power)
        tabs = QTabWidget()
        tabs.addTab(self._goal, "Efficiency and overlap")
        tabs.addTab(self._power, "Power and SNR")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tabs)

    def showEvent(self, event):
        super().showEvent(event)
        self._place_failures_box()

    def set_engine(self, engine):
        series = None if engine is None else summary_series(engine)
        minimum = None
        if engine is not None and hasattr(engine, "masksetup"):
            minimum = float(engine.masksetup.minium_efficiency)
        self._min_efficiency = minimum
        self._clear_marks()
        self._error_legend.setVisible(self._show_failures)
        curves = (self._efficiency, self._overlap, self._dumping, self._core, self._leak, self._snr)
        if series is None or len(series["x"]) == 0:
            for curve in curves:
                curve.setData([], [])
            for plot in self._plots:
                _set_integer_x(plot, [])
            self._apply_minimum_line(None, None)
            return
        x = series["x"]
        self._efficiency.setData(x, series["efficiency"])
        self._overlap.setData(x, series["overlap"])
        self._dumping.setData(x, series["dumping_percent"])
        self._core.setData(x, series["core_percent"])
        self._leak.setData(x, series["leak_percent"])
        self._snr.setData(x, series["snr"])
        for plot in self._plots:
            _set_integer_x(plot, x)
            plot.enableAutoRange(axis=pg.ViewBox.YAxis)
        self._power.right.enableAutoRange(axis=pg.ViewBox.YAxis)
        self._power._sync_right()
        self._mark_failures(x, series["passed"])
        self._apply_minimum_line(series["efficiency"], series["overlap"])

    def _apply_minimum_line(self, efficiency, overlap):
        value = self._min_efficiency
        if value is None:
            self._min_line.setVisible(False)
            self._min_label.setText("")
            return
        self._min_line.setPos(value)
        self._min_line.setVisible(True)
        self._min_label.setText(f"min efficiency {value:.3g}")
        if efficiency is None or overlap is None or len(efficiency) == 0:
            self._goal.setYRange(min(0.0, value), max(1.0, value), padding=0.05)
            self._place_min_label()
            return
        samples = np.concatenate((np.asarray(efficiency, dtype=float), np.asarray(overlap, dtype=float), [value]))
        low = float(np.min(samples))
        high = float(np.max(samples))
        if low == high:
            low -= 0.05
            high += 0.05
        pad = 0.08 * (high - low)
        self._goal.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)
        self._goal.setYRange(low - pad, high + pad, padding=0.0)
        self._place_min_label()

    def _place_min_label(self, *_args):
        if self._min_efficiency is None:
            return
        (x0, x1), _yrange = self._goal.getViewBox().viewRange()
        self._min_label.setPos(0.5 * (x0 + x1), self._min_efficiency)

    def _place_failures_box(self, *_args):
        view = self._goal.getViewBox()
        rect = view.sceneBoundingRect()
        if rect.width() < 1 or rect.height() < 1:
            return
        origin = self._goal.plotItem.mapFromScene(rect.bottomLeft())
        height = self._failures_proxy.size().height()
        self._failures_proxy.setPos(origin.x() + 8, origin.y() - height - 8)

    def _toggle_failures(self, checked: bool):
        self._show_failures = bool(checked)
        self._error_legend.setVisible(self._show_failures)
        for line in self._marks:
            line.setVisible(self._show_failures)

    def _clear_marks(self):
        for line in self._marks:
            self._goal.removeItem(line)
        self._marks.clear()

    def _mark_failures(self, x, codes):
        for mask, code in zip(x, codes):
            color = PASS_COLORS.get(int(code))
            if color is None:
                continue
            line = pg.InfiniteLine(
                pos=float(mask),
                angle=90,
                pen=pg.mkPen(color, width=1.5),
                movable=False,
            )
            line.setVisible(self._show_failures)
            self._goal.addItem(line)
            self._marks.append(line)


class ImagePane(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._plot = pg.PlotWidget(title=title)
        self._plot.setAspectLocked(True)
        self._image = pg.ImageItem(axisOrder="row-major")
        self._plot.addItem(self._image)
        self._core = pg.PlotCurveItem(pen=pg.mkPen((80, 200, 120), width=1.5))
        self._forbidden = pg.PlotCurveItem(pen=pg.mkPen((220, 60, 60), width=1.5))
        self._spectral = pg.PlotCurveItem(pen=pg.mkPen((180, 180, 180), width=1))
        self._core_text = pg.TextItem(color=(80, 200, 120), anchor=(1.0, 0.5))
        self._forbidden_text = pg.TextItem(color=(220, 60, 60), anchor=(0.5, 0.0))
        self._spectral_text = pg.TextItem(color=(180, 180, 180), anchor=(0.0, 1.0))
        for item in (
            self._spectral,
            self._spectral_text,
            self._core,
            self._forbidden,
            self._core_text,
            self._forbidden_text,
        ):
            self._plot.addItem(item)
        self._host = _SquarePlotHost(self._plot)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._host)

    def set_rgb(self, image, rect=None):
        self._image.setImage(np.asarray(image), autoLevels=False, levels=(0, 255))
        if rect is not None:
            self._image.setRect(rect)

    def set_scalar(self, image, rect=None):
        self._image.setImage(np.asarray(image), autoLevels=True)
        if rect is not None:
            self._image.setRect(rect)

    def set_circles(self, core_um, forbidden_um, spectral_um=None):
        if core_um is None:
            for curve in (self._core, self._forbidden, self._spectral):
                curve.setData([], [])
            for text in (self._core_text, self._forbidden_text, self._spectral_text):
                text.setText("")
            return
        _draw_apertures(
            self._core,
            self._forbidden,
            self._core_text,
            self._forbidden_text,
            core_um,
            forbidden_um,
            2.0 * float(core_um),
        )
        if spectral_um is None:
            self._spectral.setData([], [])
            self._spectral_text.setText("")
            return
        spectral = float(spectral_um)
        self._spectral.setData(*_square(spectral))
        self._spectral_text.setText(f"max spectral ± {spectral:.3g} um")
        self._spectral_text.setPos(spectral, spectral)

    def set_range(self, limit):
        self._plot.enableAutoRange(False)
        self._host.set_limit(limit)


class MaskViewport(QWidget):
    """One mask: SLM field, fibre field, fibre intensity, and its convergence."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Mask viewport")
        self._engine = None
        self.resize(1100, 720)

        self._index = QLabel("Mask 0")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._show_mask)

        self._slm = ImagePane("SLM field")
        self._fibre = ImagePane("Fibre field")
        self._intensity = ImagePane("Fibre 10 log10 |E|^2")
        images = QHBoxLayout()
        images.addWidget(self._slm)
        images.addWidget(self._fibre)
        images.addWidget(self._intensity)

        self._history = TwinPlot(title="Convergence")
        self._history.setLabel("bottom", "iteration")
        self._history.setLabel("left", "overlap / efficiency")
        self._history.setLabel("right", "gradient")
        self._history.addLegend()
        self._efficiency = self._history.plot(pen=pg.mkPen("w", width=2), name="efficiency")
        self._convergence = self._history.plot(pen=pg.mkPen((80, 180, 255), width=2), name="overlap")
        self._gradient = pg.PlotCurveItem(pen=pg.mkPen((80, 200, 120), width=1.5), name="gradient")
        self._gradient_neg = pg.ScatterPlotItem(
            pen=None,
            brush=pg.mkBrush(220, 60, 60),
            size=7,
            name="gradient < 0",
        )
        self._gradient_zero = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            pen=pg.mkPen((255, 255, 255, 70), width=1),
            movable=False,
        )
        self._history.add_right(self._gradient)
        self._history.add_right(self._gradient_neg)
        self._history.add_right(self._gradient_zero)
        self._history.plotItem.legend.addItem(self._gradient, "gradient")
        self._history.plotItem.legend.addItem(self._gradient_neg, "gradient < 0")
        self._history.enableAutoRange(False)
        self._history.setYRange(0.0, 1.0, padding=0.0)
        self._gradient_range = (-0.05, 0.05)
        self._history.right.disableAutoRange(axis=pg.ViewBox.YAxis)
        self._history.right.setYRange(*self._gradient_range, padding=0.0)

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(self._index)
        header.addWidget(self._slider, stretch=1)
        layout.addLayout(header)
        layout.addLayout(images, stretch=3)
        layout.addWidget(self._history, stretch=2)

    def set_engine(self, engine):
        self._engine = engine
        count = 0
        if engine is not None and hasattr(engine, "phase_masks"):
            count = int(len(engine.phase_masks))
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, count - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._gradient_range = _gradient_limits(engine)
        self._show_mask(0)

    def _show_mask(self, index: int):
        engine = self._engine
        if engine is None or not hasattr(engine, "phase_masks") or len(engine.phase_masks) == 0:
            self._index.setText("No masks")
            return
        index = int(index)
        self._index.setText(f"Mask {index} / {len(engine.phase_masks) - 1}")
        properties = engine.get_gs_properties()
        phase = engine.phase_masks[index]
        slm = slm_complex_field(engine.get_illumination(), phase)
        self._slm.set_rgb(_rgb(slm))
        self._slm.set_circles(None, None, None)

        fibre, coords = fibre_plane(
            slm,
            properties["pixel_size_fft_um"],
            properties["forbidden_radius_um"],
        )
        rect = _rect(coords, properties["pixel_size_fft_um"])
        self._fibre.set_rgb(_rgb(fibre), rect)
        power = np.abs(fibre) ** 2
        decibel = 10.0 * np.log10(np.maximum(power, 1e-12))
        self._intensity.set_scalar(decibel, rect)
        core = float(properties["core_radius_um"])
        forbidden = float(properties["forbidden_radius_um"])
        spectral = float(properties["max_spectral_distance_um"])
        for pane in (self._fibre, self._intensity):
            pane.set_circles(core, forbidden, spectral)
            pane.set_range(float(coords[-1]) + float(properties["pixel_size_fft_um"]))

        convergence, efficiency = mask_histories(engine, index)
        iterations = np.arange(convergence.size)
        gradient = np.gradient(convergence) if convergence.size else np.asarray([])
        self._convergence.setData(iterations, convergence)
        self._efficiency.setData(iterations, efficiency)
        self._gradient.setData(iterations, gradient)
        negative = gradient < 0
        self._gradient_neg.setData(iterations[negative], gradient[negative])
        last = int(iterations[-1]) if iterations.size else 1
        self._history.setXRange(0, max(last, 1), padding=0.02)
        self._history.setYRange(0.0, 1.0, padding=0.0)
        self._history.right.disableAutoRange(axis=pg.ViewBox.YAxis)
        self._history.right.setYRange(*self._gradient_range, padding=0.0)
        self._history._sync_right()

        stop = int(engine.iteration_convergence[index])
        overlap = float(convergence[-1]) if convergence.size else 0.0
        self._history.setTitle(
            f"Mask {index} | efficiency {float(engine.efficiency[index]):.3f}"
            f" | overlap {overlap:.3f} | {pass_label(engine.mask_passed[index])}"
            f" | stop {stop}"
        )


def _gradient_limits(engine) -> tuple[float, float]:
    """Min and max of each mask's overlap gradient, up to its stop iteration."""
    history = None if engine is None else getattr(engine, "convergence_history", None)
    stops = None if engine is None else getattr(engine, "iteration_convergence", None)
    if history is None or stops is None:
        return -0.05, 0.05
    history = np.asarray(history, dtype=float)
    stops = np.asarray(stops, dtype=int)
    if history.ndim != 2 or stops.size == 0 or history.shape[0] == 0:
        return -0.05, 0.05
    last = history.shape[1] - 1
    pieces = []
    for index in range(min(history.shape[0], stops.shape[0])):
        stop = int(np.clip(stops[index], 0, last))
        segment = history[index, : stop + 1]
        if segment.size == 0:
            continue
        pieces.append(np.gradient(segment))
    if not pieces:
        return -0.05, 0.05
    values = np.concatenate(pieces)
    low = float(np.min(values))
    high = float(np.max(values))
    if low == high:
        pad = 0.05 if low == 0.0 else 0.05 * abs(low)
        return low - pad, high + pad
    pad = 0.05 * (high - low)
    return low - pad, high + pad


def _coverage_mm(radius_px, length_um, samples):
    if radius_px is None or length_um is None or not samples:
        return None
    return float(radius_px) * float(length_um) / float(samples) * 1e-3


def _place_circle(curve, text, radius_mm, label, x, y):
    if radius_mm is None:
        curve.setData([], [])
        text.setText("")
        return
    curve.setData(*circle_xy(float(radius_mm)))
    text.setText(label)
    text.setPos(float(x), float(y))


def _rgb(field) -> np.ndarray:
    field = np.asarray(field)
    if field.size == 0 or np.max(np.abs(field)) == 0:
        return np.zeros(field.shape + (3,), dtype=np.uint8)
    return np.clip(complexToRGB(field) * 255.0, 0, 255).astype(np.uint8)


def _sample_style(color, name):
    """Opaque points joined by a translucent line, one point per mask."""
    red, green, blue = color
    return {
        "pen": pg.mkPen((red, green, blue, 70), width=2),
        "symbol": "o",
        "symbolSize": 7,
        "symbolBrush": color,
        "symbolPen": None,
        "name": name,
    }


def _set_integer_x(plot, x):
    values = np.asarray(x, dtype=int)
    if values.size == 0:
        plot.getAxis("bottom").setTicks(None)
        return
    step = max(1, int(np.ceil(values.size / 10)))
    shown = list(values[::step])
    if shown[-1] != int(values[-1]):
        shown.append(int(values[-1]))
    plot.getAxis("bottom").setTicks([[(int(value), str(int(value))) for value in shown]])
    plot.setXRange(float(values[0]) - 0.5, float(values[-1]) + 0.5, padding=0)


def _square(half):
    half = float(half)
    return (
        np.array([half, half, -half, -half, half]),
        np.array([half, -half, -half, half, half]),
    )


def _draw_apertures(core_curve, forbidden_curve, core_text, forbidden_text, core_radius, forbidden_radius, core_diameter):
    core_radius = float(core_radius)
    forbidden_radius = float(forbidden_radius)
    core_curve.setData(*circle_xy(core_radius))
    forbidden_curve.setData(*circle_xy(forbidden_radius))
    core_text.setText(f"core d {float(core_diameter):.3g} um")
    forbidden_text.setText(f"non-care d {2.0 * forbidden_radius:.3g} um")
    core_text.setPos(-core_radius, 0.0)
    forbidden_text.setPos(0.0, -forbidden_radius)


def _rect(coords, pixel_um: float) -> QRectF:
    origin = float(coords[0] - pixel_um / 2.0)
    span = float(coords[-1] - coords[0] + pixel_um)
    return QRectF(origin, origin, span, span)
