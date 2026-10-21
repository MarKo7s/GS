"""Qt pieces that are not plots: the settings form and the engine worker thread."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from .utils import DEFAULTS

_FLOATS = (
    ("wavelength", "Wavelength (um)", 0.01, 5.0, 4, 0.0001),
    ("fin", "fin (mm)", 0.01, 1.0e4, 3, 0.1),
    ("fout", "fout (mm)", 0.01, 1.0e4, 3, 0.1),
    ("slm_pixel_size", "SLM pixel (um)", 0.1, 100.0, 3, 0.1),
    ("MMF_mfd", "MMF MFD (um)", 0.1, 1.0e4, 3, 0.1),
    ("MMF_diameter", "MMF diameter (um)", 0.1, 1.0e4, 3, 0.1),
    ("smf_in_NA", "SMF NA", 0.01, 1.0, 3, 0.01),
    ("smf_in_diameter", "SMF diameter (um)", 1.0, 100.0, 2, 0.1),
    ("goal_value", "Overlap goal", 0.0, 1.0, 3, 0.01),
    ("min_coupling_SNR_dB", "Min coupling SNR (dB)", 0.0, 80.0, 2, 1.0),
    ("minium_efficiency", "Minimum efficiency", 0.0, 1.0, 3, 0.01),
    ("filtering_factor", "Filtering factor", 0.0, 50.0, 3, 0.1),
    ("filtering_mask_radius_factor", "Forbidden radius factor", 0.1, 50.0, 3, 0.1),
)

_INTS = (
    ("masksize", "SLM pixels", 8, 8192),
    ("MMF_mode_groups", "Mode groups", 1, 200),
    ("convergence_check_interval", "Check interval", 1, 1000),
    ("downscale", "Downscale", 1, 32),
    ("upsample", "Upsample", 0, 16),
    ("MAX_iterations", "Max iterations", 1, 5000),
    ("number_masks", "Number of masks", 1, 100000),
)


class EngineWorker(QThread):
    """One thread for every GPU call so the window stays responsive.

    ``submit`` runs ``fn`` and stores its return value on ``result``. Restarted
    for each task; do not call ``submit`` while ``isRunning``.
    """

    failed = Signal(str)
    succeeded = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fn = None
        self.result = None

    def submit(self, fn):
        if self.isRunning():
            raise RuntimeError("engine worker is already running")
        self._fn = fn
        self.result = None
        self.start()

    def run(self):
        try:
            self.result = self._fn()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.succeeded.emit()


def build_settings_form() -> tuple[QWidget, dict]:
    """Settings form. Returned widgets are keyed by the settings dict."""
    host = QWidget()
    form = QFormLayout(host)
    controls = {}

    engine_mode = QComboBox()
    engine_mode.addItem("fibremodes", "fibremodes")
    engine_mode.addItem("fields", "fields")
    form.addRow("Engine mode", engine_mode)
    controls["engine_mode"] = engine_mode

    basis = QComboBox()
    for name in ("LG", "HG", "custom"):
        basis.addItem(name, name)
    form.addRow("Mode basis", basis)
    controls["mode_basis"] = basis

    masks = QSpinBox()
    masks.setRange(1, 100000)
    masks.setValue(int(DEFAULTS["number_masks"]))
    masks_row = QWidget()
    masks_layout = QHBoxLayout(masks_row)
    masks_layout.setContentsMargins(0, 0, 0, 0)
    masks_layout.addWidget(masks)
    form.addRow("Number of masks", masks_row)
    controls["number_masks"] = masks
    controls["number_masks_row"] = masks_row
    controls["number_masks_label"] = form.labelForField(masks_row)

    start = QSpinBox()
    end = QSpinBox()
    for box, key in ((start, "mode_range_start"), (end, "mode_range_end")):
        box.setRange(0, 100000)
        box.setValue(int(DEFAULTS[key]))
        controls[key] = box
    full = QCheckBox("full catalog")
    full.setChecked(True)
    controls["full_catalog"] = full
    range_row = QWidget()
    range_layout = QHBoxLayout(range_row)
    range_layout.setContentsMargins(0, 0, 0, 0)
    range_layout.addWidget(start)
    range_layout.addWidget(QLabel("to"))
    range_layout.addWidget(end)
    range_layout.addWidget(full)
    form.addRow("Mode range", range_row)
    controls["mode_range_row"] = range_row
    controls["mode_range_label"] = form.labelForField(range_row)

    for key, label, low, high in _INTS:
        if key == "number_masks":
            continue
        box = QSpinBox()
        box.setRange(low, high)
        box.setValue(int(DEFAULTS[key]))
        form.addRow(label, box)
        controls[key] = box

    for key, label, low, high, decimals, step in _FLOATS:
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setValue(float(DEFAULTS[key]))
        form.addRow(label, box)
        controls[key] = box

    waist = QDoubleSpinBox()
    waist.setRange(0.0, 1.0e7)
    waist.setDecimals(2)
    waist.setSingleStep(10.0)
    waist.setSpecialValueText("from SMF")
    waist.setValue(0.0)
    form.addRow("Beam waist on SLM (um)", waist)
    controls["W0_slm"] = waist

    filtering = QCheckBox("Filtering")
    filtering.setChecked(bool(DEFAULTS["filtering"]))
    form.addRow(filtering)
    controls["filtering"] = filtering

    return host, controls


def read_settings(controls: dict) -> dict:
    settings = {
        "engine_mode": controls["engine_mode"].currentData(),
        "mode_basis": controls["mode_basis"].currentData(),
        "full_catalog": controls["full_catalog"].isChecked(),
        "filtering": controls["filtering"].isChecked(),
    }
    for key in (
        "masksize",
        "MMF_mode_groups",
        "convergence_check_interval",
        "downscale",
        "upsample",
        "MAX_iterations",
        "number_masks",
        "mode_range_start",
        "mode_range_end",
    ):
        settings[key] = int(controls[key].value())
    for key in (
        "wavelength",
        "fin",
        "fout",
        "slm_pixel_size",
        "MMF_mfd",
        "MMF_diameter",
        "smf_in_NA",
        "smf_in_diameter",
        "W0_slm",
        "goal_value",
        "min_coupling_SNR_dB",
        "minium_efficiency",
        "filtering_factor",
        "filtering_mask_radius_factor",
    ):
        settings[key] = float(controls[key].value())
    return settings


def write_settings(controls: dict, settings: dict) -> None:
    widgets = [widget for widget in controls.values() if hasattr(widget, "blockSignals")]
    for widget in widgets:
        widget.blockSignals(True)
    try:
        _write_settings(controls, settings)
    finally:
        for widget in widgets:
            widget.blockSignals(False)


def _write_settings(controls: dict, settings: dict) -> None:
    _set_combo(controls["engine_mode"], settings["engine_mode"])
    _set_combo(controls["mode_basis"], settings["mode_basis"])
    controls["full_catalog"].setChecked(bool(settings["full_catalog"]))
    controls["filtering"].setChecked(bool(settings["filtering"]))
    for key in (
        "masksize",
        "MMF_mode_groups",
        "convergence_check_interval",
        "downscale",
        "upsample",
        "MAX_iterations",
        "number_masks",
        "mode_range_start",
        "mode_range_end",
    ):
        controls[key].setValue(int(settings[key]))
    for key in (
        "wavelength",
        "fin",
        "fout",
        "slm_pixel_size",
        "MMF_mfd",
        "MMF_diameter",
        "smf_in_NA",
        "smf_in_diameter",
        "W0_slm",
        "goal_value",
        "min_coupling_SNR_dB",
        "minium_efficiency",
        "filtering_factor",
        "filtering_mask_radius_factor",
    ):
        controls[key].setValue(float(settings[key]))


def _set_combo(box: QComboBox, value) -> None:
    index = box.findData(value)
    if index >= 0:
        box.setCurrentIndex(index)
