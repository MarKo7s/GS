"""Qt pieces that are not plots: the settings form and the engine worker thread."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
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


def _int_box(key: str, low: int, high: int) -> QSpinBox:
    box = QSpinBox()
    box.setRange(low, high)
    box.setValue(int(DEFAULTS[key]))
    box.setFixedWidth(96)
    return box


def _float_box(key: str, low: float, high: float, decimals: int, step: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(float(DEFAULTS[key]))
    box.setFixedWidth(96)
    return box


def _menu(choices: tuple[str, ...]) -> QComboBox:
    """A menu that stays as wide as its labels, instead of stretching the column."""
    box = QComboBox()
    for name in choices:
        box.addItem(name, name)
    box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    box.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return box


def _indexed(specs):
    return {key: spec for key, *spec in ((item[0], *item[1:]) for item in specs)}


def build_settings_form() -> tuple[QWidget, dict]:
    """Settings form. Returned widgets are keyed by the settings dict."""
    host = QWidget()
    column = QVBoxLayout(host)
    column.setContentsMargins(0, 0, 0, 0)
    controls = {}
    ints = _indexed(_INTS)
    floats = _indexed(_FLOATS)

    top_host = QWidget()
    top = QFormLayout(top_host)
    top.setContentsMargins(0, 0, 0, 0)
    top.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
    engine_mode = _menu(("fibremodes", "fields"))
    top.addRow("Engine mode", engine_mode)
    controls["engine_mode"] = engine_mode

    masks = _int_box("number_masks", *ints["number_masks"][1:3])
    masks_row = QWidget()
    masks_layout = QHBoxLayout(masks_row)
    masks_layout.setContentsMargins(0, 0, 0, 0)
    masks_layout.addWidget(masks)
    top.addRow("Number of masks", masks_row)
    controls["number_masks"] = masks
    controls["number_masks_row"] = masks_row
    controls["number_masks_label"] = top.labelForField(masks_row)

    start = QSpinBox()
    end = QSpinBox()
    for box, key in ((start, "mode_range_start"), (end, "mode_range_end")):
        box.setRange(0, 100000)
        box.setValue(int(DEFAULTS[key]))
        box.setFixedWidth(96)
        controls[key] = box
    full = QCheckBox("full catalog")
    full.setChecked(True)
    controls["full_catalog"] = full
    between = QLabel("to")
    between.setAlignment(Qt.AlignmentFlag.AlignCenter)
    between.setFixedWidth(28)
    range_row = QWidget()
    range_layout = QHBoxLayout(range_row)
    range_layout.setContentsMargins(0, 0, 0, 0)
    range_layout.setSpacing(8)
    range_layout.addWidget(start)
    range_layout.addWidget(between)
    range_layout.addWidget(end)
    range_layout.addWidget(full)
    top.addRow("Mode range", range_row)
    controls["mode_range_row"] = range_row
    controls["mode_range_label"] = top.labelForField(range_row)
    _add_float(top, controls, floats, "wavelength", "Wavelength (um)")
    column.addWidget(top_host)

    def group(title: str) -> QFormLayout:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        column.addWidget(box)
        return form

    incoming = group("Input")
    _add_float(incoming, controls, floats, "fin", "fin (mm)")
    _add_float(incoming, controls, floats, "smf_in_NA", "SMF NA")
    _add_float(incoming, controls, floats, "smf_in_diameter", "SMF diameter (um)")
    waist = QDoubleSpinBox()
    waist.setRange(0.0, 1.0e7)
    waist.setDecimals(2)
    waist.setSingleStep(10.0)
    waist.setSpecialValueText("from SMF")
    waist.setValue(0.0)
    waist.setFixedWidth(96)
    incoming.addRow("Beam waist on SLM (um)", waist)
    controls["W0_slm"] = waist

    outgoing = group("Output")
    _add_float(outgoing, controls, floats, "fout", "fout (mm)")
    basis = _menu(("LG", "HG", "custom"))
    outgoing.addRow("Mode basis", basis)
    controls["mode_basis"] = basis
    _add_float(outgoing, controls, floats, "MMF_mfd", "MMF MFD (um)")
    _add_float(outgoing, controls, floats, "MMF_diameter", "MMF diameter (um)")
    _add_int(outgoing, controls, ints, "MMF_mode_groups", "Mode groups")

    slm = group("SLM")
    _add_int(slm, controls, ints, "masksize", "SLM pixels")
    _add_float(slm, controls, floats, "slm_pixel_size", "SLM pixel (um)")
    _add_int(slm, controls, ints, "downscale", "Downscale")
    _add_int(slm, controls, ints, "upsample", "Upsample")

    convergence = group("Convergence")
    _add_float(convergence, controls, floats, "goal_value", "Overlap goal")
    _add_float(convergence, controls, floats, "min_coupling_SNR_dB", "Min coupling SNR (dB)")
    _add_float(convergence, controls, floats, "minium_efficiency", "Minimum efficiency")
    _add_int(convergence, controls, ints, "MAX_iterations", "Max iterations")

    filtering_form = group("Filtering")
    filtering = QCheckBox("Filtering")
    filtering.setChecked(bool(DEFAULTS["filtering"]))
    filtering_form.addRow(filtering)
    controls["filtering"] = filtering
    _add_float(filtering_form, controls, floats, "filtering_factor", "Filtering factor")
    _add_float(filtering_form, controls, floats, "filtering_mask_radius_factor", "Forbidden radius factor")

    column.addStretch(1)
    return host, controls


def _add_int(form, controls, specs, key: str, label: str):
    low, high = specs[key][1:3]
    box = _int_box(key, low, high)
    form.addRow(label, box)
    controls[key] = box


def _add_float(form, controls, specs, key: str, label: str):
    low, high, decimals, step = specs[key][1:]
    box = _float_box(key, low, high, decimals, step)
    form.addRow(label, box)
    controls[key] = box


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
