"""GSWidget is a front end for an existing GS engine. The engine stays free of Qt."""

from __future__ import annotations

import html
import time

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gs.logger import format_properties

from .helpers import EngineWorker, build_settings_form, read_settings, write_settings
from .utils import (
    apply_saved_masks,
    build_engine,
    load_engine,
    looks_run,
    read_npz_array,
    settings_from_engine,
)
from .schematic import OpticalBench
from .viewports import FibreSketch, MaskViewport, SLMSketch, SummaryPlots


class GSWidget(QWidget):
    """Create or accept a GS engine, edit its settings, run it, and inspect the masks.

    ``engine`` is the current instance. Regenerating replaces it and leaves the
    object that was passed in untouched. ``GS.run`` always goes through
    :class:`EngineWorker`, never the GUI thread.
    """

    def __init__(self, engine=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GS")
        self.resize(QSize(1285, 1272))
        self._engine = None
        self._coefs = None
        self._results_ready = False
        self._busy = False
        self._viewport = None
        self._on_ok = None
        self._worker = EngineWorker(self)
        self._worker.succeeded.connect(self._on_succeeded)
        self._worker.failed.connect(self._on_failed)
        self._build()
        if engine is not None:
            self._attach_engine(engine, has_results=looks_run(engine))

    @property
    def engine(self):
        return self._engine

    def _build(self):
        form_host, self._controls = build_settings_form()
        self._controls["engine_mode"].currentIndexChanged.connect(self._sync_mode_rows)
        self._controls["mode_basis"].currentIndexChanged.connect(self._sync_mode_rows)
        self._controls["full_catalog"].toggled.connect(self._sync_mode_rows)
        for key in ("fin", "fout", "smf_in_NA"):
            self._controls[key].valueChanged.connect(self._sync_bench)

        self._create = QPushButton("Create")
        self._create.clicked.connect(self._create_engine)
        self._load_coefs = QPushButton("Load coefs")
        self._load_coefs.clicked.connect(self._load_coefs_file)
        self._load_basis = QPushButton("Load basis")
        self._load_basis.clicked.connect(self._load_basis_file)
        self._run = QPushButton("Run")
        self._run.clicked.connect(self._run_engine)
        self._viewport_button = QPushButton("Mask viewport")
        self._viewport_button.clicked.connect(self._open_viewport)

        self._label = QLineEdit()
        self._label.setPlaceholderText("label (optional)")
        self._upload_masks = QCheckBox("upload masks")
        self._upload_masks.setChecked(True)
        self._save = QPushButton("Save")
        self._save.clicked.connect(self._save_library)
        self._load = QPushButton("Load")
        self._load.clicked.connect(self._load_library)
        self._status = QLabel("No engine")
        self._status.setWordWrap(True)

        self._settings_box = QGroupBox("Engine settings")
        settings_layout = QVBoxLayout(self._settings_box)
        settings_layout.addWidget(self._create)
        scroll = _SettingsScroll()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(form_host)
        form_width = form_host.sizeHint().width() + 16
        scroll.setMinimumWidth(form_width)
        self._settings_box.setMinimumWidth(form_width + 28)
        settings_layout.addWidget(scroll, stretch=1)

        self._library_box = QGroupBox("Save / Load")
        library = QHBoxLayout(self._library_box)
        library.addWidget(self._label, stretch=1)
        library.addWidget(self._upload_masks)
        library.addWidget(self._save)
        library.addWidget(self._load)

        self._engine_box = QGroupBox("Engine")
        engine_layout = QVBoxLayout(self._engine_box)
        engine_layout.addWidget(self._status)
        engine_layout.addWidget(self._run)
        file_row = QHBoxLayout()
        file_row.addWidget(self._load_coefs)
        file_row.addWidget(self._load_basis)
        engine_layout.addLayout(file_row)
        engine_layout.addWidget(self._viewport_button)

        self._properties = QTextEdit()
        self._properties.setReadOnly(True)
        self._show_properties("No engine")
        self._feasibility = QLabel("")
        self._feasibility.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._bench = OpticalBench()
        self._slm_sketch = SLMSketch()
        self._sketch = FibreSketch()
        self._summary = SummaryPlots()

        sketches = QHBoxLayout()
        sketches.setContentsMargins(0, 0, 0, 0)
        sketches.addWidget(self._slm_sketch, stretch=1)
        sketches.addWidget(self._sketch, stretch=1)

        plots = QVBoxLayout()
        plots.setContentsMargins(0, 0, 0, 0)
        plots.addWidget(self._feasibility)
        plots.addWidget(self._bench)
        plots.addLayout(sketches, stretch=1)

        side = QVBoxLayout()
        side.addWidget(self._properties, stretch=1)
        side.addLayout(plots, stretch=1)

        column = QVBoxLayout()
        column.addWidget(self._settings_box, stretch=1)
        column.addWidget(self._library_box)
        column.addWidget(self._engine_box)

        top = QHBoxLayout()
        top.addLayout(column, stretch=0)
        top.addLayout(side, stretch=1)

        layout = QVBoxLayout(self)
        layout.addLayout(top, stretch=2)
        layout.addWidget(self._summary, stretch=1)
        self._sync_mode_rows()
        self._sync_bench()
        self._sync_buttons()

    def _sync_mode_rows(self, *_args):
        fields = self._controls["engine_mode"].currentData() == "fields"
        self._show_row("number_masks_row", "number_masks_label", fields)
        self._show_row("mode_range_row", "mode_range_label", not fields)
        catalog = self._controls["full_catalog"].isChecked()
        self._controls["mode_range_start"].setEnabled(not catalog)
        self._controls["mode_range_end"].setEnabled(not catalog)
        self._sync_buttons()

    def _show_row(self, row_key: str, label_key: str, visible: bool):
        self._controls[row_key].setVisible(visible)
        label = self._controls.get(label_key)
        if label is not None:
            label.setVisible(visible)

    def _sync_buttons(self):
        busy = self._busy
        engine = self._engine
        ready = bool(engine is not None and getattr(engine, "_GS_ready", False))
        fields = self._controls["engine_mode"].currentData() == "fields"
        custom = self._controls["mode_basis"].currentData() == "custom"
        self._settings_box.setEnabled(not busy)
        self._load_coefs.setVisible(fields and engine is not None)
        self._load_basis.setVisible(custom)
        self._run.setEnabled(ready and not busy and (not fields or self._coefs is not None))
        self._viewport_button.setEnabled(self._results_ready and engine is not None and not busy)
        self._save.setEnabled(engine is not None and not busy and ready)
        self._load.setEnabled(not busy)
        self._create.setEnabled(not busy)
        self._create.setText("Regenerate engine" if engine is not None else "Create")

    def _set_busy(self, busy: bool):
        self._busy = busy
        self._sync_buttons()

    def _submit(self, fn, on_ok, status: str):
        if self._worker.isRunning():
            return
        self._on_ok = on_ok
        self._status.setText(status)
        self._set_busy(True)
        self._worker.submit(fn)

    def _on_succeeded(self):
        callback = self._on_ok
        result = self._worker.result
        self._on_ok = None
        self._set_busy(False)
        if callback is not None:
            callback(result)

    def _on_failed(self, message: str):
        self._on_ok = None
        self._set_busy(False)
        self._status.setText(message)

    def _create_engine(self):
        settings = read_settings(self._controls)
        self._submit(lambda: build_engine(settings), self._on_created, "Creating engine...")

    def _on_created(self, engine):
        self._coefs = None
        self._attach_engine(engine, has_results=False)
        if getattr(engine, "_GS_ready", False):
            self._status.setText("Engine ready")
        else:
            self._status.setText("Custom basis required before the engine can run")

    def _attach_engine(self, engine, has_results: bool):
        self._engine = engine
        self._results_ready = bool(has_results)
        write_settings(self._controls, settings_from_engine(engine))
        self._sync_mode_rows()
        self._refresh_properties()
        if has_results:
            self._summary.set_engine(engine)
        else:
            self._summary.set_engine(None)
        if self._viewport is not None:
            self._viewport.set_engine(engine if has_results else None)
        if getattr(engine, "_GS_ready", False):
            self._status.setText("Engine ready")
        else:
            self._status.setText("Custom basis required before the engine can run")
        self._sync_buttons()

    def _show_properties(self, text: str):
        color = self._properties.palette().color(self._properties.foregroundRole()).name()
        escaped = html.escape(text)
        escaped = escaped.replace("PASSED", '<span style="color:#3caf5a;">PASSED</span>')
        escaped = escaped.replace("FAIL", '<span style="color:#d23b3b;">FAIL</span>')
        self._properties.setHtml(
            f'<pre style="color:{color}; font-family: Consolas, monospace; white-space: pre-wrap;">{escaped}</pre>'
        )

    def _show_feasibility(self, properties: dict | None):
        feasible = None if not properties else properties.get("setup_feasible")
        if feasible is None:
            self._feasibility.setText("")
            return
        if feasible:
            self._feasibility.setText("SETUP FEASIBLE: PASSED")
            self._feasibility.setStyleSheet("color: #3caf5a; font-weight: bold;")
            return
        self._feasibility.setText("SETUP FEASIBLE: FAIL")
        self._feasibility.setStyleSheet("color: #d23b3b; font-weight: bold;")

    def _refresh_properties(self):
        engine = self._engine
        if engine is None or not hasattr(engine, "get_gs_properties"):
            self._show_properties("No engine")
            self._show_feasibility(None)
            self._sketch.set_properties(None)
            self._slm_sketch.set_properties(None)
            self._bench.set_modes_radius(None, 0.0)
            return
        properties = engine.get_gs_properties()
        illumination = getattr(getattr(engine, "illumination", None), "ILLUMINATION", None)
        self._show_properties(format_properties(properties))
        self._show_feasibility(properties)
        self._sketch.set_properties(properties)
        self._slm_sketch.set_properties(properties, illumination)
        self._bench.set_modes_radius(
            _modes_radius_mm(properties),
            float(self._controls["fout"].value()),
        )

    def _sync_bench(self, *_args):
        self._bench.set_beam(
            float(self._controls["fin"].value()),
            float(self._controls["fout"].value()),
            float(self._controls["smf_in_NA"].value()),
        )

    def _load_coefs_file(self):
        path, _selected = QFileDialog.getOpenFileName(self, "Load field coefs", "", "NumPy (*.npz)")
        if not path or self._engine is None:
            return
        try:
            coefs = read_npz_array(path, "coefs")
        except (OSError, KeyError, ValueError) as exc:
            self._status.setText(str(exc))
            return
        expected = self._engine.get_gs_properties().get("number_of_modes")
        if coefs.ndim != 2 or (expected is not None and coefs.shape[1] != int(expected)):
            self._status.setText(f"coefs shape {coefs.shape} does not match {expected} modes")
            return
        self._coefs = coefs
        self._status.setText(f"Loaded coefs {coefs.shape}")
        self._sync_buttons()

    def _load_basis_file(self):
        if self._engine is None:
            self._status.setText("Create the engine before loading a basis")
            return
        path, _selected = QFileDialog.getOpenFileName(self, "Load custom basis", "", "NumPy (*.npz)")
        if not path:
            return
        try:
            modes = read_npz_array(path, "modes")
        except (OSError, KeyError, ValueError) as exc:
            self._status.setText(str(exc))
            return
        samples = self._engine.get_gs_properties().get("image_plane_samples")
        if modes.ndim != 3 or modes.shape[1] != samples or modes.shape[2] != samples:
            self._status.setText(f"modes shape {modes.shape} does not match {samples} x {samples}")
            return
        engine = self._engine
        self._submit(lambda: engine.set_mode_basis(modes), self._on_basis, "Setting mode basis...")

    def _on_basis(self, _result):
        pending = getattr(self._engine, "_pending_masks", None)
        restored = bool(pending) and apply_saved_masks(self._engine, pending)
        if restored:
            self._engine._pending_masks = None
            self._results_ready = True
        self._refresh_properties()
        self._summary.set_engine(self._engine if restored else None)
        self._status.setText("Mode basis set")
        self._sync_buttons()

    def _run_engine(self):
        engine = self._engine
        coefs = None if engine.engine_mode != "fields" else self._coefs

        def _go():
            started = time.perf_counter()
            engine.run(inpucoefs=coefs, verbose=False)
            return time.perf_counter() - started

        self._submit(_go, self._on_ran, "Running...")

    def _on_ran(self, elapsed):
        self._results_ready = True
        self._summary.set_engine(self._engine)
        if self._viewport is not None:
            self._viewport.set_engine(self._engine)
        self._status.setText(f"Run finished | time {_format_duration(elapsed)}")
        self._sync_buttons()

    def _open_viewport(self):
        if self._engine is None:
            return
        if self._viewport is None:
            self._viewport = MaskViewport()
        self._viewport.set_engine(self._engine)
        self._viewport.show()
        self._viewport.raise_()
        self._viewport.activateWindow()

    def _save_library(self):
        if self._engine is None:
            return
        label = self._label.text().strip() or None
        try:
            folder = self._engine.save(label=label)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._status.setText(str(exc))
            return
        self._status.setText(f"Saved {folder}")

    def _load_library(self):
        path, _selected = QFileDialog.getOpenFileName(
            self, "Load GS settings", "", "Settings (settings.json);;JSON (*.json)"
        )
        if not path:
            return
        upload = self._upload_masks.isChecked()
        self._submit(lambda: load_engine(path, upload_masks=upload), self._on_loaded, "Loading library...")

    def _on_loaded(self, result):
        engine, restored = result
        self._coefs = None
        self._attach_engine(engine, has_results=restored)
        if restored:
            self._status.setText("Loaded settings and masks")
        elif getattr(engine, "_GS_ready", False):
            self._status.setText("Loaded settings")
        else:
            self._status.setText("Loaded settings. Custom basis is still required")

    def closeEvent(self, event):
        if self._worker.isRunning():
            event.ignore()
            self._status.setText("Wait for the current task to finish before closing.")
            return
        if self._viewport is not None:
            self._viewport.close()
        super().closeEvent(event)


class _SettingsScroll(QScrollArea):
    """The form scrolls inside the box. Its full height must not size the window."""

    def sizeHint(self):
        width = 300
        widget = self.widget()
        if widget is not None:
            width = max(width, widget.sizeHint().width())
        return QSize(width, 220)

    def minimumSizeHint(self):
        widget = self.widget()
        width = 440
        if widget is not None:
            width = max(width, widget.minimumSizeHint().width())
        return QSize(width, 80)


def _modes_radius_mm(properties: dict) -> float | None:
    """Mode coverage radius in millimetres, using the downscaled SLM pitch."""
    radius_px = properties.get("modes_coverage_rad_px_approx")
    pitch = properties.get("slm_pixel_size_um")
    downscale = properties.get("downscale")
    if radius_px is None or pitch is None or downscale is None:
        return None
    return float(radius_px) * float(pitch) * float(downscale) * 1e-3


def _format_duration(seconds: float) -> str:
    total = max(0.0, float(seconds))
    if total < 60.0:
        return f"{total:.2f} s"
    whole = int(round(total))
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} m {secs:02d} s"
    return f"{minutes} m {secs:02d} s"
