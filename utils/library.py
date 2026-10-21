"""Write and reload the settings and masks of a GS run.

The engine is not imported here. ``save_library`` only reads attributes off
the object it is given, and ``settings_from_json`` returns the constructor
arguments for a new engine.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from ..toolbox.setup import GS_mask_setup, optical_setup
except ImportError:
    from toolbox.setup import GS_mask_setup, optical_setup

_OPTICAL_KEYS = (
    "wavelength",
    "fin",
    "fout",
    "masksize",
    "slm_pixel_size",
    "MMF_mfd",
    "MMF_diameter",
    "MMF_mode_groups",
    "mode_basis",
    "smf_in_NA",
    "smf_in_diameter",
    "W0_slm",
)

_MASK_KEYS = (
    "num_masks",
    "convergence_metric",
    "goal_value",
    "convergence_check_interval",
    "min_coupling_SNR_dB",
    "downscale",
    "upsample",
    "MAX_iterations",
    "minium_efficiency",
    "filtering",
    "filtering_factor",
    "filtering_mask_radius_factor",
)

_MASK_ARRAYS = (
    "phase_masks",
    "efficiency",
    "iteration_convergence",
    "convergence_history",
    "efficiency_history",
    "D_core_coefs",
    "mask_passed",
    "basis_coupling_snr",
    "dumping_field",
    "core_field",
    "leaked_power",
)

_SETTINGS_NAME = "settings.json"
_MASKS_NAME = "masks.npz"


def save_library(engine, folder=None, label=None) -> Path:
    """Write ``settings.json`` and ``masks.npz`` into one library folder.

    ``folder`` is the parent directory. When it is omitted, fibre-mode runs go
    under ``%PROGRAMDATA%\\LAB\\Algorithm\\GS\\modes_lib`` and field runs under
    ``field_lib``. The same settings always use the same folder name, so a
    later save overwrites that library.
    """
    label = _sanitize_label(label) if label else None
    parent = Path(folder) if folder is not None else _default_library_dir(engine.engine_mode)
    run_dir = parent / _folder_name(engine, label)
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = _settings_dict(engine, label)
    payload["meta"]["folder_name"] = run_dir.name
    (run_dir / _SETTINGS_NAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    arrays = {name: np.asarray(getattr(engine, name)) for name in _MASK_ARRAYS}
    pattern = getattr(engine, "_pattern_sequence_scan_array", None)
    if pattern is not None:
        arrays["pattern_sequence"] = np.asarray(pattern)
    np.savez(run_dir / _MASKS_NAME, **arrays)
    return run_dir


def settings_from_json(path):
    """Return ``(optical_setup, GS_mask_setup, engine_mode, options)``."""
    settings_path = _as_file(path, _SETTINGS_NAME)
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    optical = optical_setup(**payload["optical_setup"])
    mask = GS_mask_setup(**payload["mask_setup"])
    options = {key: _restore_option(key, value) for key, value in payload.get("options", {}).items()}
    return optical, mask, payload["engine_mode"], options


def load_masks(path) -> dict:
    """Load ``masks.npz`` from a library folder or a direct file path."""
    masks_path = _as_file(path, _MASKS_NAME)
    with np.load(masks_path) as data:
        return {key: data[key] for key in data.files}


def _default_library_dir(engine_mode: str) -> Path:
    program_data = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
    kind = "modes_lib" if engine_mode == "fibremodes" else "field_lib"
    return Path(program_data) / "LAB" / "Algorithm" / "GS" / kind


def _settings_dict(engine, label) -> dict:
    illumination = engine.illumination
    return {
        "label": label,
        "engine_mode": engine.engine_mode,
        "options": {key: _jsonable(value) for key, value in engine._options.items()},
        "optical_setup": _object_dict(engine.opticalsetup, _OPTICAL_KEYS),
        "mask_setup": _object_dict(engine.masksetup, _MASK_KEYS),
        "meta": {
            "maskcount": int(engine.maskcount),
            "w0_slm_um": _jsonable(illumination._calculated_W0),
            "illumination_mode": illumination.mode,
            "saved_at": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            "folder_name": None,
        },
    }


def _object_dict(obj, keys) -> dict:
    return {key: _jsonable(getattr(obj, key)) for key in keys}


def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot store {type(value).__name__} in settings.json")


def _restore_option(key, value):
    if key == "mode_range" and isinstance(value, list):
        return tuple(value)
    return value


def _folder_name(engine, label) -> str:
    optical = engine.opticalsetup
    mask = engine.masksetup
    prefix = "modes_lib" if engine.engine_mode == "fibremodes" else "field_lib"
    parts = [
        prefix,
        "fin", _fmt(optical.fin),
        "fout", _fmt(optical.fout),
        "wav", _fmt(optical.wavelength),
        "MFD", _fmt(optical.MMF_mfd),
        "num_masks", _fmt(engine.maskcount),
        "w0slm", _fmt(engine.illumination._calculated_W0),
        "basis", _fmt(optical.mode_basis),
        "filtering", _fmt(bool(mask.filtering)),
        "mask_radius_factor", _fmt(mask.filtering_mask_radius_factor),
    ]
    name = "_".join(parts)
    if label:
        name = f"{label}_{name}"
    return name


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:.6g}"
    return str(value)


def _sanitize_label(label: str) -> str:
    cleaned = str(label).strip().replace(" ", "_")
    cleaned = re.sub(r'[\\/:*?"<>|]+', "", cleaned)
    if not cleaned:
        raise ValueError("label is empty after removing unsupported characters")
    return cleaned


def _as_file(path, filename: str) -> Path:
    path = Path(path)
    if path.is_dir():
        path = path / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
