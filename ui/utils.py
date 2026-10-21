"""Qt-free helpers for building a GS engine and reading a finished run."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gs import GS, GS_mask_setup, load_masks, optical_setup, settings_from_json

PASS_LABELS = {
    0: "passed",
    1: "overlap",
    2: "SNR",
    3: "overlap + SNR",
    4: "gradient",
    5: "gradient + overlap",
    6: "gradient + SNR",
    7: "all",
}

# Vertical fail-line colors. Code 0 draws no line.
PASS_COLORS = {
    1: (255, 80, 80),
    2: (255, 150, 40),
    3: (230, 200, 50),
    4: (50, 200, 110),
    5: (170, 90, 255),
    6: (255, 70, 170),
    7: (0, 210, 210),
}

_RESULT_ARRAYS = (
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

OPTICAL_KEYS = (
    "wavelength",
    "fin",
    "fout",
    "masksize",
    "slm_pixel_size",
    "MMF_mfd",
    "MMF_diameter",
    "MMF_mode_groups",
    "smf_in_NA",
    "smf_in_diameter",
)

DEFAULTS = {
    "engine_mode": "fibremodes",
    "mode_basis": "LG",
    "number_masks": 2,
    "full_catalog": True,
    "mode_range_start": 0,
    "mode_range_end": 20,
    "wavelength": 0.6328,
    "fin": 4.51,
    "fout": 18.0,
    "masksize": 512,
    "slm_pixel_size": 8.0,
    "MMF_mfd": 34.0,
    "MMF_diameter": 50.0,
    "MMF_mode_groups": 10,
    "smf_in_NA": 0.1,
    "smf_in_diameter": 9.0,
    "W0_slm": 0.0,
    "goal_value": 0.99,
    "convergence_check_interval": 1,
    "min_coupling_SNR_dB": 20.0,
    "downscale": 1,
    "upsample": 0,
    "MAX_iterations": 300,
    "minium_efficiency": 0.1,
    "filtering": False,
    "filtering_factor": 2.0,
    "filtering_mask_radius_factor": 4.0,
}


def settings_from_engine(engine) -> dict:
    optical = engine.opticalsetup
    mask = engine.masksetup
    mode_range = engine._options.get("mode_range")
    number_masks = engine._options.get("number_masks")
    if number_masks is None:
        number_masks = getattr(engine, "maskcount", None) or DEFAULTS["number_masks"]
    settings = dict(DEFAULTS)
    settings.update(
        {
            "engine_mode": engine.engine_mode,
            "mode_basis": optical.mode_basis,
            "number_masks": int(number_masks),
            "full_catalog": mode_range is None,
            "mode_range_start": DEFAULTS["mode_range_start"] if not mode_range else int(mode_range[0]),
            "mode_range_end": DEFAULTS["mode_range_end"] if not mode_range else int(mode_range[1]),
            "W0_slm": 0.0 if optical.W0_slm is None else float(optical.W0_slm),
            "goal_value": float(mask.goal_value),
            "convergence_check_interval": int(mask.convergence_check_interval),
            "min_coupling_SNR_dB": float(mask.min_coupling_SNR_dB),
            "downscale": int(mask.downscale),
            "upsample": int(mask.upsample),
            "MAX_iterations": int(mask.MAX_iterations),
            "minium_efficiency": float(mask.minium_efficiency),
            "filtering": bool(mask.filtering),
            "filtering_factor": float(mask.filtering_factor),
            "filtering_mask_radius_factor": float(mask.filtering_mask_radius_factor),
        }
    )
    for key in OPTICAL_KEYS:
        settings[key] = getattr(optical, key)
    # The engine stores focal lengths in um. The form shows mm.
    settings["fin"] = float(optical.fin) * 1e-3
    settings["fout"] = float(optical.fout) * 1e-3
    return settings


def build_engine(settings: dict) -> GS:
    """Build a new engine. Convergence is always overlap."""
    waist = settings.get("W0_slm")
    if waist is None or float(waist) == 0.0:
        waist = None
    optical = optical_setup(
        wavelength=float(settings["wavelength"]),
        fin=float(settings["fin"]) * 1e3,
        fout=float(settings["fout"]) * 1e3,
        masksize=int(settings["masksize"]),
        slm_pixel_size=float(settings["slm_pixel_size"]),
        MMF_mfd=float(settings["MMF_mfd"]),
        MMF_diameter=float(settings["MMF_diameter"]),
        MMF_mode_groups=int(settings["MMF_mode_groups"]),
        mode_basis=settings["mode_basis"],
        smf_in_NA=float(settings["smf_in_NA"]),
        smf_in_diameter=float(settings["smf_in_diameter"]),
        W0_slm=waist,
    )
    fields = settings["engine_mode"] == "fields"
    mask = GS_mask_setup(
        num_masks=int(settings["number_masks"]) if fields else None,
        convergence_metric="overlap",
        goal_value=float(settings["goal_value"]),
        convergence_check_interval=int(settings["convergence_check_interval"]),
        min_coupling_SNR_dB=float(settings["min_coupling_SNR_dB"]),
        downscale=int(settings["downscale"]),
        upsample=int(settings["upsample"]),
        MAX_iterations=int(settings["MAX_iterations"]),
        minium_efficiency=float(settings["minium_efficiency"]),
        filtering=bool(settings["filtering"]),
        filtering_factor=float(settings["filtering_factor"]),
        filtering_mask_radius_factor=float(settings["filtering_mask_radius_factor"]),
    )
    if fields:
        return GS(optical, mask, engine_mode="fields", number_masks=int(settings["number_masks"]))
    mode_range = None
    if not settings.get("full_catalog", True):
        mode_range = (int(settings["mode_range_start"]), int(settings["mode_range_end"]))
    return GS(optical, mask, engine_mode="fibremodes", mode_range=mode_range)


def load_engine(path):
    """Rebuild an engine from ``settings.json`` and copy ``masks.npz`` when shapes match.

    Returns ``(engine, results_restored)``. A custom basis is not in the JSON, so the
    mask arrays wait on ``engine._pending_masks`` until ``set_mode_basis``.
    """
    optical, mask, engine_mode, options = settings_from_json(path)
    engine = GS(optical, mask, engine_mode, **options)
    folder = Path(path)
    if folder.is_file():
        folder = folder.parent
    masks_file = folder / "masks.npz"
    if not masks_file.is_file():
        return engine, False
    arrays = load_masks(folder)
    if apply_saved_masks(engine, arrays):
        return engine, True
    engine._pending_masks = arrays
    return engine, False


def apply_saved_masks(engine, arrays: dict) -> bool:
    """Copy saved arrays onto ``engine`` where the shapes already match."""
    if not hasattr(engine, "phase_masks"):
        return False
    wrote_phase = False
    for name in _RESULT_ARRAYS:
        if name not in arrays or not hasattr(engine, name):
            continue
        source = np.asarray(arrays[name])
        destination = getattr(engine, name)
        if source.shape != np.shape(destination):
            continue
        destination[...] = source
        wrote_phase = wrote_phase or name == "phase_masks"
    return wrote_phase


def read_npz_array(path, key: str) -> np.ndarray:
    with np.load(path) as data:
        if key not in data.files:
            raise KeyError(f"{path} has no array '{key}'")
        return np.asarray(data[key])


def pass_label(code) -> str:
    code = int(code)
    return f"{code}: {PASS_LABELS.get(code, 'unknown')}"


def circle_xy(radius: float, count: int = 360):
    theta = np.linspace(0.0, 2.0 * np.pi, count)
    radius = float(radius)
    return radius * np.cos(theta), radius * np.sin(theta)


def slm_complex_field(illumination, phase) -> np.ndarray:
    """Illumination times ``exp(1j * phase)``, embedding a smaller phase into the grid."""
    carrier = np.asarray(illumination)
    phase = np.asarray(phase, dtype=np.float64)
    if carrier.shape == phase.shape:
        return carrier * np.exp(1j * phase)
    if phase.shape[0] > carrier.shape[0] or phase.shape[1] > carrier.shape[1]:
        raise ValueError(
            f"phase mask {phase.shape} is larger than the illumination {carrier.shape}"
        )
    plane = np.zeros(carrier.shape, dtype=np.float64)
    y = (carrier.shape[0] - phase.shape[0]) // 2
    x = (carrier.shape[1] - phase.shape[1]) // 2
    plane[y : y + phase.shape[0], x : x + phase.shape[1]] = phase
    return carrier * np.exp(1j * plane)


def fibre_plane(field_slm, pixel_um: float, radius_um: float, margin: float = 1.2):
    """FFT to the fibre face and crop to the forbidden radius plus a margin.

    Returns ``(crop, coords_um)``. ``coords_um`` are the sample centers of the crop.
    The sample at ``kx = ky = 0`` is the fftshift origin (index ``count // 2``),
    the same sample the mode grid uses, so that coordinate is exactly zero.
    """
    field = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(np.asarray(field_slm))))
    count = field.shape[0]
    pixel_um = float(pixel_um)
    half = min(count // 2, max(1, int(np.ceil(float(radius_um) * margin / pixel_um))))
    center = count // 2
    crop = field[center - half : center + half, center - half : center + half]
    coords = (np.arange(crop.shape[0]) - half) * pixel_um
    return crop, coords


def mask_histories(engine, index: int):
    """Convergence and efficiency up to and including the stored stop iteration."""
    stop = int(engine.iteration_convergence[index])
    last = engine.convergence_history.shape[1] - 1
    stop = max(0, min(stop, last))
    convergence = np.asarray(engine.convergence_history[index, : stop + 1], dtype=float)
    efficiency = np.asarray(engine.efficiency_history[index, : stop + 1], dtype=float)
    return convergence, efficiency


def _column(engine, name: str, count: int) -> np.ndarray:
    values = getattr(engine, name, None)
    if values is None:
        return np.zeros(count, dtype=float)
    return np.asarray(values[:count], dtype=float)


def looks_run(engine) -> bool:
    """True when a finished run has left something other than the allocated zeros."""
    for name in ("efficiency", "convergence_history", "iteration_convergence", "phase_masks"):
        values = getattr(engine, name, None)
        if values is not None and np.any(np.asarray(values)):
            return True
    return False


def summary_series(engine) -> dict:
    """Per-mask curves for the two main-widget plots."""
    count = int(getattr(engine, "maskcount", 0) or 0)
    stops = _column(engine, "iteration_convergence", count).astype(int)
    history = getattr(engine, "convergence_history", None)
    if history is None or count == 0:
        overlap = np.zeros(count, dtype=float)
    else:
        last = history.shape[1] - 1
        stops = np.clip(stops, 0, last)
        overlap = np.asarray(history[np.arange(count), stops], dtype=float)
    return {
        "x": np.arange(count),
        "efficiency": _column(engine, "efficiency", count),
        "overlap": overlap,
        "snr": _column(engine, "basis_coupling_snr", count),
        "dumping_percent": _column(engine, "dumping_field", count) * 100.0,
        "core_percent": _column(engine, "core_field", count) * 100.0,
        "leak_percent": _column(engine, "leaked_power", count) * 100.0,
        "passed": _column(engine, "mask_passed", count).astype(int),
    }
