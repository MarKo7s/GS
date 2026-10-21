"""File log for GS runs. Does not import the engine."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_LOGGER_NAME = "gs"
_ready = False


def log_dir() -> Path:
    """Same application root as the mask libraries, under ``logger``."""
    program_data = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
    return Path(program_data) / "LAB" / "Algorithm" / "GS" / "logger"


def _passed(value) -> str:
    return "PASSED" if bool(value) else "FAIL"


def _percent(value) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.2f}"


def _pixels(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.0f} px"


def _feasibility_lines(properties: dict) -> list[str]:
    bar = "--------------------------------"
    if "setup_feasible" not in properties:
        return ["Feasibility setup integrity check: not run"]
    na = properties.get("NA_MMF_system_max")
    na_text = "" if na is None else f"{float(na):.2f}"
    return [
        bar,
        "Feasibility setup integrity check:",
        (
            "Illumination outside pupil power: "
            f"{_percent(properties.get('illumination_outside_pupil_power'))} % --> "
            f"{_passed(properties.get('illumination_pow_check_passed'))}"
        ),
        (
            "Modes outside aperture power: "
            f"{_percent(properties.get('mode_basis_power_outside_pupil'))} % --> "
            f"{_passed(properties.get('modes_pow_check_passed'))}"
        ),
        f"Illumination overlap support: {_passed(properties.get('illumination_overlap_support_passed'))}",
        f"Illumination coverage radius: {_pixels(properties.get('ill_coverage_rad_px_approx'))}",
        f"Modes coverage radius: {_pixels(properties.get('modes_coverage_rad_px_approx'))}",
        f"Setup feasible? {_passed(properties.get('setup_feasible'))}",
        f"This system support up to NA_MMF: {na_text}",
        bar,
    ]


def format_properties(properties: dict) -> str:
    """Text block shared by the log preamble and the widget property panel."""
    samples = properties.get("image_plane_samples")
    useful = properties.get("useful_slm_samples")
    waist_um = properties.get("beam_waist_slm_um")
    waist_mm = "" if waist_um is None else f"{float(waist_um) * 1e-3:g}"
    lines = [
        f"ENGINE MODE SELECTED: {properties.get('engine_mode')}",
        f"Mode basis: {properties.get('mode_basis')}",
        f"SLM size = {properties.get('slm_size_um')} um",
        f"Artificial SLM size = {properties.get('artificial_slm_size_um')} um",
        f"Spatial resolution {properties.get('spatial_resolution_per_um')} um-1",
        f"Image plane samples {samples} x {samples}",
        f"Useful SLM samples {useful} x {useful}",
        f"Pixel size FFT {properties.get('pixel_size_fft_um')} um",
        f"MAX spectral distance +- {properties.get('max_spectral_distance_um')} um",
        f"Beam waist in the SLM --> {waist_mm} mm , Calculated via: {properties.get('beam_waist_mode')}",
        f"Forbiden area r < {properties.get('forbidden_radius_um')} um",
        f"Core radius {properties.get('core_radius_um')} um",
        f"Core diameter {properties.get('core_diameter_um')} um",
        f"Mask count {properties.get('maskcount')}",
        f"Number of modes {properties.get('number_of_modes')}",
        f"MODES RANGE: {properties.get('mode_range')}",
        f"GS ready: {properties.get('gs_ready')}",
    ]
    lines.extend(_feasibility_lines(properties))
    return "\n".join(lines)


def get_logger() -> logging.Logger:
    global _ready
    logger = logging.getLogger(_LOGGER_NAME)
    if _ready:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        path = log_dir() / "gs.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as exc:
        _ready = True
        print(f"GS log file is unavailable: {exc}", file=sys.stderr)
        return logger
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    _ready = True
    return logger


def log_info(message: str) -> None:
    get_logger().info(message)


def log_mask_result(message: str, passed: int) -> None:
    logger = get_logger()
    if int(passed) == 0:
        logger.info(message)
    else:
        logger.warning(message)


def log_run_preamble(properties: dict) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar = "=" * 48
    log_info(f"\n{bar}\nGS run {stamp}\n{format_properties(properties)}\n{bar}")
