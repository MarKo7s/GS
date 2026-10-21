#!/usr/bin/env python3
"""Grab a GSWidget screenshot for the README (docs/images/gui.png).

Uses an in-memory stand-in for the engine, so this does not build modes or use a GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = REPO_ROOT / "docs" / "images" / "gui.png"


def demo_engine():
    optical = SimpleNamespace(
        wavelength=0.6328,
        fin=4.51e3,
        fout=18.0e3,
        masksize=512,
        slm_pixel_size=8.0,
        MMF_mfd=34.0,
        MMF_diameter=50.0,
        MMF_mode_groups=10,
        mode_basis="LG",
        smf_in_NA=0.1,
        smf_in_diameter=9.0,
        W0_slm=None,
    )
    mask = SimpleNamespace(
        goal_value=0.99,
        convergence_check_interval=1,
        min_coupling_SNR_dB=20.0,
        downscale=1,
        upsample=0,
        MAX_iterations=300,
        minium_efficiency=0.1,
        filtering=False,
        filtering_factor=2.0,
        filtering_mask_radius_factor=4.0,
    )
    count = 12
    steps = 8
    engine = SimpleNamespace(
        engine_mode="fibremodes",
        opticalsetup=optical,
        masksetup=mask,
        _options={"mode_range": None},
        _GS_ready=True,
        maskcount=count,
        illumination=SimpleNamespace(_calculated_W0=2200.0, mode="Custom Beam Waist"),
        efficiency=np.linspace(0.15, 0.85, count).astype(np.float32),
        iteration_convergence=np.full(count, steps - 1, dtype=np.int32),
        convergence_history=np.tile(np.linspace(0.2, 0.99, steps), (count, 1)).astype(np.float32),
        efficiency_history=np.tile(np.linspace(0.1, 0.7, steps), (count, 1)).astype(np.float32),
        basis_coupling_snr=np.linspace(12.0, 28.0, count).astype(np.float32),
        dumping_field=np.linspace(0.8, 0.55, count).astype(np.float32),
        core_field=np.linspace(0.18, 0.4, count).astype(np.float32),
        leaked_power=np.linspace(-0.02, 0.03, count).astype(np.float32),
        phase_masks=np.full((count, 8, 8), 0.2, dtype=np.float32),
        mask_passed=np.array([0, 1, 0, 2, 3, 4, 0, 5, 6, 0, 7, 1], dtype=np.uint8),
    )

    def get_gs_properties():
        return {
            "engine_mode": "fibremodes",
            "mode_basis": "LG",
            "gs_ready": True,
            "maskcount": count,
            "number_of_modes": 55,
            "mode_range": None,
            "slm_size_um": 4096.0,
            "artificial_slm_size_um": 4096.0,
            "spatial_resolution_per_um": 512 / 4096,
            "image_plane_samples": 512,
            "useful_slm_samples": 512,
            "slm_pixel_size_um": 8.0,
            "pixel_size_fft_um": 0.95,
            "max_spectral_distance_um": 243.0,
            "beam_waist_slm_um": 2200.0,
            "beam_waist_mode": "Custom Beam Waist",
            "core_diameter_um": 50.0,
            "core_radius_um": 25.0,
            "forbidden_radius_um": 100.0,
        }

    engine.get_gs_properties = get_gs_properties
    return engine


def build_window():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from gs.ui import GSWidget

    gui = GSWidget(demo_engine())
    gui.resize(1200, 980)
    gui.show()

    def teardown() -> None:
        gui.close()

    return gui, teardown


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture GUI screenshot for README.")
    parser.add_argument("-o", "--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--wait-ms", type=int, default=4000)
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Use QT_QPA_PLATFORM=offscreen (may blank pyqtgraph on Windows).",
    )
    args = parser.parse_args()

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import pyqtgraph as pg
    from PySide6.QtWidgets import QApplication

    pg.setConfigOptions(useOpenGL=False, antialias=False)

    app = QApplication.instance() or QApplication(sys.argv)
    window, teardown = build_window()
    window.show()
    window.raise_()
    window.activateWindow()

    deadline = time.perf_counter() + max(args.wait_ms, 200) / 1000.0
    while time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pixmap = window.grab()
    if pixmap.isNull() or pixmap.width() < 10:
        teardown()
        raise SystemExit("grab() returned an empty pixmap")
    if not pixmap.save(str(args.output), "PNG"):
        teardown()
        raise SystemExit(f"Failed to write {args.output}")

    teardown()
    print(f"Wrote {args.output}")
    app.quit()


if __name__ == "__main__":
    main()
