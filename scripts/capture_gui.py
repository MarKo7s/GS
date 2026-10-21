#!/usr/bin/env python3
"""Grab README screenshots from a finished fibre-mode library.

Builds a short LG library on the GPU, runs it, then grabs the main window
and the mask viewport. This uses the engine, not a stand-in.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_GUI = REPO_ROOT / "docs" / "images" / "gui.png"
OUT_VIEWPORT = REPO_ROOT / "docs" / "images" / "mask_viewport.png"


def run_mode_library():
    """A few LG modes on the double-clad bench, enough for real curves and fields."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from gs import GS, GS_mask_setup, optical_setup

    optical = optical_setup(
        wavelength=0.8435,
        fin=20e3,
        fout=4.7e3,
        masksize=960,
        slm_pixel_size=9.2,
        MMF_mfd=5.34,
        MMF_diameter=23,
        MMF_mode_groups=10,
        W0_slm=2.2e3,
        mode_basis="LG",
    )
    masksetup = GS_mask_setup(
        convergence_metric="overlap",
        goal_value=0.99,
        minium_efficiency=0.1,
        downscale=2,
        upsample=0,
        MAX_iterations=80,
        filtering=True,
        filtering_factor=0.5,
        filtering_mask_radius_factor=3,
    )
    engine = GS(optical, masksetup, engine_mode="fibremodes", mode_range=(0, 11))
    engine.run(verbose=False)
    return engine


def _pump(app, seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()


def _save(widget, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixmap = widget.grab()
    if pixmap.isNull() or pixmap.width() < 10:
        raise SystemExit(f"grab() returned an empty pixmap for {path}")
    if not pixmap.save(str(path), "PNG"):
        raise SystemExit(f"Failed to write {path}")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture GUI screenshots for the README.")
    parser.add_argument("--wait-ms", type=int, default=4000)
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Use QT_QPA_PLATFORM=offscreen (may blank pyqtgraph on Windows).",
    )
    args = parser.parse_args()

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    print("Running fibre-mode library...")
    engine = run_mode_library()
    print(f"Library finished: {int(engine.maskcount)} masks")

    import pyqtgraph as pg
    from PySide6.QtWidgets import QApplication

    from gs.ui import GSWidget

    pg.setConfigOptions(useOpenGL=False, antialias=False)
    app = QApplication.instance() or QApplication(sys.argv)

    gui = GSWidget(engine)
    gui.resize(1280, 980)
    gui.show()
    gui.raise_()
    gui.activateWindow()
    _pump(app, max(args.wait_ms, 200) / 1000.0)
    _save(gui, OUT_GUI)

    gui._open_viewport()
    viewport = gui._viewport
    viewport.resize(1280, 780)
    shown = min(3, max(0, int(engine.maskcount) - 1))
    viewport._slider.setValue(shown)
    viewport.show()
    viewport.raise_()
    viewport.activateWindow()
    _pump(app, max(args.wait_ms, 200) / 1000.0)
    _save(viewport, OUT_VIEWPORT)

    viewport.close()
    gui.close()
    app.quit()


if __name__ == "__main__":
    main()
