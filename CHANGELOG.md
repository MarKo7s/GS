# Changelog

All notable changes to gs are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] - 2026-10-22

### Added

- `GS.get_gs_properties()` reports the geometry and mode catalog already stored on the engine.
- Each mask keeps its final basis-to-core SNR, dumping field, core field, and core leakage, and those arrays are written with the library.
- Every `GS.run` appends to `%PROGRAMDATA%\LAB\Algorithm\GS\logger\gs.log`, including when `verbose` is false. A mask that does not pass is a warning.
- Optional `GSWidget` (`pip install gs[gui]`) to build, run, save, and inspect an engine without calling `GS.run` on the GUI thread.
- `get_gs_properties()` includes the feasibility flags (power outside the pupil, mode power outside the aperture, illumination support, and the maximum MMF NA). `GSWidget` shows them and can load a library without copying the saved masks.
- `GSWidget` draws the SMF and MMF beams only as far as the SLM. The SLM bar is the aperture, and the labels are the equivalent numerical apertures (coverage radius over focal length).

## [1.1.0] - 2026-10-19

### Changed

- Moved `pyPhasemasks.py` and `modemasks.py` into `legacy/`. Regression tests import them from there (`from GS.legacy import pyPhasemasks` / `modemasks`). The supported API remains `GS`, `optical_setup`, and `GS_mask_setup`.

## [1.0.0] - 2026-07-13

### Added

- Pip-installable package layout (`pyproject.toml`, git-tag release via `scripts/release.py`).
- Unified GS engine `pyGS.GS` with `toolbox` setup (`optical_setup`, `GS_mask_setup`), CuPy GPU path, and core-power floor stop (`minium_efficiency`).
- Dependency pin on `fibremodes[gpu] @ v1.1.1` so install pulls LG/HG mode generation and CuPy.

### Notes

- Legacy `pyPhasemasks.py` / `modemasks.py` remain in-tree at the package root; prefer `pyGS` for new work. Relocated under `legacy/` in 1.1.0.
