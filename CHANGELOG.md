# Changelog

All notable changes to gs are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
