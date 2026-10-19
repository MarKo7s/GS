# Gerchberg–Saxton (GS) for phase and amplitude light control

![alt text](images/GS_white.png)

GPU Gerchberg–Saxton phase-mask generation for multimode fibre injection / SLM control. Built on the ModeLab packaging stack: install a tagged release and pip pulls pinned siblings (notably `fibremodes`).

## Package layout

```
gs/
├── pyGS.py              unified GS engine
├── toolbox/             optical_setup, GS_mask_setup, planes, filters, modes, memory
├── legacy/              pyPhasemasks and modemasks (regression tests only)
├── test/                notebooks and regression helpers
├── scripts/release.py   git-tag release automation
├── pyproject.toml       package metadata + version
└── CHANGELOG.md
```

## ModeLab dependency stack

```
fibremodes@v1.1.1  (+ cupy-cuda13x via [gpu])
        ↑
      gs@v1.1.0
```

`gs` pins `fibremodes[gpu]` by git tag in `pyproject.toml`. Installing `gs` pulls fibremodes (LG/HG modes) and CuPy — no manual `sys.path` chaining.

## Current API (`pyGS`)

```python
import numpy as np
from gs import GS, optical_setup, GS_mask_setup

optical = optical_setup(
    wavelength=0.6328,
    fin=4.51,
    fout=18,
    masksize=512,
    slm_pixel_size=8,
    MMF_mfd=34,
    MMF_diameter=50,
    MMF_mode_groups=10,
    mode_basis="LG",
)

masksetup = GS_mask_setup(
    convergence_metric="overlap",
    goal_value=0.99,
    minium_efficiency=0.1,  # stop if spatial core power drops below this floor
    MAX_iterations=300,
    downscale=2,
)

engine = GS(optical, masksetup, engine_mode="fields", number_masks=2)
engine.run(inpucoefs=coefs, verbose=True)
# engine.phase_masks, engine.efficiency, engine.iteration_convergence, ...
```

- `engine_mode`: `'fields'` (targets from modal coefs) or `'fibremodes'` (one mask per mode / range).
- Stop when overlap reaches `goal_value`, core power falls below `minium_efficiency`, or `MAX_iterations`.

`legacy/pyPhasemasks.py` and `legacy/modemasks.py` stay in the tree so regression tests can still call the old generators. New work uses `pyGS`.

## Canonical imports

```python
import gs
from gs import GS, optical_setup, GS_mask_setup
from gs.pyGS import GS
from gs.toolbox.setup import optical_setup, GS_mask_setup
```

## Installation

### From GitHub (tagged release)

```bash
pip install "gs @ git+https://github.com/MarKo7s/GS.git@v1.1.0"
```

That resolves the pinned `fibremodes[gpu] @ …@v1.1.1` (and CuPy via that extra).

Other packages can depend on the same pin in `pyproject.toml`:

```toml
dependencies = [
  "gs @ git+https://github.com/MarKo7s/GS.git@v1.1.0",
]
```

### Local development (editable install)

```bash
git clone git@github.com:MarKo7s/GS.git
cd GS
pip install -e ".[notebooks,test]"
```

### Conda environment

```bash
conda create -n gs python=3.11 -y
conda activate gs
pip install -e ".[notebooks,test]"
```

`fibremodes[gpu]` pulls `cupy-cuda13x` for CUDA toolkit 13.x (e.g. 13.2). Do not install plain `cupy` beside that wheel. Restart Jupyter after setting `CUDA_PATH` if needed.

Register a Jupyter kernel (once):

```bash
python -m ipykernel install --user --name gs --display-name "Python (gs)"
```

`requirements.txt` remains for legacy workflows; prefer `pip install -e ".[notebooks,test]"`.

## Developer notes

### Versioning

The package version is defined in **one place only**: `pyproject.toml` → `[project].version`.

Do **not** edit `__init__.py` on each release. `gs.__version__` is read from pip metadata after install (`importlib.metadata`).

```bash
pip show gs
python -c "import gs, fibremodes; print(gs.__version__, fibremodes.__version__)"
```

Use [semantic versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

When bumping the required `fibremodes` tag, update the pin in `pyproject.toml` (and `requirements.txt`) and ship a new `gs` release.

### Releasing a new version

1. Add an entry for the new version at the top of `CHANGELOG.md`.
2. Bump `version` in `pyproject.toml`.
3. Commit all changes (including the changelog).
4. Run the release script from the repo root:

```bash
python scripts/release.py --from-changelog
```

The script reads the version from `pyproject.toml`, pushes `main`, creates annotated git tag `vX.Y.Z`, and pushes the tag. Use `--branch refactoring` (or another branch) if releasing before merge to `main`.

After that:

```bash
pip install "gs @ git+https://github.com/MarKo7s/GS.git@vX.Y.Z"
```

Dry run:

```bash
python scripts/release.py --from-changelog --dry-run
```

Optional GitHub Release (`gh` CLI):

```bash
gh release create vX.Y.Z --title "gs X.Y.Z" --notes-file CHANGELOG.md
```

**Requirements before release:** clean working tree; tag `vX.Y.Z` must not already exist on GitHub.

### Status

Further GS work (e.g. target apodization experiments) may continue later. Prefer this packaging layout for any new ModeLab modules that depend on `gs` or `fibremodes`.
