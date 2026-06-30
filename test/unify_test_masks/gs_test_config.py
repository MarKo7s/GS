"""Shared configuration for GS golden-reference mask regression tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).resolve().parent.parent
UNIFY_TEST_MASKS_DIR = Path(__file__).resolve().parent
ALGORITHMS_DIR = TEST_DIR.parent.parent
FACES_NPZ = TEST_DIR / "data_testing" / "modal_decompositions" / "faces.npz"
REFERENCE_MASKS_NPZ = UNIFY_TEST_MASKS_DIR / "reference_masks.npz"

GS_SETUP_KWARGS = {
    "wavelength": 1.3,
    "fin": 25e3,
    "fout": 8e3,
    "masksize": 960,
    "slm_pixel_size": 9.2,
    "MMF_mfd": 12.2,
    "MMFdiameter": 62.5,
    "MaxModegroups": 22,
    "maskscount": 2,
    "goal_fidelity": 0.99,
    "downscale": 2,
    "upsample": 0,
    "MAX_iterations": 300,
}

EXPECTED_ITERATIONS = np.array([78, 86], dtype=np.int32)
EXPECTED_EFFICIENCY = np.array(
    [0.27398014068603516, 0.22683385014533997], dtype=np.float64
)

PHASE_KEYS = (
    "sad_H_phase",
    "smiley_V_phase",
    "sad_H_phase_rescaled",
    "smiley_V_phase_rescaled",
)

FACES_NPZ_HELP = (
    f"Missing input file: {FACES_NPZ}\n"
    "Generate it by running test/modal_decomposition.ipynb."
)


def ensure_algorithms_on_path() -> None:
    algorithms = str(ALGORITHMS_DIR)
    if algorithms not in sys.path:
        sys.path.insert(0, algorithms)


def load_target() -> np.ndarray:
    if not FACES_NPZ.is_file():
        raise FileNotFoundError(FACES_NPZ_HELP)

    data = np.load(FACES_NPZ)
    sad = data["sad"][0]
    smiley = data["smiley"][0]
    target = np.empty((2, sad.shape[0]), np.complex64)
    target[0, :] = sad
    target[1, :] = smiley
    return target


def run_gs_pipeline():
    """Run GS mask generation and return masks plus convergence metadata."""
    ensure_algorithms_on_path()

    import cupy as cp

    from GS import pyPhasemasks

    target = load_target()
    masks = pyPhasemasks.SetupGenericMaskGSenvioment(**GS_SETUP_KWARGS)
    masks.calcMasksFromCoefs_gpu(target, printting=False)

    result = {
        "sad_H_phase": np.angle(cp.asnumpy(masks.phase_masks[0])).astype(np.float32),
        "smiley_V_phase": np.angle(cp.asnumpy(masks.phase_masks[1])).astype(np.float32),
        "sad_H_phase_rescaled": np.angle(
            cp.asnumpy(masks.phase_masks_rescaled[0])
        ).astype(np.float32),
        "smiley_V_phase_rescaled": np.angle(
            cp.asnumpy(masks.phase_masks_rescaled[1])
        ).astype(np.float32),
        "fidelity": cp.asnumpy(masks.fidelity).astype(np.float64),
        "efficiency": cp.asnumpy(masks.efficiency).astype(np.float64),
        "iteration_convergence": cp.asnumpy(masks.iteration_convergence).astype(
            np.int32
        ),
        "gs_params": json.dumps(GS_SETUP_KWARGS, sort_keys=True),
    }
    return result
