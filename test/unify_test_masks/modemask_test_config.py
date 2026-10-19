"""Shared configuration for modemask golden-reference regression tests.

Replicates test/GS_test.ipynb modemask parameters. A small mode subset is used
for fast regression; pass num_masks_out=False to generate the full library.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).resolve().parent.parent
UNIFY_TEST_MASKS_DIR = Path(__file__).resolve().parent
ALGORITHMS_DIR = TEST_DIR.parent.parent
REFERENCE_MODEMSKS_NPZ = UNIFY_TEST_MASKS_DIR / "reference_modemasks.npz"

# Same physical setup as GS_test.ipynb; subset of modes for regression speed.
MODEMSK_KWARGS = {
    "wavelength": 1.3,
    "f_SMF": 25e3,
    "f_MMF": 10e3,
    "pixel_size": 9.2,
    "MMF_d": 62.5,
    "mask_size": 960,
    "w0_slm": 1.945025401e3,
    "MMF_mfd": 13.714671176,
    "max_group": 22,
    "crosstalk": 25,
    "downscale": 1,
    "ext_FACTOR": 0,
    "num_masks_out": (16, 20),
}

REFERENCE_HELP = (
    f"Missing reference file: {REFERENCE_MODEMSKS_NPZ}\n"
    "Run: python test/unify_test_masks/generate_reference_modemasks.py"
)


def ensure_algorithms_on_path() -> None:
    algorithms = str(ALGORITHMS_DIR)
    if algorithms not in sys.path:
        sys.path.insert(0, algorithms)


def _trim_generated_masks(masks: np.ndarray) -> np.ndarray:
    """Drop unused trailing slots from the modemask output buffer."""
    flat_energy = np.abs(masks.reshape(masks.shape[0], -1)).max(axis=1)
    keep = flat_energy > 1e-10
    if not np.any(keep):
        raise ValueError("modemask pipeline produced no non-zero masks")
    return masks[keep]


def _efficiency_for_masks(
    similarity: np.ndarray, mode_indices: np.ndarray, num_masks: int
) -> np.ndarray:
    """Per-output-mask efficiency in the same order as generated masks."""
    num_masks_out = MODEMSK_KWARGS["num_masks_out"]
    if num_masks_out is False:
        mode_range = range(similarity.shape[0])
    elif isinstance(num_masks_out, tuple) and len(num_masks_out) == 2:
        mode_range = range(num_masks_out[0], num_masks_out[1])
    else:
        mode_range = range(similarity.shape[0])

    n = mode_indices[1, :]
    per_mask: list[float] = []
    for j in mode_range:
        val = float(similarity[j])
        if n[j] == 0:
            per_mask.append(val)
        else:
            per_mask.append(val)
            per_mask.append(val)

    efficiency = np.asarray(per_mask, dtype=np.float64)
    if efficiency.shape[0] != num_masks:
        raise ValueError(
            f"efficiency length {efficiency.shape[0]} != mask count {num_masks}"
        )
    return efficiency


def run_modemask_pipeline() -> dict[str, np.ndarray | str]:
    """Run modemask GS generation and return masks plus efficiency metadata."""
    ensure_algorithms_on_path()

    from GS.legacy import modemasks

    raw = modemasks.Mask_Modes_GS_gpu(**MODEMSK_KWARGS)
    masks = _trim_generated_masks(np.asarray(raw["masks"]))
    masks_phase = np.angle(masks).astype(np.float32)
    num_masks = masks_phase.shape[0]

    tm = np.asarray(raw["TM"])
    similarity = np.abs(np.diag(tm)) ** 2
    mode_indices = _load_mode_indices()
    efficiency = _efficiency_for_masks(similarity, mode_indices, num_masks)

    return {
        "masks_phase": masks_phase,
        "efficiency": efficiency.astype(np.float64),
        "modemask_params": json.dumps(MODEMSK_KWARGS, sort_keys=True, default=str),
        "num_masks": np.array([num_masks], dtype=np.int32),
    }


def _load_mode_indices() -> np.ndarray:
    """Mode-index table for the configured max_group (matches modemasks setup)."""
    from fibremodes.analytical.LG.toolbox.mode_generation_core_library import (
        graded_index_fiber_coefs,
    )

    return graded_index_fiber_coefs(MODEMSK_KWARGS["max_group"])
