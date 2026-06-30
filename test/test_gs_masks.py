"""GPU regression test for GS phase masks against golden references.

Run locally with a CUDA GPU:

    pytest test/test_gs_masks.py -m gpu -s

Comparison tables and a bar-chart plot are printed/saved on every run
(test/unify_test_masks/gs_regression_report.png).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cupy")

_TEST_DIR = Path(__file__).resolve().parent
_UNIFY_DIR = _TEST_DIR / "unify_test_masks"
if str(_UNIFY_DIR) not in sys.path:
    sys.path.insert(0, str(_UNIFY_DIR))

from gs_compare_report import emit_comparison_report  # noqa: E402
from gs_test_config import (  # noqa: E402
    EXPECTED_EFFICIENCY,
    EXPECTED_ITERATIONS,
    FACES_NPZ_HELP,
    PHASE_KEYS,
    REFERENCE_MASKS_NPZ,
    FACES_NPZ,
    UNIFY_TEST_MASKS_DIR,
    run_gs_pipeline,
)

PHASE_ATOL = 1e-5
SCALAR_ATOL = 1e-5


def _max_phase_diff(a: np.ndarray, b: np.ndarray) -> float:
    wrapped = np.angle(np.exp(1j * (a - b)))
    return float(np.max(np.abs(wrapped)))


@pytest.mark.gpu
def test_gs_masks_match_reference(capsys):
    if not FACES_NPZ.is_file():
        pytest.skip(FACES_NPZ_HELP)
    if not REFERENCE_MASKS_NPZ.is_file():
        pytest.skip(
            f"Missing reference file: {REFERENCE_MASKS_NPZ}\n"
            "Run: python test/unify_test_masks/generate_reference_masks.py"
        )

    reference = np.load(REFERENCE_MASKS_NPZ)
    generated = run_gs_pipeline()

    with capsys.disabled():
        emit_comparison_report(reference, generated, PHASE_KEYS, UNIFY_TEST_MASKS_DIR)

    for key in PHASE_KEYS:
        ref_phase = reference[key]
        gen_phase = generated[key]
        assert ref_phase.shape == gen_phase.shape, f"{key}: shape mismatch"
        if not np.allclose(gen_phase, ref_phase, atol=PHASE_ATOL, rtol=0):
            max_diff = _max_phase_diff(gen_phase, ref_phase)
            pytest.fail(
                f"{key}: phase mismatch (max wrapped diff={max_diff:.3e}, "
                f"atol={PHASE_ATOL})"
            )

    assert np.array_equal(
        generated["iteration_convergence"], reference["iteration_convergence"]
    )
    assert np.array_equal(generated["iteration_convergence"], EXPECTED_ITERATIONS)

    np.testing.assert_allclose(
        generated["efficiency"],
        reference["efficiency"],
        atol=SCALAR_ATOL,
        rtol=0,
    )
    np.testing.assert_allclose(
        generated["efficiency"],
        EXPECTED_EFFICIENCY,
        atol=SCALAR_ATOL,
        rtol=0,
    )
