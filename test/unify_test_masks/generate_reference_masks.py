"""Generate golden-reference phase masks for GS regression testing.

Replicates the pipeline in test/GS_test.ipynb. Run manually when the
baseline is intentionally updated:

    python test/unify_test_masks/generate_reference_masks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

from gs_test_config import REFERENCE_MASKS_NPZ, run_gs_pipeline


def main() -> None:
    result = run_gs_pipeline()
    np.savez_compressed(REFERENCE_MASKS_NPZ, **result)
    print(f"Saved reference masks to {REFERENCE_MASKS_NPZ}")
    print("iteration_convergence:", result["iteration_convergence"])
    print("efficiency:", result["efficiency"])


if __name__ == "__main__":
    main()
