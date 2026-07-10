"""Generate golden-reference modemask library for GS regression testing.

Replicates the modemask pipeline in test/GS_test.ipynb. Run manually when the
baseline is intentionally updated:

    python test/unify_test_masks/generate_reference_modemasks.py

Pass --full to generate the complete mode library (slow; matches num_masks_out=False
in the notebook).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

from modemask_test_config import (  # noqa: E402
    MODEMSK_KWARGS,
    REFERENCE_MODEMSKS_NPZ,
    run_modemask_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate modemask golden references")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Generate all modes (num_masks_out=False); default uses the regression subset",
    )
    args = parser.parse_args()

    if args.full:
        MODEMSK_KWARGS["num_masks_out"] = False

    result = run_modemask_pipeline()
    np.savez_compressed(REFERENCE_MODEMSKS_NPZ, **result)
    print(f"Saved reference modemasks to {REFERENCE_MODEMSKS_NPZ}")
    print("num_masks:", int(result["num_masks"][0]))
    print("efficiency:", result["efficiency"])


if __name__ == "__main__":
    main()
