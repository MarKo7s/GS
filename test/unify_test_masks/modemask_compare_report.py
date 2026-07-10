"""Reference vs test comparison report for modemask regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def max_wrapped_phase_diff(a: np.ndarray, b: np.ndarray) -> float:
    wrapped = np.angle(np.exp(1j * (a - b)))
    return float(np.max(np.abs(wrapped)))


def format_modemask_report(
    reference: dict[str, np.ndarray],
    generated: dict[str, np.ndarray],
) -> str:
    ref_masks = reference["masks_phase"]
    gen_masks = generated["masks_phase"]
    ref_eff = reference["efficiency"]
    gen_eff = generated["efficiency"]
    num_masks = ref_masks.shape[0]

    lines = [
        "",
        "Modemask regression - reference vs test",
        "=" * 72,
        f"{'Mask':<10} {'max |dphi|':>14} {'phase':>10} {'ref eff.':>12} {'test eff.':>12} {'deff':>10}",
        "-" * 72,
    ]

    for i in range(num_masks):
        max_diff = max_wrapped_phase_diff(gen_masks[i], ref_masks[i])
        phase_ok = np.allclose(gen_masks[i], ref_masks[i], atol=1e-5, rtol=0)
        ref_e = float(ref_eff[i])
        gen_e = float(gen_eff[i])
        eff_diff = gen_e - ref_e
        lines.append(
            f"{i:<10} {max_diff:>14.3e} {'OK' if phase_ok else 'MISMATCH':>10} "
            f"{ref_e:>12.6f} {gen_e:>12.6f} {eff_diff:>+10.2e}"
        )

    lines.append("=" * 72)
    return "\n".join(lines)


def save_modemask_plot(
    reference: dict[str, np.ndarray],
    generated: dict[str, np.ndarray],
    out_path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    ref_eff = reference["efficiency"]
    gen_eff = generated["efficiency"]
    x = np.arange(ref_eff.shape[0])
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Modemask regression: reference vs test efficiency", fontsize=13)
    ax.bar(x - width / 2, ref_eff, width, label="reference", color="#4c72b0")
    ax.bar(x + width / 2, gen_eff, width, label="test", color="#55a868")
    ax.set_xlabel("mask index")
    ax.set_ylabel("efficiency")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def emit_modemask_report(
    reference: dict[str, np.ndarray],
    generated: dict[str, np.ndarray],
    report_dir: Path,
) -> None:
    print(format_modemask_report(reference, generated))
    plot_path = report_dir / "modemask_regression_report.png"
    save_modemask_plot(reference, generated, plot_path)
    print(f"Saved comparison plot: {plot_path}")
