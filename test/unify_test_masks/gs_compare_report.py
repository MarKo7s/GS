"""Reference vs test comparison report for GS mask regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

MASK_LABELS = ("sad_H", "smiley_V")
PHASE_LABELS = {
    "sad_H_phase": "sad_H (downscaled)",
    "smiley_V_phase": "smiley_V (downscaled)",
    "sad_H_phase_rescaled": "sad_H (rescaled)",
    "smiley_V_phase_rescaled": "smiley_V (rescaled)",
}


def max_wrapped_phase_diff(a: np.ndarray, b: np.ndarray) -> float:
    wrapped = np.angle(np.exp(1j * (a - b)))
    return float(np.max(np.abs(wrapped)))


def build_comparison_rows(
    reference: dict[str, np.ndarray],
    generated: dict[str, np.ndarray],
    phase_keys: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, mask in enumerate(MASK_LABELS):
        rows.append(
            {
                "mask": mask,
                "ref_iterations": int(reference["iteration_convergence"][i]),
                "test_iterations": int(generated["iteration_convergence"][i]),
                "iterations_match": bool(
                    reference["iteration_convergence"][i]
                    == generated["iteration_convergence"][i]
                ),
                "ref_efficiency": float(reference["efficiency"][i]),
                "test_efficiency": float(generated["efficiency"][i]),
                "efficiency_diff": float(
                    generated["efficiency"][i] - reference["efficiency"][i]
                ),
                "ref_fidelity": float(reference["fidelity"][i]),
                "test_fidelity": float(generated["fidelity"][i]),
                "fidelity_diff": float(
                    generated["fidelity"][i] - reference["fidelity"][i]
                ),
            }
        )

    for key in phase_keys:
        ref_phase = reference[key]
        gen_phase = generated[key]
        rows.append(
            {
                "mask": PHASE_LABELS.get(key, key),
                "max_phase_diff": max_wrapped_phase_diff(gen_phase, ref_phase),
                "phase_match": bool(
                    np.allclose(gen_phase, ref_phase, atol=1e-5, rtol=0)
                ),
            }
        )
    return rows


def format_comparison_report(
    reference: dict[str, np.ndarray],
    generated: dict[str, np.ndarray],
    phase_keys: tuple[str, ...],
) -> str:
    lines = [
        "",
        "GS mask regression — reference vs test",
        "=" * 72,
        f"{'Mask':<22} {'Metric':<14} {'Reference':>12} {'Test':>12} {'Diff':>10}",
        "-" * 72,
    ]

    for i, mask in enumerate(MASK_LABELS):
        ref_iter = int(reference["iteration_convergence"][i])
        test_iter = int(generated["iteration_convergence"][i])
        match = "OK" if ref_iter == test_iter else "MISMATCH"
        lines.append(
            f"{mask:<22} {'iterations':<14} {ref_iter:>12} {test_iter:>12} {match:>10}"
        )

        ref_eff = float(reference["efficiency"][i])
        test_eff = float(generated["efficiency"][i])
        eff_diff = test_eff - ref_eff
        lines.append(
            f"{'':<22} {'efficiency':<14} {ref_eff:>12.6f} {test_eff:>12.6f} {eff_diff:>+10.2e}"
        )

        ref_fid = float(reference["fidelity"][i])
        test_fid = float(generated["fidelity"][i])
        fid_diff = test_fid - ref_fid
        lines.append(
            f"{'':<22} {'fidelity':<14} {ref_fid:>12.6f} {test_fid:>12.6f} {fid_diff:>+10.2e}"
        )
        lines.append("-" * 72)

    lines.append(f"{'Phase mask':<22} {'max |Δφ|':<14} {'match':>12}")
    lines.append("-" * 72)
    for key in phase_keys:
        label = PHASE_LABELS.get(key, key)
        max_diff = max_wrapped_phase_diff(generated[key], reference[key])
        phase_ok = np.allclose(generated[key], reference[key], atol=1e-5, rtol=0)
        lines.append(
            f"{label:<22} {max_diff:>14.3e} {'OK' if phase_ok else 'MISMATCH':>12}"
        )

    lines.append("=" * 72)
    return "\n".join(lines)


def save_comparison_plot(
    reference: dict[str, np.ndarray],
    generated: dict[str, np.ndarray],
    out_path: Path,
) -> Path:
    import matplotlib.pyplot as plt

    x = np.arange(len(MASK_LABELS))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("GS regression: reference vs test", fontsize=13)

    ref_iters = reference["iteration_convergence"]
    test_iters = generated["iteration_convergence"]
    ax = axes[0]
    ax.bar(x - width / 2, ref_iters, width, label="reference", color="#4c72b0")
    ax.bar(x + width / 2, test_iters, width, label="test", color="#55a868")
    ax.set_xticks(x, MASK_LABELS)
    ax.set_ylabel("iterations")
    ax.set_title("Convergence iterations")
    ax.legend()

    ref_eff = reference["efficiency"]
    test_eff = generated["efficiency"]
    ax = axes[1]
    ax.bar(x - width / 2, ref_eff, width, label="reference", color="#4c72b0")
    ax.bar(x + width / 2, test_eff, width, label="test", color="#55a868")
    ax.set_xticks(x, MASK_LABELS)
    ax.set_ylabel("efficiency")
    ax.set_title("Mask efficiency")
    ax.legend()

    ref_fid = reference["fidelity"]
    test_fid = generated["fidelity"]
    ax = axes[2]
    ax.bar(x - width / 2, ref_fid, width, label="reference", color="#4c72b0")
    ax.bar(x + width / 2, test_fid, width, label="test", color="#55a868")
    ax.set_xticks(x, MASK_LABELS)
    ax.set_ylabel("fidelity")
    ax.set_title("Mask fidelity")
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def emit_comparison_report(
    reference: dict[str, np.ndarray],
    generated: dict[str, np.ndarray],
    phase_keys: tuple[str, ...],
    report_dir: Path,
) -> None:
    report_text = format_comparison_report(reference, generated, phase_keys)
    print(report_text)

    plot_path = report_dir / "gs_regression_report.png"
    save_comparison_plot(reference, generated, plot_path)
    print(f"Saved comparison plot: {plot_path}")
