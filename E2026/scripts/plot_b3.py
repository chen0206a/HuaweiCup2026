"""Render B3 validation figures from saved, real result JSON files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics"
FIGURES = ROOT / "outputs" / "figures"


def main() -> None:
    source = json.loads((METRICS / "b3_summary.json").read_text(encoding="utf-8"))
    result = source["comparison"]
    diagnostics = source["diagnostics"]["error_vs_rho_single_modality"]
    ratios = (0.1, 0.2, 0.3, 0.4, 0.5)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True, sharey=True)
    colors = ("#4c78a8", "#f58518", "#e45756")
    for ax, modality in zip(axes, ("text", "audio", "vision")):
        for (name, data), color in zip(result.items(), colors):
            # The frozen B2 summary groups by ratio across three locations.
            # Per-modality ratio scores instead come from the saved scene CSV.
            import csv
            if name == "B3-B0-Reconstruction":
                path = METRICS / "b3_missing_scenarios.csv"
            else:
                path = METRICS / "b2_missing_scenarios.csv"
            with path.open(newline="", encoding="utf-8") as stream:
                rows = [row for row in csv.DictReader(stream) if row["model"] == name
                        and row["modalities"] == modality]
            values = [sum(float(row["selection_score"]) for row in rows
                          if float(row["rho"]) == rho) / 3 for rho in ratios]
            ax.plot(ratios, values, marker="o", color=color, label=name)
        ax.set(title=f"{modality} missing", xlabel="Missing ratio", xticks=ratios)
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Validation selection score")
    axes[0].legend(fontsize=8)
    fig.savefig(FIGURES / "b3_missing_ratio_score.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for col, modality in enumerate(("text", "audio", "vision")):
        values = diagnostics[modality]
        error = [values[str(rho)]["smooth_l1"] for rho in ratios]
        cosine = [values[str(rho)]["cosine"] for rho in ratios]
        axes[0, col].plot(ratios, error, marker="o", color="#e45756")
        axes[1, col].plot(ratios, cosine, marker="o", color="#4c78a8")
        axes[0, col].set(title=modality, ylabel="Masked-position SmoothL1", xticks=ratios)
        axes[1, col].set(xlabel="Missing ratio", ylabel="Masked-position cosine", xticks=ratios)
        for row in (0, 1):
            axes[row, col].grid(alpha=.25)
    fig.savefig(FIGURES / "b3_reconstruction_vs_rho.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
