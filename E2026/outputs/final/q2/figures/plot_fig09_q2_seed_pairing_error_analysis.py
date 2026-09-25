# Academic Figure Skill Asset Confirmation (based on the user-specified paper PDF)
# (a) paired dot-line plot → Figure 11 visual style → parameter inheritance
# (b) paired mean/SD plot → Figure 11 visual style → parameter inheritance
# The referenced Figure 11/12 PDFs contain no reusable plotting source script.

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Typography adapted for the Chinese manuscript and the referenced figures.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
DATA = PROJECT / "outputs/final/q2/q2_plotting_handoff_v2/plot_data"
ROBUST = pd.read_csv(DATA / "fig6_paired_robust_score.csv")
ZERO = pd.read_csv(DATA / "fig6_clean_vision_all_zero.csv")

SEEDS = [42, 43, 44]
B0 = "#AFC5CE"       # cool gray-blue
P2 = "#4F8397"       # restrained blue, matching Figure 11/12
CONNECT = "#C5C9CC"
GRID = "#D9D9D9"
TEXT = "#333333"

assert set(ROBUST.training_seed.astype(int)) == set(SEEDS) and len(ROBUST) == 3
assert len(ZERO) == 6 and set(ZERO.sample_count.astype(int)) == {15}
assert set(ZERO.model) == {"B0-WCE", "B5-P2"}
delta = ROBUST.paired_delta_P2_minus_B0.to_numpy(dtype=float)
assert np.isclose(delta.mean(), 0.0031004513796594932, atol=1e-12)
assert np.isclose(delta.std(ddof=1), 0.002906890840329915, atol=1e-12)
assert np.isfinite(ZERO[["accuracy", "macro_f1", "mae", "pearson"]].to_numpy()).all()


def panel_label(ax, letter: str, title: str) -> None:
    ax.text(-0.08, 1.075, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=TEXT, ha="left", va="bottom")
    ax.text(0.01, 1.075, title, transform=ax.transAxes,
            fontsize=8.4, fontweight="bold", color=TEXT, ha="left", va="bottom")


def paired_robust_panel(ax: plt.Axes) -> None:
    panel_label(ax, "A", "三次初始化的配对鲁棒分数")
    rows = ROBUST.set_index("training_seed").loc[SEEDS]
    x = np.arange(3, dtype=float)
    b0 = rows.B0_robust_score.to_numpy(dtype=float)
    p2 = rows.P2_robust_score.to_numpy(dtype=float)
    offset = 0.075
    for i in range(3):
        ax.plot([x[i] - offset, x[i] + offset], [b0[i], p2[i]],
                color=CONNECT, linewidth=1.0, zorder=1)
    ax.plot(x - offset, b0, linestyle="none", marker="o", markersize=4.5,
            markerfacecolor=B0, markeredgecolor="white", markeredgewidth=0.5,
            label="基线模型", zorder=3)
    ax.plot(x + offset, p2, linestyle="none", marker="o", markersize=4.8,
            markerfacecolor=P2, markeredgecolor="white", markeredgewidth=0.5,
            label="本文模型", zorder=3)
    ax.set_xticks(x, [f"种子{s}" for s in SEEDS])
    ax.set_xlim(-0.42, 2.42)
    ax.set_ylim(0.7385, 0.7475)
    ax.set_yticks([0.740, 0.742, 0.744, 0.746])
    ax.set_ylabel("鲁棒分数")
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.98), ncol=2,
              handletextpad=0.35, columnspacing=0.8, borderaxespad=0.0)
    ax.text(0.98, 0.98,
            "平均配对 Δ = +0.0031 ± 0.0029\n2/3 次提高；1 次近乎持平",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
            color=TEXT, linespacing=1.35)


def vision_zero_panel(ax: plt.Axes) -> None:
    panel_label(ax, "B", "原生视觉整段全零子集")
    colors = {"B0-WCE": B0, "B5-P2": P2}
    x_center = {"B0-WCE": 0.0, "B5-P2": 1.0}
    jitter = {42: -0.13, 43: -0.04, 44: 0.13}
    means = {}
    sds = {}
    for model in ("B0-WCE", "B5-P2"):
        values = []
        for seed in SEEDS:
            row = ZERO[(ZERO.model == model) & (ZERO.training_seed == seed)]
            assert len(row) == 1
            values.append(float(row.iloc[0].accuracy))
        means[model] = float(np.mean(values))
        sds[model] = float(np.std(values, ddof=1))

    # Paired seed lines are light; the model-colored points are the observed
    # seed accuracies, while the larger diamonds show mean ± sample SD.
    for seed in SEEDS:
        values = {}
        for model in ("B0-WCE", "B5-P2"):
            row = ZERO[(ZERO.model == model) & (ZERO.training_seed == seed)]
            values[model] = float(row.iloc[0].accuracy)
        xs = [x_center["B0-WCE"] + jitter[seed], x_center["B5-P2"] + jitter[seed]]
        ax.plot(xs, [values["B0-WCE"], values["B5-P2"]],
                color=CONNECT, linewidth=0.8, zorder=1)
        ax.scatter(xs[0], values["B0-WCE"], s=25, color=B0, edgecolor="white",
                   linewidth=0.45, zorder=3)
        ax.scatter(xs[1], values["B5-P2"], s=25, color=P2, edgecolor="white",
                   linewidth=0.45, zorder=3)
    for model in ("B0-WCE", "B5-P2"):
        ax.errorbar(x_center[model], means[model], yerr=sds[model], fmt="D",
                    markersize=5.6, color=colors[model], ecolor=TEXT,
                    markeredgecolor="white", markeredgewidth=0.55,
                    capsize=2.5, elinewidth=0.85, zorder=4)
    ax.set_xticks([0, 1], ["基线模型", "本文模型"])
    ax.set_xlim(-0.38, 1.38)
    ax.set_ylim(0.42, 0.58)
    ax.set_yticks([0.45, 0.50, 0.55])
    ax.set_ylabel("准确率")
    ax.grid(axis="y", color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.98, "每种子 n = 15", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.1, color=TEXT)


def main() -> None:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.2, 3.55), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    fig.suptitle("图9 随机初始化稳定性与薄弱场景诊断",
                 y=0.985, fontsize=10, fontweight="bold", color=TEXT)
    paired_robust_panel(ax_a)
    vision_zero_panel(ax_b)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.20, wspace=0.36)
    output = HERE / "fig09_q2_seed_pairing_error_analysis"
    fig.savefig(f"{output}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{output}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Wrote {output}.pdf and {output}.png")


if __name__ == "__main__":
    main()
