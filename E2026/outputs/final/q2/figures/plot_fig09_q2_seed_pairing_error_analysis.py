#!/usr/bin/env python
"""Q2 seed pairing and error analysis figure.

Reads existing validation result tables only; no inference or metric recomputation
from predictions is performed here. The manuscript supplies the figure number and
overall title through its caption.
"""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial", "SimHei"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "hatch.linewidth": 0.55,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": None,
    "savefig.dpi": 300,
})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[4]
FIG_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs/final/q2/q2_plotting_handoff_v2/plot_data"
ROBUST_CSV = DATA_DIR / "fig6_paired_robust_score.csv"
CM_CSV = DATA_DIR / "fig6_clean_confusion_matrices.csv"
ZERO_CSV = DATA_DIR / "fig6_clean_vision_all_zero.csv"
SCENARIO_CSV = DATA_DIR / "scenario_details_complete.csv"
BASE = FIG_DIR / "fig09_q2_seed_pairing_error_analysis"

B0 = "B0-WCE"
P2 = "B5-P2"

# Directly inherited from the Figure 11 and Figure 12 scripts.
DARK = "#333333"
GRID = "#D9D9D9"
MAIN = "#4F7F95"
ACCENT = "#F0B36D"
PALE_BLUE = "#BFDCE6"
PALE_YELLOW = "#EEE7B0"
PALE_PINK = "#E9C9CC"
BASE_COLOR = PALE_BLUE
P2_COLOR = ACCENT
PAIR_COLORS = {B0: MAIN, P2: ACCENT}
BAR_COLORS = {B0: BASE_COLOR, P2: P2_COLOR}
CMAP = LinearSegmentedColormap.from_list(
    "figure11_blue", ["#F7FBFF", "#C6DDF0", "#5B9BD5", "#084B83"], N=256
)


def style_axis(ax, *, grid_axis="y"):
    ax.set_facecolor("white")
    ax.grid(axis=grid_axis, color=GRID, lw=0.45, zorder=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(DARK)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=DARK, length=2.5, width=0.55, pad=2)


def scenario_summary(df):
    """Summarize conditions within each seed; seed is the replicate (n=3)."""
    miss = df[(df.rho > 0) & (df.location != "clean")].copy()
    masks = {
        "单模态\n缺失": miss.modalities.isin(["text", "audio", "vision"]),
        "双模态\n缺失": miss.modalities.isin(["text+audio", "text+vision", "audio+vision"]),
        "高比例缺失\nρ=0.4–0.5": miss.rho.isin([0.4, 0.5]),
        "位置扰动\n前/中/后段": miss.location.isin(["early", "middle", "late"]),
    }
    rows = []
    for category, keep in masks.items():
        subset = miss.loc[keep]
        per_seed = subset.groupby(["model", "training_seed"]).selection_score.mean()
        n_conditions = int(subset[subset.model == B0].groupby("training_seed").size().iloc[0])
        for model in (B0, P2):
            values = per_seed.loc[model].sort_index().to_numpy()
            rows.append({
                "category": category,
                "model": model,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "seed_values": values,
                "n_conditions_per_seed": n_conditions,
            })
    return pd.DataFrame(rows)


def add_panel_title(ax, label, title):
    ax.set_title(f"({label}) {title}", loc="left", fontsize=8, fontweight="bold", pad=5, color=DARK)


def main():
    robust = pd.read_csv(ROBUST_CSV).sort_values("training_seed")
    confusion = pd.read_csv(CM_CSV)
    zero = pd.read_csv(ZERO_CSV)
    scenarios = pd.read_csv(SCENARIO_CSV)

    # Basic source integrity checks against the already-completed Q2 handoff.
    assert robust.training_seed.tolist() == [42, 43, 44]
    assert len(scenarios) == 330
    assert set(confusion.training_seed) == {42, 43, 44}
    assert set(zero.sample_count) == {15}

    mm = 1 / 25.4
    # A little more page area gives the confusion matrices larger, readable cells.
    fig = plt.figure(figsize=(200 * mm, 154 * mm), facecolor="white")
    gs = fig.add_gridspec(
        2, 2, left=0.085, right=0.985, bottom=0.105, top=0.90,
        width_ratios=(1.0, 1.35), wspace=0.30, hspace=0.48,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    gs_b = gs[0, 1].subgridspec(1, 2, wspace=0.18)
    ax_b0 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[0, 1], sharex=ax_b0, sharey=ax_b0)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # (a) Horizontal dumbbell chart for paired initialization results.
    style_axis(ax_a, grid_axis="x")
    y_positions = np.array([2, 1, 0])
    ax_a.set_axisbelow(True)
    for y, row in zip(y_positions, robust.itertuples(index=False)):
        b0_value, p2_value = row.B0_robust_score, row.P2_robust_score
        ax_a.plot([b0_value, p2_value], [y, y], color="#D5E2E7", lw=6.5,
                  solid_capstyle="round", zorder=1)
        ax_a.scatter([b0_value], [y], s=88, color=BASE_COLOR, edgecolor="white",
                     linewidth=1.0, zorder=3)
        ax_a.scatter([p2_value], [y], s=88, color=P2_COLOR, edgecolor="white",
                     linewidth=1.0, zorder=4)
        paired_delta = p2_value - b0_value
        ax_a.text(0.98, y, f"Δ={paired_delta:+.4f}", transform=ax_a.get_yaxis_transform(),
                  ha="right", va="center", fontsize=7.0, color=DARK)
    ax_a.set_yticks(y_positions, ["种子42", "种子43", "种子44"])
    ax_a.set_xlabel("稳健得分", labelpad=2)
    ax_a.set_ylabel("")
    add_panel_title(ax_a, "a", "不同初始化的配对稳健得分")
    ax_a.set_xlim(0.7395, 0.7480)
    ax_a.set_ylim(-0.55, 2.55)
    ax_a.set_xticks([0.740, 0.742, 0.744, 0.746, 0.748])
    ax_a.tick_params(axis="y", length=0, pad=3)
    delta = robust.paired_delta_P2_minus_B0.to_numpy()
    ax_a.text(
        0.50, 0.035,
        f"{int((delta > 0).sum())}/3 次提高，平均 Δ={delta.mean():+.4f}±{delta.std(ddof=1):.4f}",
        transform=ax_a.transAxes, ha="center", va="bottom", fontsize=7.0, color=MAIN,
        fontweight="bold",
    )

    # (b) Paired clean-validation confusion matrices, seed 42 as the locked primary.
    cm_data = confusion[confusion.training_seed == 42]
    order = ["Negative", "Neutral", "Positive"]
    zh = ["消极", "中性", "积极"]
    for ax, model, model_name in ((ax_b0, B0, "基线模型"), (ax_b2, P2, "本文模型")):
        sub = cm_data[cm_data.model == model]
        count = sub.pivot(index="true_class", columns="predicted_class", values="count").reindex(index=order, columns=order).to_numpy(float)
        row_prop = count / count.sum(axis=1, keepdims=True)
        ax.imshow(row_prop, cmap=CMAP, vmin=0, vmax=1, aspect="equal", interpolation="nearest")
        ax.set_title(model_name, fontsize=8.2, pad=4, color=DARK, fontweight="bold")
        ax.set_xticks(range(3), zh, fontsize=7.6)
        ax.set_yticks(range(3), zh, fontsize=7.6)
        ax.set_xlabel("预测类别", fontsize=7.6, labelpad=2)
        ax.tick_params(length=0, pad=2)
        for i in range(3):
            for j in range(3):
                color = "white" if row_prop[i, j] >= 0.48 else DARK
                ax.text(j, i, f"{int(count[i,j])}\n{row_prop[i,j]:.0%}", ha="center", va="center",
                        fontsize=8.0, color=color, linespacing=1.1, fontweight="medium")
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_b0.set_ylabel("真实类别", fontsize=7.6, labelpad=12)
    ax_b2.tick_params(labelleft=False)
    fig.text(0.505, 0.906, "(b) 验证集混淆矩阵（种子42，n=728）",
             ha="left", va="bottom", fontsize=8.2, fontweight="bold", color=DARK)

    # (c) Grouped, dotted bars styled after both reference scripts. Seed means are points.
    style_axis(ax_c)
    summary = scenario_summary(scenarios)
    categories = summary.category.drop_duplicates().tolist()
    x = np.arange(len(categories))
    width = 0.30
    for offset, model in ((-width / 2, B0), (width / 2, P2)):
        part = summary[summary.model == model].set_index("category").loc[categories]
        ax_c.bar(
            x + offset, part["mean"], width=width,
            color=BAR_COLORS[model], alpha=0.82, edgecolor=DARK, linewidth=0.7,
            hatch="..", yerr=part["sd"], capsize=2.1,
            error_kw={"elinewidth": 0.65, "ecolor": DARK}, zorder=2,
        )
        for i, values in enumerate(part["seed_values"]):
            ax_c.scatter(
                np.full(3, x[i] + offset) + np.array([-0.025, 0, 0.025]), values,
                s=9, color=DARK, edgecolor="white", linewidth=0.25, zorder=4,
            )
    ax_c.set_xticks(x, categories, fontsize=6.3)
    ax_c.set_ylabel("选择得分", labelpad=3)
    ax_c.set_ylim(0, 0.82)
    add_panel_title(ax_c, "c", "不同缺失场景的选择得分")
    # The delta label makes small but real differences legible without truncating the axis.
    for i, category in enumerate(categories):
        means = summary[summary.category == category].set_index("model")["mean"]
        ax_c.text(i, 0.785, f"Δ{(means[P2]-means[B0]):+.3f}", ha="center", va="bottom",
                  fontsize=6.1, color=MAIN)

    # (d) Native vision-all-zero subset; mean ± sample SD across the three seeds.
    style_axis(ax_d)
    metrics = [("准确率", "accuracy"), ("宏平均F1", "macro_f1")]
    mx = np.arange(len(metrics))
    mwidth = 0.31
    for offset, model in ((-mwidth / 2, B0), (mwidth / 2, P2)):
        means, sds, seed_values = [], [], []
        for _, column in metrics:
            values = zero.loc[zero.model == model].sort_values("training_seed")[column].to_numpy()
            seed_values.append(values)
            means.append(values.mean())
            sds.append(values.std(ddof=1))
        ax_d.bar(
            mx + offset, means, width=mwidth,
            color=BAR_COLORS[model], alpha=0.82, edgecolor=DARK, linewidth=0.7,
            hatch="..", yerr=sds, capsize=2.1,
            error_kw={"elinewidth": 0.65, "ecolor": DARK}, zorder=2,
        )
        for i, values in enumerate(seed_values):
            ax_d.scatter(
                np.full(3, mx[i] + offset) + np.array([-0.025, 0, 0.025]), values,
                s=10, color=DARK, edgecolor="white", linewidth=0.25, zorder=4,
            )
    ax_d.set_xticks(mx, [name for name, _ in metrics], fontsize=6.7)
    ax_d.set_ylabel("指标值", labelpad=3)
    ax_d.set_ylim(0, 0.68)
    add_panel_title(ax_d, "d", "视觉全零子集（n=15）")

    # One shared model legend, matching the compact, border-free Figure 11/12 legend.
    legend_handles = [
        Patch(facecolor=BAR_COLORS[B0], edgecolor=DARK, linewidth=0.7, hatch="..", label="基线模型"),
        Patch(facecolor=BAR_COLORS[P2], edgecolor=DARK, linewidth=0.7, hatch="..", label="本文模型"),
    ]
    fig.legend(legend_handles, ["基线模型", "本文模型"], loc="upper center",
               bbox_to_anchor=(0.5, 0.927), ncol=2, frameon=False,
               fontsize=7, handlelength=1.25, columnspacing=1.8)

    fig.savefig(BASE.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches=None)
    fig.savefig(BASE.with_suffix(".pdf"), facecolor="white", bbox_inches=None)
    fig.savefig(BASE.with_suffix(".svg"), facecolor="white", bbox_inches=None)
    print("Wrote", BASE.with_suffix(".png"))
    print("Wrote", BASE.with_suffix(".pdf"))
    print("Wrote", BASE.with_suffix(".svg"))
    print(summary[["category", "model", "mean", "sd", "n_conditions_per_seed"]].to_string(index=False))


if __name__ == "__main__":
    main()
