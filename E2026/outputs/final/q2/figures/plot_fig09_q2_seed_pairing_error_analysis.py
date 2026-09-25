#!/usr/bin/env python
"""Rebuild Q2 Figure 9 from locked, existing validation result tables only."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[4]
FIG_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs/final/q2/q2_plotting_handoff_v2/plot_data"
OUT_PNG = FIG_DIR / "fig09_q2_seed_pairing_error_analysis.png"
OUT_PDF = FIG_DIR / "fig09_q2_seed_pairing_error_analysis.pdf"

ROBUST_CSV = DATA_DIR / "fig6_paired_robust_score.csv"
CM_CSV = DATA_DIR / "fig6_clean_confusion_matrices.csv"
ZERO_CSV = DATA_DIR / "fig6_clean_vision_all_zero.csv"
SCENARIO_CSV = DATA_DIR / "scenario_details_complete.csv"

B0 = "B0-WCE"
P2 = "B5-P2"
COLORS = {B0: "#91A9B7", P2: "#3978A8"}
ORANGE = "#E7A45D"
INK = "#303A40"
GRID = "#E4E9EC"
BLUE_CMAP = "Blues"


def style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#89949A")
        ax.spines[spine].set_linewidth(0.75)
    ax.tick_params(colors=INK, labelsize=9, width=0.65, length=3)


def summarize_scenario_categories(df):
    """Average scenarios within each seed, then summarize across 3 seeds."""
    miss = df[(df["rho"] > 0) & (df["location"] != "clean")].copy()
    masks = {
        "单模态\n缺失": miss["modalities"].isin(["text", "audio", "vision"]),
        "双模态\n缺失": miss["modalities"].isin(["text+audio", "text+vision", "audio+vision"]),
        "高比例缺失\nρ=0.4–0.5": miss["rho"].isin([0.4, 0.5]),
        "位置扰动\n前/中/后段": miss["location"].isin(["early", "middle", "late"]),
    }
    records = []
    for category, mask in masks.items():
        subset = miss.loc[mask]
        counts = subset.groupby(["model", "training_seed"]).size()
        per_seed = subset.groupby(["model", "training_seed"], as_index=False)["selection_score"].mean()
        for model in (B0, P2):
            vals = per_seed.loc[per_seed.model == model, "selection_score"].to_numpy()
            n_scenarios = int(counts.loc[(model, 42)])
            records.append({"category": category, "model": model,
                            "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                            "n_scenarios_per_seed": n_scenarios})
    return pd.DataFrame(records)


def main():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
    })

    robust = pd.read_csv(ROBUST_CSV)
    cm = pd.read_csv(CM_CSV)
    zero = pd.read_csv(ZERO_CSV)
    scenarios = pd.read_csv(SCENARIO_CSV)
    assert robust.training_seed.tolist() == [42, 43, 44]
    assert set(cm.training_seed) == {42, 43, 44} and set(cm.model) == {B0, P2}
    assert set(zero.sample_count) == {15}
    assert len(scenarios) == 330

    fig = plt.figure(figsize=(10.2, 7.1), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.98, bottom=0.09, top=0.86,
                          wspace=0.27, hspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = gs[0, 1].subgridspec(1, 2, wspace=0.32)
    ax_b0 = fig.add_subplot(ax_b[0, 0])
    ax_p2 = fig.add_subplot(ax_b[0, 1], sharex=ax_b0, sharey=ax_b0)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    fig.suptitle("图9 最终模型的稳定性与误差分析", fontsize=14, fontweight="bold",
                 color=INK, y=0.965)

    # (a) paired robust score by initialization
    style_axis(ax_a)
    x = np.arange(3)
    pair = robust.sort_values("training_seed")
    for i, row in enumerate(pair.itertuples(index=False)):
        ax_a.plot([i - 0.075, i + 0.075], [row.B0_robust_score, row.P2_robust_score],
                  color="#AAB4BA", lw=1.35, zorder=2)
        ax_a.scatter(i - 0.075, row.B0_robust_score, s=42, color=COLORS[B0],
                     edgecolor="white", linewidth=0.75, zorder=3)
        ax_a.scatter(i + 0.075, row.P2_robust_score, s=42, color=COLORS[P2],
                     edgecolor="white", linewidth=0.75, zorder=3)
    ax_a.set_xticks(x, ["42", "43", "44"])
    ax_a.set_xlabel("随机初始化种子")
    ax_a.set_ylabel("稳健得分")
    ax_a.set_title("(a) 三次初始化的配对结果", loc="left", pad=8, fontweight="bold")
    low = min(pair.B0_robust_score.min(), pair.P2_robust_score.min()) - .0012
    high = max(pair.B0_robust_score.max(), pair.P2_robust_score.max()) + .0012
    ax_a.set_ylim(low, high)
    delta = pair.paired_delta_P2_minus_B0
    up_count = int((delta > 0).sum())
    ax_a.text(0.98, 0.96, f"{up_count}/3 个种子提升\n平均配对差 = {delta.mean():+.4f} ± {delta.std(ddof=1):.4f}",
              transform=ax_a.transAxes, ha="right", va="top", fontsize=8.4,
              color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D7DEE2", lw=.7))

    # (b) valid clean confusion matrices for the primary locked seed (42)
    matrix_data = cm[cm.training_seed == 42]
    class_order = ["Negative", "Neutral", "Positive"]
    class_zh = ["消极", "中性", "积极"]
    plotted = []
    for ax, model, label in ((ax_b0, B0, "基线模型"), (ax_p2, P2, "本文模型")):
        sub = matrix_data[matrix_data.model == model]
        pivot = sub.pivot(index="true_class", columns="predicted_class", values="count").reindex(index=class_order, columns=class_order)
        raw = pivot.to_numpy(dtype=float)
        row_pct = raw / raw.sum(axis=1, keepdims=True)
        plotted.append(ax.imshow(row_pct, cmap=BLUE_CMAP, vmin=0, vmax=1, aspect="equal"))
        ax.set_title(label, fontsize=9, pad=5, color=INK)
        ax.set_xticks(range(3), class_zh, fontsize=8)
        ax.set_yticks(range(3), class_zh, fontsize=8)
        ax.set_xlabel("预测类别", labelpad=4, fontsize=8)
        ax.tick_params(length=0)
        for i in range(3):
            for j in range(3):
                value = row_pct[i, j]
                color = "white" if value >= .55 else INK
                ax.text(j, i, f"{int(raw[i,j])}\n{value:.0%}", ha="center", va="center",
                        fontsize=7.4, color=color, linespacing=1.05)
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_b0.set_ylabel("真实类别", labelpad=4, fontsize=8)
    ax_p2.tick_params(labelleft=False)
    ax_b0.set_title("基线模型", fontsize=9, pad=5, color=INK)
    ax_p2.set_title("本文模型", fontsize=9, pad=5, color=INK)
    fig.text(0.752, 0.872, "(b) 验证集混淆矩阵（种子42，n=728）", ha="center",
             fontsize=10, fontweight="bold", color=INK)

    # (c) grouped comparison across scenario families; the seed is the replicate
    style_axis(ax_c)
    agg = summarize_scenario_categories(scenarios)
    cats = agg.category.drop_duplicates().tolist()
    gx = np.arange(len(cats))
    width = .32
    for offset, model in ((-width / 2, B0), (width / 2, P2)):
        s = agg[agg.model == model].set_index("category").loc[cats]
        ax_c.bar(gx + offset, s["mean"], width=width, color=COLORS[model],
                 edgecolor="white", linewidth=.7, zorder=2,
                 yerr=s["sd"], capsize=2.4, error_kw={"elinewidth": .9, "ecolor": INK})
        # Individual seed summaries remain visible; scenarios are not treated as replicates.
        per_seed = scenarios[(scenarios.rho > 0) & (scenarios.location != "clean")]
        masks = {
            "单模态\n缺失": per_seed.modalities.isin(["text", "audio", "vision"]),
            "双模态\n缺失": per_seed.modalities.isin(["text+audio", "text+vision", "audio+vision"]),
            "高比例缺失\nρ=0.4–0.5": per_seed.rho.isin([.4, .5]),
            "位置扰动\n前/中/后段": per_seed.location.isin(["early", "middle", "late"]),
        }
        for k, category in enumerate(cats):
            ss = per_seed[(per_seed.model == model) & masks[category]].groupby("training_seed").selection_score.mean()
            jitter = np.array([-.035, 0, .035])
            ax_c.scatter(np.full(3, gx[k] + offset) + jitter, ss.sort_index(), s=12,
                         color=INK, edgecolor="white", linewidth=.35, zorder=4)
    ax_c.set_xticks(gx, cats, fontsize=8)
    ax_c.set_ylabel("选择得分")
    ax_c.set_ylim(0, .80)
    ax_c.set_title("(c) 不同缺失场景类别的得分", loc="left", pad=8, fontweight="bold")
    for k, category in enumerate(cats):
        vals = agg[agg.category == category].set_index("model")["mean"]
        change = vals[P2] - vals[B0]
        ax_c.text(gx[k], .765, f"Δ {change:+.4f}", ha="center", va="bottom",
                  color=ORANGE, fontsize=7.5, fontweight="bold")

    # (d) small native vision-all-zero subset; classification metrics only
    style_axis(ax_d)
    metric_names = ["准确率", "宏平均 F1"]
    metric_cols = ["accuracy", "macro_f1"]
    xx = np.arange(2)
    w = .31
    for off, model in ((-w/2, B0), (w/2, P2)):
        means, sds = [], []
        for col in metric_cols:
            values = zero.loc[zero.model == model].sort_values("training_seed")[col].to_numpy()
            means.append(values.mean())
            sds.append(values.std(ddof=1))
        ax_d.bar(xx + off, means, width=w, color=COLORS[model], edgecolor="white",
                 linewidth=.7, yerr=sds, capsize=2.5,
                 error_kw={"elinewidth": .9, "ecolor": INK}, zorder=2)
        for j, col in enumerate(metric_cols):
            values = zero.loc[zero.model == model].sort_values("training_seed")[col].to_numpy()
            ax_d.scatter(np.full(3, xx[j] + off) + np.array([-.045, 0, .045]), values,
                         s=14, color=INK, edgecolor="white", linewidth=.35, zorder=4)
    ax_d.set_xticks(xx, metric_names)
    ax_d.set_ylabel("指标值")
    ax_d.set_ylim(0, .66)
    ax_d.set_title("(d) 视觉全零子集表现（n=15）", loc="left", pad=8, fontweight="bold")

    handles = [Patch(facecolor=COLORS[B0], edgecolor="white", label="基线模型"),
               Patch(facecolor=COLORS[P2], edgecolor="white", label="本文模型")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .915), ncol=2,
               frameon=False, fontsize=9, handlelength=1.2, columnspacing=1.4)
    fig.savefig(OUT_PNG, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT_PDF, facecolor="white", bbox_inches="tight")
    print(f"PNG: {OUT_PNG}")
    print(f"PDF: {OUT_PDF}")
    print("Scenario category summaries (mean and sample SD across seed-level means):")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
