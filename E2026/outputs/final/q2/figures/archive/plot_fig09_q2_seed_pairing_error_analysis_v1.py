"""Q2 Figure 9: seed pairing, validation confusion, and native-zero diagnosis.

All values are loaded from the existing Q2 plotting handoff CSVs. This script
does not load features/checkpoints or run model inference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
PLOT_DATA = PROJECT / "outputs/final/q2/q2_plotting_handoff_v2/plot_data"
ROBUST_CSV = PLOT_DATA / "fig6_paired_robust_score.csv"
CM_CSV = PLOT_DATA / "fig6_clean_confusion_matrices.csv"
ZERO_CSV = PLOT_DATA / "fig6_clean_vision_all_zero.csv"
OUT_PNG = HERE / "fig09_q2_seed_pairing_error_analysis.png"
OUT_PDF = HERE / "fig09_q2_seed_pairing_error_analysis.pdf"

BLUE = "#4477AA"      # baseline
ORANGE = "#D87542"    # proposed model
INK = "#333333"
MUTED = "#70777D"
GRID = "#D9DDE0"
PALE_BLUE = "#EAF1F6"


def set_style() -> None:
    preferred = ["Microsoft YaHei", "Microsoft JhengHei", "SimHei", "Noto Sans CJK SC"]
    installed = {f.name for f in fm.fontManager.ttflist}
    font = next((name for name in preferred if name in installed), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font, "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 8.2,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.3,
        "ytick.labelsize": 7.3,
        "legend.fontsize": 7.2,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.75,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    robust = pd.read_csv(ROBUST_CSV)
    confusion = pd.read_csv(CM_CSV)
    zero = pd.read_csv(ZERO_CSV)
    expected_seeds = {42, 43, 44}
    assert set(robust.training_seed.astype(int)) == expected_seeds
    assert len(robust) == 3 and len(confusion) == 54 and len(zero) == 6
    assert set(zero.sample_count.astype(int)) == {15}
    assert np.isfinite(zero[["accuracy", "macro_f1", "mae", "pearson"]].to_numpy()).all()
    assert set(confusion.training_seed.astype(int)) == expected_seeds
    assert set(confusion.model) == {"B0-WCE", "B5-P2"}
    assert set(zero.model) == {"B0-WCE", "B5-P2"}

    # Verify locked summaries and paired delta from the three existing seeds.
    delta = robust.paired_delta_P2_minus_B0.to_numpy(dtype=float)
    assert np.isclose(delta.mean(), 0.0031004513796594932, atol=1e-12)
    assert np.isclose(delta.std(ddof=1), 0.002906890840329915, atol=1e-12)
    assert np.allclose(delta, [0.005599561353715088, 0.003791476553505202,
                               -0.00008968376824181057], atol=1e-12)

    # Ensure each model/seed confusion matrix covers the 728-sample valid split.
    grouped = confusion.groupby(["model", "training_seed"])["count"].sum()
    assert len(grouped) == 6 and (grouped == 728).all()
    return robust, confusion, zero


def panel_tag(ax, tag: str, title: str) -> None:
    ax.text(0.0, 1.045, tag, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.2, fontweight="bold", color=INK)
    ax.text(0.105, 1.045, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.0, fontweight="bold", color=INK)


def draw_robust(ax: plt.Axes, robust: pd.DataFrame) -> None:
    panel_tag(ax, "(A)", "鲁棒分数（按初始化配对）")
    seeds = [42, 43, 44]
    baseline = robust.set_index("training_seed").loc[seeds, "B0_robust_score"].to_numpy()
    proposed = robust.set_index("training_seed").loc[seeds, "P2_robust_score"].to_numpy()
    delta = robust.set_index("training_seed").loc[seeds, "paired_delta_P2_minus_B0"].to_numpy()
    x = np.arange(len(seeds), dtype=float)
    offsets = 0.085
    for i in range(len(seeds)):
        ax.plot([x[i] - offsets, x[i] + offsets], [baseline[i], proposed[i]],
                color="#A5A9AC", lw=1.0, zorder=1)
    ax.scatter(x - offsets, baseline, s=26, color=BLUE, edgecolor="white",
               linewidth=0.55, zorder=3, label="基线模型")
    ax.scatter(x + offsets, proposed, s=28, color=ORANGE, edgecolor="white",
               linewidth=0.55, zorder=3, label="本文模型")
    y_mid = (baseline + proposed) / 2
    for i, d in enumerate(delta):
        ax.annotate(f"Δ={d:+.5f}", (x[i], y_mid[i]), xytext=(0, 9 if d >= 0 else -13),
                    textcoords="offset points", ha="center", va="center", fontsize=6.5,
                    color=INK)
    ax.set_xticks(x, [f"种子{s}" for s in seeds])
    ax.set_ylabel("")
    ax.set_ylim(0.7388, 0.7475)
    ax.set_xlim(-0.42, 2.42)
    ax.grid(axis="y", color=GRID, lw=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=1, handletextpad=0.35,
              borderaxespad=0.2, labelspacing=0.35)
    ax.text(0.98, 0.05,
            "按初始化配对\n基线 0.741677 ± 0.001458\n本文 0.744777 ± 0.001492\nΔ 0.003100 ± 0.002907",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.45,
            color=MUTED, linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "#E0E3E5", "linewidth": 0.5})


def draw_confusion(ax: plt.Axes, confusion: pd.DataFrame) -> None:
    panel_tag(ax, "(B)", "验证混淆矩阵")
    selected = confusion[(confusion.model == "B5-P2") & (confusion.training_seed == 42)]
    order = ["Negative", "Neutral", "Positive"]
    matrix = (selected.pivot(index="true_class", columns="predicted_class", values="count")
              .reindex(index=order, columns=order).to_numpy(dtype=int))
    assert matrix.shape == (3, 3) and matrix.sum() == 728
    assert np.isclose(matrix.trace() / matrix.sum(), 0.6483516483516484, atol=1e-12)
    row_n = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / row_n
    cmap = LinearSegmentedColormap.from_list("muted_blue", ["#F7F9FB", "#B7CBD9", "#5F86A3"])
    ax.imshow(normalized, cmap=cmap, vmin=0, vmax=0.8, aspect="equal")
    for r in range(3):
        for c in range(3):
            val = normalized[r, c]
            color = "white" if val >= 0.48 else INK
            ax.text(c, r, f"{matrix[r,c]}\n{100*val:.1f}%", ha="center", va="center",
                    fontsize=7.6, color=color, linespacing=1.2)
    ax.set_xticks(range(3), ["消极", "中性", "积极"])
    ax.set_yticks(range(3), ["消极", "中性", "积极"])
    ax.set_xlabel("预测类别")
    ax.set_ylabel("")
    ax.tick_params(length=0, pad=3)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)


def draw_zero_subset(ax: plt.Axes, zero: pd.DataFrame) -> None:
    panel_tag(ax, "(C)", "原生视觉全零诊断")
    metrics = [("accuracy", "准确率"), ("macro_f1", "宏平均F1"), ("pearson", "皮尔逊相关")]
    subgs = ax.get_subplotspec().subgridspec(3, 1, hspace=0.23)
    seeds = [42, 43, 44]
    x = np.arange(3, dtype=float)
    colors = {"B0-WCE": BLUE, "B5-P2": ORANGE}
    labels = {"B0-WCE": "基线模型", "B5-P2": "本文模型"}
    axes = []
    for row, (metric, label) in enumerate(metrics):
        a = ax.figure.add_subplot(subgs[row, 0])
        axes.append(a)
        for i, seed in enumerate(seeds):
            vals = {}
            for model in ("B0-WCE", "B5-P2"):
                result = zero[(zero.training_seed == seed) & (zero.model == model)]
                assert len(result) == 1
                vals[model] = float(result.iloc[0][metric])
            a.plot([x[i]-0.055, x[i]+0.055], [vals["B0-WCE"], vals["B5-P2"]],
                   color="#B8BDC1", lw=0.8, zorder=1)
        for model, offset in (("B0-WCE", -0.055), ("B5-P2", 0.055)):
            values = [float(zero[(zero.training_seed == seed) & (zero.model == model)][metric].iloc[0])
                      for seed in seeds]
            a.plot(x + offset, values, marker="o", ms=3.7, lw=1.1,
                   color=colors[model], label=labels[model] if row == 0 else None,
                   mec="white", mew=0.45, zorder=3)
        all_values = zero[metric].to_numpy(dtype=float)
        span = max(0.035, float(all_values.max() - all_values.min()))
        pad = span * 0.28
        a.set_ylim(float(all_values.min()) - pad, float(all_values.max()) + pad)
        a.set_xlim(-0.4, 2.4)
        a.set_xticks(x, ["42", "43", "44"] if row == 2 else [])
        if row == 2:
            a.set_xlabel("随机种子", labelpad=1)
        a.set_ylabel("")
        a.text(0.025, 0.84, label, transform=a.transAxes, ha="left", va="top",
               fontsize=6.7, color=INK,
               bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.6})
        a.grid(axis="y", color=GRID, lw=0.45, alpha=0.8)
        a.set_axisbelow(True)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        if row < 2:
            a.tick_params(axis="x", length=0)
        if row == 0:
            # Model colors are defined once in panel A and shared across panels.
            pass
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)


def main() -> None:
    set_style()
    robust, confusion, zero = source_data()
    fig = plt.figure(figsize=(7.2, 4.45), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.17, 1.12, 1.42],
                          left=0.075, right=0.985, top=0.84, bottom=0.23, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    draw_robust(ax_a, robust)
    draw_confusion(ax_b, confusion)
    draw_zero_subset(ax_c, zero)
    fig.suptitle("图9 随机初始化稳定性与误差分析", y=0.97,
                 fontsize=11.0, fontweight="bold", color=INK)
    fig.text(0.5, 0.085,
             "附件2验证集；(B) 本文模型种子42完整验证预测，行是真实类别、列为预测类别，格内为计数 / 行归一化比例。",
             ha="center", va="center", fontsize=6.5, color=MUTED)
    fig.text(0.5, 0.052,
             "(A) 误差项为三次初始化的样本标准差（非置信区间）；(C) 原生视觉整段全零子集 n=15，按种子配对展示。",
             ha="center", va="center", fontsize=6.5, color=MUTED)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
