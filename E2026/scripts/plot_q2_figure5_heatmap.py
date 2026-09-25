"""Draw the Q2 missing location heatmap from Attachment2 validation summaries.

Figure numbering and the overall title are supplied by the manuscript caption.
"""

from __future__ import annotations

# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})


def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)


mpl.use("Agg")

import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "final" / "q2" / "q2_plotting_handoff_v2" / "plot_data"
OUT = ROOT / "outputs" / "final" / "q2" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BASE = OUT / "figure5_q2_missing_location_heatmap"

FILES = {
    "single_seed": DATA / "fig5_modality_by_location_per_seed_complete.csv",
    "single_summary": DATA / "fig5_modality_by_location_mean_sd.csv",
    "double_seed": DATA / "fig5_double_modality_location_per_seed.csv",
    "double_summary": DATA / "fig5_double_modality_location_mean_sd.csv",
}
MODELS = ("B0-WCE", "B5-P2")
SEEDS = (42, 43, 44)
LOCATIONS = ("early", "middle", "late")
SINGLE = ("text", "audio", "vision")
DOUBLE = ("text+audio", "text+vision", "audio+vision")
METRICS = ("macro_f1", "pearson", "mae", "accuracy")
TITLES = ("宏平均 F1", "皮尔逊相关", "绝对误差改善", "准确率")
CN_ROWS = {
    "text": "文本", "audio": "音频", "vision": "视觉",
    "text+audio": "文本＋音频", "text+vision": "文本＋视觉", "audio+vision": "音频＋视觉",
}
CN_LOCS = ("前段", "中段", "后段")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_group(seed_df: pd.DataFrame, summary: pd.DataFrame, modalities: tuple[str, ...], scenario_count: int) -> None:
    metric_cols = (*METRICS, "selection_score")
    expected_seed = {(m, s, mod, loc) for m in MODELS for s in SEEDS for mod in modalities for loc in LOCATIONS}
    expected_summary = {(m, mod, loc) for m in MODELS for mod in modalities for loc in LOCATIONS}
    actual_seed = list(seed_df[["model", "training_seed", "modalities", "location"]].itertuples(index=False, name=None))
    actual_summary = list(summary[["model", "modalities", "location"]].itertuples(index=False, name=None))
    if len(actual_seed) != 54 or set(actual_seed) != expected_seed or len(set(actual_seed)) != 54:
        raise ValueError("逐种子表不是完整的两模型 × 三种子 × 三模态 × 三位置网格")
    if len(actual_summary) != 18 or set(actual_summary) != expected_summary or len(set(actual_summary)) != 18:
        raise ValueError("均值表不是完整的两模型 × 三模态 × 三位置网格")
    if not (seed_df["scenario_count_averaged"] == scenario_count).all():
        raise ValueError("场景聚合数量与已补齐的 Figure 5 表不一致")
    if not (summary["n_seeds"] == 3).all():
        raise ValueError("均值表未按三种子汇总")
    for df, cols in ((seed_df, metric_cols), (summary, tuple(f"{m}_mean" for m in metric_cols) + tuple(f"{m}_sample_sd" for m in metric_cols))):
        if not np.isfinite(df.loc[:, cols].to_numpy(dtype=float)).all():
            raise ValueError("输入指标含非有限值")
    indexed = summary.set_index(["model", "modalities", "location"])
    for key, group in seed_df.groupby(["model", "modalities", "location"], sort=False):
        row = indexed.loc[key]
        if tuple(sorted(group["training_seed"].tolist())) != SEEDS:
            raise ValueError(f"种子集合错误：{key}")
        for metric in metric_cols:
            if not np.isclose(group[metric].mean(), row[f"{metric}_mean"], atol=1e-12, rtol=0):
                raise ValueError(f"三种子均值与汇总表不一致：{key} {metric}")
            if not np.isclose(group[metric].std(ddof=1), row[f"{metric}_sample_sd"], atol=1e-12, rtol=0):
                raise ValueError(f"三种子 sample SD 与汇总表不一致：{key} {metric}")


def load_deltas() -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    frames = {name: pd.read_csv(path) for name, path in FILES.items()}
    validate_group(frames["single_seed"], frames["single_summary"], SINGLE, 5)
    validate_group(frames["double_seed"], frames["double_summary"], DOUBLE, 1)
    matrices: dict[str, dict[str, np.ndarray]] = {}
    records = []
    for group_name, mod_order, summary in (
        ("single", SINGLE, frames["single_summary"]),
        ("double", DOUBLE, frames["double_summary"]),
    ):
        indexed = summary.set_index(["model", "modalities", "location"])
        matrices[group_name] = {}
        for metric in METRICS:
            matrix = np.empty((3, 3), dtype=float)
            for i, modality in enumerate(mod_order):
                for j, location in enumerate(LOCATIONS):
                    baseline = float(indexed.loc[(MODELS[0], modality, location), f"{metric}_mean"])
                    proposed = float(indexed.loc[(MODELS[1], modality, location), f"{metric}_mean"])
                    matrix[i, j] = baseline - proposed if metric == "mae" else proposed - baseline
            matrices[group_name][metric] = matrix
        for i, modality in enumerate(mod_order):
            for j, location in enumerate(LOCATIONS):
                records.append({"group": group_name, "modalities": modality, "location": location,
                                **{f"delta_{metric}": matrices[group_name][metric][i, j] for metric in METRICS}})
    return matrices, pd.DataFrame.from_records(records)


def make_figure(matrices: dict[str, dict[str, np.ndarray]], values: pd.DataFrame) -> float:
    # Override the baseline's Latin font only for this Chinese paper figure.
    mpl.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"], "axes.unicode_minus": False})
    all_values = values[[f"delta_{m}" for m in METRICS]].to_numpy(dtype=float)
    limit = math.ceil(float(np.max(np.abs(all_values))) / 0.002) * 0.002
    cmap = LinearSegmentedColormap.from_list("q2_cool_white_warm", ["#6D9DB8", "#F9F9F7", "#DEA078"], N=256)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    fig = plt.figure(figsize=(183 / 25.4, 107 / 25.4), facecolor="white")
    grid = fig.add_gridspec(2, 4, left=0.175, right=0.885, bottom=0.145, top=0.90,
                           wspace=0.16, hspace=0.42)
    for row_index, (group_name, modalities) in enumerate((("single", SINGLE), ("double", DOUBLE))):
        for col_index, (metric, title) in enumerate(zip(METRICS, TITLES)):
            ax = fig.add_subplot(grid[row_index, col_index])
            matrix = matrices[group_name][metric]
            ax.pcolormesh(np.arange(4), np.arange(4), matrix, cmap=cmap, norm=norm,
                          shading="flat", edgecolors="white", linewidth=0.9)
            ax.set_xlim(0, 3)
            ax.set_ylim(3, 0)
            ax.set_aspect("equal")
            ax.set_xticks(np.arange(3) + 0.5, CN_LOCS)
            ax.tick_params(axis="x", length=0, pad=2, labelsize=6.9, colors="#333333")
            if col_index == 0:
                ax.set_yticks(np.arange(3) + 0.5, [CN_ROWS[m] for m in modalities])
                ax.tick_params(axis="y", length=0, pad=4, labelsize=6.9, colors="#333333")
            else:
                ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                ax.set_title(title, fontsize=8.2, fontweight="bold", pad=9, color="#2D3338")
            for i in range(3):
                for j in range(3):
                    shown = f"{matrix[i, j]:+.3f}".replace("-", "−")
                    ax.text(j + 0.5, i + 0.5, shown, ha="center", va="center",
                            fontsize=6.25, color="#263039")

    fig.text(0.017, 0.745, "（a）\n单模态缺失", fontsize=7.6, fontweight="bold", va="center", color="#30363B")
    fig.text(0.017, 0.295, "（b）\n双模态缺失", fontsize=7.6, fontweight="bold", va="center", color="#30363B")
    color_axis = fig.add_axes((0.917, 0.22, 0.017, 0.54))
    strip_count = 64
    step = 2 * limit / strip_count
    for i in range(strip_count):
        lower = -limit + i * step
        color_axis.add_patch(Rectangle((0, lower), 1, step, facecolor=cmap(norm(lower + step/2)), edgecolor="none"))
    color_axis.add_patch(Rectangle((0, -limit), 1, 2*limit, facecolor="none", edgecolor="#555555", linewidth=0.5))
    color_axis.set_xlim(0, 1)
    color_axis.set_ylim(-limit, limit)
    color_axis.set_xticks([])
    color_axis.yaxis.tick_right()
    color_axis.yaxis.set_label_position("right")
    color_axis.set_yticks([-limit, -limit/2, 0.0, limit/2, limit],
                         [f"{v:+.3f}".replace("-", "−") if v != 0 else "0" for v in [-limit, -limit/2, 0.0, limit/2, limit]])
    color_axis.tick_params(axis="y", labelsize=6.1, length=2.5, width=0.5, colors="#333333")
    color_axis.set_ylabel("改进值（正值更优）", fontsize=7.1, labelpad=6, color="#333333")
    for spine in color_axis.spines.values():
        spine.set_visible(False)
    fig.text(0.53, 0.055, "三种子均值差；绝对误差按“基线−本文”计算", ha="center", fontsize=6.6, color="#5C6268")

    save_cns_figure(fig, str(BASE))
    svg_path = BASE.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", dpi=300,
                metadata={"Date": "2026-09-24"})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                        encoding="utf-8")
    plt.close(fig)
    return limit


def write_notes(values: pd.DataFrame, limit: float) -> None:
    sign_counts = {metric: {"positive": int((values[f"delta_{metric}"] > 0).sum()),
                            "negative": int((values[f"delta_{metric}"] < 0).sum())} for metric in METRICS}
    sources = "\n".join(f"- `{path.relative_to(ROOT).as_posix()}`（SHA256 `{sha256(path)}`）" for path in FILES.values())
    notes = f"""# Figure 5 热图说明

## 输入

{sources}

逐种子表和均值／样本标准差表已交叉复核：每组为 2 模型 × 3 种子 × 3 模态组合 × 3 位置；单模态单元格在每个种子内平均 5 个预设缺失比例场景，双模态单元格在每个种子内对应 1 个场景。图中不把场景当作独立重复。

## 数值与色标

- 宏平均 F1、皮尔逊相关、准确率：`本文模型三种子均值 − 基线模型三种子均值`。
- 绝对误差改善：`基线模型 MAE 三种子均值 − 本文模型 MAE 三种子均值`。
- 全部 8 个热图共用以零为中心的色标 `[{(-limit):+.3f}, {limit:+.3f}]`；冷色为负、暖色为正。每格显示带符号的三位小数；显示为 `±0.000` 时实际绝对值可能小于 `0.0005`。
- 图中包含准确率面板，以呈现分类指标之间的取舍。未以颜色或数值标记统计显著性。

按 18 个模态×位置组合计，正／负单元格数：宏平均 F1 {sign_counts['macro_f1']['positive']}/{sign_counts['macro_f1']['negative']}，皮尔逊相关 {sign_counts['pearson']['positive']}/{sign_counts['pearson']['negative']}，绝对误差改善 {sign_counts['mae']['positive']}/{sign_counts['mae']['negative']}，准确率 {sign_counts['accuracy']['positive']}/{sign_counts['accuracy']['negative']}。这些是验证集三种子均值的方向统计，不代表每种子或每场景均改善。

## 图注草稿

图5 不同缺失位置与模态组合下本文模型相对基线模型的性能变化。上排为单模态缺失，下排为双模态缺失；色块表示附件2验证集三随机种子均值之差，绝对误差按“基线−本文”计算，因此正值均表示本文模型更优。收益依赖指标及模态／位置组合：宏平均 F1 和相关性多数为正，但准确率在若干组合下降；总体改善幅度较小且存在种子敏感性，不表示所有缺失场景稳定提升。
"""
    (OUT / "figure5_q2_missing_location_heatmap_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    matrices, values = load_deltas()
    values.to_csv(OUT / "figure5_q2_missing_location_heatmap_values.csv", index=False, float_format="%.12g")
    limit = make_figure(matrices, values)
    write_notes(values, limit)
    print(f"FIGURE5_HEATMAP_COMPLETE color_scale=[{-limit:+.3f},{limit:+.3f}] values={len(values)}")


if __name__ == "__main__":
    main()
