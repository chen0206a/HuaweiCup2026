"""Render the final Chinese Q3 faithfulness figure from locked HEAF metrics.

Only language, palette, layout and drawing style change. The figure reads the
original metrics JSON and preserves all plotted means, counts and bootstrap CIs.
The manuscript supplies the figure number and overall title through its caption.
"""

from __future__ import annotations

import hashlib
import json
import shutil
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch


ROOT = Path('D:\\华为杯\\E2026')
OUT = ROOT / "outputs" / "final" / "q3" / "figures"
ARCHIVE = OUT / "archive"
PREVIEW = Path('D:\\华为杯\\outputs\\q3_story_refine\\qa\\figure_preview')
METRICS = ROOT / "outputs" / "q3" / "heaf_validation_metrics.json"
BASE = Path('D:\\华为杯\\outputs\\q3_story_refine\\figures\\q3\\fig15_q3_faithfulness_validation')

TEXT = "#BFDCE6"
VISION = "#E9C9CC"
AUDIO = "#EEE7B0"
MAIN = "#4F7F95"
ACCENT = "#F0B36D"
DARK = "#333333"
GRID = "#D9D9D9"
MODALITIES = (("text", "文本", TEXT), ("vision", "视觉", VISION), ("audio", "语音", AUDIO))
RATIOS = ("0.10", "0.20", "0.30", "0.40")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_previous() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        current = BASE.with_suffix(f".{ext}")
        archived = ARCHIVE / f"figure8_q3_faithfulness_v1.{ext}"
        if not current.is_file():
            raise FileNotFoundError(current)
        if archived.exists():
            # A rerun after the Chinese version exists must leave the original
            # archive untouched, even though the current output now differs.
            continue
        shutil.copy2(current, archived)
        if sha256(current) != sha256(archived):
            raise RuntimeError(f"Cannot verify archived Figure 8 {ext}")
    for ext in ("png", "pdf", "svg"):
        current = BASE.with_suffix(f".{ext}")
        archived = ARCHIVE / f"figure8_q3_faithfulness_scheme_c.{ext}"
        if not archived.exists():
            shutil.copy2(current, archived)


def load_metrics() -> dict:
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    if data["status"] != "HEAF_VALIDATION_PASSED":
        raise RuntimeError("The locked HEAF validation has not passed")
    if data["data"]["n_valid"] != 728:
        raise RuntimeError("Unexpected Attachment2 valid sample count")
    if tuple(data["faithfulness"]["deletion_curve"]) != RATIOS:
        raise RuntimeError("Unexpected deletion-curve fractions")
    for ratio in RATIOS:
        node = data["faithfulness"]["deletion_curve"][ratio]["class_margin"]
        for field in (
            "top_mean", "random_mean", "top_minus_random_mean",
            "top_grouped_bootstrap_95ci", "random_grouped_bootstrap_95ci",
            "top_minus_random_grouped_bootstrap_95ci",
        ):
            if field not in node:
                raise RuntimeError(f"Missing locked metric: {ratio} {field}")
    return data


def style_axis(ax) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.spines["left"].set_color(DARK)
    ax.spines["bottom"].set_color(DARK)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", colors=DARK, labelsize=7, width=0.6, length=2.3)
    ax.xaxis.label.set_color(DARK)
    ax.yaxis.label.set_color(DARK)


def draw_primary_counts(ax, metrics: dict) -> None:
    count = metrics["primary"]
    x = np.arange(3, dtype=float)
    width = 0.35
    colors = [item[2] for item in MODALITIES]
    class_counts = [count["classification_counts"][key] for key, _, _ in MODALITIES]
    reg_counts = [count["regression_counts"][key] for key, _, _ in MODALITIES]
    bars_class = ax.bar(x - width / 2, class_counts, width, color=colors,
                        edgecolor=DARK, linewidth=0.7, hatch="..", zorder=3)
    bars_reg = ax.bar(x + width / 2, reg_counts, width, color=colors,
                      edgecolor=DARK, linewidth=0.7, hatch="...", zorder=3)
    ax.bar_label(bars_class, padding=2, fontsize=7, color=DARK)
    ax.bar_label(bars_reg, padding=2, fontsize=7, color=DARK)
    ax.set_xticks(x, [label for _, label, _ in MODALITIES])
    ax.set_xlabel("模态", labelpad=3)
    ax.set_ylabel("样本数", labelpad=3)
    ax.set_ylim(0, 800)
    ax.set_yticks([0, 200, 400, 600, 800])
    ax.legend(handles=[
        Patch(facecolor=TEXT, edgecolor=DARK, linewidth=0.7, hatch="..", label="分类主导模态"),
        Patch(facecolor=TEXT, edgecolor=DARK, linewidth=0.7, hatch="...", label="回归主导模态"),
    ], loc="upper right", fontsize=6.6, handlelength=1.2, labelspacing=0.25)
    style_axis(ax)


def draw_deletion_curve(ax, metrics: dict) -> None:
    nodes = metrics["faithfulness"]["deletion_curve"]
    x = np.array([10, 20, 30, 40], dtype=float)
    for prefix, name, color in (
        ("top", "高影响位置", MAIN),
        ("random", "随机位置", ACCENT),
    ):
        mean = np.array([nodes[r]["class_margin"][f"{prefix}_mean"] for r in RATIOS])
        ci = np.array([nodes[r]["class_margin"][f"{prefix}_grouped_bootstrap_95ci"] for r in RATIOS])
        ax.fill_between(x, ci[:, 0], ci[:, 1],
                        color=TEXT if prefix == "top" else ACCENT,
                        alpha=0.22 if prefix == "top" else 0.18,
                        linewidth=0, zorder=1)
        ax.plot(x, mean, color=color, linewidth=2.4, marker="o", markersize=4.5,
                markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.45,
                label=name, zorder=3)
    ax.set_xticks(x, ["10%", "20%", "30%", "40%"])
    ax.set_xlabel("删除比例", labelpad=3)
    ax.set_ylabel("分类对数优势下降", labelpad=3)
    ax.set_ylim(-0.035, 0.68)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.legend(loc="upper left", fontsize=7.2, handlelength=2.2, labelspacing=0.45)
    ax.text(0.02, 0.76, "阴影：95%置信区间", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.9, color=DARK)
    style_axis(ax)


def draw_margin_gain(ax, metrics: dict) -> None:
    nodes = metrics["faithfulness"]["deletion_curve"]
    x = np.array([10, 20, 30, 40], dtype=float)
    means = np.array([nodes[r]["class_margin"]["top_minus_random_mean"] for r in RATIOS])
    ci = np.array([nodes[r]["class_margin"]["top_minus_random_grouped_bootstrap_95ci"] for r in RATIOS])
    if not np.all((ci[:, 0] <= means) & (means <= ci[:, 1])):
        raise RuntimeError("Paired bootstrap interval does not contain the reported mean")
    ax.fill_between(x, ci[:, 0], ci[:, 1], color=TEXT, alpha=0.22,
                    linewidth=0, zorder=1)
    ax.errorbar(x, means, yerr=np.vstack((means - ci[:, 0], ci[:, 1] - means)),
                color=MAIN, linewidth=2.4, marker="o", markersize=4.5,
                markerfacecolor=MAIN, markeredgecolor="white", markeredgewidth=0.45,
                elinewidth=0.75, capsize=2.3, zorder=3)
    ax.axhline(0, color="#999999", linewidth=0.6, linestyle="--", zorder=1)
    ax.set_xticks(x, ["10%", "20%", "30%", "40%"])
    ax.set_xlabel("删除比例", labelpad=3)
    ax.set_ylabel("对数优势下降差", labelpad=3)
    ax.set_ylim(-0.045, 0.58)
    ax.set_yticks([0.0, 0.2, 0.4])
    style_axis(ax)


def draw(metrics: dict):
    mm = 1 / 25.4
    fig = plt.figure(figsize=(183 * mm, 110 * mm), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.28],
                  left=0.105, right=0.96, top=0.91, bottom=0.135,
                  wspace=0.37, hspace=0.56)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    draw_primary_counts(ax_a, metrics)
    draw_deletion_curve(ax_b, metrics)
    draw_margin_gain(ax_c, metrics)
    for ax, title in (
        (ax_a, "（A）主导模态分布"),
        (ax_b, "（B）删除比例与分类对数优势变化"),
        (ax_c, "（C）高影响位置与随机位置对比"),
    ):
        ax.set_title(title, loc="left", fontsize=8.2, fontweight="bold", color=DARK, pad=9)
    return fig


def normalize_svg_whitespace(path: Path) -> None:
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                    encoding="utf-8")


def main() -> None:
    metrics = load_metrics()
    fig = draw(metrics)
    fig.savefig(BASE.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches=None)
    fig.savefig(BASE.with_suffix(".pdf"), dpi=300, facecolor="white", bbox_inches=None)
    svg = BASE.with_suffix(".svg")
    fig.savefig(svg, facecolor="white", bbox_inches=None)
    normalize_svg_whitespace(svg)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASE.with_suffix(".png"), PREVIEW / BASE.with_suffix(".png").name)
    plt.close(fig)
    print(f"Wrote Chinese Figure 8 to {BASE}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
