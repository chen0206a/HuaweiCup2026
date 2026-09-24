# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (A) deletion trend → assets/figures/LineTrend/plot_trend.py → param inherit (source data shape differs)
# (B) grouped modality counts → assets/figures/GroupedBarChart/plot_GroupedBarChartv1.py → param inherit (source data shape differs)
# (C) top-minus-random comparison → assets/figures/BarComparison/plot_comparison_Trajectory.py → param inherit (source data shape differs)
# Multipanel layout → assets/figures/multipanel/ is absent locally; use matplotlib GridSpec with asymmetric mixed layout.
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.

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

from pathlib import Path
import csv
import json
import shutil
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[5]
Q3 = ROOT / "outputs" / "q3"
OUT = ROOT / "outputs" / "final" / "q3" / "figures"
DATA = OUT / "data"
PREVIEW = OUT / "preview"
METRICS_PATH = Q3 / "heaf_validation_metrics.json"
MODALITIES = ["Text", "Audio", "Vision"]
MODALITY_KEYS = ["text", "audio", "vision"]
TOP = CATEGORICAL[0]
RANDOM = GREY


def write_rows(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_svg_whitespace(path):
    svg = Path(path)
    lines = svg.read_text(encoding="utf-8").splitlines()
    svg.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def prepare_data(metrics):
    faith = metrics["faithfulness"]
    curve_rows = []
    gain_rows = []
    for ratio in ["0.10", "0.20", "0.30", "0.40"]:
        entry = faith["deletion_curve"][ratio]
        cm = entry["class_margin"]
        for method, prefix, color in [
            ("Top interval deletion", "top", TOP),
            ("Random same-length deletion", "random", RANDOM),
        ]:
            ci = cm[f"{prefix}_grouped_bootstrap_95ci"]
            curve_rows.append({
                "deletion_ratio_percent": int(float(ratio) * 100),
                "method": method,
                "mean_class_margin_drop": cm[f"{prefix}_mean"],
                "median_class_margin_drop": cm[f"{prefix}_median"],
                "ci95_lower": ci[0],
                "ci95_upper": ci[1],
                "n_samples": entry["n_samples"],
                "n_video_ids": entry["n_video_ids"],
                "bootstrap_unit": entry["bootstrap_unit"],
                "bootstrap_reps": entry["bootstrap_reps"],
                "source": "Attachment2 validation audit; frozen B5-P2 seed42",
            })
        delta_ci = cm["top_minus_random_grouped_bootstrap_95ci"]
        gain_rows.append({
            "deletion_ratio_percent": int(float(ratio) * 100),
            "top_minus_random_mean_class_margin_drop": cm["top_minus_random_mean"],
            "top_minus_random_median_class_margin_drop": cm["top_minus_random_median"],
            "ci95_lower": delta_ci[0],
            "ci95_upper": delta_ci[1],
            "fraction_top_gt_random": cm["fraction_top_gt_random"],
            "negative_effect_fraction": cm["negative_effect_fraction"],
            "n_samples": entry["n_samples"],
            "n_video_ids": entry["n_video_ids"],
            "bootstrap_unit": entry["bootstrap_unit"],
            "bootstrap_reps": entry["bootstrap_reps"],
            "source": "Attachment2 validation audit; frozen B5-P2 seed42",
        })

    primary = metrics["primary"]
    class_counts = primary["classification_counts"]
    reg_counts = primary["regression_counts"]
    count_rows = []
    for modality, key in zip(MODALITIES, MODALITY_KEYS):
        count_rows.append({
            "modality": modality,
            "classification_primary_count": class_counts[key],
            "regression_primary_count": reg_counts[key],
            "n_valid": metrics["data"]["n_valid"],
            "source": "Attachment2 validation; all valid samples",
        })

    write_rows(DATA / "figure8_deletion_curve.csv", curve_rows, list(curve_rows[0]))
    write_rows(DATA / "figure8_primary_modality_counts.csv", count_rows, list(count_rows[0]))
    write_rows(DATA / "figure8_margin_gain_summary.csv", gain_rows, list(gain_rows[0]))
    return curve_rows, count_rows, gain_rows


def panel_label(ax, label):
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold", color=BLACK)


def draw_figure(metrics, curve_rows, count_rows, gain_rows):
    mm = 1 / 25.4
    fig = plt.figure(figsize=(183 * mm, 104 * mm), constrained_layout=False)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.45, 1.0], height_ratios=[1, 0.88],
                  left=0.085, right=0.985, top=0.92, bottom=0.15, wspace=0.40, hspace=0.64)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    ratios = np.array([10, 20, 30, 40])
    for method, color, marker in [("Top interval deletion", TOP, "o"),
                                  ("Random same-length deletion", RANDOM, "s")]:
        rows = [r for r in curve_rows if r["method"] == method]
        means = np.array([r["mean_class_margin_drop"] for r in rows])
        lower = np.array([r["ci95_lower"] for r in rows])
        upper = np.array([r["ci95_upper"] for r in rows])
        ax_a.plot(ratios, means, color=color, lw=1.8, marker=marker, ms=3.8,
                  label=method, zorder=3)
        ax_a.fill_between(ratios, lower, upper, color=color, alpha=0.15, linewidth=0, zorder=1)
    ax_a.set_xticks(ratios, [f"{x}%" for x in ratios])
    ax_a.set_xlabel("Deletion ratio")
    ax_a.set_ylabel("Class-margin drop")
    ax_a.set_title("Deletion faithfulness on validation audit", loc="left", pad=7)
    ax_a.legend(loc="upper left", frameon=False, fontsize=6.3, handlelength=1.8)
    ax_a.text(0.99, 0.02, "Mean ± video-group bootstrap 95% CI\n396 clips · 126 video IDs",
              transform=ax_a.transAxes, ha="right", va="bottom", fontsize=5.8, color="#555555")
    ax_a.grid(axis="y", color="#E5E5E5", lw=0.4, zorder=0)
    panel_label(ax_a, "A")

    x = np.arange(3)
    width = 0.34
    class_vals = np.array([r["classification_primary_count"] for r in count_rows])
    reg_vals = np.array([r["regression_primary_count"] for r in count_rows])
    bars1 = ax_b.bar(x - width / 2, class_vals, width, color=TOP, label="Classification", zorder=3)
    bars2 = ax_b.bar(x + width / 2, reg_vals, width, color="#6B879B", label="Regression", zorder=3)
    ax_b.bar_label(bars1, padding=2, fontsize=6)
    ax_b.bar_label(bars2, padding=2, fontsize=6)
    ax_b.set_xticks(x, MODALITIES)
    ax_b.set_ylabel("Primary modality count")
    ax_b.set_ylim(0, max(class_vals.max(), reg_vals.max()) * 1.18)
    ax_b.set_title("Primary modality counts (n = 728)", loc="left", pad=7)
    ax_b.legend(frameon=False, fontsize=6.1, loc="upper right", handlelength=1.2)
    ax_b.grid(axis="y", color="#E5E5E5", lw=0.4, zorder=0)
    panel_label(ax_b, "B")

    means = np.array([r["top_minus_random_mean_class_margin_drop"] for r in gain_rows])
    low = np.array([r["ci95_lower"] for r in gain_rows])
    high = np.array([r["ci95_upper"] for r in gain_rows])
    yerr = np.vstack([means - low, high - means])
    ax_c.errorbar(ratios, means, yerr=yerr, color=TOP, lw=1.5, marker="o", ms=3.8,
                 capsize=2.2, elinewidth=0.8, zorder=3)
    ax_c.axhline(0, color=GREY, lw=0.6, ls="--", zorder=1)
    ax_c.set_xticks(ratios, [f"{x}%" for x in ratios])
    ax_c.set_xlabel("Deletion ratio")
    ax_c.set_ylabel("Top − random margin drop")
    ax_c.set_title("Average margin gain", loc="left", pad=7)
    ax_c.grid(axis="y", color="#E5E5E5", lw=0.4, zorder=0)
    panel_label(ax_c, "C")

    fig.text(0.5, 0.035,
             "Top interval deletion changes the frozen model margin more than same-length random deletion; error bars are video-group bootstrap 95% CIs.",
             ha="center", va="bottom", fontsize=6.0, color="#444444")
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if metrics["status"] != "HEAF_VALIDATION_PASSED":
        raise RuntimeError("Q3 HEAF validation status is not passed")
    curve_rows, count_rows, gain_rows = prepare_data(metrics)
    fig = draw_figure(metrics, curve_rows, count_rows, gain_rows)
    base = OUT / "figure8_q3_faithfulness"
    save_cns_figure(fig, str(base))
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    normalize_svg_whitespace(f"{base}.svg")
    shutil.copy2(f"{base}.png", PREVIEW / "figure8_q3_faithfulness.png")
    plt.close(fig)
    print(f"Wrote Figure 8 and source data to {OUT}")


if __name__ == "__main__":
    main()
