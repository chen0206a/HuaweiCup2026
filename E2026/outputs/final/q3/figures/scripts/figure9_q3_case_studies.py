# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (A-D) case-study panels → assets/figures/LineTrend/plot_trend.py → param inherit for temporal line styling.
# (A-D) modality and interaction bars → assets/figures/GroupedBarChart/ and BarComparison/ → param inherit for restrained bars.
# Asymmetric multipanel → assets/figures/multipanel/ contains no production script; use matplotlib GridSpec.
# Heatmap asset was reviewed but not used: three signed pair-interaction values are clearer as direct bars.
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
import json
import shutil
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "outputs" / "final" / "q3" / "figures"
DATA = OUT / "data"
PREVIEW = OUT / "preview"
CARDS = ROOT / "outputs" / "q3" / "final" / "attachment4_explanations.jsonl"
SUMMARY = ROOT / "outputs" / "q3" / "final" / "attachment4_summary.json"
MODALITY_ORDER = ["text", "audio", "vision"]
MODALITY_LABELS = ["Text", "Audio", "Vision"]
MODALITY_COLORS = {"text": CATEGORICAL[0], "audio": CATEGORICAL[3], "vision": CATEGORICAL[2]}
INTERACTION_KEYS = ["text_audio", "text_vision", "audio_vision"]
INTERACTION_LABELS = ["T–A", "T–V", "A–V"]
INTERACTION_COLORS = [CATEGORICAL[0], CATEGORICAL[4], CATEGORICAL[3]]
CASE_MAP = {
    "14": ("text_primary_high_faithfulness", "Text-primary high-faithfulness hero case"),
    "02": ("vision_primary", "Vision-primary case; raw-media grounding unavailable"),
    "16": ("strong_interaction", "Strongest absolute classification pair interaction"),
    "19": ("weak_or_failure_boundary", "Weak/failure boundary at q=10% deletion"),
}


def sample_key(sample_id):
    return str(sample_id).strip().lstrip("0") or "0"


def normalize_svg_whitespace(path):
    svg = Path(path)
    lines = svg.read_text(encoding="utf-8").splitlines()
    svg.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def prepare_cards(summary):
    cards = {}
    for line in CARDS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cards[sample_key(row["sample_id"])] = row
    nominations = summary["unlabeled_summary"]["typical_explanation_candidates"]
    expected = {key: sample_key(nominations[name]["sample_id"])
                for key, (name, _) in CASE_MAP.items() if nominations.get(name) is not None}
    for key, expected_id in expected.items():
        if sample_key(key) != expected_id:
            raise ValueError(f"Locked representative selection mismatch: {key} != {expected_id}")

    for key, (selection_name, description) in CASE_MAP.items():
        canonical_key = sample_key(key)
        if canonical_key not in cards:
            raise KeyError(f"Attachment4 explanation card missing locked case ID {key}")
        card = dict(cards[canonical_key])
        candidate = nominations.get(selection_name)
        card["figure_case"] = {
            "panel": {"14": "A", "02": "B", "16": "C", "19": "D"}[key],
            "description": description,
            "selection_provenance": candidate,
            "selection_source": "outputs/q3/final/attachment4_summary.json representative_candidates",
        }
        (DATA / f"figure9_sample{key}_card.json").write_text(
            json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return {key: cards[sample_key(key)] for key in CASE_MAP}


def panel_label(ax, label):
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold", color=BLACK)


def format_score(value, digits=2):
    return f"{value:+.{digits}f}"


def draw_contribution(ax, card, show_regression=False, compact=False):
    values = card["modality_contribution"]["classification_log_odds"]
    y = np.arange(3)
    labels = MODALITY_LABELS
    vals = [values[k] for k in MODALITY_ORDER]
    ax.barh(y, vals, color=[MODALITY_COLORS[k] for k in MODALITY_ORDER], height=0.60, zorder=3)
    ax.axvline(0, color=GREY, lw=0.55, zorder=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    if not compact:
        ax.set_xlabel("Classification Shapley (log-odds)", labelpad=2)
    ax.tick_params(axis="both", labelsize=5.5 if compact else 6.2, length=2)
    ax.grid(axis="x", color="#E5E5E5", lw=0.35, zorder=0)
    for yi, val in zip(y, vals):
        offset = 0.025 if val >= 0 else -0.025
        ax.text(val + offset, yi, f"{val:+.2f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=5.2 if compact else 5.7)
    if show_regression:
        reg = card["modality_contribution"]["regression"]
        ax.text(0.0, -0.34,
                "Regression φ  " + " · ".join(f"{lab[0]} {format_score(reg[k], 2)}" for lab, k in zip(labels, MODALITY_ORDER)),
                transform=ax.transAxes, ha="left", va="top", fontsize=5.4, color="#444444")
    ax.margins(x=0.20)


def draw_interactions(ax, card, compact=False):
    vals = [card["pairwise_interaction"]["classification_log_odds"][k]["value"] for k in INTERACTION_KEYS]
    colors = list(INTERACTION_COLORS)
    strongest = int(np.argmax(np.abs(vals)))
    colors[strongest] = CATEGORICAL[4]
    y = np.arange(3)
    ax.barh(y, vals, color=colors, height=0.58, zorder=3)
    ax.axvline(0, color=GREY, lw=0.55, zorder=1)
    ax.set_yticks(y, INTERACTION_LABELS)
    ax.invert_yaxis()
    if not compact:
        ax.set_xlabel("Pair interaction", labelpad=2)
    ax.tick_params(axis="both", labelsize=5.3 if compact else 5.8, length=2)
    ax.grid(axis="x", color="#E5E5E5", lw=0.35, zorder=0)
    for yi, val in zip(y, vals):
        offset = 0.025 if val >= 0 else -0.025
        ax.text(val + offset, yi, f"{val:+.2f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=5.0 if compact else 5.4)
    ax.margins(x=0.24)


def draw_temporal(ax, card, compact=False):
    interval = card["key_interval"]
    curve = interval["temporal_curve"]
    valid_length = int(interval["valid_length"])
    finite_indices = [i for i, val in enumerate(curve) if val is not None and np.isfinite(val)]
    xs = np.array(finite_indices, dtype=float) + int(interval["window_length"]) / 2
    ys = np.array([curve[i] for i in finite_indices], dtype=float)
    ax.axvspan(interval["start_index"], interval["end_index"], color="#F1A340", alpha=0.18, lw=0, zorder=0)
    ax.plot(xs, ys, color=CATEGORICAL[0], lw=1.15 if compact else 1.45, marker="o",
            ms=1.8 if compact else 2.4, zorder=3)
    ax.axhline(0, color=GREY, lw=0.5, ls="--", zorder=1)
    ax.set_xlim(0, valid_length)
    ax.set_xlabel("Window center slot" if compact else "Sliding-window center slot", labelpad=2)
    ax.set_ylabel("Margin drop" if compact else "Class-margin drop", labelpad=2)
    ax.tick_params(axis="both", labelsize=5.3 if compact else 6.0, length=2)
    ax.grid(axis="y", color="#E5E5E5", lw=0.35, zorder=0)
    ax.text(0.99, 0.97,
            f"Key feature interval [{interval['start_index']}, {interval['end_index']})",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.2 if compact else 5.7,
            color="#654000")


def draw_prediction_text(ax, card):
    pred = card["prediction"]
    pclass = card["primary_modality_classification"]["name"].title()
    rclass = card["primary_modality_regression"]["name"].title()
    ax.axis("off")
    ax.text(0.02, 0.98, f"{pred['predicted_class_name']}", va="top", ha="left",
            fontsize=13, fontweight="bold", color=BLACK)
    ax.text(0.02, 0.78, f"Intensity  {pred['predicted_intensity']:+.3f}", va="top", fontsize=7.0)
    ax.text(0.02, 0.62, f"Confidence  {pred['confidence']:.3f}", va="top", fontsize=7.0)
    ax.text(0.02, 0.39, f"Class primary  {pclass}", va="top", fontsize=6.4)
    ax.text(0.02, 0.25, f"Regression primary  {rclass}", va="top", fontsize=6.4)
    q10 = card["faithfulness"]["deletion_curve"][0]
    q10_gain = q10["top"]["class_log_odds"] - q10["random_mean"]["class_log_odds"]
    ax.text(0.02, 0.10,
            f"Key [{card['key_interval']['start_index']}, {card['key_interval']['end_index']}) · q10 Δmargin {q10_gain:+.3f}",
            va="top", fontsize=5.8, color="#555555")


def draw_hero(fig, outer, card):
    inner = outer.subgridspec(1, 3, width_ratios=[0.75, 1.15, 1.65], wspace=0.46)
    ax_card = fig.add_subplot(inner[0, 0])
    mid = inner[0, 1].subgridspec(2, 1, height_ratios=[1.0, 0.85], hspace=0.78)
    ax_shap = fig.add_subplot(mid[0, 0])
    ax_inter = fig.add_subplot(mid[1, 0])
    ax_time = fig.add_subplot(inner[0, 2])
    draw_prediction_text(ax_card, card)
    draw_contribution(ax_shap, card, show_regression=True)
    ax_shap.set_title("Classification Shapley", loc="left", pad=3, fontsize=6.5)
    draw_interactions(ax_inter, card)
    ax_inter.set_title("Pair interactions", loc="left", pad=3, fontsize=6.5)
    draw_temporal(ax_time, card)
    ax_time.set_title("Temporal occlusion curve", loc="left", pad=3, fontsize=7)
    evidence = card["raw_evidence"]
    if evidence["grounding_status"] != "verified" or not evidence["text_fragment"]:
        raise ValueError("Hero sample 14 must retain verified text evidence")
    snippet = evidence["text_fragment"]
    note = evidence["mapping_note"]
    import re
    match = re.search(r"raw_text_character_span=\{'start_char': (\d+), 'end_char': (\d+)\}", note)
    span_text = f"character span [{match.group(1)}, {match.group(2)})" if match else "verified tokenizer-offset span"
    return f'Verified text evidence · {span_text}:  “{snippet}”'


def draw_small_case(fig, spec, card, panel, case_type, case_title):
    sub = spec.subgridspec(4, 1, height_ratios=[0.24, 0.44, 0.86, 0.22], hspace=0.42)
    ax_header = fig.add_subplot(sub[0, 0])
    ax_header.axis("off")
    ax_header.set_title(f"{panel}  {case_title}", loc="left", pad=1, fontsize=6.3, fontweight="semibold")
    pred = card["prediction"]
    ax_header.text(0.0, 0.94,
                   f"{pred['predicted_class_name']}  ·  intensity {pred['predicted_intensity']:+.2f}  ·  conf. {pred['confidence']:.2f}",
                   va="top", fontsize=5.8, fontweight="semibold", color=BLACK)
    ax1 = fig.add_subplot(sub[1, 0])
    ax2 = fig.add_subplot(sub[2, 0])
    ax_note = fig.add_subplot(sub[3, 0])
    ax_note.axis("off")
    if case_type == "vision":
        ax_note.add_patch(plt.Rectangle((0, 0.03), 1, 0.94, transform=ax_note.transAxes,
                                        facecolor="#F0F0F0", edgecolor="#BDBDBD", lw=0.5, clip_on=False))
    if case_type == "interaction":
        draw_interactions(ax1, card, compact=True)
        ax1.set_title("Classification pair interactions", loc="left", pad=1, fontsize=6.2)
    else:
        draw_contribution(ax1, card, compact=True)
        ax1.set_title("Classification modality Shapley", loc="left", pad=1, fontsize=6.2)
    draw_temporal(ax2, card, compact=True)
    if case_type == "vision":
        ax_note.text(0, 0.5,
                     "Primary modality: Vision\nRaw-media grounding: UNVERIFIED\nOnly feature-space temporal evidence is available.",
                     ha="left", va="center", fontsize=5.0, color="#444444", linespacing=1.1)
    elif case_type == "interaction":
        vals = [card["pairwise_interaction"]["classification_log_odds"][k]["value"] for k in INTERACTION_KEYS]
        strongest_idx = int(np.argmax(np.abs(vals)))
        ax_note.text(0, 0.5, f"Largest |interaction|: {INTERACTION_LABELS[strongest_idx]}\nValue: {vals[strongest_idx]:+.3f}",
                     ha="left", va="center", fontsize=5.0, color="#444444", linespacing=1.15)
    elif case_type == "weak":
        q10 = card["faithfulness"]["deletion_curve"][0]["top"]["class_log_odds"] - card["faithfulness"]["deletion_curve"][0]["random_mean"]["class_log_odds"]
        ax_note.text(0, 0.5, f"Boundary / weaker explanation\nq=10% top−random margin: {q10:+.3f}",
                     ha="left", va="center", fontsize=5.0, color="#444444", linespacing=1.15)
def draw_figure(cards):
    mm = 1 / 25.4
    fig = plt.figure(figsize=(183 * mm, 205 * mm), constrained_layout=False)
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.12, 0.16, 1.35],
                  left=0.075, right=0.985, top=0.955, bottom=0.065,
                  wspace=0.33, hspace=0.27)
    fig.text(0.075, 0.965, "A  Text-primary hero case · Sample 14", ha="left", va="bottom",
             fontsize=7.2, fontweight="semibold", color=BLACK)
    quote = draw_hero(fig, gs[0, :], cards["14"])
    qax = fig.add_subplot(gs[1, :])
    qax.axis("off")
    qax.add_patch(plt.Rectangle((0, 0.08), 1, 0.83, transform=qax.transAxes,
                                facecolor="#EDF4F8", edgecolor="#CBD9E2", lw=0.55, clip_on=False))
    qax.text(0.018, 0.50, quote, transform=qax.transAxes, ha="left", va="center",
             fontsize=6.4, color="#263746")

    panels = [
        ("02", "B", "vision", "Vision-primary · Sample 02"),
        ("16", "C", "interaction", "Strong interaction · Sample 16"),
        ("19", "D", "weak", "Weak boundary · Sample 19"),
    ]
    for col, (key, panel, case_type, case_title) in enumerate(panels):
        draw_small_case(fig, gs[2, col], cards[key], panel, case_type, case_title)

    fig.text(0.5, 0.016,
             "Unverified A/V evidence remains in feature-slot space; no frame times are inferred. Curves show class-margin change under continuous-window occlusion.",
             ha="center", va="bottom", fontsize=5.4, color="#555555")
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if summary["status"] != "Q3_FINAL_INFERENCE_COMPLETE":
        raise RuntimeError("Attachment4 final inference status is not complete")
    cards = prepare_cards(summary)
    fig = draw_figure(cards)
    base = OUT / "figure9_q3_case_studies"
    save_cns_figure(fig, str(base))
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    normalize_svg_whitespace(f"{base}.svg")
    shutil.copy2(f"{base}.png", PREVIEW / "figure9_q3_case_studies.png")
    plt.close(fig)
    print(f"Wrote Figure 9 and case-card JSON files to {OUT}")


if __name__ == "__main__":
    main()
