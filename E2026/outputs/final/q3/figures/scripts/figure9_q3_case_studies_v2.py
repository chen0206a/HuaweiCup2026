"""Chinese two-case Figure 9 from frozen Attachment4 explanations and context frames.

The video frames are fixed-duration context images selected by
extract_figure9_context_frames.py. They have no feature-slot grounding claim.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

# Typography, palette and export conventions inherited from Figure 8 / the
# academic-figure-skill LineTrend, GroupedBarChart and multipanel assets.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Arial", "SimHei"],
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
    "hatch.linewidth": 0.55,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": None,  # preserve the intended 183 mm manuscript width
    "savefig.dpi": 300,
})

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / "outputs" / "final" / "q3" / "figures"
DATA = OUT / "data"
ARCHIVE = OUT / "archive"
EXPLANATIONS = ROOT / "outputs" / "q3" / "final" / "attachment4_explanations.jsonl"
MANIFEST = DATA / "figure9_video_frame_manifest.json"
BASE = OUT / "figure9_q3_case_studies_v2"

BLACK = "#333333"
GREY = "#666666"
LIGHT_GREY = "#D9D9D9"
COLORS = {"text": "#BFDCE6", "audio": "#EEE7B0", "vision": "#E9C9CC"}
MAIN = "#4F7F95"
ACCENT = "#F0B36D"
MODALITY_LABELS = {"text": "文本", "audio": "音频", "vision": "视觉"}
CLASS_LABELS = {"Negative": "负面", "Neutral": "中性", "Positive": "正面"}
ORDER = ("text", "audio", "vision")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cards() -> dict[str, dict]:
    cards = {
        row["sample_id"]: row
        for row in (json.loads(s) for s in EXPLANATIONS.read_text(encoding="utf-8").splitlines())
        if row["sample_id"] in ("14", "02")
    }
    if set(cards) != {"14", "02"}:
        raise RuntimeError("Both fixed Figure 9 cases must occur in final JSONL")
    if cards["14"]["primary_modality_classification"]["name"] != "text":
        raise RuntimeError("Sample 14 must be text-primary")
    if cards["02"]["primary_modality_classification"]["name"] != "vision":
        raise RuntimeError("Sample 02 must be vision-primary")
    if cards["14"]["raw_evidence"]["grounding_status"] != "verified":
        raise RuntimeError("Sample 14 text grounding must be verified")
    raw02 = cards["02"]["raw_evidence"]
    if raw02["grounding_status"] != "unverified" or any(
        raw02[k] is not None for k in ("text_fragment", "audio_time_range", "video_frame_time")
    ):
        raise RuntimeError("Sample 02 raw visual grounding must remain unverified and null")
    for sample_id, card in cards.items():
        if card["key_interval"]["modality"] != card["primary_modality_classification"]["name"]:
            raise RuntimeError(f"Key interval modality mismatch: {sample_id}")
    return cards


def validate_text_fragment(card: dict) -> str:
    evidence = card["raw_evidence"]
    match = re.search(
        r"raw_text_character_span=\{'start_char': (\d+), 'end_char': (\d+)\}",
        evidence["mapping_note"],
    )
    if not match:
        raise RuntimeError("Verified sample 14 span is absent")
    start, end = map(int, match.groups())
    video = (ROOT / evidence["media_file"]).resolve()
    feature = video.parent.parent / "14.pkl"
    if not feature.is_file():
        raise RuntimeError(f"Cannot check raw_text span: {feature}")
    with feature.open("rb") as stream:
        raw_text = str(pickle.load(stream)["raw_text"])
    fragment = evidence["text_fragment"]
    if raw_text[start:end] != fragment:
        raise RuntimeError("Text fragment differs from verified raw_text character span")
    return fragment


def validate_frames(cards: dict[str, dict]) -> dict[str, list[Path]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("frame_selection_independent_of_heaf"):
        raise RuntimeError("Frame selection provenance is missing")
    records = manifest["frames"]
    expected = {"14": [0.50], "02": [0.25, 0.50, 0.75]}
    frames = {"14": [], "02": []}
    for sample_id, positions in expected.items():
        sample_records = [r for r in records if r["sample_id"] == sample_id]
        if [r["requested_relative_position"] for r in sample_records] != positions:
            raise RuntimeError(f"Fixed frame positions differ: {sample_id}")
        media = str((ROOT / cards[sample_id]["raw_evidence"]["media_file"]).resolve())
        for record in sample_records:
            if (record["purpose"], record["grounding_status"]) != (
                "context_only", "not_keyframe_mapping"
            ) or record["mp4_path"] != media:
                raise RuntimeError(f"Context-frame provenance mismatch: {sample_id}")
            image = OUT / record["output_frame"]
            if not image.is_file():
                raise FileNotFoundError(image)
            frames[sample_id].append(image)
    if len(records) != 4:
        raise RuntimeError("Exactly four context frames are required")
    return frames


def archive_old_figure() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        original = OUT / f"figure9_q3_case_studies.{extension}"
        archived = ARCHIVE / original.name
        if not original.is_file():
            raise FileNotFoundError(original)
        if archived.exists():
            if sha256(archived) != sha256(original):
                raise RuntimeError(f"Archived old Figure 9 differs: {archived}")
        else:
            shutil.copy2(original, archived)

    # Preserve the already-reviewed two-case layout before the palette-only
    # harmonization with Chinese Figure 8. Reruns never replace this snapshot.
    for extension in ("png", "pdf", "svg"):
        original_v2 = BASE.with_suffix(f".{extension}")
        archived_v2 = ARCHIVE / f"figure9_q3_case_studies_v2_before_scheme_c.{extension}"
        if not archived_v2.exists() and original_v2.is_file():
            shutil.copy2(original_v2, archived_v2)
    for extension in ("png", "pdf", "svg"):
        current = BASE.with_suffix(f".{extension}")
        archived = ARCHIVE / f"figure9_q3_case_studies_v2_scheme_c.{extension}"
        if not archived.exists():
            shutil.copy2(current, archived)


def add_scene(fig, bounds, path: Path) -> None:
    ax = fig.add_axes(bounds)
    ax.imshow(plt.imread(path), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#B5B5B5")
        spine.set_linewidth(0.45)


def add_prediction(fig, card: dict, x: float, y: float) -> None:
    pred = card["prediction"]
    primary = card["primary_modality_classification"]["name"]
    label = CLASS_LABELS[pred["predicted_class_name"]]
    fig.text(x, y, f"预测类别  {label}     情感强度  {pred['predicted_intensity']:+.3f}",
             fontsize=7.4, ha="left", va="top", color=BLACK)
    fig.text(x, y - 0.025, f"置信度  {pred['confidence']:.3f}     分类主模态  {MODALITY_LABELS[primary]}",
             fontsize=7.4, ha="left", va="top", color=BLACK)


def add_shapley(fig, card: dict, bounds) -> None:
    ax = fig.add_axes(bounds)
    vals = [float(card["modality_contribution"]["classification_log_odds"][key]) for key in ORDER]
    y = np.arange(3)
    ax.barh(y, vals, color=[COLORS[key] for key in ORDER], height=0.53,
            edgecolor=BLACK, linewidth=0.7, hatch="..", zorder=2)
    ax.axvline(0, lw=0.65, color=BLACK, zorder=1)
    ax.set_yticks(y, [MODALITY_LABELS[key] for key in ORDER])
    ax.invert_yaxis()
    span = max(0.25, max(abs(value) for value in vals))
    ax.set_xlim(-span * 1.36, span * 1.36)
    for yi, val in enumerate(vals):
        ax.text(val + (0.045 if val >= 0 else -0.045) * span, yi,
                f"{val:+.2f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=7.2, color=BLACK)
    ax.set_xlabel("分类对数赔率贡献", fontsize=7.0, labelpad=1.0)
    ax.tick_params(axis="x", labelsize=6.6, length=2)
    ax.tick_params(axis="y", labelsize=7.2, length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["bottom"].set_linewidth(0.5)


def add_temporal(fig, card: dict, bounds) -> None:
    ax = fig.add_axes(bounds)
    interval = card["key_interval"]
    valid = int(interval["valid_length"])
    # Use the JSONL curve values verbatim and their actual slot indices.
    values = interval["temporal_curve"][:valid]
    if len(values) != valid or not all(v is not None and np.isfinite(v) for v in values):
        raise RuntimeError("Temporal curve does not match valid slot prefix")
    start, end = int(interval["start_index"]), int(interval["end_index"])
    if not (0 <= start < end <= valid):
        raise RuntimeError("Invalid feature-space key interval")
    ax.axvspan(start - 0.5, end - 0.5, color=ACCENT,
               alpha=0.18, lw=0, zorder=0)
    ax.plot(np.arange(valid), values, color=MAIN,
            linewidth=1.8, marker="o", markersize=2.3, zorder=2)
    ax.axhline(0, color="#999999", lw=0.55, ls="--", zorder=1)
    ax.grid(axis="y", color=LIGHT_GREY, lw=0.45, zorder=0)
    ax.set_xlim(-0.5, valid - 0.5)
    ax.set_xlabel("特征槽位", fontsize=7.0, labelpad=1.0)
    ax.set_ylabel("类别边际下降", fontsize=7.0, labelpad=0.5)
    ax.tick_params(axis="both", labelsize=6.6, length=2)
    ax.spines["left"].set_color(BLACK)
    ax.spines["bottom"].set_color(BLACK)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    fig.text(bounds[0], bounds[1] - 0.072, f"关键区间（槽位） [{start}, {end})",
             fontsize=7.0, ha="left", va="top", color=GREY)


def draw(cards: dict[str, dict], frames: dict[str, list[Path]], fragment: str):
    mm = 1 / 25.4
    fig = plt.figure(figsize=(183 * mm, 140 * mm), facecolor="white")
    fig.text(0.06, 0.968, "A", fontsize=9.5, fontweight="bold", ha="left", va="top", color=BLACK)
    fig.text(0.088, 0.968, "文本主导典型案例（样本14）", fontsize=9.0,
             fontweight="bold", ha="left", va="top", color=BLACK)
    fig.text(0.06, 0.487, "B", fontsize=9.5, fontweight="bold", ha="left", va="top", color=BLACK)
    fig.text(0.088, 0.487, "视觉主导边界案例（样本02）", fontsize=9.0,
             fontweight="bold", ha="left", va="top", color=BLACK)
    fig.add_artist(plt.Line2D([0.06, 0.96], [0.509, 0.509], transform=fig.transFigure,
                              color=LIGHT_GREY, linewidth=0.55))

    # Row A. The video still is a scene illustration, independent of the
    # text feature interval and of all HEAF scores.
    add_scene(fig, [0.061, 0.718, 0.288, 0.204], frames["14"][0])
    fig.text(0.061, 0.707, "原视频场景", fontsize=7.2, ha="left", va="top", color=GREY)
    add_prediction(fig, cards["14"], 0.061, 0.673)
    fig.text(0.398, 0.922, "分类 Shapley 贡献", fontsize=8.0,
             fontweight="semibold", ha="left", va="bottom", color=BLACK)
    add_shapley(fig, cards["14"], [0.398, 0.704, 0.259, 0.204])
    fig.text(0.706, 0.922, "时间遮挡曲线", fontsize=8.0,
             fontweight="semibold", ha="left", va="bottom", color=BLACK)
    add_temporal(fig, cards["14"], [0.706, 0.704, 0.251, 0.204])
    fig.add_artist(Rectangle((0.055, 0.559), 0.905, 0.052, transform=fig.transFigure,
                             facecolor=COLORS["text"], edgecolor=LIGHT_GREY,
                             linewidth=0.55, alpha=0.42, zorder=0))
    fig.text(0.061, 0.585, "已验证文本证据", fontsize=7.6,
             fontweight="semibold", ha="left", va="center", color=BLACK)
    fig.text(0.221, 0.585, f"“{fragment}”", fontsize=7.6,
             ha="left", va="center", color=BLACK)

    # Row B. All three stills have the same thin gray border. None is marked
    # as a model-selected frame or aligned with the key feature interval.
    for i, path in enumerate(frames["02"]):
        add_scene(fig, [0.061 + i * 0.098, 0.274, 0.092, 0.140], path)
    fig.text(0.061, 0.262, "原视频上下文（非关键帧定位）", fontsize=7.0,
             ha="left", va="top", color=GREY)
    add_prediction(fig, cards["02"], 0.061, 0.225)
    fig.text(0.398, 0.443, "分类 Shapley 贡献", fontsize=8.0,
             fontweight="semibold", ha="left", va="bottom", color=BLACK)
    add_shapley(fig, cards["02"], [0.398, 0.224, 0.259, 0.204])
    fig.text(0.706, 0.443, "时间遮挡曲线", fontsize=8.0,
             fontweight="semibold", ha="left", va="bottom", color=BLACK)
    add_temporal(fig, cards["02"], [0.706, 0.224, 0.251, 0.204])
    fig.add_artist(Rectangle((0.055, 0.067), 0.905, 0.05, transform=fig.transFigure,
                             facecolor="#F7F7F5", edgecolor=LIGHT_GREY,
                             linewidth=0.55, zorder=0))
    fig.text(0.061, 0.092, "原始视觉位置未验证，仅展示特征槽位区间",
             fontsize=7.6, fontweight="semibold", ha="left", va="center", color=BLACK)
    return fig


def normalize_svg_whitespace(path: Path) -> None:
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                    encoding="utf-8")


def main() -> None:
    archive_old_figure()
    cards = load_cards()
    fragment = validate_text_fragment(cards["14"])
    frames = validate_frames(cards)
    fig = draw(cards, frames, fragment)
    fig.savefig(BASE.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches=None)
    fig.savefig(BASE.with_suffix(".pdf"), dpi=300, facecolor="white", bbox_inches=None)
    svg = BASE.with_suffix(".svg")
    fig.savefig(svg, facecolor="white", bbox_inches=None)
    normalize_svg_whitespace(svg)
    plt.close(fig)
    print(f"Wrote {BASE}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
