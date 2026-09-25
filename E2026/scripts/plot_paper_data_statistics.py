"""Generate Q1/Q2 descriptive figures for the E2026 manuscript.

All plotted observations are read from the final Q1 manifest plus label workbook,
and from Q2 train/valid arrays in aligned_50.pkl. For Q2, the test split is used
only to count sample IDs; no test labels or features are accessed.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
E2026 = WORKSPACE / "E2026"
PAPER = (
    WORKSPACE
    / "Q1_论文绘图数据与LaTeX交付包"
    / "Q1_论文绘图数据与LaTeX交付包"
    / "paper"
)
Q1_MANIFEST = (
    PAPER.parent
    / "outputs"
    / "q1_final_local"
    / "05_appendix"
    / "q1_sample_summary_100.csv"
)
Q2_PICKLE = E2026 / "data" / "raw" / "aligned_50.pkl"
Q1_FIG_DIR = PAPER / "figures" / "q1"
Q2_FIG_DIR = PAPER / "figures" / "q2"

BLUE = "#4F7F95"
BLUE_LIGHT = "#BFDCE6"
BLUE_PALE = "#E4F0F4"
ORANGE = "#F0B36D"
ORANGE_LIGHT = "#F6DDBB"
GREEN = "#78A98B"
INK = "#333333"
GRID = "#D9D9D9"
MUTED = "#7A8288"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_label_workbook() -> Path:
    matches = list(
        (WORKSPACE / "第二十三届中国研究生数学建模竞赛 - 中文题目").rglob(
            "label-100.xlsx"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one label-100.xlsx, found {len(matches)}: {matches}"
        )
    return matches[0]


def style_axes(ax: plt.Axes, *, ygrid: bool = True) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#858B8F")
    ax.spines["bottom"].set_color("#858B8F")
    ax.tick_params(colors=INK, labelsize=8.2, length=3, width=0.7)
    ax.grid(ygrid, axis="y", color=GRID, linewidth=0.55, alpha=0.82)
    ax.set_axisbelow(True)


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=10, color=INK, pad=7, weight="semibold")


def export_figure(fig: plt.Figure, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_q1() -> dict:
    workbook = find_label_workbook()
    manifest = pd.read_csv(Q1_MANIFEST)
    labels = pd.read_excel(workbook, sheet_name="label")

    required_manifest = {
        "sample_id",
        "video_id",
        "clip_id",
        "duration",
        "text_feature_source",
        "timestamp_fallback_count",
        "text_valid_bins",
        "audio_valid_bins",
        "vision_valid_bins",
    }
    required_labels = {"video_id", "clip_id", "label", "annotation"}
    if not required_manifest.issubset(manifest.columns):
        raise ValueError(f"Q1 manifest columns missing: {required_manifest - set(manifest.columns)}")
    if not required_labels.issubset(labels.columns):
        raise ValueError(f"Q1 label workbook columns missing: {required_labels - set(labels.columns)}")
    if len(manifest) != 100 or manifest["sample_id"].nunique() != 100:
        raise ValueError("Q1 final manifest must contain 100 unique sample IDs")
    if manifest[["video_id", "clip_id"]].duplicated().any() or labels[["video_id", "clip_id"]].duplicated().any():
        raise ValueError("Q1 source contains duplicated video_id/clip_id keys")

    joined = manifest.merge(
        labels[["video_id", "clip_id", "label", "annotation"]],
        on=["video_id", "clip_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(joined) != 100 or not (joined["_merge"] == "both").all():
        raise ValueError("Q1 manifest and label workbook do not match one-to-one")
    joined = joined.drop(columns="_merge")
    if joined["annotation"].isna().any() or not set(joined["annotation"]).issubset(
        {"Negative", "Neutral", "Positive"}
    ):
        raise ValueError("Unexpected Q1 annotation values")
    if not np.isfinite(joined["duration"].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite Q1 duration")
    for modality in ("text", "audio", "vision"):
        values = joined[f"{modality}_valid_bins"].to_numpy(dtype=int)
        if np.any((values < 0) | (values > 50)):
            raise ValueError(f"Invalid {modality} valid-bin count")

    source_csv = Q1_FIG_DIR / "fig05_q1_data_statistics_source.csv"
    source_cols = [
        "sample_id",
        "video_id",
        "clip_id",
        "duration",
        "label",
        "annotation",
        "text_valid_bins",
        "audio_valid_bins",
        "vision_valid_bins",
        "text_feature_source",
        "timestamp_fallback_count",
    ]
    joined[source_cols].sort_values("sample_id").to_csv(source_csv, index=False)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "SimHei"],
            "font.size": 9,
            "axes.labelcolor": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.0))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.11, wspace=0.31, hspace=0.38)

    ax = axes[0, 0]
    durations = joined["duration"].to_numpy(dtype=float)
    ax.hist(durations, bins=12, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=0.65)
    mean_duration = float(durations.mean())
    median_duration = float(np.median(durations))
    ax.axvline(mean_duration, color=ORANGE, linewidth=1.25, linestyle="--", label=f"均值 {mean_duration:.2f} s")
    ax.axvline(median_duration, color=BLUE, linewidth=1.25, linestyle=":", label=f"中位数 {median_duration:.2f} s")
    ax.set_xlabel("原始视频时长 / s", fontsize=9)
    ax.set_ylabel("样本数", fontsize=9)
    ax.legend(frameon=False, fontsize=7.8, loc="upper right")
    panel_title(ax, "a", "原始视频时长分布")
    style_axes(ax)

    ax = axes[0, 1]
    categories = ["Negative", "Neutral", "Positive"]
    counts = joined["annotation"].value_counts().reindex(categories, fill_value=0)
    bars = ax.bar(["消极", "中性", "积极"], counts.to_numpy(), color=[BLUE, ORANGE, GREEN], width=0.58, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, padding=2, fontsize=8.2, color=INK)
    ax.set_ylabel("样本数", fontsize=9)
    ax.set_ylim(0, max(counts.to_numpy()) * 1.22)
    panel_title(ax, "b", "情感类别分布")
    style_axes(ax)

    ax = axes[1, 0]
    modality_cols = ["text_valid_bins", "audio_valid_bins", "vision_valid_bins"]
    data = [joined[c].to_numpy(dtype=float) for c in modality_cols]
    positions = np.arange(1, 4)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.2},
        whiskerprops={"color": MUTED, "linewidth": 0.8},
        capprops={"color": MUTED, "linewidth": 0.8},
    )
    for patch, color in zip(bp["boxes"], [BLUE_LIGHT, ORANGE_LIGHT, "#CBE3D3"]):
        patch.set_facecolor(color)
        patch.set_edgecolor(MUTED)
        patch.set_linewidth(0.8)
    rng = np.random.default_rng(20260925)
    for pos, values, color in zip(positions, data, [BLUE, ORANGE, GREEN]):
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        ax.scatter(pos + jitter, values, s=9, alpha=0.60, color=color, edgecolors="white", linewidths=0.25, zorder=3)
    ax.set_xticks(positions, ["文本", "语音", "视觉"])
    ax.set_ylim(0, 52)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_ylabel("有效时间窗数", fontsize=9)
    panel_title(ax, "c", "三模态有效时间窗")
    style_axes(ax)

    ax = axes[1, 1]
    sources = ["official_transcript", "media_asr"]
    no_fallback = []
    fallback = []
    for source in sources:
        rows = joined[joined["text_feature_source"] == source]
        no_fallback.append(int((rows["timestamp_fallback_count"] == 0).sum()))
        fallback.append(int((rows["timestamp_fallback_count"] > 0).sum()))
    x = np.arange(2)
    ax.bar(x, no_fallback, color=BLUE, width=0.55, label="无回退记录")
    ax.bar(x, fallback, bottom=no_fallback, color=ORANGE, width=0.55, label="有回退记录")
    for index, (base, top) in enumerate(zip(no_fallback, fallback)):
        if base:
            ax.text(index, base / 2, str(base), ha="center", va="center", color="white", fontsize=7.5, weight="semibold")
    ax.set_xticks(x, [f"官方文本\n回退 {fallback[0]}", f"媒体ASR\n回退 {fallback[1]}"])
    ax.set_ylabel("样本数", fontsize=9)
    ax.set_ylim(0, max(np.array(no_fallback) + np.array(fallback)) * 1.23)
    ax.legend(frameon=False, fontsize=7.8, loc="upper center", ncol=2)
    panel_title(ax, "d", "文本来源与时间戳回退")
    style_axes(ax)

    stem = "fig05_q1_data_statistics"
    export_figure(fig, Q1_FIG_DIR, stem)
    stats = {
        "rows": int(len(joined)),
        "unique_sample_ids": int(joined["sample_id"].nunique()),
        "manifest_xlsx_one_to_one": True,
        "duration_s": {
            "mean": mean_duration,
            "median": median_duration,
            "min": float(durations.min()),
            "max": float(durations.max()),
        },
        "class_counts": {k: int(v) for k, v in counts.items()},
        "valid_bins": {
            modality: {
                "mean": float(joined[f"{modality}_valid_bins"].mean()),
                "median": float(joined[f"{modality}_valid_bins"].median()),
                "min": int(joined[f"{modality}_valid_bins"].min()),
                "max": int(joined[f"{modality}_valid_bins"].max()),
            }
            for modality in ("text", "audio", "vision")
        },
        "text_sources": {
            source: {
                "total": int(no_fallback[i] + fallback[i]),
                "no_fallback": no_fallback[i],
                "fallback": fallback[i],
            }
            for i, source in enumerate(sources)
        },
        "timestamp_fallback_record_counts": {
            str(int(k)): int(v)
            for k, v in joined["timestamp_fallback_count"].value_counts().sort_index().items()
        },
        "data_sources": {
            "manifest": str(Q1_MANIFEST.relative_to(WORKSPACE)),
            "label_workbook": str(workbook.relative_to(WORKSPACE)),
            "manifest_sha256": sha256(Q1_MANIFEST),
            "label_workbook_sha256": sha256(workbook),
            "figure_source_csv": str(source_csv.relative_to(WORKSPACE)),
        },
    }
    (Q1_FIG_DIR / "fig05_q1_data_statistics_source.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def build_q2() -> dict:
    if not Q2_PICKLE.exists():
        raise FileNotFoundError(Q2_PICKLE)
    # This object is intentionally accessed only for train/valid labels and
    # masks. The test split contributes its ID count only.
    with Q2_PICKLE.open("rb") as stream:
        data = pickle.load(stream)
    if not {"train", "valid", "test"}.issubset(data):
        raise ValueError("Q2 aligned-50 pickle is missing a split")

    records: list[pd.DataFrame] = []
    lengths_by_split: dict[str, np.ndarray] = {}
    split_counts = {"train": len(data["train"]["id"]), "valid": len(data["valid"]["id"])}
    for split in ("train", "valid"):
        part = data[split]
        cls = np.asarray(part["classification_labels"])
        reg = np.asarray(part["regression_labels"], dtype=np.float64)
        text_bert = np.asarray(part["text_bert"])
        if text_bert.ndim != 3 or text_bert.shape[1:] != (3, 50):
            raise ValueError(f"Unexpected {split} text_bert shape: {text_bert.shape}")
        if cls.shape != (len(part["id"]),) or reg.shape != cls.shape:
            raise ValueError(f"Q2 {split} labels do not match IDs")
        cls_rounded = cls.astype(int)
        if not np.array_equal(cls, cls_rounded) or not set(np.unique(cls_rounded)).issubset({0, 1, 2}):
            raise ValueError(f"Unexpected {split} classification label encoding")
        if not np.isfinite(reg).all():
            raise ValueError(f"Non-finite {split} regression labels")

        valid_field = text_bert[:, 1, :]
        valid_mask = valid_field != 0
        lengths = valid_mask.sum(axis=1).astype(int)
        # The established field is a prefix-valid mask; verify rather than
        # infer padding from feature values.
        expected_prefix = np.arange(50)[None, :] < lengths[:, None]
        if not np.array_equal(valid_mask, expected_prefix):
            raise ValueError(f"{split} text_bert[:,1,:] is not prefix-valid")
        lengths_by_split[split] = lengths
        records.append(
            pd.DataFrame(
                {
                    "split": split,
                    "id": list(part["id"]),
                    "classification_label": cls_rounded,
                    "classification_name": pd.Categorical(
                        cls_rounded, categories=[0, 1, 2]
                    ).rename_categories(["Negative", "Neutral", "Positive"]),
                    "regression_label": reg,
                    "valid_length": lengths,
                }
            )
        )

    # Do not access test labels, features, or text. Only count sample IDs.
    split_counts["test"] = len(data["test"]["id"])
    del data
    q2_data = pd.concat(records, ignore_index=True)
    if not q2_data["id"].is_unique:
        raise ValueError("Q2 train/valid IDs are not unique across splits")
    source_csv = Q2_FIG_DIR / "fig10_q2_data_statistics_train_valid_source.csv"
    q2_data.to_csv(source_csv, index=False)
    split_csv = Q2_FIG_DIR / "fig10_q2_data_statistics_split_counts.csv"
    pd.DataFrame([{"split": split, "sample_count": count} for split, count in split_counts.items()]).to_csv(
        split_csv, index=False
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "SimHei"],
            "font.size": 9,
            "axes.labelcolor": INK,
            "text.color": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.0))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.11, wspace=0.31, hspace=0.38)

    ax = axes[0, 0]
    split_order = ["train", "valid", "test"]
    split_names = ["训练", "验证", "测试"]
    split_values = [split_counts[k] for k in split_order]
    bars = ax.bar(split_names, split_values, color=[BLUE, ORANGE, "#AEB7BC"], width=0.58)
    ax.bar_label(bars, padding=2, fontsize=8.2, color=INK, fmt="%d")
    ax.set_ylabel("样本数", fontsize=9)
    ax.set_ylim(0, max(split_values) * 1.16)
    panel_title(ax, "a", "数据划分规模")
    style_axes(ax)

    ax = axes[0, 1]
    labels = ["Negative", "Neutral", "Positive"]
    positions = np.arange(3)
    width = 0.34
    train_counts = q2_data[q2_data["split"] == "train"]["classification_label"].value_counts().reindex([0, 1, 2], fill_value=0)
    valid_counts = q2_data[q2_data["split"] == "valid"]["classification_label"].value_counts().reindex([0, 1, 2], fill_value=0)
    b_train = ax.bar(positions - width / 2, train_counts.to_numpy(), width, color=BLUE, label="训练")
    b_valid = ax.bar(positions + width / 2, valid_counts.to_numpy(), width, color=ORANGE, label="验证")
    ax.bar_label(b_train, padding=1.5, fontsize=7.5, color=INK)
    ax.bar_label(b_valid, padding=1.5, fontsize=7.5, color=INK)
    ax.set_xticks(positions, ["消极", "中性", "积极"])
    ax.set_ylabel("样本数", fontsize=9)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left", ncol=2)
    ax.set_ylim(0, max(train_counts.max(), valid_counts.max()) * 1.20)
    panel_title(ax, "b", "三分类标签分布")
    style_axes(ax)

    ax = axes[1, 0]
    train_reg = q2_data.loc[q2_data["split"] == "train", "regression_label"].to_numpy()
    valid_reg = q2_data.loc[q2_data["split"] == "valid", "regression_label"].to_numpy()
    bins = np.linspace(-3, 3, 25)
    ax.hist(train_reg, bins=bins, density=True, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=0.5, alpha=0.76, label="训练")
    ax.hist(valid_reg, bins=bins, density=True, color=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=0.5, alpha=0.68, label="验证")
    ax.set_xlim(-3, 3)
    ax.set_xlabel("连续情感强度", fontsize=9)
    ax.set_ylabel("概率密度", fontsize=9)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    panel_title(ax, "c", "连续情感强度分布")
    style_axes(ax)

    ax = axes[1, 1]
    length_bins = np.arange(-0.5, 51.5, 1)
    ax.hist(lengths_by_split["train"], bins=length_bins, density=True, histtype="step", linewidth=1.35, color=BLUE, label="训练")
    ax.hist(lengths_by_split["valid"], bins=length_bins, density=True, histtype="step", linewidth=1.35, color=ORANGE, label="验证")
    ax.set_xlim(0, 50)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_xlabel("有效序列长度", fontsize=9)
    ax.set_ylabel("概率密度", fontsize=9)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    panel_title(ax, "d", "按文本有效位统计的序列长度")
    style_axes(ax)

    stem = "fig10_q2_data_statistics"
    export_figure(fig, Q2_FIG_DIR, stem)
    class_counts = {
        split: {
            labels[int(index)]: int(count)
            for index, count in q2_data[q2_data["split"] == split]["classification_label"].value_counts().sort_index().items()
        }
        for split in ("train", "valid")
    }
    regression_summary = {
        split: {
            "count": int(len(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std_population": float(values.std(ddof=0)),
            "quantiles": {str(q): float(values.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        }
        for split, values in q2_data.groupby("split")["regression_label"]
    }
    length_summary = {
        split: {
            "count": int(len(lengths)),
            "min": int(lengths.min()),
            "median": float(np.median(lengths)),
            "max": int(lengths.max()),
        }
        for split, lengths in lengths_by_split.items()
    }
    stats = {
        "split_counts": split_counts,
        "class_counts_train_valid_only": class_counts,
        "regression_summary_train_valid_only": regression_summary,
        "valid_length_summary_train_valid_only": length_summary,
        "test_policy": "sample count only; test labels/features/predictions not read or used",
        "data_sources": {
            "aligned50_pickle": str(Q2_PICKLE.relative_to(WORKSPACE)),
            "aligned50_sha256": sha256(Q2_PICKLE),
            "train_valid_figure_source_csv": str(source_csv.relative_to(WORKSPACE)),
            "split_count_csv": str(split_csv.relative_to(WORKSPACE)),
        },
    }
    (Q2_FIG_DIR / "fig10_q2_data_statistics_source.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def main() -> None:
    if not PAPER.exists():
        raise FileNotFoundError(PAPER)
    q1_stats = build_q1()
    q2_stats = build_q2()
    print(json.dumps({"q1": q1_stats, "q2": q2_stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
