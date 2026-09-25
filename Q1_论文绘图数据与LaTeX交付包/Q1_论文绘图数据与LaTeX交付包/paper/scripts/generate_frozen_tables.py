"""Generate LaTeX tables from the frozen Q1 CSV exports without recalculating metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
PROJECT = PAPER.parent
SOURCE = PROJECT / "outputs" / "q1_final_local" / "02_paper_tables"
OUT = PAPER / "generated"


def read_csv(name: str) -> list[dict[str, str]]:
    path = SOURCE / name
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Frozen source table has no rows: {path}")
    return rows


def number(value: str, places: int = 4) -> str:
    # Formatting only: the table stores the supplied mean/std, without recomputation.
    return f"{float(value):.{places}f}"


def mean_std(row: dict[str, str], mean: str, std: str) -> str:
    return rf"{number(row[mean])} $\pm$ {number(row[std])}"


def write(name: str, body: str) -> None:
    (OUT / name).write_text(body.rstrip() + "\n", encoding="utf-8")


def table_feature() -> None:
    rows = read_csv("table_q1_feature_scheme.csv")
    expected = {"Text", "Audio", "Vision"}
    if {r["Modality"] for r in rows} != expected:
        raise ValueError("Unexpected modality rows in frozen feature table")
    names = {"Text": "文本", "Audio": "语音", "Vision": "视觉"}
    encoder_names = {
        "FacebookAI/roberta-base": "RoBERTa-base",
        "microsoft/wavlm-base-plus": "WavLM-base-plus",
        "google/siglip2-base-patch16-224": "SigLIP2-B/16",
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{三模态特征构建与时间覆盖结果}",
        r"\label{tab:q1_feature_scheme}",
        r"\begin{tabularx}{\textwidth}{@{}lXrrcr@{}}",
        r"\toprule",
        r"模态 & 特征提取方法 & 原生特征数 & 特征维数 & \shortstack{单样本\\对齐形状} & 平均时间覆盖率 \\",
        r"\midrule",
    ]
    for row in rows:
        modality = names[row["Modality"]]
        encoder = r"\texttt{" + encoder_names[row["Encoder"]] + "}"
        aligned = rf"$\QOneBins\times{row['Aligned dimension']}$"
        lines.append(
            f"{modality} & {encoder} & {int(row['Native rows'])} & "
            f"{int(row['Native dimension'])} & {aligned} & "
            f"{number(row['Mean coverage'])} " + chr(92) * 2
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    write("table_q1_feature.tex", "\n".join(lines))


def table_alignment() -> None:
    rows = read_csv("table_alignment_ablation.csv")
    expected = ["maximum_overlap_hard", "nearest_center", "overlap_weighted_mean"]
    if [r["Method"] for r in rows] != expected:
        raise ValueError("Unexpected alignment method order in frozen table")
    labels = {
        expected[0]: r"\shortstack{最大时间\\重叠法}",
        expected[1]: r"\shortstack{最近中心法}",
        expected[2]: r"\shortstack{重叠加权\\平均法}",
    }
    issue = rows[1]["temporal_semantic_issue"]
    match = re.search(r"\b(\d+)\b", issue)
    if not match or "zero-overlap" not in issue:
        raise ValueError("Could not verify the frozen nearest-center mapping note")
    zero_overlap = int(match.group(1))
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{不同时序对齐方法的定量比较}",
        r"\label{tab:q1_alignment_ablation}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}>{\centering\arraybackslash}p{2.6cm}*{4}{>{\centering\arraybackslash}p{3.0cm}}@{}}",
        r"\toprule",
        r"对齐方法 & 准确率 & 宏平均 F1 值 & \shortstack{平均绝对误差\\（MAE）} & \shortstack{皮尔逊\\相关系数} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{labels[row['Method']]} & {mean_std(row, 'Accuracy', 'accuracy_std')} & "
            f"{mean_std(row, 'Macro-F1', 'macro_f1_std')} & "
            f"{mean_std(row, 'MAE', 'mae_std')} & "
            f"{mean_std(row, 'Pearson', 'pearson_std')} " + chr(92) * 2
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        rf"\par\smallskip\noindent\footnotesize 注：最近中心法有 {zero_overlap} 个有效文本时间窗选择了与目标时间窗零交叠的特征。",
        r"\end{table}",
    ])
    write("table_q1_alignment.tex", "\n".join(lines))


def table_visual() -> None:
    rows = read_csv("table_visual_backbone.csv")
    expected = ["SigLIP2", "DINOv3"]
    if [r["Encoder"] for r in rows] != expected:
        raise ValueError("Unexpected vision encoder rows in frozen table")
    labels = {"SigLIP2": "SigLIP2", "DINOv3": r"\shortstack{DINOv3\\（ViT-B/16）}"}
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{不同视觉特征提取方法的比较}",
        r"\label{tab:q1_visual_backbone}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}p{2.3cm}*{4}{>{\centering\arraybackslash}p{2.9cm}}>{\centering\arraybackslash}p{1.7cm}@{}}",
        r"\toprule",
        r"视觉编码器 & 准确率 & \shortstack{宏平均\\F1} & MAE & 皮尔逊 r & \shortstack{提取时长\\（秒）} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{labels[row['Encoder']]} & {mean_std(row, 'Accuracy', 'accuracy_std')} & "
            f"{mean_std(row, 'Macro-F1', 'macro_f1_std')} & "
            f"{mean_std(row, 'MAE', 'mae_std')} & "
            f"{mean_std(row, 'Pearson', 'pearson_std')} & {number(row['Extraction time (s)'], 2)} " + chr(92) * 2
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write("table_q1_visual.tex", "\n".join(lines))


def table_modality() -> None:
    rows = read_csv("table_modality_ablation.csv")
    expected = ["T", "A", "V", "T+A", "T+V", "A+V", "T+A+V"]
    if [r["Modality set"] for r in rows] != expected:
        raise ValueError("Unexpected modality configuration order in frozen table")
    labels = {
        "T": "文本",
        "A": "语音",
        "V": "视觉",
        "T+A": "文本+语音",
        "T+V": "文本+视觉",
        "A+V": "语音+视觉",
        "T+A+V": "文本+语音+视觉",
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{不同模态组合的定量比较}",
        r"\label{tab:q1_modality_ablation}",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}p{3.3cm}*{4}{>{\centering\arraybackslash}p{3.0cm}}@{}}",
        r"\toprule",
        r"模态组合 & 准确率 & 宏平均 F1 值 & \shortstack{平均绝对误差\\（MAE）} & \shortstack{皮尔逊\\相关系数} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{labels[row['Modality set']]} & {mean_std(row, 'accuracy_mean', 'accuracy_std')} & "
            f"{mean_std(row, 'macro_f1_mean', 'macro_f1_std')} & "
            f"{mean_std(row, 'mae_mean', 'mae_std')} & "
            f"{mean_std(row, 'pearson_mean', 'pearson_std')} " + chr(92) * 2
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write("table_q1_modality.tex", "\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for build in (table_feature, table_alignment, table_visual, table_modality):
        build()
    files = [
        "table_q1_feature_scheme.csv",
        "table_alignment_ablation.csv",
        "table_visual_backbone.csv",
        "table_modality_ablation.csv",
    ]
    manifest = {
        "source_root": "outputs/q1_final_local/02_paper_tables",
        "sources": {
            name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
            for name in files
        },
        "transformation": "Direct CSV field formatting for LaTeX; no metric recomputation.",
    }
    (OUT / "q1_table_sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
