"""Render frozen Q1/Q2/Q3 result files into manuscript tables; no inference."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


PAPER = Path(__file__).resolve().parent.parent
ROOT = PAPER.parents[2]
Q1_FINAL = PAPER.parent / "outputs/q1_final_local"
Q2_FINAL = ROOT / "E2026/outputs/final/q2"
Q3_FINAL = ROOT / "E2026/outputs/q3/final"
GENERATED = PAPER / "generated"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_ids(rows: list[dict[str, str]], key: str, expected: int) -> set[str]:
    ids = [row[key] for row in rows]
    assert len(ids) == expected and len(set(ids)) == expected
    assert all(ids)
    return set(ids)


def tex_id(value: str) -> str:
    assert not any(character in value for character in "{}\\")
    return rf"\texttt{{\detokenize{{{value}}}}}"


def write_table(filename: str, lines: list[str]) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def q1_table() -> dict:
    source = Q1_FINAL / "05_appendix/q1_sample_summary_100.csv"
    paper_copy = PAPER / "tables/q1/q1_sample_summary_100.csv"
    manifest_path = Q1_FINAL / "00_manifest/q1_paper_numbers.json"
    rows = read_csv(source)
    manifest = read_json(manifest_path)
    ids = check_ids(rows, "sample_id", 100)
    assert source.read_bytes() == paper_copy.read_bytes()
    assert manifest["sample_count"]["value"] == len(rows)
    assert {row["processing_status"] for row in rows} == {"complete"}
    assert {int(row["aligned_length"]) for row in rows} == {50}
    assert all(
        int(row[f"{modality}_dimension"]) == 768
        for row in rows for modality in ("text", "audio", "vision")
    )
    assert all(int(row["vision_valid_bins"]) == 50 for row in rows)
    for modality in ("text", "audio", "vision"):
        observed = statistics.mean(float(row[f"{modality}_coverage"]) for row in rows)
        frozen = manifest["modality_coverage"][modality]["value"]
        assert math.isclose(observed, frozen, abs_tol=1e-12)
        assert all(0 <= int(row[f"{modality}_valid_bins"]) <= 50 for row in rows)
    source_counts = Counter(row["text_feature_source"] for row in rows)
    assert source_counts == {
        key: value["value"] for key, value in manifest["text_source_counts"].items()
    }
    names = {"official_transcript": "官方文本", "media_asr": "媒体ASR"}
    assert set(source_counts) == set(names)
    assert sum(int(row["timestamp_fallback_count"]) for row in rows) == (
        manifest["timestamp_fallback_count"]["value"]
    )
    lines = [
        r"\begingroup\small\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{@{}lrrrlr@{}}",
        r"\caption{附件1全部100条样本的特征构建结果（逐样本汇总）}\label{tab:q1_all_samples}\\",
        r"\toprule",
        r"样本ID & 时长/s & 文本有效窗 & 语音有效窗 & 文本来源 & 时间戳回退次数\\",
        r"\midrule\endfirsthead",
        r"\multicolumn{6}{@{}l}{续表~\thetable}\\",
        r"\toprule",
        r"样本ID & 时长/s & 文本有效窗 & 语音有效窗 & 文本来源 & 时间戳回退次数\\",
        r"\midrule\endhead",
        r"\midrule\multicolumn{6}{r@{}}{续下页}\\\endfoot",
        r"\bottomrule\endlastfoot",
    ]
    for row in rows:
        duration = float(row["duration"])
        fallback = int(row["timestamp_fallback_count"])
        assert math.isfinite(duration) and duration > 0 and fallback >= 0
        lines.append(
            f'{tex_id(row["sample_id"])} & {duration:.3f} & '
            f'{int(row["text_valid_bins"])} & {int(row["audio_valid_bins"])} & '
            f'{names[row["text_feature_source"]]} & {fallback}' + r"\\"
        )
    lines += [r"\end{longtable}", r"\endgroup"]
    write_table("table_q1_all_samples.tex", lines)
    return {
        "row_count": len(rows), "unique_ids": len(ids), "source": repo_path(source),
        "source_sha256": sha256(source), "paper_copy_sha256": sha256(paper_copy),
        "manifest": repo_path(manifest_path), "manifest_sha256": sha256(manifest_path),
        "source_counts": dict(source_counts),
        "fallback_sample_count": sum(int(r["timestamp_fallback_count"]) > 0 for r in rows),
        "warning_sample_count": sum(bool(r["text_warning"]) for r in rows),
        "coverage": {m: statistics.mean(float(r[f"{m}_coverage"]) for r in rows)
                     for m in ("text", "audio", "vision")},
    }


def q2_public_table() -> dict:
    baseline_dir = Q2_FINAL / "public_baselines"
    clean_path = baseline_dir / "baseline_clean_mean_sd.csv"
    per_seed_path = baseline_dir / "baseline_clean_per_seed.csv"
    p2_path = ROOT / "E2026/outputs/metrics/b5_p2_multiseed_summary.json"
    clean_rows = read_csv(clean_path)
    per_seed = read_csv(per_seed_path)
    p2 = read_json(p2_path)
    assert {r["model"] for r in clean_rows} == {"TFN", "MulT", "MISA"}
    assert p2["training_seeds"] == [42, 43, 44]
    parameter_counts = {}
    for model in ("TFN", "MulT", "MISA"):
        model_rows = [r for r in per_seed if r["model"] == model]
        assert {int(r["seed"]) for r in model_rows} == {42, 43, 44}
        counts = {int(r["parameters"]) for r in model_rows}
        assert len(counts) == 1
        parameter_counts[model] = counts.pop()
        frozen_row = next(r for r in clean_rows if r["model"] == model)
        for field in ("accuracy", "macro_f1", "mae", "pearson"):
            values = [float(r[field]) for r in model_rows]
            assert math.isclose(statistics.mean(values), float(frozen_row[f"{field}_mean"]), abs_tol=1e-12)
            assert math.isclose(statistics.stdev(values), float(frozen_row[f"{field}_sd"]), abs_tol=1e-12)
    p2_counts = {int(r["p2_parameter_count"]) for r in p2["seed43_44_details"].values()}
    assert len(p2_counts) == 1
    parameter_counts["本文模型"] = p2_counts.pop()
    rows_by_model = {r["model"]: r for r in clean_rows}
    fields = ("accuracy", "macro_f1", "mae", "pearson")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\small\setlength{\tabcolsep}{4pt}",
        r"\caption{公开多模态模型在附件2完整输入验证集上的对比（三次初始化均值$\pm$样本标准差）}",
        r"\label{tab:q2_public_baselines}",
        r"\begin{tabular}{@{}lrcccc@{}}",
        r"\toprule",
        r"方法 & 参数量 & 准确率$\uparrow$ & 宏平均F1$\uparrow$ & MAE$\downarrow$ & 相关系数$\uparrow$\\",
        r"\midrule",
    ]
    for model in ("TFN", "MulT", "MISA", "本文模型"):
        if model == "本文模型":
            metric = p2["models"]["B5-P2"]["clean"]
            values = [(metric[field]["mean"], metric[field]["std"]) for field in fields]
        else:
            row = rows_by_model[model]
            values = [(float(row[f"{field}_mean"]), float(row[f"{field}_sd"]))
                      for field in fields]
        values_tex = " & ".join(f"${mean:.4f}\\pm{sd:.4f}$" for mean, sd in values)
        lines.append(f'{model} & {parameter_counts[model]:,} & {values_tex}' + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("table_q2_public_baselines.tex", lines)
    return {
        "row_count": 4, "source": [repo_path(p) for p in (clean_path, per_seed_path, p2_path)],
        "sha256": {repo_path(p): sha256(p) for p in (clean_path, per_seed_path, p2_path)},
        "parameters": parameter_counts,
    }


CLASS_NAMES = {"Negative": "消极", "Neutral": "中性", "Positive": "积极"}


def attachment3_table() -> dict:
    directory = Q2_FINAL / "attachment3"
    predictions_path = directory / "attachment3_predictions.csv"
    audit_path = directory / "attachment3_predictions_audit.csv"
    inventory_path = directory / "attachment3_input_audit.json"
    summary_path = directory / "attachment3_summary.json"
    interface_path = directory / "attachment3_text_interface_audit.json"
    predictions = read_csv(predictions_path)
    audit_rows = read_csv(audit_path)
    inventory = read_json(inventory_path)
    summary = read_json(summary_path)
    interface = read_json(interface_path)
    ids = check_ids(predictions, "sample_id", 30)
    assert check_ids(audit_rows, "sample_id", 30) == ids
    assert {r["filename"].removesuffix(".pkl") for r in inventory["files"]} == ids
    assert summary["sample_count"] == 30 and interface["status"] == "PASS"
    by_id = {r["sample_id"]: r for r in audit_rows}
    for row in predictions:
        other = by_id[row["sample_id"]]
        for field in ("predicted_class_id", "predicted_class_name"):
            assert row[field] == other[field]
        assert float(row["predicted_intensity"]) == float(other["predicted_intensity"])
        assert row["predicted_class_name"] in CLASS_NAMES
    assert dict(Counter(r["predicted_class_name"] for r in predictions)) == summary["class_counts"]
    for name, count in summary["class_counts"].items():
        assert math.isclose(100 * count / 30, summary["class_percent"][name], abs_tol=1e-12)
    intensities = [float(r["predicted_intensity"]) for r in predictions]
    confidences = [float(by_id[r["sample_id"]]["confidence"]) for r in predictions]
    assert math.isclose(statistics.mean(intensities), summary["intensity"]["mean"], abs_tol=1e-12)
    assert math.isclose(statistics.stdev(intensities), summary["intensity"]["sample_sd"], abs_tol=1e-12)
    assert math.isclose(min(intensities), summary["intensity"]["min"], abs_tol=1e-12)
    assert math.isclose(max(intensities), summary["intensity"]["max"], abs_tol=1e-12)
    assert math.isclose(statistics.mean(confidences), summary["confidence"]["mean"], abs_tol=1e-12)
    lines = [
        r"\begingroup\small\setlength{\tabcolsep}{7pt}",
        r"\begin{longtable}{@{}llrr@{}}",
        r"\caption{附件3全部30条无标签样本的最终预测}\label{tab:q2_attachment3_all}\\",
        r"\toprule",
        r"样本ID & 预测类别 & 连续情感强度 & 分类置信度\\",
        r"\midrule\endfirsthead",
        r"\multicolumn{4}{@{}l}{续表~\thetable}\\",
        r"\toprule",
        r"样本ID & 预测类别 & 连续情感强度 & 分类置信度\\",
        r"\midrule\endhead",
        r"\midrule\multicolumn{4}{r@{}}{续下页}\\\endfoot",
        r"\bottomrule\endlastfoot",
    ]
    for row in predictions:
        confidence = float(by_id[row["sample_id"]]["confidence"])
        assert 0 <= confidence <= 1
        lines.append(
            f'{tex_id(row["sample_id"])} & {CLASS_NAMES[row["predicted_class_name"]]} & '
            f'{float(row["predicted_intensity"]):+.6f} & {confidence:.4f}' + r"\\"
        )
    lines += [r"\end{longtable}", r"\endgroup"]
    write_table("table_q2_attachment3_all.tex", lines)
    sources = (predictions_path, audit_path, inventory_path, summary_path, interface_path)
    return {
        "row_count": len(predictions), "unique_ids": len(ids),
        "source": [repo_path(p) for p in sources],
        "sha256": {repo_path(p): sha256(p) for p in sources},
        "input_inventory_id_match": True, "class_counts": summary["class_counts"],
        "interface_status": interface["status"],
    }


def attachment4_table() -> dict:
    predictions_path = Q3_FINAL / "attachment4_predictions_explanations.csv"
    summary_path = Q3_FINAL / "attachment4_summary.json"
    rows = read_csv(predictions_path)
    summary = read_json(summary_path)
    ids = check_ids(rows, "sample_id", 20)
    inventory_ids = {r["sample_id"] for r in summary["input_integrity"]["records"]}
    assert ids == inventory_ids
    assert summary["scope"]["n_samples"] == 20
    locked = summary["unlabeled_summary"]
    assert Counter(r["predicted_class_name"] for r in rows) == {
        key: value["count"] for key, value in
        locked["predicted_class_counts_and_proportions"].items()
    }
    for value in locked["predicted_class_counts_and_proportions"].values():
        assert math.isclose(value["count"] / 20, value["proportion"], abs_tol=1e-12)
    for field, summary_field in (
        ("primary_modality_classification", "classification_primary_modality_counts"),
        ("primary_modality_regression", "regression_primary_modality_counts"),
        ("grounding_status", "grounding_status_counts"),
    ):
        assert Counter(r[field] for r in rows) == locked[summary_field]
    assert sum(r["primary_modality_agreement"] == "True" for r in rows) == 17
    modalities = {"text": "文本", "audio": "语音", "vision": "视觉"}
    prediction_lines = [
        r"\begin{table}[H]",
        r"\centering\small\setlength{\tabcolsep}{7pt}",
        r"\caption{附件4全部20条无标签样本的预测结果}",
        r"\label{tab:q3_attachment4_all}",
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"样本ID & 预测类别 & 情感强度 & 分类置信度\\",
        r"\midrule",
    ]
    explanation_lines = [
        r"\begin{table}[H]",
        r"\centering\small\setlength{\tabcolsep}{3.5pt}",
        r"\caption{附件4全部20条样本的分类贡献与证据层级；区间为从零开始的对齐特征槽位$[a,b)$}",
        r"\label{tab:q3_attachment4_explanations}",
        r"\begin{tabular}{@{}lrrrlcl@{}}",
        r"\toprule",
        r"ID & $\phi_T$ & $\phi_A$ & $\phi_V$ & 主导模态 & 关键区间 & 证据层级\\",
        r"\midrule",
    ]
    for row in rows:
        modality = row["primary_modality_classification"]
        status = row["grounding_status"]
        assert modality in modalities and row["predicted_class_name"] in CLASS_NAMES
        assert status in {"verified", "unverified"}
        if status == "verified":
            assert modality == "text" and row["text_fragment"] not in {"", "NA"}
            evidence = "原文字符级"
        else:
            assert modality == "vision" and row["video_frame_time"] in {"", "NA"}
            evidence = "特征行级"
        start, end = int(row["key_start_index"]), int(row["key_end_index"])
        assert 0 <= start < end <= 50
        intensity = float(row["predicted_intensity"])
        confidence = float(row["confidence"])
        assert math.isfinite(intensity) and 0 <= confidence <= 1
        phi = tuple(float(row[f"shapley_{name}"]) for name in ("text", "audio", "vision"))
        assert all(math.isfinite(value) for value in phi)
        prediction_lines.append(
            f'{tex_id(row["sample_id"])} & {CLASS_NAMES[row["predicted_class_name"]]} & '
            f'{intensity:+.6f} & {confidence:.4f}' + r"\\"
        )
        explanation_lines.append(
            f'{tex_id(row["sample_id"])} & {phi[0]:+.4f} & {phi[1]:+.4f} & {phi[2]:+.4f} & '
            f'{modalities[modality]} & $[{start},{end})$ & {evidence}' + r"\\"
        )
    prediction_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    explanation_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table("table_q3_attachment4_all.tex", prediction_lines)
    write_table("table_q3_attachment4_explanations.tex", explanation_lines)
    return {
        "row_count": len(rows), "unique_ids": len(ids),
        "source": [repo_path(predictions_path), repo_path(summary_path)],
        "sha256": {repo_path(p): sha256(p) for p in (predictions_path, summary_path)},
        "input_inventory_id_match": True,
        "classification_regression_agreement": 17,
    }


def main() -> None:
    audit = {
        "q1": q1_table(),
        "q2_public": q2_public_table(),
        "attachment3": attachment3_table(),
        "attachment4": attachment4_table(),
    }
    (GENERATED / "full_result_table_sources.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Frozen result tables:", {key: value["row_count"] for key, value in audit.items()})


if __name__ == "__main__":
    main()
