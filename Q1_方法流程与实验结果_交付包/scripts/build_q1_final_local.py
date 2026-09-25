"""Build a local, source-traceable Q1 publication package from frozen outputs only.

No model, training, ASR, GPU, or network modules are imported by this script.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/q1_final_local"
F = OUT / "01_final_features"
T = OUT / "02_paper_tables"
G = OUT / "03_paper_figures"
C = OUT / "04_case_studies"
A = OUT / "05_appendix"
R = OUT / "06_reproducibility"
AUDIT = OUT / "07_audit"
METHODS = ("maximum_overlap_hard", "nearest_center", "overlap_weighted_mean")
MODALITIES = ("text", "audio", "vision")
METRICS = ("accuracy", "macro_f1", "mae", "pearson")


def rel(path):
    return str(Path(path).relative_to(ROOT)).replace("\\", "/")


def load_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def load_csv(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows, fields=None):
    if not rows:
        raise ValueError(f"Empty output table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(value, source_file, source_field, source_row=None):
    return {"value": value, "source_file": source_file, "source_field": source_field,
            "source_row": source_row}


def value(reference):
    return reference["value"]


def numeric_row(row, source, row_key, fields):
    return {field: ref(float(row[field]), source, field, row_key) for field in fields}


def preflight():
    """Stop before writing any package file on conflicting frozen baselines."""
    alignment = load_csv("outputs/q1_alignment_ablation/paper_alignment_comparison.csv")
    visual = load_csv("outputs/q1_visual_encoder_ablation/visual_encoder_comparison.csv")
    modality = load_csv("outputs/q1_modality_quality_ablation/modality_ablation_summary.csv")
    quality = load_csv("outputs/q1_modality_quality_ablation/paper_quality_alignment.csv")
    known = [next(r for r in alignment if r["configuration"] == "maximum_overlap_hard"),
             next(r for r in visual if r["encoder"] == "SigLIP2"),
             next(r for r in modality if r["configuration"] == "T+A+V"),
             next(r for r in quality if r["configuration"] == "T+A+V")]
    drifts = {}
    for metric in METRICS:
        for suffix in ("mean", "std"):
            field = f"{metric}_{suffix}"
            vals = [float(r[field]) for r in known]
            drifts[field] = max(vals) - min(vals)
            if drifts[field] > 1e-6:
                raise ValueError(f"Conflicting frozen baseline {field}: {vals}")
    native = load_json("outputs/q1_features_100_final/validation_report.json")
    source = load_csv("outputs/q1_text_final/text_feature_sources/text_feature_source_audit.csv")
    if Counter(r["text_feature_source"] for r in source) != native["text_source_counts"]:
        raise ValueError("Final text-source counts conflict")
    if sum(int(r["used_segment_fallback_count"]) for r in source) != native["selected_text_feature_segment_fallback_count"]:
        raise ValueError("Final selected fallback count conflicts")
    matrix = load_csv("outputs/q1_alignment_ablation/alignment_metrics.csv")
    semantic = {int(r["nearest_center_zero_overlap_assignments"]) for r in matrix
                if r["method"] == "nearest_center"}
    if len(semantic) != 1 or not re.search(rf"\b{next(iter(semantic))}\b 个文本有效 bin", (ROOT / "outputs/q1_alignment_ablation/report.md").read_text(encoding="utf-8")):
        raise ValueError(f"Nearest-center issue count conflicts across rows: {semantic}")
    if len(source) != 100 or native["sample_count"] != 100:
        raise ValueError("100-sample final source is incomplete")
    return {"baseline_max_metric_discrepancy": max(drifts.values()),
            "nearest_center_global_zero_overlap_assignments": next(iter(semantic)),
            "baseline_source_rows": [
                "outputs/q1_alignment_ablation/paper_alignment_comparison.csv:maximum_overlap_hard",
                "outputs/q1_visual_encoder_ablation/visual_encoder_comparison.csv:SigLIP2",
                "outputs/q1_modality_quality_ablation/modality_ablation_summary.csv:T+A+V",
                "outputs/q1_modality_quality_ablation/paper_quality_alignment.csv:T+A+V"]}


def build_samples_and_features():
    inventory = {r["sample_id"]: r for r in load_json("outputs/q1_cpu/manifest.json")}
    sources = {r["sample_id"]: r for r in load_csv("outputs/q1_text_final/text_feature_sources/text_feature_source_audit.csv")}
    ids = sorted(inventory)
    if len(ids) != 100 or len(set(ids)) != 100 or set(ids) != set(sources):
        raise ValueError("Final sample inventory/source join differs from 100 unique IDs")
    arrays = {m: np.zeros((100, 50, 768), dtype=np.float32) for m in MODALITIES}
    masks = {m: np.zeros((100, 50), dtype=bool) for m in MODALITIES}
    sample_rows = []
    revisions = {}
    for n, sid in enumerate(ids):
        inv, source = inventory[sid], sources[sid]
        native = {m: load_json(f"outputs/q1_features_100_final/{m}/{sid}.json") for m in MODALITIES}
        for m in MODALITIES:
            meta = native[m]
            if meta["sample_id"] != sid or meta["source_mp4_sha256"] != inv["source_sha256"]:
                raise ValueError(f"Native ID or MP4 hash mismatch: {sid}/{m}")
            if int(meta["dimension"]) != 768 or len(meta["records"]) != int(meta["shape"][0]):
                raise ValueError(f"Native shape mismatch: {sid}/{m}")
            revisions.setdefault(m, (meta["encoder"]["name"], meta["encoder"]["version"]))
            if revisions[m] != (meta["encoder"]["name"], meta["encoder"]["version"]):
                raise ValueError(f"Encoder revision changed across samples: {m}")
        with np.load(ROOT / f"outputs/q1_alignment_ablation/aligned/maximum_overlap_hard/{sid}.npz", allow_pickle=False) as data:
            for m in MODALITIES:
                arr, mask = data[f"{m}_values"], data[f"{m}_valid"]
                if arr.shape != (50, 768) or mask.shape != (50,) or not np.isfinite(arr).all():
                    raise ValueError(f"Nonfinite or incorrect aligned shape: {sid}/{m}")
                if np.any(arr[~mask] != 0):
                    raise ValueError(f"Invalid bin has nonzero feature: {sid}/{m}")
                arrays[m][n] = arr
                masks[m][n] = mask
        fallback = sum(r["timestamp_source"] == "segment_fallback" for r in native["text"]["records"])
        if fallback != int(source["used_segment_fallback_count"]):
            raise ValueError(f"Selected text fallback count differs: {sid}")
        warning = source["warning"] or "; ".join(native["text"].get("warnings", []))
        row = {"sample_id": sid, "video_id": inv["video_id"], "clip_id": inv["clip_id"],
               "duration": float(inv["duration"]), "text_feature_source": source["text_feature_source"],
               "official_alignment_valid": source["official_alignment_valid"],
               "media_text_valid": source["media_text_valid"], "timestamp_valid": source["timestamp_valid"],
               "official_consistency_score": float(source["official_consistency_score"]) if source["official_consistency_score"] else "",
               "text_warning": warning, "aligned_length": 50, "timestamp_fallback_count": fallback,
               "source_mp4_hash": inv["source_sha256"], "processing_status": "complete"}
        for m in MODALITIES:
            row[f"{m}_native_length"] = len(native[m]["records"])
            row[f"{m}_dimension"] = int(native[m]["dimension"])
            valid = int(masks[m][n].sum())
            row[f"{m}_valid_bins"] = valid
            row[f"{m}_coverage"] = valid / 50
            row[f"{m}_empty_bin_ratio"] = 1 - valid / 50
        sample_rows.append(row)
    save_csv(A / "q1_sample_summary_100.csv", sample_rows)
    save_csv(F / "sample_ids.csv", [{"index": i, "sample_id": sid} for i, sid in enumerate(ids)])
    np.savez_compressed(F / "aligned_features_fp32.npz", **arrays)
    np.savez_compressed(F / "masks.npz", **{f"{m}_mask": masks[m] for m in MODALITIES})
    half = {m: arrays[m].astype(np.float16) for m in MODALITIES}
    np.savez_compressed(F / "aligned_features_fp16.npz", **half)
    errors = {}
    for m in MODALITIES:
        restored = half[m].astype(np.float32)
        diff = np.abs(arrays[m] - restored)
        errors[m] = {"max_abs_error": float(diff.max()), "mean_abs_error": float(diff.mean()),
                     "rmse": float(np.sqrt(np.mean(np.square(diff, dtype=np.float64)))),
                     "fp16_nan": int(np.isnan(half[m]).sum()), "fp16_inf": int(np.isinf(half[m]).sum()),
                     "mask_equality": True}
        if not np.isfinite(half[m]).all() or np.any(restored[~masks[m]] != 0):
            raise ValueError(f"FP16 compression invalid: {m}")
    manifest = {"sample_order": ids, "dtype": "float32", "shape": [100, 50, 768],
                "encoder": {m: revisions[m][0] for m in MODALITIES},
                "encoder_revision": {m: revisions[m][1] for m in MODALITIES},
                "alignment_method": "maximum_overlap_hard", "K": 50,
                "source_directory": "outputs/q1_alignment_ablation/aligned/maximum_overlap_hard",
                "generation_script": "scripts/build_q1_final_local.py",
                "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in
                          [F / "aligned_features_fp32.npz", F / "aligned_features_fp16.npz", F / "masks.npz", F / "sample_ids.csv"]},
                "fp16_candidate_only": True, "dinov3_in_final_package": False}
    save_json(F / "feature_manifest.json", manifest)
    (F / "README.md").write_text(
        "# Frozen Q1 features\n\nRows follow `sample_ids.csv` (zero-based index). Each modality in "
        "`aligned_features_fp32.npz` has shape (100,50,768), with matching masks in `masks.npz`. "
        "Invalid bins contain zero. FP16 is a candidate transport copy; see the compression audit. "
        "The formal visual encoder is SigLIP2. Native timestamps and source traces remain in the "
        "frozen source directory; these dense matrices alone do not replace them.\n", encoding="utf-8")
    lines = ["# FP16 compression audit", "", "FP16 is a candidate compact copy; FP32 remains formal.", "",
             "| Modality | max abs error | mean abs error | RMSE | NaN | Inf | mask equality |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for m, e in errors.items():
        lines.append(f"| {m} | {e['max_abs_error']:.8g} | {e['mean_abs_error']:.8g} | {e['rmse']:.8g} | {e['fp16_nan']} | {e['fp16_inf']} | {e['mask_equality']} |")
    lines += ["", f"FP32 bytes: {(F / 'aligned_features_fp32.npz').stat().st_size}",
              f"FP16 bytes: {(F / 'aligned_features_fp16.npz').stat().st_size}"]
    (AUDIT / "fp16_compression_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ids, sample_rows, arrays, masks, revisions, errors


def build_numbers(sample_rows, revisions, fp16_errors, preflight_result):
    validation_path = "outputs/q1_features_100_final/validation_report.json"
    validation = load_json(validation_path)
    numbers = {"sample_count": ref(validation["sample_count"], validation_path, "sample_count"),
               "final_models": {}, "model_revisions": {}, "feature_dimensions": {},
               "native_row_counts": {}, "aligned_length": ref(int(next(iter(validation["aligned_shape_counts"]))[1:].split(",")[0]), validation_path, "aligned_shape_counts key"),
               "modality_coverage": {}, "text_source_counts": {},
               "official_alignment_valid_count": ref(validation["text_validity_counts"]["official_alignment_valid"], validation_path, "text_validity_counts.official_alignment_valid"),
               "timestamp_fallback_count": ref(validation["selected_text_feature_segment_fallback_count"], validation_path, "selected_text_feature_segment_fallback_count"),
               "raw_asr_timestamp_fallback_count": ref(validation["raw_asr_segment_fallback_word_count"], validation_path, "raw_asr_segment_fallback_word_count"),
               "alignment_ablation": [], "shift_sensitivity": [], "visual_encoder_ablation": [],
               "modality_ablation": [], "quality_alignment_ablation": [], "runtime": {}, "memory": {},
               "test_results": {}, "fp16_compression": {}}
    for m in MODALITIES:
        meta_source = f"outputs/q1_features_100_final/{m}/{sample_rows[0]['sample_id']}.json"
        numbers["final_models"][m] = ref(revisions[m][0], meta_source, "encoder.name", sample_rows[0]["sample_id"])
        numbers["model_revisions"][m] = ref(revisions[m][1], meta_source, "encoder.version", sample_rows[0]["sample_id"])
        native_dims = validation["native"][m]["dimensions"]
        if len(native_dims) != 1:
            raise ValueError(f"Multiple frozen native dimensions for {m}")
        numbers["feature_dimensions"][m] = ref(int(next(iter(native_dims))), validation_path, f"native.{m}.dimensions key")
        numbers["native_row_counts"][m] = ref(validation["native"][m]["feature_rows"], validation_path, f"native.{m}.feature_rows")
        numbers["modality_coverage"][m] = ref(validation["hard_alignment_mean_valid_bin_ratio"][m], validation_path, f"hard_alignment_mean_valid_bin_ratio.{m}")
        numbers["runtime"][m] = ref(validation["server_timings_and_peak_cuda_memory"][m]["elapsed_including_load_seconds"], validation_path, f"server_timings_and_peak_cuda_memory.{m}.elapsed_including_load_seconds")
        numbers["memory"][m] = ref(validation["server_timings_and_peak_cuda_memory"][m]["peak_allocated_bytes"], validation_path, f"server_timings_and_peak_cuda_memory.{m}.peak_allocated_bytes")
        numbers["fp16_compression"][m] = {key: ref(v, "outputs/q1_final_local/07_audit/fp16_compression_audit.md", key, m)
                                           for key, v in fp16_errors[m].items()}
    for k, v in validation["text_source_counts"].items():
        numbers["text_source_counts"][k] = ref(v, validation_path, f"text_source_counts.{k}")
    whisper_path = "outputs/q1_text_audit/summary.json"
    whisper = load_json(whisper_path)
    numbers["final_models"]["whisper_timestamps"] = ref(whisper["model_name"], whisper_path, "model_name")
    numbers["model_revisions"]["whisper_timestamps"] = ref("unavailable", whisper_path, "checkpoint revision not recorded")
    numbers["runtime"]["whisper_asr_seconds"] = ref(whisper["run"]["elapsed_seconds"], whisper_path, "run.elapsed_seconds")
    numbers["memory"]["whisper_asr_peak_cuda_bytes"] = ref(whisper["run"]["peak_cuda_memory_bytes"], whisper_path, "run.peak_cuda_memory_bytes")
    ap = "outputs/q1_alignment_ablation/paper_alignment_comparison.csv"
    am = "outputs/q1_alignment_ablation/alignment_metrics.csv"
    align_summary = load_csv(ap)
    align_metrics = load_csv(am)
    for method in METHODS:
        row = next(r for r in align_summary if r["configuration"] == method)
        item = {"method": ref(method, ap, "configuration", method)}
        item.update(numeric_row(row, ap, f"configuration={method}", [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]))
        for modality in MODALITIES:
            subset = [r for r in align_metrics if r["method"] == method and r["modality"] == modality]
            if len(subset) != 100:
                raise ValueError(f"Alignment metric rows incomplete: {method}/{modality}")
            coverage = float(np.mean([float(r["coverage_ratio"]) for r in subset]))
            traceability = float(np.mean([float(r["traceability_rate"]) for r in subset]))
            item[f"{modality}_coverage"] = ref(coverage, am, "mean(coverage_ratio)", f"method={method};modality={modality};100 rows")
            item[f"{modality}_traceability"] = ref(traceability, am, "mean(traceability_rate)", f"method={method};modality={modality};100 rows")
        repeated_issue_counts = {int(r["nearest_center_zero_overlap_assignments"]) for r in align_metrics if r["method"] == method}
        if len(repeated_issue_counts) != 1:
            raise ValueError(f"Inconsistent global nearest-center issue count: {method}")
        item["nearest_center_zero_overlap_assignments"] = ref(next(iter(repeated_issue_counts)), am,
            "nearest_center_zero_overlap_assignments (global, repeated per row)", f"method={method}")
        numbers["alignment_ablation"].append(item)
    sp = "outputs/q1_alignment_ablation/shift_curve_data.csv"
    for r in load_csv(sp):
        item = {key: ref(r[key], sp, key, f"{r['shift_modality']}:{r['shift_seconds']}:{r['metric']}")
                for key in ("shift_modality", "metric")}
        for key in ("shift_seconds", "mean", "std", "fold_count"):
            item[key] = ref(float(r[key]), sp, key, f"{r['shift_modality']}:{r['shift_seconds']}:{r['metric']}")
        numbers["shift_sensitivity"].append(item)
    vp = "outputs/q1_visual_encoder_ablation/visual_encoder_comparison.csv"
    vi = "outputs/q1_visual_encoder_ablation/visual_encoder_metrics.csv"
    visual_intrinsic = {r["encoder"]: r for r in load_csv(vi)}
    for r in load_csv(vp):
        if r["encoder"] not in {"SigLIP2", "DINOv3"}:
            continue
        name = r["encoder"]
        item = {"encoder": ref(name, vp, "encoder", name)}
        item.update(numeric_row(r, vp, f"encoder={name}", [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]))
        for key in ("native_dimension", "total_native_rows", "coverage", "empty_bin_ratio", "traceability_rate", "extraction_seconds", "peak_gpu_bytes"):
            item[key] = ref(float(visual_intrinsic[name][key]), vi, key, f"encoder={name}")
        numbers["visual_encoder_ablation"].append(item)
    mp = "outputs/q1_modality_quality_ablation/modality_ablation_summary.csv"
    for r in load_csv(mp):
        item = {"configuration": ref(r["configuration"], mp, "configuration", r["configuration"])}
        item.update(numeric_row(r, mp, f"configuration={r['configuration']}", [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]))
        numbers["modality_ablation"].append(item)
    qp = "outputs/q1_modality_quality_ablation/paper_quality_alignment.csv"
    qm = "outputs/q1_modality_quality_ablation/quality_alignment_metrics.csv"
    quality_metric = load_csv(qm)[0]
    for r in load_csv(qp):
        item = {"configuration": ref(r["configuration"], qp, "configuration", r["configuration"])}
        item.update(numeric_row(r, qp, f"configuration={r['configuration']}", [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]))
        for key in ("changed_text_bins", "changed_share_of_valid", "valid_text_bins", "native_fallback_interval_count"):
            item[key] = ref(float(quality_metric[key]), qm, key, "single summary row")
        numbers["quality_alignment_ablation"].append(item)
    vis_manifest_path = "outputs/q1_visual_encoder_ablation/run_manifest.json"
    vis_manifest = load_json(vis_manifest_path)
    numbers["model_revisions"]["dinov3_ablation"] = ref(vis_manifest["model"]["official_revision"], vis_manifest_path, "model.official_revision")
    numbers["final_models"]["dinov3_ablation"] = ref(vis_manifest["model"]["checkpoint"], vis_manifest_path, "model.checkpoint")
    numbers["runtime"]["dinov3_extraction_seconds"] = next(r["extraction_seconds"] for r in numbers["visual_encoder_ablation"] if value(r["encoder"]) == "DINOv3")
    numbers["runtime"]["dinov3_probe_seconds"] = ref(vis_manifest["probe_seconds"], vis_manifest_path, "probe_seconds")
    numbers["memory"]["dinov3_extraction"] = ref(vis_manifest["extraction_peak_gpu_bytes"], vis_manifest_path, "extraction_peak_gpu_bytes")
    mod_manifest_path = "outputs/q1_modality_quality_ablation/run_manifest.json"
    mod_manifest = load_json(mod_manifest_path)
    numbers["runtime"]["modality_quality_probe_seconds"] = ref(mod_manifest["probe_seconds"], mod_manifest_path, "probe_seconds")
    numbers["test_results"] = {
        "native_sample_count": ref(validation["sample_count"], validation_path, "sample_count"),
        "native_all_finite": ref(validation["all_features_finite"], validation_path, "all_features_finite"),
        "source_hash_errors": ref(validation["source_hash_errors"], validation_path, "source_hash_errors"),
        "alignment_errors": ref(validation["alignment_errors"], validation_path, "alignment_errors"),
        "baseline_max_abs_metric_drift": ref(mod_manifest["baseline_max_abs_metric_drift"], mod_manifest_path, "baseline_max_abs_metric_drift"),
        "modality_quality_mask_equal": ref(mod_manifest["mask_equal"], mod_manifest_path, "mask_equal")}
    save_json(OUT / "00_manifest/q1_paper_numbers.json", numbers)
    return numbers


def make_tables(numbers):
    align = numbers["alignment_ablation"]
    visual = numbers["visual_encoder_ablation"]
    modality = numbers["modality_ablation"]
    quality = numbers["quality_alignment_ablation"]
    tab1 = []
    timestamp = {"text": "Whisper word / owning segment fallback", "audio": "WAV sample stride/receptive field", "vision": "observed MP4 frame PTS"}
    for m in MODALITIES:
        tab1.append({"Modality": m.title(), "Encoder": value(numbers["final_models"][m]),
                     "Native dimension": value(numbers["feature_dimensions"][m]),
                     "Native rows": value(numbers["native_row_counts"][m]),
                     "Aligned dimension": value(numbers["feature_dimensions"][m]),
                     "Mean coverage": value(numbers["modality_coverage"][m]),
                     "Timestamp source": timestamp[m]})
    save_csv(T / "table_q1_feature_scheme.csv", tab1)
    tab2 = []
    for r in align:
        method = value(r["method"])
        row = {"Method": method}
        for metric in METRICS:
            display = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "mae": "MAE", "pearson": "Pearson"}[metric]
            row[display] = value(r[f"{metric}_mean"])
            row[f"{metric}_std"] = value(r[f"{metric}_std"])
        for m in MODALITIES:
            row[f"{m}_coverage"] = value(r[f"{m}_coverage"])
            row[f"{m}_traceability"] = value(r[f"{m}_traceability"])
        row["Coverage"] = np.mean([row[f"{m}_coverage"] for m in MODALITIES])
        row["Traceability"] = np.mean([row[f"{m}_traceability"] for m in MODALITIES])
        row["temporal_semantic_issue"] = (f"{int(value(r['nearest_center_zero_overlap_assignments']))} valid text bins assigned a zero-overlap feature"
                                          if method == "nearest_center" else "none observed")
        tab2.append(row)
    save_csv(T / "table_alignment_ablation.csv", tab2)
    tab3 = []
    for r in visual:
        row = {"Encoder": value(r["encoder"])}
        for metric in METRICS:
            display = {"accuracy": "Accuracy", "macro_f1": "Macro-F1", "mae": "MAE", "pearson": "Pearson"}[metric]
            row[display] = value(r[f"{metric}_mean"])
            row[f"{metric}_std"] = value(r[f"{metric}_std"])
        row.update({"Extraction time (s)": value(r["extraction_seconds"]), "Peak memory (bytes)": int(value(r["peak_gpu_bytes"])),
                    "Native rows": int(value(r["total_native_rows"])), "Traceability": value(r["traceability_rate"])})
        tab3.append(row)
    save_csv(T / "table_visual_backbone.csv", tab3)
    tab4 = []
    for r in modality:
        row = {"Modality set": value(r["configuration"])}
        for metric in METRICS:
            row[f"{metric}_mean"] = value(r[f"{metric}_mean"])
            row[f"{metric}_std"] = value(r[f"{metric}_std"])
        tab4.append(row)
    save_csv(T / "table_modality_ablation.csv", tab4)
    tab5 = []
    for r in quality:
        row = {"Method": "maximum_overlap_hard" if value(r["configuration"]) == "T+A+V" else value(r["configuration"])}
        for metric in METRICS:
            row[f"{metric}_mean"] = value(r[f"{metric}_mean"])
            row[f"{metric}_std"] = value(r[f"{metric}_std"])
        row["changed_text_bins"] = int(value(r["changed_text_bins"])) if row["Method"] != "maximum_overlap_hard" else 0
        row["changed_ratio"] = value(r["changed_share_of_valid"]) if row["Method"] != "maximum_overlap_hard" else 0
        tab5.append(row)
    save_csv(T / "table_quality_alignment.csv", tab5)
    mb = load_csv("outputs/q1_modality_quality_ablation/modality_bootstrap_ci.csv")
    selected = [r for r in mb if r["reference"] == "T+A+V" and r["comparator"] in {"T+A", "T+V", "T"}]
    save_csv(T / "table_modality_bootstrap_key_comparisons.csv", selected)
    qb = load_csv("outputs/q1_modality_quality_ablation/quality_alignment_bootstrap_ci.csv")
    save_csv(T / "table_quality_alignment_bootstrap.csv", qb)
    save_csv(A / "modality_bootstrap_all_comparisons.csv", mb)
    return {"scheme": tab1, "alignment": tab2, "visual": tab3, "modality": tab4, "quality": tab5}


def plot(fig, basename):
    fig.savefig(G / f"{basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(G / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(numbers, tables):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.spines.right": False, "axes.spines.top": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "figure.facecolor": "white"})
    (G / "q1_pipeline.mmd").write_text("""flowchart LR
    A[Original MP4] --> B[Media audit and PTS timeline]
    B --> C[Whisper ASR and text reliability]
    C --> D[RoBERTa text native features + real time]
    B --> E[WavLM audio native features + WAV sample time]
    B --> F[SigLIP2 visual native features + frame PTS]
    D --> G[K=50 common timeline]
    E --> G
    F --> G
    G --> H[Maximum overlap hard alignment]
    H --> I[Features, masks and source traces]
""", encoding="utf-8")
    coverage = [{"modality": m, "mean_coverage": value(numbers["modality_coverage"][m])} for m in MODALITIES]
    save_csv(G / "coverage_source.csv", coverage)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar([r["modality"].title() for r in coverage], [r["mean_coverage"] for r in coverage], color=["#426A8B", "#729EAC", "#A8B9B0"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Mean valid-bin coverage"); ax.set_title("Coverage across 100 clips")
    plot(fig, "coverage")
    for metric, ylabel in (("Accuracy", "Accuracy"), ("Macro-F1", "Macro-F1"), ("MAE", "MAE"), ("Pearson", "Pearson r")):
        key = metric
        data = [{"method": r["Method"], "mean": r[key], "std": r[("macro_f1" if key == "Macro-F1" else key.lower()) + "_std"]} for r in tables["alignment"]]
        filename = "alignment_" + ("macro_f1" if metric == "Macro-F1" else metric.lower())
        save_csv(G / f"{filename}_source.csv", data)
        fig, ax = plt.subplots(figsize=(4.7, 2.6))
        ax.errorbar(range(len(data)), [r["mean"] for r in data], yerr=[r["std"] for r in data], fmt="o", capsize=3, color="#426A8B")
        ax.set_xticks(range(len(data)), ["Hard", "Nearest", "Weighted"]); ax.set_ylabel(ylabel)
        ax.set_title("Alignment ablation: 25-fold mean ± SD")
        plot(fig, filename)
    shifts = numbers["shift_sensitivity"]
    for modality in ("audio", "vision"):
        data = [{"shift_seconds": value(r["shift_seconds"]), "macro_f1_mean": value(r["mean"]), "macro_f1_std": value(r["std"])}
                for r in shifts if value(r["shift_modality"]) == modality and value(r["metric"]) == "macro_f1"]
        data.sort(key=lambda r: r["shift_seconds"])
        save_csv(G / f"{modality}_shift_macro_f1_source.csv", data)
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        ax.errorbar([r["shift_seconds"] for r in data], [r["macro_f1_mean"] for r in data],
                    yerr=[r["macro_f1_std"] for r in data], marker="o", capsize=3, color="#426A8B")
        ax.set_xlabel("Shift Δt (s)"); ax.set_ylabel("Macro-F1"); ax.set_title(f"{modality.title()} time shift")
        plot(fig, f"{modality}_shift_macro_f1")
    data = [{"encoder": r["Encoder"], "macro_f1_mean": r["Macro-F1"], "macro_f1_std": r["macro_f1_std"]} for r in tables["visual"]]
    save_csv(G / "visual_backbone_macro_f1_source.csv", data)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.bar([r["encoder"] for r in data], [r["macro_f1_mean"] for r in data], yerr=[r["macro_f1_std"] for r in data], capsize=3, color=["#426A8B", "#A8B9B0"])
    ax.set_ylabel("Macro-F1"); ax.set_title("Visual backbone: 25-fold mean ± SD")
    plot(fig, "visual_backbone_macro_f1")
    for metric, ylabel in (("macro_f1", "Macro-F1"), ("pearson", "Pearson r")):
        data = [{"modality_set": r["Modality set"], "mean": r[f"{metric}_mean"], "std": r[f"{metric}_std"]} for r in tables["modality"]]
        save_csv(G / f"modality_{metric}_source.csv", data)
        fig, ax = plt.subplots(figsize=(5.1, 2.7))
        ax.bar(range(len(data)), [r["mean"] for r in data], yerr=[r["std"] for r in data], capsize=2, color="#729EAC")
        ax.set_xticks(range(len(data)), [r["modality_set"] for r in data]); ax.set_ylabel(ylabel)
        ax.set_title(f"Modality ablation: {ylabel}, 25-fold mean ± SD")
        plot(fig, f"modality_{metric}")


def cases(ids, sample_rows):
    ranked = [r for r in sample_rows if r["official_alignment_valid"] == "True" and r["media_text_valid"] == "True"
              and r["timestamp_valid"] == "True" and not re.search("severe|unresolved|failure", r["text_warning"], re.I)]
    ranked.sort(key=lambda r: (-min(r["text_coverage"], r["audio_coverage"], r["vision_coverage"]),
                               -(r["text_native_length"] + r["vision_native_length"]), r["sample_id"]))
    if not ranked:
        raise ValueError("No traceable main case meeting recorded validity criteria")
    case = ranked[0]
    sid = case["sample_id"]
    timeline = load_json(f"outputs/q1_cpu/samples/{sid}/timeline.json")
    native = {m: load_json(f"outputs/q1_features_100_final/{m}/{sid}.json") for m in MODALITIES}
    with np.load(ROOT / f"outputs/q1_alignment_ablation/aligned/maximum_overlap_hard/{sid}.npz", allow_pickle=False) as data:
        traces = {m: json.loads(str(data[f"{m}_trace_json"])) for m in MODALITIES}
        masks = {m: data[f"{m}_valid"] for m in MODALITIES}
    candidate_bins = [k for k in range(50) if all(masks[m][k] for m in MODALITIES)]
    if not candidate_bins:
        raise ValueError("Main case has no jointly observed bin")
    bin_idx = candidate_bins[len(candidate_bins) // 2]
    bin_info = timeline["bins"][bin_idx]
    trace_rows = []
    selected = {}
    for k, b in enumerate(timeline["bins"]):
        for m in MODALITIES:
            tr = traces[m][k]
            j = tr["source_feature_indices"][0] if tr["source_feature_indices"] else None
            record = native[m]["records"][j] if j is not None else None
            trace_rows.append({"sample_id": sid, "bin_index": k, "bin_start": b["start"], "bin_end": b["end"],
                               "modality": m, "valid": bool(masks[m][k]), "selected_native_index": "" if j is None else j,
                               "source_start": "" if record is None else record["start"],
                               "source_end": "" if record is None else record["end"],
                               "overlap_seconds": "" if j is None else tr["overlap_seconds"][0],
                               "frame_index": record.get("frame_index", "") if record else "",
                               "frame_pts": record.get("frame_pts", "") if record else "",
                               "wav_sample_start": record.get("wav_sample_start", "") if record else "",
                               "wav_sample_end": record.get("wav_sample_end", "") if record else "",
                               "text": record.get("text", "") if record else "",
                               "timestamp_source": record.get("timestamp_source", "") if record else "",
                               "source_mp4": timeline["source_mp4"], "source_mp4_hash": timeline["source_sha256"]})
            if k == bin_idx:
                selected[m] = {"native_feature_index": j, "record": record, "overlap_seconds": tr["overlap_seconds"][0]}
    save_csv(C / "main_case_trace.csv", trace_rows)
    main = {"sample_id": sid, "selection_basis": "official alignment valid, no severe warning, highest minimum modality coverage then observed native count",
            "official_text": native["text"]["encoder"]["parameters"]["official_text_unchanged"],
            "selected_text_source": native["text"]["encoder"]["parameters"]["text_feature_source"],
            "duration": timeline["duration"], "source_mp4": timeline["source_mp4"], "source_mp4_sha256": timeline["source_sha256"],
            "coverage": {m: case[f"{m}_coverage"] for m in MODALITIES}, "selected_bin": {"index": bin_idx, "start": bin_info["start"],
            "end": bin_info["end"], "modalities": selected}, "trace_csv": "main_case_trace.csv"}
    save_json(C / "main_case.json", main)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    duration = float(timeline["duration"])
    for b in timeline["bins"]:
        ax.axvline(float(b["start"]), color="#d9e0e5", lw=.35, zorder=0)
    ax.axvspan(float(bin_info["start"]), float(bin_info["end"]), color="#f3d28d", alpha=.55, zorder=0)
    for rec in native["text"]["records"]:
        ax.plot([rec["start"], rec["end"]], [3, 3], color="#3f6686", lw=3)
    for rec in native["audio"]["records"]:
        ax.plot([rec["start"], rec["end"]], [2, 2], color="#6a9ca8", lw=.35, alpha=.45)
    for rec in native["vision"]["records"]:
        ax.plot([rec["frame_pts"]], [1], marker="|", color="#7f927a", ms=7, ls="")
    ax.plot([selected["text"]["record"]["start"], selected["text"]["record"]["end"]], [3, 3], color="#c47d37", lw=4)
    ax.plot([selected["audio"]["record"]["start"], selected["audio"]["record"]["end"]], [2, 2], color="#c47d37", lw=4)
    ax.plot([selected["vision"]["record"]["frame_pts"]], [1], marker="o", color="#c47d37", ms=5, ls="")
    ax.set_xlim(0, duration); ax.set_ylim(.5, 3.5); ax.set_yticks([1, 2, 3], ["SigLIP2 frame PTS", "WavLM native intervals", "Selected text intervals"])
    # Overall figure title and number are supplied by the manuscript caption.
    ax.set_xlabel("Original MP4 presentation time (s)")
    midpoint = (float(bin_info["start"]) + float(bin_info["end"])) / 2
    zstart, zend = max(0, midpoint - 0.35), min(duration, midpoint + 0.35)
    inset = ax.inset_axes([0.67, 0.55, 0.3, 0.34])
    inset.axvspan(float(bin_info["start"]), float(bin_info["end"]), color="#f3d28d", alpha=.6)
    for m, y, color in (("text", 3, "#3f6686"), ("audio", 2, "#6a9ca8")):
        for rec in native[m]["records"]:
            if float(rec["end"]) > zstart and float(rec["start"]) < zend:
                inset.plot([rec["start"], rec["end"]], [y, y], color=color, lw=1.6 if m == "audio" else 3)
        rec = selected[m]["record"]
        inset.plot([rec["start"], rec["end"]], [y, y], color="#c47d37", lw=4)
    for rec in native["vision"]["records"]:
        if zstart <= float(rec["frame_pts"]) <= zend:
            inset.plot([rec["frame_pts"]], [1], marker="|", color="#7f927a", ms=6, ls="")
    inset.plot([selected["vision"]["record"]["frame_pts"]], [1], marker="o", color="#c47d37", ms=4, ls="")
    inset.set_xlim(zstart, zend); inset.set_ylim(.5, 3.5); inset.set_yticks([])
    inset.set_title(f"Bin {bin_idx} detail", fontsize=7); inset.tick_params(axis="x", labelsize=6)
    fig.tight_layout(); fig.savefig(C / "main_case_timeline.png", dpi=300); fig.savefig(C / "main_case_timeline.pdf"); plt.close(fig)
    anomaly_candidates = [r for r in sample_rows if r["text_feature_source"] == "media_asr" and r["official_consistency_score"] != ""]
    anomaly = min(anomaly_candidates, key=lambda r: (r["official_consistency_score"], r["sample_id"]))
    aid = anomaly["sample_id"]
    text_meta = load_json(f"outputs/q1_features_100_final/text/{aid}.json")
    aparams = text_meta["encoder"]["parameters"]
    anom = f"""# Text anomaly case: {aid}

- Official transcript (unchanged): {aparams['official_text_unchanged']}
- Observed ASR from the same MP4: {aparams['observed_asr_text']}
- Official consistency score: {anomaly['official_consistency_score']}
- Final text feature source: {anomaly['text_feature_source']}
- Timestamp handling: existing same-sample Whisper word intervals; {anomaly['timestamp_fallback_count']} native feature intervals use coarse owning-segment fallback.
- Source MP4 SHA-256: `{anomaly['source_mp4_hash']}`.

The official transcript was retained for audit. Its words could not be reliably mapped to this clip's observed ASR timeline, so the frozen Q1 pipeline encoded the media-observed text. No forced alignment or cross-clip timestamp transfer was performed. This file only organizes existing evidence; it does not re-evaluate the audio.
"""
    (C / "text_anomaly_case.md").write_text(anom, encoding="utf-8")
    return main, anomaly


def reproducibility(numbers, main_case):
    versions = []
    def add(name, revision, source):
        versions.append({"component": name, "version_or_revision": revision if revision else "unavailable", "source_file": source})
    for m in MODALITIES:
        add(value(numbers["final_models"][m]), value(numbers["model_revisions"][m]), numbers["model_revisions"][m]["source_file"])
    add("openai/whisper-large-v3-turbo", "unavailable", "outputs/q1_text_audit/summary.json")
    add(value(numbers["final_models"]["dinov3_ablation"]), value(numbers["model_revisions"]["dinov3_ablation"]),
        "outputs/q1_visual_encoder_ablation/run_manifest.json")
    add("transformers (GPU)", "4.57.3", "outputs/q1_visual_encoder_ablation/run_manifest.json:model.transformers_version")
    mod_manifest = load_json("outputs/q1_modality_quality_ablation/run_manifest.json")
    add("torch (GPU)", mod_manifest.get("torch_version", "unavailable"), "outputs/q1_modality_quality_ablation/run_manifest.json:torch_version")
    runlog = (ROOT / "outputs/run_log.md").read_text(encoding="utf-8")
    cpu_python = re.search(r"Python ([0-9.]+)", runlog)
    cpu_numpy = re.search(r"NumPy ([0-9.]+)", runlog)
    cpu_ffmpeg = re.search(r"FFmpeg/ffprobe ([0-9.]+)", runlog)
    add("Python (CPU preparation)", cpu_python.group(1) if cpu_python else "unavailable", "outputs/run_log.md")
    add("NumPy (CPU preparation)", cpu_numpy.group(1) if cpu_numpy else "unavailable", "outputs/run_log.md")
    add("FFmpeg/ffprobe (CPU preparation)", cpu_ffmpeg.group(1) if cpu_ffmpeg else "unavailable", "outputs/run_log.md")
    add("Python (GPU run)", "unavailable", "not recorded in frozen run manifest")
    add("NumPy (GPU run)", "unavailable", "not recorded in frozen run manifest")
    save_csv(R / "model_versions.csv", versions)
    (R / "software_versions.txt").write_text("\n".join(f"{v['component']}: {v['version_or_revision']} [{v['source_file']}]" for v in versions) + "\n", encoding="utf-8")
    (R / "commands.md").write_text("""# Reproduction commands

The Q1 algorithms and encoder results are frozen. The following commands only rebuild and verify the local paper package from saved outputs:

```powershell
python scripts/build_q1_final_local.py
node scripts/build_q1_summary_xlsx.mjs
python scripts/verify_q1_final_local.py
```

Historic extraction commands are documented in the workspace README and run logs; do not rerun them for this package. `sample_ids.csv` fixes the row order. No remote server connection is needed.
""", encoding="utf-8")
    (R / "directory_map.md").write_text("""# Local package map

- `00_manifest`: source-traceable paper numbers.
- `01_final_features`: formal FP32 aligned matrices, masks and FP16 candidate.
- `02_paper_tables`: five main tables plus compact bootstrap comparisons.
- `03_paper_figures`: Mermaid flow source, plot CSVs, PNG/PDF figures.
- `04_case_studies`: traceable main and anomalous-text examples.
- `05_appendix`: 100-sample CSV/XLSX summary and full modality bootstrap.
- `06_reproducibility`: recorded revisions, software versions and commands.
- `07_audit`: compression check, file hashes, inventory and final audit.

Frozen native metadata, sample timelines and complete per-bin trace remain in their original authoritative output directories and are referenced, not moved.
""", encoding="utf-8")
    (R / "README.md").write_text("""# Reproducibility boundary

This package is derived from persisted Q1 results. All paper table/figure numbers flow through `00_manifest/q1_paper_numbers.json`, which includes source paths and fields. The 100-sample appendix and trace cases are joined directly from final native/CPU metadata. Model revisions are copied from frozen manifests; unavailable software versions are explicitly marked. The formal vision backbone is SigLIP2; DINOv3 appears only in its sensitivity table.
""", encoding="utf-8")


def status(numbers, tables, main_case, anomaly):
    status_text = f"""# Q1 final local status

1. Formal scheme: {value(numbers['final_models']['text'])} / {value(numbers['final_models']['audio'])} / {value(numbers['final_models']['vision'])}; `maximum_overlap_hard`, K={value(numbers['aligned_length'])}.
2. Processing: {value(numbers['sample_count'])}/100 original MP4 clips have final native and aligned features.
3. Formal aligned shapes: text/audio/vision each `(100,50,768)` FP32, with separate masks. FP16 is a candidate copy.
4. Mean valid-bin coverage: text {value(numbers['modality_coverage']['text']):.4f}, audio {value(numbers['modality_coverage']['audio']):.4f}, vision {value(numbers['modality_coverage']['vision']):.4f}.
5. Text source: {value(numbers['text_source_counts']['official_transcript'])} official-mapped, {value(numbers['text_source_counts']['media_asr'])} media ASR; official transcript remains unchanged in native metadata. {value(numbers['timestamp_fallback_count'])} selected text feature intervals use coarse segment fallback.
6. Alignment: maximum-overlap hard keeps positive time overlap, source indices and masks; nearest-center had {int(value(numbers['alignment_ablation'][1]['nearest_center_zero_overlap_assignments']))} valid text-bin zero-overlap selections. Probe differences were small.
7. Visual backbone: SigLIP2 remained formal; DINOv3 is a separate single-variable sensitivity experiment. SigLIP2 had higher four-metric downstream means and shorter native extraction time in the frozen run.
8. Modality ablation: T+A achieved the highest classification means in the frozen 25-fold sanity-check; T+A+V did not uniformly dominate single/dual modalities.
9. Text quality-aware hard changed {int(value(numbers['quality_alignment_ablation'][1]['changed_text_bins']))} valid text bins; four-metric changes were mixed, so formal hard alignment remained frozen.
10. Generated five core tables, coverage/alignment/shift/backbone/modality figures, traceable main case `{main_case['sample_id']}` and text anomaly case `{anomaly['sample_id']}`.
11. Local directory: `outputs/q1_final_local/`.
12. Limits: 100 clips and repeated-CV probe are small-sample sanity checks; related clips may correlate. FP16 quantization introduces measured error and is not the formal feature version. Original source traces stay in the frozen output tree.
"""
    (OUT / "Q1_FINAL_STATUS.md").write_text(status_text, encoding="utf-8")


def main():
    preflight_result = preflight()
    for p in (OUT / "00_manifest", F, T, G, C, A, R, AUDIT):
        p.mkdir(parents=True, exist_ok=True)
    ids, samples, arrays, masks, revisions, errors = build_samples_and_features()
    numbers = build_numbers(samples, revisions, errors, preflight_result)
    tables = make_tables(numbers)
    make_figures(numbers, tables)
    main_case, anomaly = cases(ids, samples)
    reproducibility(numbers, main_case)
    status(numbers, tables, main_case, anomaly)
    print(f"Built frozen Q1 local package for {len(ids)} samples at {OUT}")


if __name__ == "__main__":
    main()
