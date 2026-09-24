"""Run HEAF Q3-1 on Attachment2 valid only, with explicit stop gates."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

from src.data.dataset import MODALITIES
from src.q3.coalitions import COALITIONS, explain_batch, masked_coalition
from src.q3.faithfulness import deletion_curve, summarize_curve, summarize_top_random
from src.q3.frozen_predictor import FrozenP2Predictor
from src.q3.run_validation import ROOT, clean_reproduction, valid_dataset
from src.q3.temporal_occlusion import evaluate_windows


OUTPUT = ROOT / "outputs/q3"
EXPERIMENT = ROOT / "experiments/q3/exp_001_heaf_validation"
CONFIG = ROOT / "configs/final/q3_heaf_validation.yaml"
METRICS = OUTPUT / "heaf_validation_metrics.json"


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=jsonable,
                               allow_nan=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=jsonable,
                                    allow_nan=False) + "\n")


def video_id(sample_id: str) -> str:
    if "$_$" not in sample_id:
        raise ValueError(f"sample ID lacks official video/clip separator: {sample_id!r}")
    return sample_id.split("$_$", 1)[0]


def split_ids(dataset) -> dict:
    design, audit = [], []
    for index, raw_id in enumerate(dataset.ids):
        sample_id = str(raw_id)
        group = video_id(sample_id)
        token = hashlib.sha256(f"20260924|{group}".encode()).digest()
        (design if token[0] & 1 == 0 else audit).append(index)
    groups_design = {video_id(str(dataset.ids[i])) for i in design}
    groups_audit = {video_id(str(dataset.ids[i])) for i in audit}
    if groups_design & groups_audit or not design or not audit:
        raise RuntimeError("video_id overlap or empty Q3 validation subset")
    return {"design_indices": design, "audit_indices": audit,
            "design_sample_ids": [str(dataset.ids[i]) for i in design],
            "audit_sample_ids": [str(dataset.ids[i]) for i in audit],
            "design_video_ids": sorted(groups_design), "audit_video_ids": sorted(groups_audit),
            "assignment": "sha256('20260924|video_id')[0] & 1: 0 design, 1 audit",
            "overlapping_video_ids": []}


def sample_batch(dataset, index: int) -> dict:
    item = dataset[index]
    return {key: item[key].unsqueeze(0) for key in (*MODALITIES, "padding_mask")}


def assert_mask_semantics(batch: dict) -> dict:
    saved = {key: batch[key].clone() for key in (*MODALITIES, "padding_mask", "native_zero_mask")}
    pad = batch["padding_mask"]
    for coalition in COALITIONS:
        masked = masked_coalition(batch, coalition)
        if not torch.equal(masked["padding_mask"], pad) or not torch.equal(masked["native_zero_mask"], saved["native_zero_mask"]):
            raise RuntimeError("coalition changed padding or native-zero records")
        for modality in MODALITIES:
            if modality in coalition:
                if not torch.equal(masked[modality], saved[modality]):
                    raise RuntimeError("coalition changed an included modality")
            elif not torch.equal(masked[modality][~pad], saved[modality][~pad]) or not torch.all(masked[modality][pad] == 0):
                raise RuntimeError("coalition violated valid-only zeroing")
    if any(not torch.equal(batch[key], value) for key, value in saved.items()):
        raise RuntimeError("coalition modified source tensors in place")
    return {"passed": True, "coalitions_checked": len(COALITIONS),
            "samples_checked": int(pad.shape[0]), "native_zero_records_unchanged": True}


def compute_explanations(predictor, dataset, indices: list[int], *, check_mask: bool = False) -> tuple[dict, dict]:
    loader = DataLoader(Subset(dataset, indices), batch_size=128, shuffle=False, num_workers=0)
    records = {}
    max_error = {"classification": 0.0, "regression": 0.0}
    mask_check = None
    for batch in loader:
        if check_mask:
            current_check = assert_mask_semantics(batch)
            if mask_check is None:
                mask_check = current_check
            else:
                mask_check["samples_checked"] += current_check["samples_checked"]
        explained = explain_batch(predictor, batch)
        max_error["classification"] = max(max_error["classification"], float(np.max(np.abs(explained["efficiency_error_class"]))))
        max_error["regression"] = max(max_error["regression"], float(np.max(np.abs(explained["efficiency_error_reg"]))))
        for row, raw_id in enumerate(batch["id"]):
            sample_id = str(raw_id)
            records[sample_id] = {
                "sample_id": sample_id, "video_id": video_id(sample_id),
                "predicted_class": int(explained["fixed_class"][row]),
                "full_logits": explained["logits"][row, -1],
                "full_regression": float(explained["regression"][row, -1]),
                "class_margin_full": float(explained["class_margin"][row, -1]),
                "class_margin_empty": float(explained["class_margin"][row, 0]),
                "regression_empty": float(explained["regression"][row, 0]),
                "phi_class": explained["phi_class"][row],
                "phi_reg": explained["phi_reg"][row],
                "primary_class": MODALITIES[int(explained["primary_class"][row])],
                "primary_reg": MODALITIES[int(explained["primary_reg"][row])],
                "supports_class": bool(explained["supports_class"][row]),
                "primary_agreement": bool(explained["primary_class"][row] == explained["primary_reg"][row]),
                "interaction_class": {key: {name: float(value[name][row]) for name in value}
                                      for key, value in explained["interaction_class"].items()},
                "interaction_reg": {key: {name: float(value[name][row]) for name in value}
                                    for key, value in explained["interaction_reg"].items()},
            }
    return records, {"n_samples": len(records), "max_absolute_efficiency_error": max_error,
                     "mask_semantics": mask_check}


def evaluate_one(predictor, dataset, index: int, record: dict, rho: float) -> dict:
    return evaluate_windows(predictor, sample_batch(dataset, index), record["primary_class"],
                            rho, record["predicted_class"], record["full_logits"],
                            record["full_regression"], record["sample_id"])


def interaction_summary(records: list[dict]) -> dict:
    result = {}
    for target in ("class", "reg"):
        result[target] = {}
        for pair in ("text_audio", "text_vision", "audio_vision"):
            values = np.asarray([row[f"interaction_{target}"][pair]["value"] for row in records])
            result[target][pair] = {"mean": float(values.mean()), "median": float(np.median(values)),
                                    "negative_fraction": float(np.mean(values < 0)),
                                    "positive_fraction": float(np.mean(values > 0))}
    return result


def compact_window(window: dict) -> dict:
    return {"modality": window["modality"], "rho": window["rho"],
            "valid_length": window["valid_length"], "window_length": window["window_length"],
            "top_start": window["top_start"], "top_end": window["top_end"],
            "top": window["top"], "random_mean": window["random_mean"],
            "supports_predicted_class": window["supports_predicted_class"],
            "temporal_curve": [float(value) if np.isfinite(value) else None for value in window["curve"]],
            "regression_temporal_curve": [float(value) if np.isfinite(value) else None for value in window["regression_curve"]],
            "temporal_windows": [{"start_index": int(start), "end_index": int(start + window["window_length"]),
                                  "class_logit_drop": float(window["drops"]["class_logit"][i]),
                                  "class_log_odds_drop": float(window["drops"]["class_margin"][i]),
                                  "confidence_drop": float(window["drops"]["confidence"][i]),
                                  "regression_shift": float(window["drops"]["regression"][i])}
                                 for i, start in enumerate(window["starts"])]}


def interval_iou(a: dict, b: dict) -> float:
    if a["modality"] != b["modality"]:
        return 0.0
    overlap = max(0, min(a["top_end"], b["top_end"]) - max(a["top_start"], b["top_start"]))
    union = max(a["top_end"], b["top_end"]) - min(a["top_start"], b["top_start"])
    return overlap / union


def rank_correlation(a, b) -> float | None:
    correlation = float(spearmanr(a, b).statistic)
    return correlation if np.isfinite(correlation) else None


def render_report(metrics: dict) -> str:
    lines = ["# Q3-1 HEAF validation", "", f"Status: **{metrics['status']}**.",
             "", "Only Attachment2 valid was indexed and evaluated. No training or Attachment3/4 use.",
             "The official aligned pickle is monolithic, so loading valid deserializes the container; test is never indexed or evaluated.", ""]
    clean = metrics.get("clean_reproduction")
    if clean:
        lines += ["## Frozen predictor gate", "", f"Passed: **{clean['passed']}** (absolute tolerance {clean['absolute_tolerance']}).",
                  f"Actual: Accuracy {clean['actual']['accuracy']:.9f}, Macro-F1 {clean['actual']['macro_f1']:.9f}, MAE {clean['actual']['mae']:.9f}, Pearson {clean['actual']['pearson']:.9f}.",
                  f"Maximum locked-metric delta: {max(abs(x) for x in clean['delta'].values()):.3g}.", ""]
    if "split" in metrics:
        split = metrics["split"]
        lines += ["## Grouped validation split", "", f"Design: {split['n_design']} clips / {split['n_design_videos']} video IDs; audit: {split['n_audit']} clips / {split['n_audit_videos']} video IDs. No video ID overlap.", ""]
    if "shapley" in metrics:
        sh = metrics["shapley"]
        lines += ["## Exact coalitions", "", f"Eight coalitions/sample; maximum efficiency errors: class {sh['max_absolute_efficiency_error']['classification']:.3g}, regression {sh['max_absolute_efficiency_error']['regression']:.3g}. Mask semantics: {sh['mask_semantics']['passed']} across {sh['mask_semantics']['samples_checked']} samples.",
                  "Negative pair values are reported only as negative interactions.", ""]
        if "primary" in metrics:
            p = metrics["primary"]
            lines += [f"Classification primary counts: {p['classification_counts']}; regression primary counts: {p['regression_counts']}; agreement {p['classification_regression_agreement']:.3f}.", ""]
        if "interaction" in metrics:
            lines += ["| Target | Pair | Mean interaction | Negative fraction |", "|---|---|---:|---:|"]
            for target, pairs in metrics["interaction"].items():
                for pair, row in pairs.items():
                    lines.append(f"| {target} | {pair} | {row['mean']:.6f} | {row['negative_fraction']:.3f} |")
            lines += [""]
    if "rho_selection" in metrics:
        sel = metrics["rho_selection"]
        lines += ["## Window selection (design subset only)", "", f"Chosen ρ: **{sel['chosen_rho']:.2f}**; stride 1; 32 random same-length draws, seed 20260924.",
                  "| ρ | Mean top−random class-margin drop |", "|---:|---:|"]
        lines += [f"| {rho} | {row['mean_top_minus_random_class_margin']:.6f} |" for rho, row in sel["candidates"].items()]
        lines += [""]
    if "faithfulness" in metrics:
        interval = metrics["faithfulness"]["interval"]
        lines += ["## Faithfulness (audit subset only)", "",
                  "| Outcome | Top mean | Random mean | Top−random mean | Grouped 95% CI | Fraction top>random | Negative top fraction |",
                  "|---|---:|---:|---:|---|---:|---:|"]
        for key in ("class_margin", "confidence", "class_logit", "regression", "regression_absolute"):
            x = interval[key]
            lines.append(f"| {key} | {x['top_mean']:.6f} | {x['random_mean']:.6f} | {x['top_minus_random_mean']:.6f} | {x['top_minus_random_grouped_bootstrap_95ci']} | {x['fraction_top_gt_random']:.3f} | {x['negative_effect_fraction']:.3f} |")
        lines += ["", "The interval selected by the largest margin drop is compared with random intervals from that same evaluated window set. Its margin advantage is partly built into selection; treat it as a selection sanity check, not independent proof of explanation quality.",
                  "", "### Deletion curve", "",
                  "| Deleted valid positions | Mean margin top−random | Grouped 95% CI | Top>random fraction | Negative top fraction | Mean confidence top−random |",
                  "|---:|---:|---|---:|---:|---:|"]
        for fraction, point in metrics["faithfulness"]["deletion_curve"].items():
            margin = point["class_margin"]
            confidence = point["confidence"]
            lines.append(f"| {float(fraction):.0%} | {margin['top_minus_random_mean']:.6f} | {margin['top_minus_random_grouped_bootstrap_95ci']} | {margin['fraction_top_gt_random']:.3f} | {margin['negative_effect_fraction']:.3f} | {confidence['top_minus_random_mean']:.6f} |")
        lines += ["", "Grouped intervals and sample-level results are in the JSON/JSONL artifacts.", ""]
    if "seed_sensitivity" in metrics:
        lines += ["## Seed sensitivity on fixed audit subset", "",
                  "ρ and audit IDs were not reselected. Seed42 remains the main predictor.", ""]
        for seed, row in metrics["seed_sensitivity"].items():
            lines.append(f"- Seed {seed}: predicted-class agreement {row['predicted_class_agreement']:.3f}; class primary agreement {row['classification_primary_agreement']:.3f}; regression primary agreement {row['regression_primary_agreement']:.3f}; class/regression Shapley rank correlation {row['classification_shapley_spearman_mean']:.3f}/{row['regression_shapley_spearman_mean']:.3f}; mean same-modality interval IoU {row['mean_interval_iou']:.3f}; top−random direction agreement {row['faithfulness_direction_agreement']:.3f}.")
        lines += ["", "Rank comparisons may involve different predicted classes; conditional same-class values are in metrics.json. Direction agreement for the selected interval has the same selection limitation as the main interval check.", ""]
    if "representative_failure_cases" in metrics:
        lines += ["## Representative audit cases", "", "Positive interval cases (largest top−random margin):"]
        lines += [f"- `{row['sample_id']}` ({row['primary_modality']}): {row['top_minus_random_class_margin']:+.4f}" for row in metrics["representative_positive_cases"]]
        lines += ["", "Failure cases (smallest 10% deletion top−random margin):"]
        lines += [f"- `{row['sample_id']}` ({row['primary_modality']}): {row['q10_top_minus_random_class_margin']:+.4f}" for row in metrics["representative_failure_cases"]]
        lines += [""]
    if metrics.get("stop_reason"):
        lines += ["", "## Stop reason", "", metrics["stop_reason"], ""]
    lines += ["", "Interpretation: these are frozen-model zero-intervention explanations, not real-world causal effects or raw-media grounding.", ""]
    return "\n".join(lines)


def save_progress(metrics: dict, cfg: dict) -> None:
    write_json(METRICS, metrics)
    write_json(EXPERIMENT / "metrics.json", metrics)
    (OUTPUT / "heaf_validation_report.md").write_text(render_report(metrics), encoding="utf-8")
    (EXPERIMENT / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run() -> dict:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "outputs/final/q2/q2_checkpoint_manifest.json").read_text(encoding="utf-8"))
    predictor = FrozenP2Predictor(ROOT / cfg["predictor"]["checkpoint"],
                                  cfg["predictor"]["sha256"], 42)
    dataset = valid_dataset()
    metrics = {"status": "RUNNING", "stage": "clean_reproduction",
               "predictor": {"seed": 42, "sha256": predictor.sha256},
               "data": {"source": "Attachment2", "split_indexed": "valid", "n_valid": len(dataset)}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    metrics["clean_reproduction"] = clean_reproduction(predictor, dataset)
    save_progress(metrics, cfg)
    print("clean gate", metrics["clean_reproduction"]["passed"], flush=True)
    if not metrics["clean_reproduction"]["passed"]:
        metrics.update(status="BLOCKED_BY_CLEAN_REPRODUCTION", stop_reason="Frozen P2 clean metrics exceed 1e-6 tolerance; no coalitions run.")
        save_progress(metrics, cfg)
        return metrics

    split = split_ids(dataset)
    write_json(EXPERIMENT / "validation_ids.json", split)
    metrics["split"] = {"n_design": len(split["design_indices"]), "n_audit": len(split["audit_indices"]),
                        "n_design_videos": len(split["design_video_ids"]), "n_audit_videos": len(split["audit_video_ids"]),
                        "id_file": "validation_ids.json", "video_id_overlap": 0}
    metrics["stage"] = "coalitions"
    save_progress(metrics, cfg)
    all_indices = list(range(len(dataset)))
    try:
        records, diagnostics = compute_explanations(predictor, dataset, all_indices, check_mask=True)
    except (RuntimeError, ValueError) as exc:
        metrics.update(status="BLOCKED_BY_COALITION", stop_reason=str(exc))
        save_progress(metrics, cfg)
        return metrics
    metrics["shapley"] = diagnostics
    metrics["interaction"] = interaction_summary(list(records.values()))
    metrics["primary"] = {
        "classification_counts": dict(Counter(row["primary_class"] for row in records.values())),
        "regression_counts": dict(Counter(row["primary_reg"] for row in records.values())),
        "classification_regression_agreement": float(np.mean([row["primary_agreement"] for row in records.values()])),
        "classification_negative_fallback_fraction": float(np.mean([not row["supports_class"] for row in records.values()])),
    }
    print("coalitions", diagnostics["n_samples"], diagnostics["max_absolute_efficiency_error"], flush=True)
    metrics["stage"] = "rho_design_selection"
    save_progress(metrics, cfg)

    candidate_rows: dict[float, list[dict]] = {0.10: [], 0.20: [], 0.30: []}
    for serial, index in enumerate(split["design_indices"], 1):
        row = records[str(dataset.ids[index])]
        for rho in candidate_rows:
            window = evaluate_one(predictor, dataset, index, row, rho)
            candidate_rows[rho].append({"sample_id": row["sample_id"], "video_id": row["video_id"],
                                        "top_minus_random_class_margin": window["top"]["class_margin"] - window["random_mean"]["class_margin"],
                                        "window_length": window["window_length"]})
        if serial % 100 == 0:
            print("design windows", serial, "/", len(split["design_indices"]), flush=True)
    candidate_summary = {f"{rho:.2f}": {"mean_top_minus_random_class_margin": float(np.mean([x["top_minus_random_class_margin"] for x in rows])),
                                        "mean_window_length": float(np.mean([x["window_length"] for x in rows]))}
                         for rho, rows in candidate_rows.items()}
    chosen = min(candidate_rows, key=lambda rho: (-candidate_summary[f"{rho:.2f}"]["mean_top_minus_random_class_margin"], rho))
    cfg["temporal"]["chosen_rho"] = chosen
    CONFIG.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    metrics["rho_selection"] = {"chosen_rho": chosen, "metric": cfg["temporal"]["selection_metric"],
                                "candidates": candidate_summary, "tie_break": "smaller_rho"}
    write_jsonl(EXPERIMENT / "design_window_scores.jsonl", [
        {"rho": rho, **row} for rho, rows in candidate_rows.items() for row in rows])
    metrics["stage"] = "faithfulness_audit"
    save_progress(metrics, cfg)
    print("chosen rho", chosen, candidate_summary, flush=True)

    audit_rows = []
    for serial, index in enumerate(split["audit_indices"], 1):
        record = records[str(dataset.ids[index])]
        window = evaluate_one(predictor, dataset, index, record, chosen)
        curve = deletion_curve(predictor, sample_batch(dataset, index), window,
                               record["predicted_class"], record["full_logits"],
                               record["full_regression"], record["sample_id"])
        audit_rows.append({**record, "interval": compact_window(window),
                           "deletion_curve": curve})
        if serial % 100 == 0:
            print("audit faithfulness", serial, "/", len(split["audit_indices"]), flush=True)
    write_jsonl(EXPERIMENT / "audit_sample_diagnostics.jsonl", audit_rows)
    metrics["faithfulness"] = {
        "interval": summarize_top_random(audit_rows, "interval"),
        "deletion_curve": summarize_curve(audit_rows),
        "random_draws": 32, "random_seed": 20260924,
        "no_discriminating_random_interval_count": sum(row["interval"]["valid_length"] == row["interval"]["window_length"] for row in audit_rows),
    }
    ordered = sorted(audit_rows, key=lambda row: row["interval"]["top"]["class_margin"] - row["interval"]["random_mean"]["class_margin"])
    failure_ordered = sorted(audit_rows, key=lambda row: row["deletion_curve"][0]["top"]["class_margin"] - row["deletion_curve"][0]["random_mean"]["class_margin"])
    metrics["representative_failure_cases"] = [{"sample_id": row["sample_id"], "video_id": row["video_id"],
                                                  "predicted_class": row["predicted_class"], "primary_modality": row["primary_class"],
                                                  "q10_top_minus_random_class_margin": row["deletion_curve"][0]["top"]["class_margin"] - row["deletion_curve"][0]["random_mean"]["class_margin"]}
                                                 for row in failure_ordered[:5]]
    metrics["representative_positive_cases"] = [{"sample_id": row["sample_id"], "video_id": row["video_id"],
                                                   "predicted_class": row["predicted_class"], "primary_modality": row["primary_class"],
                                                   "top_minus_random_class_margin": row["interval"]["top"]["class_margin"] - row["interval"]["random_mean"]["class_margin"]}
                                                  for row in reversed(ordered[-5:])]
    interval = metrics["faithfulness"]["interval"]
    margin_gain = interval["class_margin"]["top_minus_random_mean"]
    confidence_gain = interval["confidence"]["top_minus_random_mean"]
    save_progress(metrics, cfg)
    if margin_gain <= 0 or confidence_gain <= 0:
        metrics.update(status="BLOCKED_BY_FAITHFULNESS", stage="stopped",
                       stop_reason=f"Audit top deletion did not exceed same-length random: margin difference {margin_gain:.6g}, confidence difference {confidence_gain:.6g}; seed sensitivity not run.")
        cfg["stage"] = "RHO_LOCKED_FAITHFULNESS_FAILED"
        CONFIG.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        save_progress(metrics, cfg)
        return metrics
    print("faithfulness gate passed", margin_gain, confidence_gain, flush=True)

    metrics["stage"] = "seed_sensitivity"
    save_progress(metrics, cfg)
    sensitivity = {}
    main_by_id = {row["sample_id"]: row for row in audit_rows}
    for seed in (43, 44):
        item = next(entry for entry in manifest["checkpoints"] if entry["model"] == "B5-P2" and entry["seed"] == seed)
        candidate = FrozenP2Predictor(ROOT / item["checkpoint"]["relative_path"], item["checkpoint"]["sha256"], seed)
        other_records, diag = compute_explanations(candidate, dataset, split["audit_indices"])
        compare = []
        candidate_rows = []
        for serial, index in enumerate(split["audit_indices"], 1):
            sample_id = str(dataset.ids[index])
            main = main_by_id[sample_id]
            other = other_records[sample_id]
            other_window = evaluate_one(candidate, dataset, index, other, chosen)
            other_interval = compact_window(other_window)
            main_direction = np.sign(main["interval"]["top"]["class_margin"] - main["interval"]["random_mean"]["class_margin"])
            other_direction = np.sign(other_interval["top"]["class_margin"] - other_interval["random_mean"]["class_margin"])
            compare.append({"sample_id": sample_id, "video_id": main["video_id"],
                            "predicted_class_agree": main["predicted_class"] == other["predicted_class"],
                            "primary_class_agree": main["primary_class"] == other["primary_class"],
                            "primary_reg_agree": main["primary_reg"] == other["primary_reg"],
                            "shapley_class_rank_correlation": rank_correlation(main["phi_class"], other["phi_class"]),
                            "shapley_reg_rank_correlation": rank_correlation(main["phi_reg"], other["phi_reg"]),
                            "interval_iou": interval_iou(main["interval"], other_interval),
                            "faithfulness_direction_agree": bool(main_direction == other_direction)})
            candidate_rows.append({**other, "interval": other_interval})
            if serial % 100 == 0:
                print("seed sensitivity", seed, serial, "/", len(split["audit_indices"]), flush=True)
        write_jsonl(EXPERIMENT / f"seed{seed}_audit_diagnostics.jsonl", candidate_rows)
        write_jsonl(EXPERIMENT / f"seed{seed}_comparison.jsonl", compare)
        cls_rank = [row["shapley_class_rank_correlation"] for row in compare if row["shapley_class_rank_correlation"] is not None]
        cls_rank_same = [row["shapley_class_rank_correlation"] for row in compare if row["predicted_class_agree"] and row["shapley_class_rank_correlation"] is not None]
        reg_rank = [row["shapley_reg_rank_correlation"] for row in compare if row["shapley_reg_rank_correlation"] is not None]
        sensitivity[str(seed)] = {"checkpoint_sha256": candidate.sha256, "audit_count": len(compare),
                                  "predicted_class_agreement": float(np.mean([row["predicted_class_agree"] for row in compare])),
                                  "classification_primary_agreement": float(np.mean([row["primary_class_agree"] for row in compare])),
                                  "regression_primary_agreement": float(np.mean([row["primary_reg_agree"] for row in compare])),
                                  "classification_shapley_spearman_mean": float(np.mean(cls_rank)) if cls_rank else None,
                                  "classification_shapley_spearman_same_predicted_class_mean": float(np.mean(cls_rank_same)) if cls_rank_same else None,
                                  "regression_shapley_spearman_mean": float(np.mean(reg_rank)) if reg_rank else None,
                                  "mean_interval_iou": float(np.mean([row["interval_iou"] for row in compare])),
                                  "faithfulness_direction_agreement": float(np.mean([row["faithfulness_direction_agree"] for row in compare])),
                                  "shapley_diagnostics": diag}
        metrics["seed_sensitivity"] = sensitivity
        save_progress(metrics, cfg)
    metrics.update(status="HEAF_VALIDATION_PASSED", stage="complete")
    cfg["stage"] = "RHO_LOCKED_VALIDATION_PASSED"
    CONFIG.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    save_progress(metrics, cfg)
    return metrics


if __name__ == "__main__":
    outcome = run()
    print("Q3-1 status", outcome["status"], flush=True)
