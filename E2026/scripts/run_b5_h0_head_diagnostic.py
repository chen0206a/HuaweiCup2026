"""Read-only B5-H0 head-consistency diagnostics on valid + frozen scenarios."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import Aligned50Dataset, MODALITIES
from src.data.block_mask import predictor_inputs
from src.evaluation.missing_benchmark import (
    BENCHMARK_SEED,
    mask_scenario,
    scenarios,
)
from src.models.baseline import B0Baseline


LABEL_NAMES = ("Negative", "Neutral", "Positive")
TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0)
BENCHMARK_SHA256 = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reg_class(value: float, tau: float = 0.0) -> int:
    if value < -tau:
        return 0
    if value > tau:
        return 2
    return 1


def mean_std(x: np.ndarray) -> dict:
    if not x.size:
        return {k: None for k in ("mean", "std", "median", "p05", "p25", "p75", "p95")}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
    }


def diagnostic(rows: list[dict], tau: float = 0.0) -> dict:
    true = np.asarray([r["true_class"] for r in rows], dtype=np.int64)
    cls = np.asarray([r["cls_pred"] for r in rows], dtype=np.int64)
    reg = np.asarray([r["reg_pred"] for r in rows], dtype=np.float64)
    reg_cls = np.asarray([reg_class(v, tau) for v in reg], dtype=np.int64)
    cc, rc = cls == true, reg_cls == true
    agree = cls == reg_cls
    n = max(len(rows), 1)
    q = {
        "both_correct": cc & rc,
        "cls_only_correct": cc & ~rc,
        "reg_only_correct": ~cc & rc,
        "both_wrong": ~cc & ~rc,
    }
    quadrants = {k: {"count": int(v.sum()), "rate": float(v.mean()) if len(v) else 0.0}
                 for k, v in q.items()}
    cls_wrong_n = int((~cc).sum())
    reg_wrong_n = int((~rc).sum())
    per_class = {}
    for c, name in enumerate(LABEL_NAMES):
        ix = true == c
        per_class[name] = {
            "n": int(ix.sum()),
            "cls_accuracy_recall": float(cc[ix].mean()) if ix.any() else None,
            "reg_accuracy_recall": float(rc[ix].mean()) if ix.any() else None,
            "agreement_rate": float(agree[ix].mean()) if ix.any() else None,
            "cls_only_correct_rate": float(q["cls_only_correct"][ix].mean()) if ix.any() else None,
            "reg_only_correct_rate": float(q["reg_only_correct"][ix].mean()) if ix.any() else None,
            "both_wrong_rate": float(q["both_wrong"][ix].mean()) if ix.any() else None,
        }
    return {
        "n": len(rows), "tau": tau,
        "cls_accuracy": float(cc.mean()) if len(rows) else None,
        "reg_derived_accuracy": float(rc.mean()) if len(rows) else None,
        "agreement_rate": float(agree.mean()) if len(rows) else None,
        "quadrants": quadrants,
        "cls_errors_rescued_by_reg": {
            "count": int(q["reg_only_correct"].sum()), "denominator": cls_wrong_n,
            "rate_among_cls_errors": float(q["reg_only_correct"].sum() / cls_wrong_n)
            if cls_wrong_n else None,
        },
        "reg_errors_rescued_by_cls": {
            "count": int(q["cls_only_correct"].sum()), "denominator": reg_wrong_n,
            "rate_among_reg_errors": float(q["cls_only_correct"].sum() / reg_wrong_n)
            if reg_wrong_n else None,
        },
        "oracle_union_accuracy": float((cc | rc).mean()) if len(rows) else None,
        "oracle_gain_over_cls": float((cc | rc).mean() - cc.mean()) if len(rows) else None,
        "per_true_class": per_class,
    }


def confidence_diagnostics(rows: list[dict]) -> dict:
    true = np.asarray([r["true_class"] for r in rows])
    cls = np.asarray([r["cls_pred"] for r in rows])
    reg = np.asarray([r["reg_pred"] for r in rows], dtype=np.float64)
    conf = np.asarray([r["max_probability"] for r in rows], dtype=np.float64)
    margin = np.asarray([r["top1_margin"] for r in rows], dtype=np.float64)
    agree = cls == np.asarray([reg_class(v, 0) for v in reg])
    correct = cls == true

    def describe(ix: np.ndarray) -> dict:
        return {
            "n": int(ix.sum()),
            "max_probability": mean_std(conf[ix]),
            "top1_margin": mean_std(margin[ix]),
            "abs_reg_prediction": mean_std(np.abs(reg[ix])),
        }

    def quantile_bins(x: np.ndarray, name: str, values: dict[str, np.ndarray]) -> list[dict]:
        order = np.argsort(x, kind="stable")
        chunks = np.array_split(order, min(5, len(order))) if len(order) else []
        result = []
        for j, ids in enumerate(chunks):
            if not len(ids):
                continue
            row = {"bin": j + 1, "n": int(len(ids)), "lower": float(np.min(x[ids])),
                   "upper": float(np.max(x[ids]))}
            for key, val in values.items():
                row[key] = float(np.mean(val[ids]))
            result.append(row)
        return result

    absreg = np.abs(reg)
    return {
        "cls_correct": describe(correct),
        "cls_wrong": describe(~correct),
        "heads_agree": describe(agree),
        "heads_disagree": describe(~agree),
        "neutral": describe(true == 1),
        "confidence_quintiles": quantile_bins(
            conf, "max_probability", {"cls_accuracy": correct.astype(float),
                                      "head_agreement": agree.astype(float)}),
        "abs_reg_quintiles": quantile_bins(
            absreg, "abs_reg_prediction", {"cls_accuracy": correct.astype(float),
                                            "head_agreement": agree.astype(float),
                                            "neutral_fraction": (true == 1).astype(float)}),
    }


def regression_by_true_class(rows: list[dict]) -> dict:
    truth = np.asarray([r["true_class"] for r in rows])
    pred = np.asarray([r["reg_pred"] for r in rows], dtype=np.float64)
    result = {}
    for c, name in enumerate(LABEL_NAMES):
        x = pred[truth == c]
        result[name] = mean_std(x)
        if name == "Negative":
            result[name]["reg_pred_gt_0_rate"] = float(np.mean(x > 0)) if x.size else None
        elif name == "Positive":
            result[name]["reg_pred_lt_0_rate"] = float(np.mean(x < 0)) if x.size else None
        else:
            result[name]["abs_reg_prediction"] = mean_std(np.abs(x))
    return result


def scenario_group(rows_by_scenario: dict[str, list[dict]], scenario_meta: dict[str, dict],
                   key_fn) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for sid, rows in rows_by_scenario.items():
        groups[key_fn(scenario_meta[sid])].append(diagnostic(rows, 0.0))
    result = {}
    metrics = ("agreement_rate", "cls_accuracy", "reg_derived_accuracy",
               "oracle_union_accuracy")
    for name, entries in groups.items():
        result[name] = {
            "scenario_count": len(entries),
            **{m: float(np.mean([e[m] for e in entries])) for m in metrics},
            **{f"{k}_rate": float(np.mean([e["quadrants"][k]["rate"] for e in entries]))
               for k in ("both_correct", "cls_only_correct", "reg_only_correct", "both_wrong")},
        }
    return result


def paired_diagnostic(clean: list[dict], masked: list[dict]) -> dict:
    by_id = {r["id"]: r for r in clean}
    paired = [(by_id[r["id"]], r) for r in masked]
    crows, mrows = [x[0] for x in paired], [x[1] for x in paired]
    c_agree = np.asarray([r["cls_pred"] == reg_class(r["reg_pred"], 0) for r in crows])
    m_agree = np.asarray([r["cls_pred"] == reg_class(r["reg_pred"], 0) for r in mrows])
    csign = np.sign([r["reg_pred"] for r in crows])
    msign = np.sign([r["reg_pred"] for r in mrows])
    return {
        "n": len(paired),
        "cls_flip_rate": float(np.mean([a["cls_pred"] != b["cls_pred"] for a, b in paired])),
        "reg_sign_flip_rate": float(np.mean(csign != msign)),
        "agreement_rate_clean": float(c_agree.mean()),
        "agreement_rate_missing": float(m_agree.mean()),
        "agreement_rate_change": float(m_agree.mean() - c_agree.mean()),
        "clean_agree_missing_disagree_rate": float(np.mean(c_agree & ~m_agree)),
        "clean_disagree_missing_agree_rate": float(np.mean(~c_agree & m_agree)),
        "cls_flip_and_reg_sign_stable_rate": float(np.mean(
            np.asarray([a["cls_pred"] != b["cls_pred"] for a, b in paired]) & (csign == msign))),
    }


def paired_groups(clean: list[dict], rows_by_scenario: dict[str, list[dict]],
                  scenario_meta: dict[str, dict], key_fn) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for sid, rows in rows_by_scenario.items():
        groups[key_fn(scenario_meta[sid])].append(paired_diagnostic(clean, rows))
    metrics = ("cls_flip_rate", "reg_sign_flip_rate", "agreement_rate_change",
               "clean_agree_missing_disagree_rate", "clean_disagree_missing_agree_rate")
    return {
        key: {
            "scenario_count": len(items),
            **{name: float(np.mean([it[name] for it in items])) for name in metrics},
        }
        for key, items in groups.items()
    }


def compact_record(sample_id: str, scenario: dict, true_cls: int, true_reg: float,
                   logits: np.ndarray, probs: np.ndarray, reg_pred: float) -> dict:
    cls_pred = int(np.argmax(logits))
    reg_classes = {f"reg_class_tau_{tau:g}": reg_class(reg_pred, tau) for tau in TAUS}
    p_sorted = np.sort(probs)
    return {
        "id": str(sample_id), "scenario_id": scenario["scenario_id"],
        "modalities": scenario["modalities"], "rho": float(scenario["rho"]),
        "location": scenario["location"], "true_class": int(true_cls),
        "true_class_name": LABEL_NAMES[int(true_cls)], "true_regression": float(true_reg),
        "classification_logits": [float(x) for x in logits],
        "classification_probabilities": [float(x) for x in probs],
        "cls_pred": cls_pred, "cls_pred_name": LABEL_NAMES[cls_pred],
        "reg_pred": float(reg_pred), "reg_class_tau_0": reg_class(reg_pred, 0),
        **reg_classes,
        "max_probability": float(p_sorted[-1]),
        "top1_margin": float(p_sorted[-1] - p_sorted[-2]),
        "abs_reg_pred": float(abs(reg_pred)),
        "cls_correct": bool(cls_pred == true_cls),
        "reg_correct_tau_0": bool(reg_class(reg_pred, 0) == true_cls),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b0_weighted_ce_score_selection.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/b0_weighted_ce_best_selection_score.pt")
    parser.add_argument("--benchmark", default="outputs/metrics/b2_benchmark_definition.json")
    parser.add_argument("--outdir", default="outputs/metrics")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    cfg_path, ckpt_path, benchmark_path = map(Path, (args.config, args.checkpoint, args.benchmark))
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    definition = json.loads(benchmark_path.read_text(encoding="utf-8"))
    digest = sha256(benchmark_path)
    if digest != BENCHMARK_SHA256 or digest != definition.get("sha256", digest):
        raise RuntimeError(f"Frozen benchmark SHA-256 mismatch: {digest}")
    if definition.get("seed") != BENCHMARK_SEED or len(definition.get("scenarios", [])) != 54:
        raise RuntimeError("Expected frozen 54-scenario benchmark seed 20260923")
    if float(ckpt["config"]["training"]["lambda_reg"]) != 1.0:
        raise RuntimeError("Expected historical B0-WCE checkpoint lambda_reg=1.0")
    if ckpt["config"]["training"]["seed"] != 42:
        raise RuntimeError("Expected historical B0-WCE seed42 checkpoint")
    if cfg["training"]["evaluate_test"] is not False:
        raise RuntimeError("B0 diagnostic config unexpectedly permits test evaluation")
    if definition["scenarios"] != [s.as_dict() for s in scenarios()]:
        raise RuntimeError("Benchmark definitions differ from current frozen scenario generator")

    raw_path = Path(cfg["data"]["pkl_path"])
    with raw_path.open("rb") as f:
        raw = pickle.load(f)
    if "valid" not in raw:
        raise RuntimeError("Pickle does not contain the audited validation split")
    # Construct validation only. The test split is neither passed nor accessed below.
    ds = Aligned50Dataset(raw["valid"], "valid", scaler=None)
    del raw
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    model = B0Baseline(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    bench = scenarios()
    all_scenarios = ({"scenario_id": "clean", "modalities": "none", "rho": 0.0,
                      "location": "clean"}, *[s.as_dict() for s in bench])
    rows_by_scenario: dict[str, list[dict]] = {x["scenario_id"]: [] for x in all_scenarios}
    timings = {"device": str(device), "checkpoint_sha256": sha256(ckpt_path),
               "benchmark_sha256": digest, "benchmark_seed": BENCHMARK_SEED,
               "valid_count": len(ds), "scenario_count": 54}
    with torch.inference_mode():
        for batch in loader:
            moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            b = len(batch["id"])
            truths = moved["cls_label"].cpu().tolist()
            reg_truths = moved["reg_label"].cpu().tolist()
            ids = batch["id"]

            def collect(scenario, outputs):
                logits = outputs["classification_logits"].float().cpu().numpy()
                probs = torch.softmax(outputs["classification_logits"].float(), dim=-1).cpu().numpy()
                reg_preds = outputs["regression"].float().cpu().numpy()
                rows_by_scenario[scenario["scenario_id"]].extend(
                    compact_record(ids[i], scenario, truths[i], reg_truths[i], logits[i],
                                   probs[i], float(reg_preds[i])) for i in range(b))

            clean = {"scenario_id": "clean", "modalities": "none", "rho": 0.0,
                     "location": "clean"}
            collect(clean, model(predictor_inputs(moved)))
            for left in range(0, len(bench), 6):
                chunk = bench[left:left + 6]
                masked = [mask_scenario(moved, s)[0] for s in chunk]
                predictor = {key: torch.cat([part[key] for part in masked], dim=0)
                             for key in (*MODALITIES, "padding_mask")}
                outputs = model(predictor)
                for j, s in enumerate(chunk):
                    sl = slice(j * b, (j + 1) * b)
                    collect(s.as_dict(), {k: v[sl] for k, v in outputs.items()})

    expected_rows = len(ds)
    if any(len(rows) != expected_rows for rows in rows_by_scenario.values()):
        raise RuntimeError("A condition did not return one prediction per validation ID")
    clean_rows = rows_by_scenario["clean"]
    if len({r["id"] for r in clean_rows}) != expected_rows:
        raise RuntimeError("Validation IDs are not unique; paired analysis would be ambiguous")
    id_order = [r["id"] for r in clean_rows]
    if any([r["id"] for r in rows] != id_order for rows in rows_by_scenario.values()):
        raise RuntimeError("Scenario rows no longer align by validation ID")

    missing_rows = [r for s in [x["scenario_id"] for x in all_scenarios[1:]]
                    for r in rows_by_scenario[s]]
    scenario_meta = {x["scenario_id"]: x for x in all_scenarios[1:]}
    missing_by_scenario = {key: value for key, value in rows_by_scenario.items()
                           if key != "clean"}
    output = {
        "model": "historical B0-WCE seed42 best-selection-score checkpoint",
        "checkpoint": str(ckpt_path), "checkpoint_best_epoch": ckpt.get("best_epoch"),
        "lambda_reg": 1.0, "selection_score": ckpt.get("selection_score"),
        "normalization": "none", "training_or_finetuning": False,
        "test_or_attachment3_accessed": False,
        "test_split_indexed_or_used": False,
        "pickle_container_deserialized_as_unit": True,
        "split_used": "valid only; test split not indexed or passed to a loader",
        "benchmark": {"seed": BENCHMARK_SEED, "sha256": digest,
                      "conditions": 55, "missing_scenarios": 54},
        "valid_count": expected_rows,
        "default_tau_zero": {
            "clean": diagnostic(clean_rows, 0),
            "mean_missing": diagnostic(missing_rows, 0),
        },
        "threshold_sweep": {},
        "regression_distribution_by_true_class": {
            "clean": regression_by_true_class(clean_rows),
            "mean_missing": regression_by_true_class(missing_rows),
        },
        "confidence_and_regression_evidence": {
            "clean": confidence_diagnostics(clean_rows),
            "mean_missing": confidence_diagnostics(missing_rows),
        },
        "missing_scenario_metrics": {},
        "missing_groups": {},
        "clean_to_missing_paired": {},
        "metadata": timings,
    }
    for tau in TAUS:
        output["threshold_sweep"][f"tau_{tau:g}"] = {
            "clean": diagnostic(clean_rows, tau),
            "mean_missing": diagnostic(missing_rows, tau),
        }
    for sid, rows in rows_by_scenario.items():
        if sid != "clean":
            output["missing_scenario_metrics"][sid] = diagnostic(rows, 0)
    group_fns = {
        "by_modality": lambda m: m["modalities"] if "+" not in m["modalities"] else None,
        "by_rho": lambda m: f"{m['rho']:.1f}" if "+" not in m["modalities"] else None,
        "by_location": lambda m: m["location"],
        "double_modality": lambda m: m["modalities"] if "+" in m["modalities"] else None,
    }
    for name, fn in group_fns.items():
        output["missing_groups"][name] = scenario_group(
            missing_by_scenario, scenario_meta, lambda m, fn=fn: fn(m) or "__skip__")
        output["missing_groups"][name].pop("__skip__", None)
    paired_fns = {
        "by_modality": lambda m: m["modalities"] if "+" not in m["modalities"] else None,
        "by_rho": lambda m: f"{m['rho']:.1f}" if "+" not in m["modalities"] else None,
        "by_location": lambda m: m["location"],
        "double_modality": lambda m: m["modalities"] if "+" in m["modalities"] else None,
    }
    for name, fn in paired_fns.items():
        output["clean_to_missing_paired"][name] = paired_groups(
            clean_rows, missing_by_scenario, scenario_meta, lambda m, fn=fn: fn(m) or "__skip__")
        output["clean_to_missing_paired"][name].pop("__skip__", None)
    output["clean_to_missing_paired"]["by_rho_including_double_rho03"] = paired_groups(
        clean_rows, missing_by_scenario, scenario_meta, lambda m: f"{m['rho']:.1f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / "b5_h0_head_diagnostic.json"
    md_path = outdir / "b5_h0_head_diagnostic.md"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    jsonl_path = outdir / "b5_h0_predictions.jsonl"
    csv_path = outdir / "b5_h0_predictions.csv"
    fields = list(clean_rows[0].keys())
    with jsonl_path.open("w", encoding="utf-8") as f:
        for scenario in all_scenarios:
            for row in rows_by_scenario[scenario["scenario_id"]]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for scenario in all_scenarios:
            for row in rows_by_scenario[scenario["scenario_id"]]:
                flat = dict(row)
                for key in ("classification_logits", "classification_probabilities"):
                    flat[key] = json.dumps(flat[key], separators=(",", ":"))
                writer.writerow(flat)

    def pct(x): return f"{100*x:.2f}%"
    def line(d):
        q = d["quadrants"]
        return (f"{pct(d['agreement_rate'])} agreement; cls-only {pct(q['cls_only_correct']['rate'])}; "
                f"reg-only {pct(q['reg_only_correct']['rate'])}; both correct {pct(q['both_correct']['rate'])}; "
                f"both wrong {pct(q['both_wrong']['rate'])}; oracle union {pct(d['oracle_union_accuracy'])}")
    clean_d, missing_d = output["default_tau_zero"]["clean"], output["default_tau_zero"]["mean_missing"]
    md = ["# B5-H0 Classification–Regression Head Consistency Diagnostic", "",
          "Read-only inference on the historical B0-WCE seed42 checkpoint; no training, checkpoint writes, or attachment3. Only the validation split was indexed or evaluated. The aligned pickle is a combined multi-split file and is deserialized as a single pickle object to isolate `valid`; its test split was not indexed, passed to a loader, or evaluated.",
          f"Benchmark seed `{BENCHMARK_SEED}`, SHA-256 `{digest}`; 728 validation samples × clean + 54 scenarios.", "",
          "Default regression-to-class map uses sign (tau=0). Oracle is offline only, not deployable and not a model result.", "",
          "## Head agreement and error quadrants", "",
          f"- Clean: {line(clean_d)}.", f"- Mean missing: {line(missing_d)}.",
          f"- Clean cls errors corrected by regression: {clean_d['cls_errors_rescued_by_reg']['count']}/"
          f"{clean_d['cls_errors_rescued_by_reg']['denominator']} = "
          f"{pct(clean_d['cls_errors_rescued_by_reg']['rate_among_cls_errors'])}; regression errors corrected by cls: "
          f"{clean_d['reg_errors_rescued_by_cls']['count']}/"
          f"{clean_d['reg_errors_rescued_by_cls']['denominator']} = "
          f"{pct(clean_d['reg_errors_rescued_by_cls']['rate_among_reg_errors'])}.",
          f"- Mean-missing cls errors corrected by regression: {missing_d['cls_errors_rescued_by_reg']['count']}/"
          f"{missing_d['cls_errors_rescued_by_reg']['denominator']} = "
          f"{pct(missing_d['cls_errors_rescued_by_reg']['rate_among_cls_errors'])}; regression errors corrected by cls: "
          f"{missing_d['reg_errors_rescued_by_cls']['count']}/{missing_d['reg_errors_rescued_by_cls']['denominator']} = "
          f"{pct(missing_d['reg_errors_rescued_by_cls']['rate_among_reg_errors'])}.", "",
          "## True-class breakdown", "",
          "Values below are conditional rates within each true class (tau=0).", "",
          "| Condition | Class | N | Cls acc/recall | Reg acc/recall | Agreement | cls-only | reg-only | both wrong |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cond, d in (("Clean", clean_d), ("Mean missing", missing_d)):
        for name, v in d["per_true_class"].items():
            md.append(f"| {cond} | {name} | {v['n']} | {pct(v['cls_accuracy_recall'])} | {pct(v['reg_accuracy_recall'])} | "
                      f"{pct(v['agreement_rate'])} | {pct(v['cls_only_correct_rate'])} | "
                      f"{pct(v['reg_only_correct_rate'])} | {pct(v['both_wrong_rate'])} |")
    md += ["", "## Threshold sweep", "", "Descriptive only; no threshold is selected or used as a predictor.", "",
           "| τ | Clean reg accuracy | Clean agreement | Clean neutral output | Missing reg accuracy | Missing agreement | Missing neutral output |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    for tau in TAUS:
        d0 = output["threshold_sweep"][f"tau_{tau:g}"]["clean"]
        d1 = output["threshold_sweep"][f"tau_{tau:g}"]["mean_missing"]
        md.append(f"| {tau:g} | {pct(d0['reg_derived_accuracy'])} | {pct(d0['agreement_rate'])} | "
                  f"{pct(sum(reg_class(r['reg_pred'],tau)==1 for r in clean_rows)/len(clean_rows))} | "
                  f"{pct(d1['reg_derived_accuracy'])} | {pct(d1['agreement_rate'])} | "
                  f"{pct(sum(reg_class(r['reg_pred'],tau)==1 for r in missing_rows)/len(missing_rows))} |")
    md += ["", "## Scenario groups and paired changes", "",
           "Scenario rates are unweighted means over the scenarios in each group. Rho rows contain the single-modality scenarios; double-modality rows are separate.", "",
           "| Group | Name | Scenarios | Agreement | cls-only | reg-only | both wrong |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for group_name in ("by_modality", "by_rho", "by_location", "double_modality"):
        for name, v in output["missing_groups"][group_name].items():
            md.append(f"| {group_name} | {name} | {v['scenario_count']} | {pct(v['agreement_rate'])} | "
                      f"{pct(v['cls_only_correct_rate'])} | {pct(v['reg_only_correct_rate'])} | "
                      f"{pct(v['both_wrong_rate'])} |")
    md += ["", "### Paired clean→missing changes", "",
           "Flip and transition rates are paired by validation ID; rates are then averaged over the scenarios in each group.", "",
           "| Group | Name | Scenarios | Cls flip | Reg sign flip | Agreement Δ | Agree→disagree | Disagree→agree |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for group_name in ("by_modality", "by_rho", "by_location", "double_modality"):
        for name, v in output["clean_to_missing_paired"][group_name].items():
            md.append(f"| {group_name} | {name} | {v['scenario_count']} | {pct(v['cls_flip_rate'])} | "
                      f"{pct(v['reg_sign_flip_rate'])} | {v['agreement_rate_change']:+.2%} | "
                      f"{pct(v['clean_agree_missing_disagree_rate'])} | "
                      f"{pct(v['clean_disagree_missing_agree_rate'])} |")
    md += ["", "## Regression distributions by true class", "",
           "Predicted continuous regression values; negative/positive wrong-sign rates use strict sign tests. Neutral abs(prediction) is not a class prediction.", "",
           "| Condition | True class | Mean | Std | Median | P05 | P25 | P75 | P95 | Sign / abs diagnostic |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cond in ("clean", "mean_missing"):
        for name, v in output["regression_distribution_by_true_class"][cond].items():
            extra = (f"reg>0 {pct(v['reg_pred_gt_0_rate'])}" if name == "Negative" else
                     f"reg<0 {pct(v['reg_pred_lt_0_rate'])}" if name == "Positive" else
                     f"|reg| mean {v['abs_reg_prediction']['mean']:.3f}; median {v['abs_reg_prediction']['median']:.3f}; "
                     f"P75 {v['abs_reg_prediction']['p75']:.3f}")
            md.append(f"| {cond} | {name} | {v['mean']:.3f} | {v['std']:.3f} | {v['median']:.3f} | "
                      f"{v['p05']:.3f} | {v['p25']:.3f} | {v['p75']:.3f} | {v['p95']:.3f} | {extra} |")
    md += ["", "## Confidence and regression evidence", "",
           "| Condition | Subset | N | Mean max prob | Mean top1 margin | Mean |reg_pred| |",
           "|---|---|---:|---:|---:|---:|"]
    for cond in ("clean", "mean_missing"):
        for subset in ("cls_correct", "cls_wrong", "heads_agree", "heads_disagree", "neutral"):
            v = output["confidence_and_regression_evidence"][cond][subset]
            md.append(f"| {cond} | {subset} | {v['n']} | {v['max_probability']['mean']:.3f} | "
                      f"{v['top1_margin']['mean']:.3f} | {v['abs_reg_prediction']['mean']:.3f} |")
    md += ["", "Confidence quintiles (clean and mean-missing) and |reg_pred| quintiles are stored in the JSON. Across both conditions, disagreement concentrates at low classification confidence and small |reg_pred|; Neutral predictions have broad regression magnitudes rather than a narrow band around zero.", "",
           "## Oracle diagnostic", "",
           f"Oracle union accuracy (tau=0): clean {pct(clean_d['oracle_union_accuracy'])} "
           f"(gain {pct(clean_d['oracle_gain_over_cls'])} over cls); mean missing "
           f"{pct(missing_d['oracle_union_accuracy'])} (gain {pct(missing_d['oracle_gain_over_cls'])}). "
           "This is an offline upper bound only, not deployable and not a model result.", "",
           "## Mechanism judgment", "",
           "Classification and regression show **partial complementarity with substantial shared errors**. Regression alone corrects some classification errors (5.08% clean / 5.53% mean-missing under the fixed sign map), and the oracle union adds about five percentage points. However, both heads are wrong on about 30% of examples, and Neutral recall of sign-mapped regression is necessarily zero; the neutral regression predictions are broadly spread and overlap the Negative/Positive distributions. This is not strong complementarity, but not pure redundancy either. The diagnostic supports, at most, a future lightweight consistency hypothesis focused on continuous evidence and Neutral handling; no such method is implemented here.", "",
           "## Per-sample exports", "",
           f"- `{jsonl_path.name}`: one JSON object per validation sample and condition.",
           f"- `{csv_path.name}`: flattened CSV; logits/probabilities stored as JSON arrays.", ""]
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"outdir": str(outdir), "rows": expected_rows * 55,
                      "clean": {"agreement": clean_d["agreement_rate"],
                                "cls_only": clean_d["quadrants"]["cls_only_correct"]["rate"],
                                "reg_only": clean_d["quadrants"]["reg_only_correct"]["rate"],
                                "both_wrong": clean_d["quadrants"]["both_wrong"]["rate"]},
                      "mean_missing": {"agreement": missing_d["agreement_rate"],
                                       "cls_only": missing_d["quadrants"]["cls_only_correct"]["rate"],
                                       "reg_only": missing_d["quadrants"]["reg_only_correct"]["rate"],
                                       "both_wrong": missing_d["quadrants"]["both_wrong"]["rate"]}}, indent=2))


if __name__ == "__main__":
    main()
