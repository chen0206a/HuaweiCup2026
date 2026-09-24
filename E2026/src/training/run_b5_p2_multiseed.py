"""Paired seed-43/44 replication of B5-P2 attention residual pooling."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.dataset import MODALITIES
from src.data.preprocess import build_datasets_and_loaders
from src.evaluation.missing_benchmark import (
    BENCHMARK_SEED,
    evaluate_benchmark,
    scenarios,
    summary,
)
from src.models.baseline import B0Baseline, multitask_loss
from src.models.pooling_residual import B5PoolingResidual
from src.training.run_b5_pooling import (
    attention_diagnostics,
    b0_equality,
    check_identity,
    fresh_train_loader,
    vision_subset,
    write_json,
)
from src.training.train_b1 import seed_everything

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "outputs" / "metrics"
CHECKPOINTS = ROOT / "outputs" / "checkpoints"
BENCHMARK_SHA256 = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"
SCORE_FIELDS = ("accuracy", "macro_f1", "mae", "pearson", "selection_score")
B0_PREFIXES = ("modality_projection.", "fusion.", "classification_head.", "regression_head.")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(cfg: dict, config_path: Path) -> dict:
    if cfg["training"]["seeds"] != [43, 44]:
        raise ValueError("This replication is fixed to training seeds 43 and 44")
    train_cfg = cfg["training"]
    if (cfg["preprocessing"]["normalization"] != "none" or
            train_cfg["epochs"] != 80 or train_cfg["patience"] != 12 or
            train_cfg["learning_rate"] != 0.001 or train_cfg["weight_decay"] != 0.0001 or
            train_cfg["lambda_reg"] != 1.0 or train_cfg["class_weighting"] != "balanced_train" or
            train_cfg["augmentation"] != "none" or not train_cfg["freeze_b0_parameters"]):
        raise ValueError("B5-P2 protocol differs from frozen exp_008 settings")
    definition_path = Path(cfg["benchmark"]["definition"])
    digest = sha256(definition_path)
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if (digest != BENCHMARK_SHA256 or digest != cfg["benchmark"]["sha256"] or
            definition.get("seed") != BENCHMARK_SEED or
            definition.get("scenarios") != [scene.as_dict() for scene in scenarios()]):
        raise RuntimeError("Frozen validation benchmark does not match exp_008")
    original = yaml.safe_load((config_path.parent / "b5_pooling.yaml").read_text(encoding="utf-8"))
    for key in ("data", "preprocessing", "model"):
        expected = original[key]
        if key == "data":
            expected = {k: v for k, v in expected.items() if k != "pkl_path"}
            actual = {k: v for k, v in cfg[key].items() if k != "pkl_path"}
        else:
            actual = cfg[key]
        if actual != expected:
            raise ValueError(f"B5-P2 {key} configuration changed from exp_008")
    for key in ("epochs", "patience", "learning_rate", "weight_decay", "lambda_reg",
                "class_weighting", "augmentation", "freeze_b0_parameters"):
        if train_cfg[key] != original["training"][key]:
            raise ValueError(f"B5-P2 training.{key} changed from exp_008")
    return definition


def assert_b0_frozen(model: B5PoolingResidual, source_state: dict) -> dict:
    pred_state = model.state_dict()
    names = [key for key in source_state if key.startswith(B0_PREFIXES)]
    changed = [key for key in names if not torch.equal(
        pred_state[key].detach().cpu(), source_state[key].detach().cpu())]
    nonfrozen = [name for name, p in model.named_parameters()
                 if name.startswith(B0_PREFIXES) and p.requires_grad]
    return {"passed": not changed and not nonfrozen, "tensor_count": len(names),
            "changed_tensors": changed, "unexpected_trainable_b0_parameters": nonfrozen}


@torch.no_grad()
def eval_saved_checkpoint(path: Path, mode: str, cfg: dict, source_state: dict,
                          valid_loader, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = B5PoolingResidual(mode, **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    for name, parameter in model.named_parameters():
        if name.startswith(B0_PREFIXES):
            parameter.requires_grad_(False)
    model.eval()
    rows = evaluate_benchmark(model, valid_loader, device)
    frozen = assert_b0_frozen(model, source_state)
    if not frozen["passed"]:
        raise RuntimeError(f"Checkpoint contains a changed frozen B0 tensor: {path}")
    return {
        "validation": summary(rows),
        "vision_all_zero": vision_subset(rows),
        "scenario_details": rows,
        "gamma": {m: float(model.gamma[m].detach().cpu()) for m in MODALITIES},
        "frozen_b0_equality": frozen,
        "checkpoint_epoch": int(ckpt["epoch"]),
    }


def run_seed(seed: int, cfg: dict, datasets: dict, loaders: dict,
             baseline_metrics: dict, device: torch.device) -> dict:
    checkpoint_path = Path(cfg["training"]["baseline_checkpoints"][seed])
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_cfg = source.get("config", {})
    if (int(source_cfg.get("training", {}).get("seed", -1)) != seed or
            source_cfg.get("model") != cfg["model"] or
            float(source_cfg.get("training", {}).get("lambda_reg", -1)) != 1.0 or
            source_cfg.get("training", {}).get("class_weighting") != "balanced_train"):
        raise RuntimeError(f"Baseline checkpoint does not match seed{seed} B0-WCE protocol")
    if source.get("normalization", "none") != "none":
        raise RuntimeError(f"seed{seed} B0 checkpoint uses unexpected normalization")

    seed_everything(seed)
    b0 = B0Baseline(**cfg["model"]).to(device).eval()
    b0.load_state_dict(source["model_state_dict"], strict=True)
    b0_state = {k: v.detach().clone() for k, v in source["model_state_dict"].items()}
    b0_rows = evaluate_benchmark(b0, loaders["valid"], device)
    b0_validation = summary(b0_rows)
    for group in ("clean", "mean_missing"):
        for metric in SCORE_FIELDS:
            if abs(b0_validation[group][metric] - baseline_metrics["validation"][group][metric]) > 1e-10:
                raise RuntimeError(f"seed{seed} B0 checkpoint no longer reproduces saved validation")
    if abs(b0_validation["robust_score"] - baseline_metrics["validation"]["robust_score"]) > 1e-10:
        raise RuntimeError(f"seed{seed} B0 checkpoint no longer reproduces robust validation")

    model = B5PoolingResidual("mean_attention", **cfg["model"]).to(device)
    incompatible = model.load_state_dict(source["model_state_dict"], strict=False)
    if (incompatible.unexpected_keys or any(not k.startswith(("gamma.", "attention_scorer."))
                                             for k in incompatible.missing_keys)):
        raise RuntimeError(f"seed{seed} B0 initialization mismatch: {incompatible}")
    for name, parameter in model.named_parameters():
        if name.startswith(B0_PREFIXES):
            parameter.requires_grad_(False)
    identity = check_identity(b0, model, loaders["valid"], device)
    if not identity["passed"]:
        raise RuntimeError(f"seed{seed} P2 gamma=0 initialization is not identical to B0")

    counts = np.bincount(datasets["train"].cls_labels, minlength=3)
    if np.any(counts == 0):
        raise RuntimeError("train split lacks a class")
    class_weights = torch.as_tensor(len(datasets["train"]) / (3 * counts),
                                    dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=float(cfg["training"]["learning_rate"]),
                                  weight_decay=float(cfg["training"]["weight_decay"]))
    train_loader = fresh_train_loader(datasets["train"], int(cfg["data"]["batch_size"]), seed)
    prefix = f"{cfg['outputs']['prefix']}_seed{seed}"
    clean_path = CHECKPOINTS / f"{prefix}_best_clean_score.pt"
    robust_path = CHECKPOINTS / f"{prefix}_best_robust_score.pt"
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    best_clean = best_robust = -float("inf")
    best_clean_epoch = best_robust_epoch = stale = 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        model.modality_projection.eval()
        model.fusion.eval()
        model.classification_head.eval()
        model.regression_head.eval()
        if any(module.training for module in (model.modality_projection, model.fusion,
                                               model.classification_head, model.regression_head)):
            raise RuntimeError("Frozen B0 submodule accidentally entered train mode")
        if any(p.requires_grad for name, p in model.named_parameters()
               if name.startswith(B0_PREFIXES)):
            raise RuntimeError("A frozen B0 parameter unexpectedly requires grad")
        loss_sum, seen = 0.0, 0
        for batch in train_loader:
            moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model({k: moved[k] for k in (*MODALITIES, "padding_mask")})
            loss = multitask_loss(outputs, moved, lambda_reg=1.0,
                                  class_weights=class_weights)["total"]
            loss.backward()
            optimizer.step()
            n = len(moved["cls_label"])
            loss_sum += float(loss.item()) * n
            seen += n
        validation_start = time.perf_counter()
        rows = evaluate_benchmark(model, loaders["valid"], device)
        validation_seconds = time.perf_counter() - validation_start
        metrics = summary(rows)
        clean_score, robust_score = metrics["clean"]["selection_score"], metrics["robust_score"]
        entry = {"epoch": epoch, "train_loss": loss_sum / seen,
                 "validation_seconds": validation_seconds, **metrics}
        history.append(entry)
        checkpoint = {
            "model_state_dict": model.state_dict(), "config": cfg,
            "mode": "mean_attention", "seed": seed, "epoch": epoch,
            "clean_score": clean_score, "robust_score": robust_score,
            "benchmark_sha256": BENCHMARK_SHA256,
            "baseline_checkpoint": str(checkpoint_path),
            "train_class_counts": counts.tolist(),
            "class_weights": class_weights.detach().cpu(),
        }
        if clean_score > best_clean:
            best_clean, best_clean_epoch = clean_score, epoch
            torch.save(checkpoint, clean_path)
        if robust_score > best_robust:
            best_robust, best_robust_epoch, stale = robust_score, epoch, 0
            torch.save(checkpoint, robust_path)
        else:
            stale += 1
        print(json.dumps({"seed": seed, "epoch": epoch, "train_loss": loss_sum / seen,
                          "clean_score": clean_score, "robust_score": robust_score,
                          "validation_seconds": validation_seconds}), flush=True)
        if stale >= int(cfg["training"]["patience"]):
            break
    training_seconds = time.perf_counter() - start

    clean_eval = eval_saved_checkpoint(clean_path, "mean_attention", cfg, b0_state,
                                       loaders["valid"], device)
    robust_eval = eval_saved_checkpoint(robust_path, "mean_attention", cfg, b0_state,
                                        loaders["valid"], device)
    if clean_eval["checkpoint_epoch"] != best_clean_epoch or robust_eval["checkpoint_epoch"] != best_robust_epoch:
        raise RuntimeError("Saved P2 selection checkpoint epoch mismatch")
    restored = B5PoolingResidual("mean_attention", **cfg["model"]).to(device).eval()
    saved = torch.load(robust_path, map_location="cpu", weights_only=False)
    restored.load_state_dict(saved["model_state_dict"], strict=True)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    first_batch = next(iter(loaders["valid"]))
    example = {k: first_batch[k].to(device) for k in (*MODALITIES, "padding_mask")}
    with torch.inference_mode():
        normal = model.eval()(example)
        roundtrip = restored(example)
    checkpoint_roundtrip = all(torch.equal(normal[k], roundtrip[k]) for k in normal)
    if not checkpoint_roundtrip:
        raise RuntimeError("P2 checkpoint round-trip prediction mismatch")
    attention = attention_diagnostics(restored, loaders["valid"], device)
    frozen = robust_eval["frozen_b0_equality"]
    if not frozen["passed"]:
        raise RuntimeError(f"seed{seed} changed frozen B0 tensors")
    return {
        "model": "B5-P2", "seed": seed,
        "baseline_checkpoint": str(checkpoint_path),
        "baseline_checkpoint_sha256": sha256(checkpoint_path),
        "baseline_best_epoch": baseline_metrics["best_epoch"],
        "baseline_validation": baseline_metrics["validation"],
        "baseline_vision_all_zero": vision_subset(b0_rows),
        "p2_parameter_count": sum(p.numel() for p in model.parameters()),
        "p2_trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "initial_identity": identity,
        "freeze_protocol": {"requires_grad_false": True,
                            "frozen_modules_eval_after_model_train": True,
                            "frozen_b0_equality_after_training": frozen,
                            "tensor_count": frozen["tensor_count"]},
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "best_clean_epoch": best_clean_epoch,
        "best_robust_epoch": best_robust_epoch,
        "epochs_run": len(history), "training_seconds": training_seconds,
        "mean_validation_seconds": float(np.mean([h["validation_seconds"] for h in history])),
        "best_clean_score": best_clean,
        "best_robust_score": best_robust,
        "best_clean_validation": clean_eval["validation"],
        "best_robust_validation": robust_eval["validation"],
        "best_clean_vision_all_zero": clean_eval["vision_all_zero"],
        "best_robust_vision_all_zero": robust_eval["vision_all_zero"],
        "best_clean_gamma": clean_eval["gamma"], "best_robust_gamma": robust_eval["gamma"],
        "best_robust_attention_diagnostics": attention,
        "best_robust_scenario_details": robust_eval["scenario_details"],
        "checkpoints": {"best_clean": str(clean_path), "best_robust": str(robust_path)},
        "train_class_counts": counts.tolist(),
        "class_weights": class_weights.detach().cpu().tolist(),
        "history": history,
    }


def load_existing_baselines(cfg: dict) -> tuple[dict[int, dict], dict[int, dict]]:
    metrics_by_seed: dict[int, dict] = {}
    if not (METRICS / "b5_pooling_metrics.json").exists():
        raise FileNotFoundError("Seed42 exp_008 B5-P metrics are required")
    pool = json.loads((METRICS / "b5_pooling_metrics.json").read_text(encoding="utf-8"))
    if pool.get("benchmark_sha256") != BENCHMARK_SHA256 or pool.get("training_seed") != 42:
        raise RuntimeError("Seed42 B5-P result does not match the frozen benchmark")
    baseline42 = {"seed": 42, "best_epoch": 3,
                  "checkpoint": "/root/workspace/E2026/outputs/checkpoints/b0_weighted_ce_best_selection_score.pt",
                  "validation": pool["P0"]["validation"],
                  "vision_all_zero": pool["P0"]["vision_all_zero"]}
    p2_42 = pool["P2"]
    p2_hist = json.loads((METRICS / "b5_pooling_history.json").read_text(encoding="utf-8"))["P2"]
    best_clean_row = max(p2_hist, key=lambda row: row["clean"]["selection_score"])
    p2_42_clean = {
        "epoch": int(best_clean_row["epoch"]),
        "validation": {key: best_clean_row[key] for key in
                       ("clean", "mean_missing", "robust_score", "by_ratio", "by_location",
                        "by_modality", "double_stress")},
    }
    metrics_by_seed[42] = baseline42
    seed_baselines = {42: baseline42}
    for seed in (43, 44):
        path = METRICS / f"b21_b0_seed_{seed}_validation.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("seed") != seed or data.get("benchmark_sha256") != BENCHMARK_SHA256:
            raise RuntimeError(f"Saved B0 seed{seed} metrics mismatch")
        seed_baselines[seed] = data
    return seed_baselines, {"seed42_b5_p2": p2_42, "seed42_best_clean_history": p2_42_clean}


def summarize_three_seeds(cfg: dict, results: dict[int, dict], baselines: dict[int, dict],
                          seed42_data: dict) -> dict:
    p2_by_seed = {
        42: seed42_data["seed42_b5_p2"]["validation"],
        43: results[43]["best_robust_validation"],
        44: results[44]["best_robust_validation"],
    }
    b0_by_seed = {seed: data["validation"] for seed, data in baselines.items()}
    metrics = ("accuracy", "macro_f1", "mae", "pearson", "selection_score")
    output = {
        "training_seeds": [42, 43, 44],
        "benchmark_seed": BENCHMARK_SEED,
        "benchmark_sha256": BENCHMARK_SHA256,
        "std_definition": "sample standard deviation, ddof=1",
        "comparison_checkpoint_rule": "B0-WCE corresponding-seed best-selection-score; P2 corresponding-seed best-robust-score",
        "models": {}, "paired_deltas": {},
        "per_seed": {},
        "seed43_44_details": {str(seed): results[seed] for seed in (43, 44)},
        "seed42_best_clean_checkpoint": seed42_data["seed42_best_clean_history"],
    }
    for model_name, values in (("B0-WCE", b0_by_seed), ("B5-P2", p2_by_seed)):
        model_summary = {}
        for cond in ("clean", "mean_missing"):
            model_summary[cond] = {}
            for metric in metrics:
                vals = [values[s][cond][metric] for s in (42, 43, 44)]
                model_summary[cond][metric] = {"mean": float(np.mean(vals)),
                                                "std": float(np.std(vals, ddof=1)),
                                                "values": vals}
        robust = [values[s]["robust_score"] for s in (42, 43, 44)]
        model_summary["robust_score"] = {"mean": float(np.mean(robust)),
                                          "std": float(np.std(robust, ddof=1)),
                                          "values": robust}
        output["models"][model_name] = model_summary
    deltas = {}
    for seed in (42, 43, 44):
        d = p2_by_seed[seed]["robust_score"] - b0_by_seed[seed]["robust_score"]
        deltas[str(seed)] = float(d)
        output["per_seed"][str(seed)] = {
            "B0": b0_by_seed[seed], "P2": p2_by_seed[seed],
            "delta_robust_P2_minus_B0": float(d),
            "B0_best_epoch": int(baselines[seed]["best_epoch"]),
            "P2_best_robust_epoch": int(seed42_data["seed42_b5_p2"]["best_robust_epoch"])
            if seed == 42 else results[seed]["best_robust_epoch"],
            "P2_best_clean_epoch": int(seed42_data["seed42_b5_p2"]["best_clean_epoch"])
            if seed == 42 else results[seed]["best_clean_epoch"],
        }
    dv = list(deltas.values())
    output["paired_deltas"] = {
        "robust_score": {"per_seed": deltas, "mean": float(np.mean(dv)),
                         "sample_std": float(np.std(dv, ddof=1)),
                         "positive_seeds": int(sum(v > 0 for v in dv)),
                         "direction_consistent": bool(all(v > 0 for v in dv))},
    }
    return output


def render_report(summary_data: dict, seed42_data: dict, results: dict[int, dict]) -> str:
    lines = ["# B5-P2 attention residual pooling: three-seed replication", "",
             "Paired B0-WCE/P2 validation comparison; P2 checkpoints selected by robust score. "
             "Frozen benchmark clean + 54 scenarios, seed 20260923. No test/attachment3.", "",
             "## Paired seed results", "",
             "| Training seed | B0 robust | P2 robust | Paired Δ robust | B0 epoch | P2 best-clean epoch | P2 best-robust epoch |", "",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in (42, 43, 44):
        row = summary_data["per_seed"][str(seed)]
        lines.append(f"| {seed} | {row['B0']['robust_score']:.6f} | {row['P2']['robust_score']:.6f} | "
                     f"{row['delta_robust_P2_minus_B0']:+.6f} | {row['B0_best_epoch']} | "
                     f"{row['P2_best_clean_epoch']} | {row['P2_best_robust_epoch']} |")
    lines += ["", f"Mean paired Δrobust = {summary_data['paired_deltas']['robust_score']['mean']:+.6f} "
              f"± {summary_data['paired_deltas']['robust_score']['sample_std']:.6f} (sample SD); "
              f"positive in {summary_data['paired_deltas']['robust_score']['positive_seeds']}/3 seeds.", "",
              "## Three-seed mean ± sample SD", "",
              "| Model | Split | Accuracy | Macro-F1 | MAE | Pearson | Selection score | Robust score |", "",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model in ("B0-WCE", "B5-P2"):
        for cond in ("clean", "mean_missing"):
            m = summary_data["models"][model][cond]
            vals = [f"{m[k]['mean']:.4f} ± {m[k]['std']:.4f}" for k in SCORE_FIELDS]
            robust = summary_data["models"][model]["robust_score"]
            lines.append(f"| {model} | {cond} | " + " | ".join(vals) +
                         f" | {robust['mean']:.4f} ± {robust['std']:.4f} |")
    lines += ["", "## Per-seed four metrics", "",
              "Values are selected B0 checkpoint and paired P2 best-robust checkpoint. Full values and groups are in JSON.", "",
              "| Seed | Model | Condition | Acc | Macro-F1 | MAE | Pearson | Score |", "",
              "|---:|---|---|---:|---:|---:|---:|---:|"]
    for seed in (42, 43, 44):
        for model in ("B0", "P2"):
            val = summary_data["per_seed"][str(seed)][model]
            label = "B0-WCE" if model == "B0" else "P2"
            for cond in ("clean", "mean_missing"):
                m = val[cond]
                lines.append(f"| {seed} | {label} | {cond} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                             f"{m['mae']:.4f} | {m['pearson']:.4f} | {m['selection_score']:.4f} |")
    lines += ["", "## P2 training diagnostics", "",
              "| Seed | γ text | γ audio | γ vision | Clean epoch | Robust epoch | Seconds | Frozen B0 exact |", "",
              "|---:|---:|---:|---:|---:|---:|---:|---|"]
    p2_data = {42: seed42_data["seed42_b5_p2"], 43: results[43], 44: results[44]}
    for seed, item in p2_data.items():
        g = item["gamma"] if seed == 42 else item["best_robust_gamma"]
        same = True if seed == 42 else item["freeze_protocol"]["frozen_b0_equality_after_training"]["passed"]
        sec = item.get("training_seconds", None)
        seconds_text = f"{sec:.1f}" if sec is not None else "reused"
        lines.append(f"| {seed} | {g['text']:+.4f} | {g['audio']:+.4f} | {g['vision']:+.4f} | "
                     f"{item['best_clean_epoch']} | {item['best_robust_epoch']} | "
                     f"{seconds_text} | {'PASS' if same else 'FAIL'} |")
    lines += ["", "### Attention entropy and concentration", "",
              "Normalized entropy, mean max temporal weight, and fraction with max weight ≥0.9 are reported for P2 best-robust; seed42 values are reused from exp_008.", "",
              "| Seed | Condition | Modality | Normalized entropy | Mean max weight | Fraction max≥.9 |", "",
              "|---:|---|---|---:|---:|---:|"]
    for seed, item in p2_data.items():
        diag = item["attention_diagnostics"] if seed == 42 else item["best_robust_attention_diagnostics"]
        for condition, mods in diag.items():
            for modality, d in mods.items():
                lines.append(f"| {seed} | {condition} | {modality} | "
                             f"{d['mean_normalized_entropy_length_gt_one']:.4f} | "
                             f"{d['mean_max_weight']:.4f} | "
                             f"{d['fraction_max_weight_ge_0_9_length_gt_one']:.4f} |")
    lines += ["", "## Subgroup selection score (mean ± sample SD across seeds)", "",
              "| Group | B0-WCE | P2 | Δ(P2−B0) |", "",
              "|---|---:|---:|---:|"]
    subgroup_specs = (("by_modality", ("text", "audio", "vision")),
                      ("by_ratio", ("0.1", "0.2", "0.3", "0.4", "0.5")),
                      ("by_location", ("early", "middle", "late")),
                      ("double_stress", ("text+audio", "text+vision", "audio+vision")))
    for section, groups in subgroup_specs:
        for group in groups:
            b0_scores = [summary_data["per_seed"][str(s)]["B0"][section][group]["selection_score"]
                         for s in (42, 43, 44)]
            p2_scores = [summary_data["per_seed"][str(s)]["P2"][section][group]["selection_score"]
                         for s in (42, 43, 44)]
            bm, bs = float(np.mean(b0_scores)), float(np.std(b0_scores, ddof=1))
            pm, ps = float(np.mean(p2_scores)), float(np.std(p2_scores, ddof=1))
            lines.append(f"| {section.replace('_', ' ')}: {group} | {bm:.4f} ± {bs:.4f} | "
                         f"{pm:.4f} ± {ps:.4f} | {pm-bm:+.4f} |")
    lines += ["", "## Vision-all-zero subset", "",
              "The 15-sample subset is diagnostic only; it does not change training or selection.", "",
              "| Seed | Model | Split | Acc | Macro-F1 | MAE | Pearson |", "",
              "|---:|---|---|---:|---:|---:|---:|"]
    for seed in (42, 43, 44):
        if seed == 42:
            seed42_pool = json.loads((METRICS / "b5_pooling_metrics.json").read_text(encoding="utf-8"))
            subsets = {"B0": seed42_pool["P0"]["vision_all_zero"],
                       "P2": seed42_data["seed42_b5_p2"]["vision_all_zero"]}
        else:
            item = results[seed]
            subsets = {"B0": item["baseline_vision_all_zero"], "P2": item["best_robust_vision_all_zero"]}
        for model, values in subsets.items():
            for split in ("clean", "mean_missing"):
                m = values[split]
                lines.append(f"| {seed} | {model} | {split} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                             f"{m['mae']:.4f} | {m['pearson']:.4f} |")
    lines += ["", "## Subgroup details", "",
              "The per-seed B0/P2 subgroup scores and four metrics are preserved in JSON under `per_seed` and `seed43_44_details`. Each P2 seed also includes attention diagnostics.", "",
              "## Stability interpretation", "",
              "Seed-42 screening showed +0.005600. The seed43/44 paired deltas determine whether that gain replicates. The category (all positive / 2-of-3 / only seed42 positive) is filled from the observed paired deltas below.", ""]
    deltas = summary_data["paired_deltas"]["robust_score"]["per_seed"]
    positives = sum(v > 0 for v in deltas.values())
    if positives == 3:
        judgment = "A: all three seeds are positive; P2 is a stable seed-level Strong-B0 candidate under this validation benchmark."
    elif positives == 2:
        judgment = "B: two seeds are positive and one is negative; report seed sensitivity and compare the paired mean and negative-seed magnitude without overstating stability."
    else:
        judgment = "C: only one or fewer seeds are positive; seed42's gain does not replicate consistently, so stop the P2 main line."
    lines[-1] = judgment
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b5_p2_multiseed.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(cfg, config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # include_test=False: only train and valid datasets/loaders are constructed.
    datasets, loaders, _ = build_datasets_and_loaders(
        cfg["data"]["pkl_path"], normalization="none",
        batch_size=int(cfg["data"]["batch_size"]), num_workers=0,
        seed=43, include_test=False)
    baseline_files = {}
    for seed in (43, 44):
        baseline_files[seed] = json.loads(
            (METRICS / f"b21_b0_seed_{seed}_validation.json").read_text(encoding="utf-8"))
    results, histories = {}, {}
    all_baselines, seed42_data = load_existing_baselines(cfg)
    for seed in (43, 44):
        seed_everything(seed)
        result = run_seed(seed, cfg, datasets, loaders, baseline_files[seed], device)
        results[seed] = result
        histories[str(seed)] = result["history"]
        all_baselines[seed]["vision_all_zero"] = result["baseline_vision_all_zero"]
        payload = {"seed": seed, **result}
        write_json(METRICS / f"b5_p2_seed{seed}_metrics.json", payload)
        write_json(METRICS / cfg["outputs"]["history"], histories)
    combined = {"benchmark_sha256": BENCHMARK_SHA256,
                "benchmark_seed": BENCHMARK_SEED,
                "training_seeds_run": [43, 44],
                "models": {str(k): v for k, v in results.items()},
                "baseline_by_seed": {str(k): v for k, v in all_baselines.items()},
                "seed42_reused": seed42_data}
    write_json(METRICS / cfg["outputs"]["metrics"], combined)
    three = summarize_three_seeds(cfg, results, all_baselines, seed42_data)
    write_json(METRICS / cfg["outputs"]["summary"], three)
    (METRICS / cfg["outputs"]["report"]).write_text(
        render_report(three, seed42_data, results), encoding="utf-8")
    print(json.dumps(three["paired_deltas"], indent=2))


if __name__ == "__main__":
    main()
