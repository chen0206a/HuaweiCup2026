"""B5-L1: seed-42 B0-from-scratch sweep over the fixed regression-loss weight grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.preprocess import build_datasets_and_loaders
from src.evaluation.missing_benchmark import evaluate_benchmark, scenarios, summary
from src.models.baseline import B0Baseline, multitask_loss
from src.training.evaluate import evaluate_loader
from src.training.train_b1 import seed_everything

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "outputs" / "metrics"
CHECKPOINTS = ROOT / "outputs" / "checkpoints"
BENCHMARK_SHA256 = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"
SCORE_FIELDS = ("accuracy", "macro_f1", "mae", "pearson", "selection_score")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_protocol(cfg: dict, baseline_cfg: dict, checkpoint: dict) -> list[float]:
    train = cfg["training"]
    old_train = baseline_cfg["training"]
    old_model = baseline_cfg["model"]
    actual_lambda0 = float(checkpoint["config"]["training"]["lambda_reg"])
    expected_settings = {
        "seed": int(train["seed"]) == int(old_train["seed"]),
        "epochs": int(train["epochs"]) == int(old_train["epochs"]),
        "patience": int(train["patience"]) == int(old_train["patience"]),
        "learning_rate": float(train["learning_rate"]) == float(old_train["learning_rate"]),
        "weight_decay": float(train["weight_decay"]) == float(old_train["weight_decay"]),
        "model": cfg["model"] == old_model,
        "batch_size": int(cfg["data"]["batch_size"]) == int(baseline_cfg["data"]["batch_size"]),
        "normalization": baseline_cfg["preprocessing"]["normalization"] == "none",
        "weighted_ce": old_train["class_weighting"] == "balanced_train",
    }
    if not all(expected_settings.values()):
        raise ValueError(f"B5-L1 settings differ from actual B0-WCE config: {expected_settings}")
    if checkpoint["config"] != baseline_cfg or actual_lambda0 != float(old_train["lambda_reg"]):
        raise ValueError("B0-WCE checkpoint config differs from its saved baseline config")
    if (train["seed"] != 42 or train["initialization"] != "random_seeded_b0_from_scratch" or
            train["augmentation"] != "none" or train["class_weighting"] != "balanced_train" or
            cfg["preprocessing"]["normalization"] != "none" or
            train["optimizer"] != "AdamW" or train["scheduler"] != "none"):
        raise ValueError("B5-L1 configuration differs from the specified B0 training protocol")
    path = Path(cfg["benchmark"]["definition"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    definition = json.loads(path.read_text(encoding="utf-8"))
    if (digest != BENCHMARK_SHA256 or digest != cfg["benchmark"]["sha256"] or
            definition["seed"] != 20260923 or
            definition["scenarios"] != [scene.as_dict() for scene in scenarios()]):
        raise RuntimeError("frozen clean + 54 scenario benchmark has changed")
    lambda0 = actual_lambda0
    if lambda0 != float(train["lambda_reg_baseline"]):
        raise ValueError("configured lambda baseline does not equal the actual B0 checkpoint value")
    values = [lambda0 * float(multiplier) for multiplier in train["lambda_reg_multipliers"]]
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError("B5-L1 requires four unique, predeclared lambda values")
    return values


def fresh_train_loader(dataset, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                      generator=generator)


def initialization_snapshot(model: B0Baseline) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in state.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def sample_order(dataset, batch_size: int, seed: int) -> list[str]:
    return [sample_id for batch in fresh_train_loader(dataset, batch_size, seed)
            for sample_id in batch["id"]]


def gradient_diagnostic(model: B0Baseline, batch: dict, weights: torch.Tensor,
                        device: torch.device) -> dict:
    model.train()
    moved = {key: (value.to(device) if isinstance(value, torch.Tensor) else value)
             for key, value in batch.items()}
    outputs = model(moved)
    ce = torch.nn.functional.cross_entropy(outputs["classification_logits"],
                                           moved["cls_label"], weight=weights)
    reg = torch.nn.functional.smooth_l1_loss(outputs["regression"], moved["reg_label"])
    params = [parameter for parameter in model.fusion.parameters() if parameter.requires_grad]
    grad_cls = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
    grad_reg = torch.autograd.grad(reg, params, allow_unused=True)

    def flatten(grads):
        return torch.cat([grad.reshape(-1) for grad in grads if grad is not None])

    g_cls, g_reg = flatten(grad_cls), flatten(grad_reg)
    norm_cls, norm_reg = g_cls.norm(), g_reg.norm()
    cosine = torch.dot(g_cls, g_reg) / (norm_cls * norm_reg).clamp_min(1e-12)
    return {"parameter_group": "fusion MLP only", "parameter_count": sum(p.numel() for p in params),
            "batch_size": int(moved["cls_label"].shape[0]),
            "weighted_ce": float(ce.detach().cpu()), "smooth_l1": float(reg.detach().cpu()),
            "classification_gradient_l2": float(norm_cls.detach().cpu()),
            "regression_gradient_l2": float(norm_reg.detach().cpu()),
            "gradient_norm_ratio_reg_over_cls": float((norm_reg / norm_cls.clamp_min(1e-12)).detach().cpu()),
            "cosine_similarity": float(cosine.detach().cpu())}


def neutral_summary(rows: list[dict], subset: list[dict] | None = None) -> dict:
    selected = rows if subset is None else subset
    return {"recall": float(np.mean([row["per_class"]["Neutral"]["recall"] for row in selected])),
            "f1": float(np.mean([row["per_class"]["Neutral"]["f1"] for row in selected]))}


def vision_zero_summary(rows: list[dict]) -> dict:
    available = [row["vision_all_zero_metrics"] for row in rows if row["vision_all_zero_metrics"]]
    if not available:
        return {"count": 0, "clean": None, "mean_missing": None}
    return {"count": int(rows[0]["vision_all_zero_count"]),
            "clean": {key: rows[0]["vision_all_zero_metrics"][key] for key in SCORE_FIELDS[:4]},
            "mean_missing": {key: float(np.mean([row["vision_all_zero_metrics"][key]
                                                  for row in rows[1:]]))
                             for key in SCORE_FIELDS[:4]}}


def train_one(lambda_reg: float, cfg: dict, initial_state: dict,
              datasets: dict, valid_loader, weights: torch.Tensor,
              device: torch.device, initial_digest: str) -> tuple[dict, list[dict]]:
    seed = int(cfg["training"]["seed"])
    seed_everything(seed)
    model = B0Baseline(**cfg["model"]).to(device)
    model.load_state_dict(initial_state)
    if not state_equal(initial_state, initialization_snapshot(model)):
        raise RuntimeError("model did not load the common random initialization exactly")
    train_loader = fresh_train_loader(datasets["train"], int(cfg["data"]["batch_size"]), seed)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(cfg["training"]["learning_rate"]),
                                  weight_decay=float(cfg["training"]["weight_decay"]))
    tag = f"lambda_{lambda_reg:g}"
    clean_path = CHECKPOINTS / f"b5_l1_{tag}_best_clean_score.pt"
    robust_path = CHECKPOINTS / f"b5_l1_{tag}_best_robust_score.pt"
    best_clean = best_robust = -float("inf")
    best_clean_epoch = best_robust_epoch = stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        train_n = 0
        ce_sum = reg_sum = total_sum = 0.0
        for batch in train_loader:
            moved = {key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                     for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(moved)
            losses = multitask_loss(outputs, moved, lambda_reg=lambda_reg,
                                    class_weights=weights)
            losses["total"].backward()
            optimizer.step()
            n = int(moved["cls_label"].shape[0])
            train_n += n
            ce_sum += float(losses["cross_entropy"].detach()) * n
            reg_sum += float(losses["smooth_l1"].detach()) * n
            total_sum += float(losses["total"].detach()) * n
        train_losses = {"classification": ce_sum / train_n,
                        "regression": reg_sum / train_n,
                        "weighted_regression_contribution": lambda_reg * reg_sum / train_n,
                        "total": total_sum / train_n,
                        "weighted_regression_to_classification_ratio":
                            (lambda_reg * reg_sum / max(ce_sum, 1e-12))}
        clean_loss_metrics = evaluate_loader(model, valid_loader, device,
                                             lambda_reg=lambda_reg, class_weights=weights)
        rows = evaluate_benchmark(model, valid_loader, device)
        metrics = summary(rows)
        clean_score = metrics["clean"]["selection_score"]
        robust_score = metrics["robust_score"]
        row = {"epoch": epoch, "train_losses": train_losses,
               "valid_clean_losses": clean_loss_metrics["loss"],
               "clean": metrics["clean"], "mean_missing": metrics["mean_missing"],
               "robust_score": robust_score,
               "neutral_clean": metrics["clean"],
               "neutral_mean_missing": neutral_summary(rows[1:])}
        history.append(row)
        checkpoint = {"model_state_dict": model.state_dict(), "config": cfg,
                      "lambda_reg": lambda_reg, "epoch": epoch,
                      "clean_score": clean_score, "robust_score": robust_score,
                      "initial_state_sha256": initial_digest,
                      "train_class_weights": weights.detach().cpu()}
        if clean_score > best_clean:
            best_clean, best_clean_epoch = clean_score, epoch
            torch.save(checkpoint, clean_path)
        if robust_score > best_robust:
            best_robust, best_robust_epoch, stale = robust_score, epoch, 0
            torch.save(checkpoint, robust_path)
        else:
            stale += 1
        print(json.dumps({"lambda_reg": lambda_reg, "epoch": epoch,
                          "train_losses": train_losses,
                          "clean_score": clean_score,
                          "mean_missing_score": metrics["mean_missing"]["selection_score"],
                          "robust_score": robust_score}, ensure_ascii=False), flush=True)
        if stale >= int(cfg["training"]["patience"]):
            break
    elapsed = time.perf_counter() - started

    def evaluate_checkpoint(path: Path) -> tuple[dict, list[dict]]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        selected_model = B0Baseline(**cfg["model"]).to(device).eval()
        selected_model.load_state_dict(checkpoint["model_state_dict"])
        selected_rows = evaluate_benchmark(selected_model, valid_loader, device)
        return summary(selected_rows), selected_rows

    clean_summary, clean_rows = evaluate_checkpoint(clean_path)
    robust_summary, robust_rows = evaluate_checkpoint(robust_path)
    if clean_summary["clean"]["selection_score"] != best_clean or robust_summary["robust_score"] != best_robust:
        raise RuntimeError(f"{tag} selected checkpoint re-evaluation changed score")
    clean_checkpoint = torch.load(clean_path, map_location="cpu", weights_only=False)
    robust_checkpoint = torch.load(robust_path, map_location="cpu", weights_only=False)
    is_lambda_baseline = lambda_reg == float(cfg["training"]["lambda_reg_baseline"])
    result = {"candidate_name": "B0-LambdaBaseline-seed42" if is_lambda_baseline else tag,
              "lambda_reg": lambda_reg, "initial_state_sha256": initial_digest,
              "class_weights_sha256": hashlib.sha256(weights.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "best_clean_epoch": best_clean_epoch, "best_robust_epoch": best_robust_epoch,
              "epochs_run": len(history), "training_seconds": elapsed,
              "best_clean_validation": clean_summary,
              "best_robust_validation": robust_summary,
              "best_clean_neutral": {"clean": neutral_summary(clean_rows[:1]),
                                     "mean_missing": neutral_summary(clean_rows[1:])},
              "best_robust_neutral": {"clean": neutral_summary(robust_rows[:1]),
                                      "mean_missing": neutral_summary(robust_rows[1:])},
              "best_clean_vision_all_zero": vision_zero_summary(clean_rows),
              "best_robust_vision_all_zero": vision_zero_summary(robust_rows),
              "best_clean_scenario_details": clean_rows,
              "best_robust_scenario_details": robust_rows,
              "best_clean_losses": history[best_clean_epoch - 1],
              "best_robust_losses": history[best_robust_epoch - 1],
              "checkpoints": {"best_clean": str(clean_path), "best_robust": str(robust_path)},
              "checkpoint_epochs": {"best_clean": int(clean_checkpoint["epoch"]),
                                    "best_robust": int(robust_checkpoint["epoch"])}}
    return result, history


def render_report(result: dict) -> str:
    candidates = result["candidates"]
    baseline_tag = result["baseline_lambda_tag"]
    baseline = candidates[baseline_tag]["best_robust_validation"]
    lines = ["# B5-L1 classification/regression loss balance", "",
             "B0 from-scratch initialization, seed 42; clean train; unchanged clean + 54 scenario benchmark.",
             "Screening uses each run's best-robust checkpoint. Test and attachment3 were not loaded.", "",
             "| lambda_reg | Clean Acc | Clean F1 | Clean MAE | Clean Pearson | Clean score | Missing Acc | Missing F1 | Missing MAE | Missing Pearson | Missing score | Robust | Δ robust vs lambda0 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, d in candidates.items():
        c, m = d["best_robust_validation"]["clean"], d["best_robust_validation"]["mean_missing"]
        delta = d["best_robust_validation"]["robust_score"] - baseline["robust_score"]
        vals = [d["lambda_reg"], *[c[k] for k in SCORE_FIELDS[:4]], c["selection_score"],
                *[m[k] for k in SCORE_FIELDS[:4]], m["selection_score"],
                d["best_robust_validation"]["robust_score"], delta]
        lines.append("| " + " | ".join([f"{vals[0]:g}"] + [f"{x:.4f}" for x in vals[1:]]) + " |")
    lines += ["", "## Clean-selected checkpoints", "",
              "| lambda_reg | best-clean epoch | Clean score | Robust score at best-clean |",
              "|---:|---:|---:|---:|"]
    for tag, d in candidates.items():
        s = d["best_clean_validation"]
        lines.append(f"| {d['lambda_reg']:g} | {d['best_clean_epoch']} | "
                     f"{s['clean']['selection_score']:.4f} | {s['robust_score']:.4f} |")
    for section in ("by_modality", "by_ratio", "by_location", "double_stress"):
        lines += ["", f"## {section}", "",
                  "| lambda | group | score | Neutral recall | Neutral F1 | Acc | Macro-F1 | MAE | Pearson |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
        for d in candidates.values():
            summary_result = d["best_robust_validation"]
            for group, metrics in summary_result[section].items():
                scene_rows = d["best_robust_scenario_details"]
                if section == "by_modality":
                    matching = [r for r in scene_rows[1:] if r["modalities"] == group]
                elif section == "double_stress":
                    matching = [r for r in scene_rows[1:] if r["modalities"] == group]
                elif section == "by_ratio":
                    matching = [r for r in scene_rows[1:]
                                if str(r["rho"]) == group and "+" not in r["modalities"]]
                else:
                    matching = [r for r in scene_rows[1:] if r["location"] == group]
                neutral = neutral_summary(matching) if matching else {"recall": 0.0, "f1": 0.0}
                lines.append(f"| {d['lambda_reg']:g} | {group} | {metrics['selection_score']:.4f} | "
                             f"{neutral['recall']:.4f} | {neutral['f1']:.4f} | "
                             + " | ".join(f"{metrics[k]:.4f}" for k in SCORE_FIELDS[:4]) + " |")
    lines += ["", "## vision_all_zero subset and loss balance", "",
              "| lambda | subset condition | Acc | Macro-F1 | MAE | Pearson | Neutral recall | Neutral F1 | Train CE | Train SmoothL1 | lambda×SmoothL1 | weighted-reg/CE | best-clean epoch | best-robust epoch | seconds |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d in candidates.values():
        loss = d["best_robust_losses"]["train_losses"]
        for condition in ("clean", "mean_missing"):
            metrics = d["best_robust_vision_all_zero"][condition]
            neutral = d["best_robust_neutral"][condition]
            lines.append(f"| {d['lambda_reg']:g} | {condition} | " +
                         " | ".join(f"{metrics[key]:.4f}" for key in SCORE_FIELDS[:4]) +
                         f" | {neutral['recall']:.4f} | {neutral['f1']:.4f} | "
                         f"{loss['classification']:.4f} | {loss['regression']:.4f} | "
                         f"{loss['weighted_regression_contribution']:.4f} | "
                         f"{loss['weighted_regression_to_classification_ratio']:.4f} | "
                         f"{d['best_clean_epoch']} | {d['best_robust_epoch']} | {d['training_seconds']:.2f} |")
    lines += ["", "## Initialization and gradient diagnostic", "",
              f"Lambda0: {result['lambda0']}; candidates: {', '.join(str(v) for v in result['lambda_values'])}.",
              f"B0 protocol: {result['actual_b0_protocol']['classification_loss']} + lambda×"
              f"{result['actual_b0_protocol']['regression_loss']}; "
              f"{result['actual_b0_protocol']['optimizer']} LR={result['actual_b0_protocol']['learning_rate']}, "
              f"weight decay={result['actual_b0_protocol']['weight_decay']}, "
              f"batch={result['actual_b0_protocol']['batch_size']}, "
              f"max epochs/patience={result['actual_b0_protocol']['max_epochs']}/"
              f"{result['actual_b0_protocol']['patience']}; initialization={result['actual_b0_protocol']['initialization']}.",
              f"Same initial model state across four runs: {result['initial_state_equality']['passed']} "
              f"({result['initial_state_equality']['tensor_count']} tensors).",
              f"Same full train-sample order: {result['same_train_order']['passed']}; class weights identical: "
              f"{result['class_weights_identical']}; benchmark SHA-256: {result['benchmark_sha256']}.",
              f"At the common initial model, fusion gradient norms were CE {result['gradient_diagnostic']['classification_gradient_l2']:.5f}, "
              f"regression {result['gradient_diagnostic']['regression_gradient_l2']:.5f}, cosine "
              f"{result['gradient_diagnostic']['cosine_similarity']:.5f}. This is descriptive only.",
              "", "## Historical seed-42 B0 comparison", "",
              f"Historical clean-selected B0 score: {result['historical_b0']['clean_selection_score']:.6f}; "
              f"historical frozen-benchmark robust score: {result['historical_b0']['robust_score']:.6f}.",
              f"Fresh lambda0 best-clean score: {candidates[baseline_tag]['best_clean_validation']['clean']['selection_score']:.6f}; "
              f"fresh lambda0 best-robust score: {baseline['robust_score']:.6f}."]
    return "\n".join(lines) + "\n"


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    baseline_cfg = yaml.safe_load(Path(cfg["training"]["baseline_config"]).read_text(encoding="utf-8"))
    historical_checkpoint = torch.load(cfg["training"]["baseline_checkpoint"],
                                       map_location="cpu", weights_only=False)
    lambda_values = verify_protocol(cfg, baseline_cfg, historical_checkpoint)
    seed = int(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, loaders, _ = build_datasets_and_loaders(
        cfg["data"]["pkl_path"], normalization="none",
        batch_size=int(cfg["data"]["batch_size"]), num_workers=0,
        seed=seed, include_test=False)
    class_counts = np.bincount(datasets["train"].cls_labels, minlength=3)
    if np.any(class_counts == 0):
        raise RuntimeError("train split lacks a class")
    weights_cpu = torch.as_tensor(len(datasets["train"]) / (3.0 * class_counts), dtype=torch.float32)
    weights = weights_cpu.to(device)

    # Independently instantiate all four from the same reset seed before any training.
    initial_states = []
    for _ in lambda_values:
        seed_everything(seed)
        initial_model = B0Baseline(**cfg["model"])
        initial_states.append(initialization_snapshot(initial_model))
    equality = all(state_equal(initial_states[0], state) for state in initial_states[1:])
    if not equality:
        raise RuntimeError("candidate initial model states are not tensor-identical")
    initial_state = initial_states[0]
    initial_digest = state_digest(initial_state)

    orders = [sample_order(datasets["train"], int(cfg["data"]["batch_size"]), seed)
              for _ in lambda_values]
    same_order = all(order == orders[0] for order in orders[1:])
    if not same_order:
        raise RuntimeError("train DataLoader sample order differs across lambda candidates")

    # One fixed first shuffled training batch; no diagnostic gradients affect optimizer updates.
    diagnostic_batch = next(iter(fresh_train_loader(
        datasets["train"], int(cfg["data"]["batch_size"]), seed)))
    seed_everything(seed)
    diagnostic_model = B0Baseline(**cfg["model"]).to(device)
    diagnostic_model.load_state_dict(initial_state)
    grad_diag = gradient_diagnostic(diagnostic_model, diagnostic_batch, weights, device)

    old_metric = json.loads((METRICS / "b0_weighted_ce_score_selection_metrics.json").read_text(encoding="utf-8"))
    old_benchmark = json.loads((METRICS / "b2_inherent_summary.json").read_text(encoding="utf-8"))
    historical_b0 = {"clean_selection_score": old_metric["best_selection_score"]["value"],
                     "clean_selection_epoch": old_metric["best_selection_score"]["epoch"],
                     "clean_metrics": old_metric["best_selection_score"]["validation_metrics"],
                     "robust_score": old_benchmark["B0-WCE"]["summary"]["robust_score"],
                     "robust_metrics": old_benchmark["B0-WCE"]["summary"]}
    results, histories = {}, {}
    baseline_tag = f"lambda_{float(cfg['training']['lambda_reg_baseline']):g}"
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    for lambda_reg in lambda_values:
        tag = f"lambda_{lambda_reg:g}"
        # Reinitialize RNG before model creation/training for paired dropout and sample order.
        result, history = train_one(lambda_reg, cfg, initial_state, datasets,
                                    loaders["valid"], weights, device, initial_digest)
        results[tag], histories[tag] = result, history
        write_json(METRICS / cfg["outputs"]["history"], histories)
        partial = {"lambda0": float(cfg["training"]["lambda_reg_baseline"]),
                   "lambda_values": lambda_values, "candidates": results}
        write_json(METRICS / cfg["outputs"]["metrics"], partial)
    result = {"training_seed": seed,
              "benchmark_seed": 20260923,
              "benchmark_sha256": BENCHMARK_SHA256,
              "lambda0": float(cfg["training"]["lambda_reg_baseline"]),
              "lambda_values": lambda_values,
              "actual_b0_protocol": {
                  "config": cfg["training"]["baseline_config"],
                  "checkpoint": cfg["training"]["baseline_checkpoint"],
                  "checkpoint_epoch": int(historical_checkpoint.get("best_epoch", -1)),
                  "classification_loss": "Weighted CrossEntropy, balanced weights from train counts",
                  "regression_loss": "SmoothL1Loss(beta=1.0 default)",
                  "optimizer": "AdamW",
                  "learning_rate": float(baseline_cfg["training"]["learning_rate"]),
                  "weight_decay": float(baseline_cfg["training"]["weight_decay"]),
                  "batch_size": int(baseline_cfg["data"]["batch_size"]),
                  "max_epochs": int(baseline_cfg["training"]["epochs"]),
                  "patience": int(baseline_cfg["training"]["patience"]),
                  "normalization": baseline_cfg["preprocessing"]["normalization"],
                  "initialization": "B0Baseline random initialization after seed_everything(42); no checkpoint finetuning",
                  "source_checkpoint_config_lambda_reg": float(historical_checkpoint["config"]["training"]["lambda_reg"]),
              },
              "baseline_config": cfg["training"]["baseline_config"],
              "baseline_checkpoint": cfg["training"]["baseline_checkpoint"],
              "baseline_lambda_tag": baseline_tag,
              "initial_state_equality": {"passed": equality,
                                         "tensor_count": len(initial_state),
                                         "sha256": initial_digest},
              "same_train_order": {"passed": same_order,
                                   "samples_per_epoch": len(orders[0]),
                                   "order_sha256": hashlib.sha256("\n".join(orders[0]).encode()).hexdigest()},
              "class_weights_identical": len({entry["class_weights_sha256"]
                                               for entry in results.values()}) == 1,
              "class_counts": class_counts.tolist(), "class_weights": weights_cpu.tolist(),
              "gradient_diagnostic": grad_diag,
              "historical_b0": historical_b0,
              "test_or_attachment3_used": False,
              "candidates": results}
    write_json(METRICS / cfg["outputs"]["metrics"], result)
    write_json(METRICS / cfg["outputs"]["history"], histories)
    (METRICS / cfg["outputs"]["report"]).write_text(render_report(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "b5_l1_lambda.yaml")
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({tag: {"lambda_reg": value["lambda_reg"],
                            "robust": value["best_robust_validation"]["robust_score"],
                            "epoch": value["best_robust_epoch"]}
                      for tag, value in result["candidates"].items()}), flush=True)


if __name__ == "__main__":
    main()
