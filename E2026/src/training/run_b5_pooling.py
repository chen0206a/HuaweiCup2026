"""Seed-42 B5-P pooling screening on the unchanged B2 validation benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import MODALITIES
from src.data.preprocess import build_datasets_and_loaders
from src.evaluation.missing_benchmark import evaluate_benchmark, mask_scenario, scenarios, summary
from src.models.baseline import B0Baseline, multitask_loss
from src.models.pooling_residual import B5PoolingResidual
from src.training.train_b1 import seed_everything

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "outputs" / "metrics"
CHECKPOINTS = ROOT / "outputs" / "checkpoints"
BENCHMARK_SHA256 = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"
SCORE_FIELDS = ("accuracy", "macro_f1", "mae", "pearson", "selection_score")
B0_KEYS = ("modality_projection.", "fusion.", "classification_head.", "regression_head.")


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_config(cfg: dict) -> None:
    if (cfg["training"]["seed"] != 42 or cfg["preprocessing"]["normalization"] != "none" or
            cfg["training"]["augmentation"] != "none" or
            cfg["training"]["class_weighting"] != "balanced_train"):
        raise ValueError("B5-P requires seed42, no normalization/augmentation, balanced train CE")
    definition_path = Path(cfg["benchmark"]["definition"])
    digest = hashlib.sha256(definition_path.read_bytes()).hexdigest()
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    if (digest != BENCHMARK_SHA256 or digest != cfg["benchmark"]["sha256"] or
            definition["seed"] != 20260923 or
            definition["scenarios"] != [scene.as_dict() for scene in scenarios()]):
        raise RuntimeError("frozen 54-scenario benchmark changed")


def init_candidate(mode: str, cfg: dict, source: dict, device: torch.device) -> B5PoolingResidual:
    model = B5PoolingResidual(mode, **cfg["model"]).to(device)
    incompatible = model.load_state_dict(source["model_state_dict"], strict=False)
    if incompatible.unexpected_keys or any(not key.startswith(("gamma.", "attention_scorer."))
                                       for key in incompatible.missing_keys):
        raise RuntimeError(f"B0 initialization mismatch: {incompatible}")
    if cfg["training"]["freeze_b0_parameters"]:
        for name, parameter in model.named_parameters():
            if name.startswith(B0_KEYS):
                parameter.requires_grad_(False)
    return model


@torch.no_grad()
def check_identity(source_model: B0Baseline, candidate: B5PoolingResidual,
                   valid_loader, device: torch.device) -> dict:
    source_model.eval()
    candidate.eval()
    largest = {"classification_logits": 0.0, "regression": 0.0}
    exact = True
    seen = 0
    for batch in valid_loader:
        x = {key: batch[key].to(device) for key in (*MODALITIES, "padding_mask")}
        base, new = source_model(x), candidate(x)
        for key in largest:
            largest[key] = max(largest[key], float((base[key] - new[key]).abs().max().item()))
            exact = exact and torch.equal(base[key], new[key])
        seen += x["padding_mask"].shape[0]
    return {"passed": exact, "exact_prediction_match": exact,
            "maximum_absolute_difference": largest, "valid_samples": seen}


def b0_equality(model: B5PoolingResidual, state: dict[str, torch.Tensor]) -> dict:
    changed = [name for name, tensor in state.items()
               if not torch.equal(model.state_dict()[name].detach().cpu(), tensor.detach().cpu())]
    return {"passed": len(changed) == 0, "tensor_count": len(state), "changed_tensors": changed}


def fresh_train_loader(dataset, batch_size: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=0, generator=generator)


@torch.no_grad()
def attention_diagnostics(model: B5PoolingResidual, valid_loader,
                          device: torch.device) -> dict:
    """Evaluate attention concentration for clean and all frozen missing scenes."""
    model.eval()
    aggregation = {condition: {m: {"entropy": [], "normalized_entropy": [],
                                  "max_weight": [], "long_enough": []}
                               for m in MODALITIES} for condition in ("clean", "mean_missing")}

    def collect(condition: str, batch: dict) -> None:
        weights = model.pool_diagnostics(batch)
        length = batch["padding_mask"].sum(dim=1)
        for modality in MODALITIES:
            w = weights[modality]
            entropy = -(w * w.clamp_min(1e-12).log()).sum(dim=1)
            maximum = w.amax(dim=1)
            longer = length > 1
            normal = entropy[longer] / length[longer].float().log()
            row = aggregation[condition][modality]
            row["entropy"].extend(entropy.cpu().tolist())
            row["normalized_entropy"].extend(normal.cpu().tolist())
            row["max_weight"].extend(maximum.cpu().tolist())
            row["long_enough"].extend(longer.cpu().tolist())

    for batch in valid_loader:
        moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        collect("clean", {key: moved[key] for key in (*MODALITIES, "padding_mask")})
        for scenario in scenarios():
            masked, _ = mask_scenario(moved, scenario)
            collect("mean_missing", {key: masked[key] for key in (*MODALITIES, "padding_mask")})

    result = {}
    for condition, modalities in aggregation.items():
        result[condition] = {}
        for modality, row in modalities.items():
            maxima = np.asarray(row["max_weight"])
            long_enough = np.asarray(row["long_enough"], dtype=bool)
            result[condition][modality] = {
                "count": int(len(maxima)),
                "length_gt_one_count": int(long_enough.sum()),
                "mean_entropy": float(np.mean(row["entropy"])),
                "mean_normalized_entropy_length_gt_one": float(np.mean(row["normalized_entropy"])),
                "mean_max_weight": float(maxima.mean()),
                "p95_max_weight": float(np.quantile(maxima, 0.95)),
                "fraction_max_weight_ge_0_9_length_gt_one": float(np.mean(maxima[long_enough] >= 0.9)),
            }
    return result


def vision_subset(rows: list[dict]) -> dict:
    return {"count": rows[0]["vision_all_zero_count"],
            "clean": {key: rows[0]["vision_all_zero_metrics"][key] for key in SCORE_FIELDS[:4]},
            "mean_missing": {key: float(np.mean([
                row["vision_all_zero_metrics"][key] for row in rows[1:]]))
                for key in SCORE_FIELDS[:4]}}


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, loaders, _ = build_datasets_and_loaders(
        cfg["data"]["pkl_path"], normalization="none",
        batch_size=int(cfg["data"]["batch_size"]), num_workers=0,
        seed=42, include_test=False)
    source = torch.load(cfg["training"]["init_checkpoint"], map_location="cpu", weights_only=False)
    if int(source["config"]["training"]["seed"]) != 42 or source["config"]["model"] != cfg["model"]:
        raise RuntimeError("source is not the requested seed-42 B0-WCE architecture")
    base = B0Baseline(**cfg["model"]).to(device).eval()
    base.load_state_dict(source["model_state_dict"])
    b0_state = {k: v.detach().clone() for k, v in source["model_state_dict"].items()}
    baseline_rows = json.loads((METRICS / "b2_b0_inherent_detail.json").read_text(encoding="utf-8"))
    p0_rows = evaluate_benchmark(base, loaders["valid"], device)
    p0_match = p0_rows == baseline_rows
    if not p0_match:
        raise RuntimeError("P0 B0 checkpoint did not reproduce frozen benchmark result")
    result = {"benchmark_sha256": BENCHMARK_SHA256,
              "training_seed": 42, "benchmark_seed": 20260923,
              "preprocessing": "none", "training_augmentation": "none",
              "freeze_b0_parameters": bool(cfg["training"]["freeze_b0_parameters"]),
              "P0": {"parameter_count": sum(p.numel() for p in base.parameters()),
                     "benchmark_match": p0_match, "validation": summary(p0_rows),
                     "vision_all_zero": vision_subset(p0_rows)}}
    histories = {}
    counts = np.bincount(datasets["train"].cls_labels, minlength=3)
    if np.any(counts == 0):
        raise RuntimeError("train split lacks a class")
    class_weights = torch.as_tensor(len(datasets["train"]) / (3 * counts),
                                    dtype=torch.float32, device=device)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    for label, mode in (("P1", "mean_max"), ("P2", "mean_attention")):
        seed_everything(42)
        model = init_candidate(mode, cfg, source, device)
        identity = check_identity(base, model, loaders["valid"], device)
        if not identity["passed"]:
            raise RuntimeError(f"{label} gamma=0 identity failed: {identity}")
        # Optimizer type, LR, weight decay, CE, regression loss and data order match B0.
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=float(cfg["training"]["learning_rate"]),
                                      weight_decay=float(cfg["training"]["weight_decay"]))
        train_loader = fresh_train_loader(datasets["train"],
                                          int(cfg["data"]["batch_size"]), 42)
        clean_path = CHECKPOINTS / f"{cfg['outputs']['checkpoint_prefix']}_{label.lower()}_best_clean_score.pt"
        robust_path = CHECKPOINTS / f"{cfg['outputs']['checkpoint_prefix']}_{label.lower()}_best_robust_score.pt"
        best_clean = best_robust = -float("inf")
        best_clean_epoch = best_robust_epoch = stale = 0
        history = []
        train_start = time.perf_counter()
        for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
            model.train()
            if cfg["training"]["freeze_b0_parameters"]:
                model.modality_projection.eval()
                model.fusion.eval()
                model.classification_head.eval()
                model.regression_head.eval()
            loss_total = 0.0
            seen = 0
            for batch in train_loader:
                moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                         for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                output = model({key: moved[key] for key in (*MODALITIES, "padding_mask")})
                loss = multitask_loss(output, moved,
                                      lambda_reg=float(cfg["training"]["lambda_reg"]),
                                      class_weights=class_weights)["total"]
                loss.backward()
                optimizer.step()
                size = len(moved["cls_label"])
                loss_total += float(loss.item()) * size
                seen += size
            valid_start = time.perf_counter()
            rows = evaluate_benchmark(model, loaders["valid"], device)
            valid_seconds = time.perf_counter() - valid_start
            metrics = summary(rows)
            clean_score, robust_score = metrics["clean"]["selection_score"], metrics["robust_score"]
            history.append({"epoch": epoch, "train_loss": loss_total / seen,
                            "validation_seconds": valid_seconds, **metrics})
            checkpoint = {"model_state_dict": model.state_dict(), "config": cfg,
                          "mode": mode, "epoch": epoch, "clean_score": clean_score,
                          "robust_score": robust_score, "benchmark_sha256": BENCHMARK_SHA256,
                          "train_class_counts": counts.tolist(),
                          "class_weights": class_weights.detach().cpu()}
            if clean_score > best_clean:
                best_clean, best_clean_epoch = clean_score, epoch
                torch.save(checkpoint, clean_path)
            if robust_score > best_robust:
                best_robust, best_robust_epoch, stale = robust_score, epoch, 0
                torch.save(checkpoint, robust_path)
            else:
                stale += 1
            print(json.dumps({"model": label, "epoch": epoch,
                              "train_loss": loss_total / seen,
                              "clean_score": clean_score, "robust_score": robust_score,
                              "validation_seconds": valid_seconds}), flush=True)
            if stale >= int(cfg["training"]["patience"]):
                break
        train_seconds = time.perf_counter() - train_start
        best_state = torch.load(robust_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_state["model_state_dict"])
        model.eval()
        frozen_equality = b0_equality(model, b0_state) if cfg["training"]["freeze_b0_parameters"] else None
        if frozen_equality is not None and not frozen_equality["passed"]:
            raise RuntimeError(f"{label} changed frozen B0 tensors")
        final_rows = evaluate_benchmark(model, loaders["valid"], device)
        final_summary = summary(final_rows)
        if final_summary["robust_score"] != best_robust:
            raise RuntimeError(f"{label} checkpoint round trip changed robust score")
        restored = B5PoolingResidual(mode, **cfg["model"]).to(device).eval()
        restored.load_state_dict(best_state["model_state_dict"])
        first_batch = next(iter(loaders["valid"]))
        first_input = {k: first_batch[k].to(device) for k in (*MODALITIES, "padding_mask")}
        with torch.no_grad():
            a, b = model(first_input), restored(first_input)
        roundtrip = all(torch.equal(a[k], b[k]) for k in a)
        if not roundtrip:
            raise RuntimeError(f"{label} checkpoint round trip changed predictions")
        gamma = {m: float(model.gamma[m].detach().cpu()) for m in MODALITIES}
        result[label] = {"mode": mode, "parameter_count": sum(p.numel() for p in model.parameters()),
                         "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                         "initial_identity": identity, "frozen_b0_equality": frozen_equality,
                         "checkpoint_roundtrip": roundtrip,
                         "best_clean_epoch": best_clean_epoch, "best_robust_epoch": best_robust_epoch,
                         "epochs_run": len(history), "training_seconds": train_seconds,
                         "mean_validation_seconds": float(np.mean([h["validation_seconds"] for h in history])),
                         "gamma": gamma, "validation": final_summary,
                         "vision_all_zero": vision_subset(final_rows),
                         "scenario_details": final_rows,
                         "checkpoints": {"best_clean": str(clean_path), "best_robust": str(robust_path)}}
        if mode == "mean_attention":
            result[label]["attention_diagnostics"] = attention_diagnostics(model, loaders["valid"], device)
        histories[label] = history
        write_json(METRICS / cfg["outputs"]["history"], histories)
        write_json(METRICS / cfg["outputs"]["metrics"], result)
    (METRICS / cfg["outputs"]["report"]).write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: dict) -> str:
    lines = ["# B5-P pooling screening, training seed 42", "",
             "P0 is the original B0-WCE best-selection checkpoint. P1 and P2 start from that checkpoint.",
             "Frozen clean + 54 missing scenarios; benchmark seed 20260923. No test/attachment3.",
             f"B0 prediction parameters frozen in P1/P2: {result['freeze_b0_parameters']}.", "",
             "| Model | Condition | Accuracy | Macro-F1 | MAE | Pearson | Score | Robust |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("P0", "P1", "P2"):
        validation = result[name]["validation"]
        for condition in ("clean", "mean_missing"):
            m = validation[condition]
            lines.append(f"| {name} | {condition} | " + " | ".join(f"{m[k]:.4f}" for k in SCORE_FIELDS) +
                         f" | {validation['robust_score']:.4f} |")
    for section in ("by_modality", "by_ratio", "by_location", "double_stress"):
        lines += ["", f"## {section}", "",
                  "| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for name in ("P0", "P1", "P2"):
            for group, m in result[name]["validation"][section].items():
                lines.append(f"| {name} | {group} | " + " | ".join(f"{m[k]:.4f}" for k in SCORE_FIELDS) + " |")
    lines += ["", "## vision_all_zero validation subset", "",
              "| Model | Condition | Accuracy | Macro-F1 | MAE | Pearson |",
              "|---|---|---:|---:|---:|---:|"]
    for name in ("P0", "P1", "P2"):
        for condition in ("clean", "mean_missing"):
            m = result[name]["vision_all_zero"][condition]
            lines.append(f"| {name} | {condition} | " +
                         " | ".join(f"{m[k]:.4f}" for k in SCORE_FIELDS[:4]) + " |")
    lines += ["", "## Residuals and timing", "",
              "| Model | Parameters | Trainable | Gamma T/A/V | Best clean epoch | Best robust epoch | Epochs | Train seconds | Mean validation seconds |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for name in ("P0", "P1", "P2"):
        d = result[name]
        if name == "P0":
            lines.append(f"| P0 | {d['parameter_count']} | — | — | — | — | — | — | — |")
        else:
            g = d["gamma"]
            lines.append(f"| {name} | {d['parameter_count']} | {d['trainable_parameter_count']} | " +
                         "/".join(f"{g[m]:+.6f}" for m in MODALITIES) +
                         f" | {d['best_clean_epoch']} | {d['best_robust_epoch']} | {d['epochs_run']} | " +
                         f"{d['training_seconds']:.2f} | {d['mean_validation_seconds']:.2f} |")
    lines += ["", "## P2 attention concentration", "",
              "| Condition | Modality | Mean entropy | Mean normalized entropy (L>1) | Mean max weight | P95 max weight | Fraction max≥0.9 (L>1) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for condition, modalities in result["P2"]["attention_diagnostics"].items():
        for modality, d in modalities.items():
            lines.append(f"| {condition} | {modality} | {d['mean_entropy']:.4f} | "
                         f"{d['mean_normalized_entropy_length_gt_one']:.4f} | "
                         f"{d['mean_max_weight']:.4f} | {d['p95_max_weight']:.4f} | "
                         f"{d['fraction_max_weight_ge_0_9_length_gt_one']:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "b5_pooling.yaml")
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({name: {"robust_score": result[name]["validation"]["robust_score"],
                             "best_epoch": result[name].get("best_robust_epoch")}
                      for name in ("P0", "P1", "P2")}), flush=True)


if __name__ == "__main__":
    main()
