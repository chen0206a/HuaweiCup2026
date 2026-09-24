"""B5-F1 seed-42 low-rank fusion screening on the frozen B2 benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import MODALITIES
from src.data.preprocess import build_datasets_and_loaders
from src.evaluation.missing_benchmark import evaluate_benchmark, mask_scenario, scenarios, summary
from src.models.baseline import B0Baseline, multitask_loss
from src.models.fusion_interaction import B5FusionInteraction, PAIR_NAMES
from src.training.train_b1 import seed_everything
from src.utils.metrics import compute_metrics, validation_selection_score

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "outputs" / "metrics"
CHECKPOINTS = ROOT / "outputs" / "checkpoints"
BENCHMARK_SHA256 = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"
SCORE_FIELDS = ("accuracy", "macro_f1", "mae", "pearson", "selection_score")
PREDICTOR_KEYS = (*MODALITIES, "padding_mask")


@dataclass
class EncodedChunk:
    scene_ids: tuple[str, ...]
    batch_size: int
    representations: torch.Tensor
    z_b0: torch.Tensor
    cls_true: list[int]
    reg_true: list[float]
    vision_zero: list[bool]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_config(cfg: dict) -> None:
    train = cfg["training"]
    if (int(train["seed"]) != 42 or int(cfg["model"]["rank"]) != 16 or
            cfg["preprocessing"]["normalization"] != "none" or
            train["augmentation"] != "none" or
            train["class_weighting"] != "balanced_train" or
            train["freeze_b0_parameters"] is not True):
        raise ValueError("B5-F1 requires seed42/rank16/frozen B0/clean train/weighted CE")
    path = Path(cfg["benchmark"]["definition"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    definition = json.loads(path.read_text(encoding="utf-8"))
    if (digest != BENCHMARK_SHA256 or digest != cfg["benchmark"]["sha256"] or
            definition["seed"] != 20260923 or
            definition["scenarios"] != [s.as_dict() for s in scenarios()]):
        raise RuntimeError("frozen B2 benchmark definition changed")


def predictor(batch: dict) -> dict:
    return {key: batch[key] for key in PREDICTOR_KEYS}


@torch.no_grad()
def initialize(cfg: dict, device: torch.device) -> tuple[B5FusionInteraction, B0Baseline, dict]:
    source = torch.load(cfg["training"]["init_checkpoint"], map_location="cpu", weights_only=False)
    b0_config = dict(source["config"]["model"])
    if int(source["config"]["training"]["seed"]) != 42 or b0_config != {
            key: cfg["model"][key] for key in ("hidden_dim", "fusion_dim", "dropout")}:
        raise RuntimeError("source checkpoint is not seed-42 B0-WCE")
    b0 = B0Baseline(**b0_config).to(device).eval()
    b0.load_state_dict(source["model_state_dict"])
    model = B5FusionInteraction(**cfg["model"]).to(device)
    mismatch = model.load_state_dict(source["model_state_dict"], strict=False)
    if mismatch.unexpected_keys or not mismatch.missing_keys or any(
            not key.startswith(("interaction_projection.", "interaction_output."))
            for key in mismatch.missing_keys):
        raise RuntimeError(f"B0 initialization mismatch: {mismatch}")
    model.freeze_b0()
    return model, b0, source


@torch.no_grad()
def identity_check(model: B5FusionInteraction, b0: B0Baseline,
                   valid_loader, device: torch.device) -> dict:
    model.eval()
    b0.eval()
    exact = True
    maximum = {"classification_logits": 0.0, "regression": 0.0}
    count = 0
    for batch in valid_loader:
        x = {key: batch[key].to(device) for key in PREDICTOR_KEYS}
        expected, actual = b0(x), model(x)
        for key in maximum:
            maximum[key] = max(maximum[key], float((expected[key] - actual[key]).abs().max().item()))
            exact = exact and torch.equal(expected[key], actual[key])
        count += x["padding_mask"].shape[0]
    return {"passed": exact, "valid_count": count, "maximum_absolute_difference": maximum}


@torch.no_grad()
def build_train_cache(model: B5FusionInteraction, dataset, batch_size: int,
                      device: torch.device) -> TensorDataset:
    """Sequential source-dataset order, then the same seed-42 DataLoader shuffle."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    parts = {key: [] for key in ("representations", "z_b0", "cls_label", "reg_label")}
    for batch in loader:
        moved = {key: batch[key].to(device) for key in PREDICTOR_KEYS}
        representations, z_b0 = model.encode_b0(moved)
        parts["representations"].append(representations.cpu())
        parts["z_b0"].append(z_b0.cpu())
        parts["cls_label"].append(batch["cls_label"])
        parts["reg_label"].append(batch["reg_label"])
    return TensorDataset(*(torch.cat(parts[key], dim=0)
                           for key in ("representations", "z_b0", "cls_label", "reg_label")))


@torch.no_grad()
def build_validation_cache(model: B5FusionInteraction, loader,
                           device: torch.device) -> list[EncodedChunk]:
    """Encode B0 exactly once in the same batches and scene order as full validation."""
    model.eval()
    benchmark = scenarios()
    cache = []
    for batch in loader:
        moved = {key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                 for key, value in batch.items()}
        n = len(moved["cls_label"])
        labels = moved["cls_label"].cpu().tolist()
        regression = moved["reg_label"].cpu().tolist()
        flags = moved["vision_all_zero"].cpu().tolist()
        h, z = model.encode_b0(predictor(moved))
        cache.append(EncodedChunk(("clean",), n, h, z, labels, regression, flags))
        for left in range(0, len(benchmark), 6):
            scene_chunk = benchmark[left:left + 6]
            masked = [mask_scenario(moved, scene)[0] for scene in scene_chunk]
            current = {key: torch.cat([part[key] for part in masked], dim=0)
                       for key in PREDICTOR_KEYS}
            h, z = model.encode_b0(current)
            cache.append(EncodedChunk(tuple(scene.scenario_id for scene in scene_chunk),
                                      n, h, z, labels, regression, flags))
    return cache


@torch.no_grad()
def evaluate_cached(model: B5FusionInteraction, cache: list[EncodedChunk]) -> list[dict]:
    model.eval()
    keys = ("clean",) + tuple(scene.scenario_id for scene in scenarios())
    collected = {key: {"cls_true": [], "cls_pred": [], "reg_true": [],
                       "reg_pred": [], "vision_zero": []} for key in keys}
    for chunk in cache:
        prediction = model.predict_encoded(chunk.representations, chunk.z_b0)
        for i, scene_id in enumerate(chunk.scene_ids):
            lo, hi = i * chunk.batch_size, (i + 1) * chunk.batch_size
            data = collected[scene_id]
            data["cls_true"].extend(chunk.cls_true)
            data["cls_pred"].extend(prediction["classification_logits"][lo:hi].argmax(-1).cpu().tolist())
            data["reg_true"].extend(chunk.reg_true)
            data["reg_pred"].extend(prediction["regression"][lo:hi].cpu().tolist())
            data["vision_zero"].extend(chunk.vision_zero)
    rows = []
    for scene in (None, *scenarios()):
        key = "clean" if scene is None else scene.scenario_id
        data = collected[key]
        metrics = compute_metrics(data["cls_true"], data["cls_pred"],
                                  data["reg_true"], data["reg_pred"])
        flagged = np.flatnonzero(data["vision_zero"])
        subset = (compute_metrics(np.asarray(data["cls_true"])[flagged],
                                  np.asarray(data["cls_pred"])[flagged],
                                  np.asarray(data["reg_true"])[flagged],
                                  np.asarray(data["reg_pred"])[flagged])
                  if flagged.size else None)
        rows.append({"scenario_id": key,
                     "modalities": "none" if scene is None else "+".join(scene.modalities),
                     "rho": 0.0 if scene is None else scene.rho,
                     "location": "clean" if scene is None else scene.location,
                     "sample_count": len(data["cls_true"]),
                     "selection_score": validation_selection_score(metrics),
                     "vision_all_zero_count": int(flagged.size),
                     "vision_all_zero_metrics": subset, **metrics})
    clean = rows[0]
    for row in rows:
        for key in SCORE_FIELDS:
            row[f"delta_{key}"] = row[key] - clean[key]
    return rows


def exact_rows(cached: list[dict], full: list[dict]) -> dict:
    if len(cached) != 55 or len(full) != 55:
        raise RuntimeError("validation must include clean plus 54 scenes")
    mismatches = [(i, key) for i in range(55) for key in full[i]
                  if cached[i].get(key) != full[i][key]]
    return {"passed": not mismatches, "rows": 55, "mismatch_count": len(mismatches),
            "mismatches": mismatches[:20]}


def frozen_equality(model: B5FusionInteraction, source_state: dict) -> dict:
    state = model.state_dict()
    changed = [key for key in source_state if not torch.equal(
        state[key].detach().cpu(), source_state[key].detach().cpu())]
    return {"passed": not changed, "tensor_count": len(source_state), "changed_tensors": changed}


def vision_subset(rows: list[dict]) -> dict:
    return {"count": rows[0]["vision_all_zero_count"],
            "clean": {key: rows[0]["vision_all_zero_metrics"][key] for key in SCORE_FIELDS[:4]},
            "mean_missing": {key: float(np.mean([
                row["vision_all_zero_metrics"][key] for row in rows[1:]]))
                for key in SCORE_FIELDS[:4]}}


@torch.no_grad()
def interaction_diagnostics(model: B5FusionInteraction,
                            cache: list[EncodedChunk]) -> dict:
    result = {}
    for condition in ("clean", "mean_missing"):
        ratios = []
        pair_norms = {name: [] for name in PAIR_NAMES}
        for chunk in cache:
            if ("clean" in chunk.scene_ids) != (condition == "clean"):
                continue
            prediction = model.predict_encoded(chunk.representations, chunk.z_b0)
            ratio = prediction["delta_z"].norm(dim=-1) / (
                prediction["z_b0"].norm(dim=-1) + 1e-8)
            ratios.append(ratio.cpu().numpy())
            contributions = model.pair_contributions(prediction["pairwise"])
            for name in PAIR_NAMES:
                pair_norms[name].append(contributions[name].norm(dim=-1).cpu().numpy())
        values = np.concatenate(ratios)
        result[condition] = {
            "count": int(values.size),
            "residual_norm_ratio": {"mean": float(values.mean()),
                                    "sample_std": float(values.std(ddof=1)),
                                    "p05": float(np.quantile(values, 0.05)),
                                    "p50": float(np.quantile(values, 0.5)),
                                    "p95": float(np.quantile(values, 0.95))},
            "mean_pair_delta_z_norm": {name: float(np.concatenate(chunks).mean())
                                       for name, chunks in pair_norms.items()},
        }
    return result


def render_report(result: dict) -> str:
    baseline, candidate = result["F0"]["validation"], result["F1"]["validation"]
    lines = ["# Q2 B5-F1 low-rank pairwise fusion screening", "",
             "Seed 42, original B0-WCE checkpoint; B0 prediction parameters frozen in eval mode.",
             "Clean train only; fixed B2 clean + 54 missing validation scenarios. No test or attachment3.", "",
             "| Model | Condition | Accuracy | Macro-F1 | MAE | Pearson | Score | Robust |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, block in (("F0", baseline), ("F1", candidate)):
        for condition in ("clean", "mean_missing"):
            m = block[condition]
            lines.append(f"| {name} | {condition} | " +
                         " | ".join(f"{m[key]:.4f}" for key in SCORE_FIELDS) +
                         f" | {block['robust_score']:.4f} |")
    lines += ["", f"F1 minus F0 robust score: {result['robust_delta_vs_f0']:+.6f}."]
    for section in ("by_modality", "by_ratio", "by_location", "double_stress"):
        lines += ["", f"## {section}", "",
                  "| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for name, block in (("F0", baseline), ("F1", candidate)):
            for group, values in block[section].items():
                lines.append(f"| {name} | {group} | " +
                             " | ".join(f"{values[key]:.4f}" for key in SCORE_FIELDS) + " |")
    lines += ["", "## vision_all_zero subset", "",
              "| Model | Condition | Accuracy | Macro-F1 | MAE | Pearson |",
              "|---|---|---:|---:|---:|---:|"]
    for name in ("F0", "F1"):
        for condition in ("clean", "mean_missing"):
            m = result[name]["vision_all_zero"][condition]
            lines.append(f"| {name} | {condition} | " +
                         " | ".join(f"{m[key]:.4f}" for key in SCORE_FIELDS[:4]) + " |")
    lines += ["", "## Interaction contribution", "",
              "| Condition | R mean | R sample std | R P05 | R P50 | R P95 | TA mean norm | TV mean norm | AV mean norm |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for condition, data in result["F1"]["interaction_diagnostics"].items():
        ratio, pair = data["residual_norm_ratio"], data["mean_pair_delta_z_norm"]
        lines.append(f"| {condition} | " +
                     " | ".join(f"{ratio[key]:.5f}" for key in ("mean", "sample_std", "p05", "p50", "p95")) +
                     " | " + " | ".join(f"{pair[key]:.5f}" for key in PAIR_NAMES) + " |")
    f1 = result["F1"]
    lines += ["", "## Verification and timing", "",
              f"Identity: {f1['initial_identity']['passed']}; frozen B0 tensors: "
              f"{f1['frozen_equality']['passed']} ({f1['frozen_equality']['tensor_count']}); "
              f"cached/full rows: {f1['cached_full_equality']['passed']}; "
              f"checkpoint round trip: {f1['checkpoint_roundtrip']}.",
              f"Parameters: {f1['parameter_count']} total, {f1['trainable_parameter_count']} trainable.",
              f"Best clean epoch {f1['best_clean_epoch']}; best robust epoch {f1['best_robust_epoch']}; "
              f"epochs run {f1['epochs_run']}.",
              f"Training {f1['timing_seconds']['training']:.2f}s; validation cache build "
              f"{f1['timing_seconds']['validation_cache_build']:.2f}s; cached validation "
              f"{f1['timing_seconds']['cached_validation_mean_per_epoch']:.2f}s/epoch; "
              f"full validation verification {f1['timing_seconds']['full_validation_verification']:.2f}s."]
    return "\n".join(lines) + "\n"


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verify_config(cfg)
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, loaders, _ = build_datasets_and_loaders(
        cfg["data"]["pkl_path"], normalization="none",
        batch_size=int(cfg["data"]["batch_size"]), num_workers=0,
        seed=42, include_test=False)
    model, base, source = initialize(cfg, device)
    frozen_before = {key: value.detach().cpu().clone()
                     for key, value in source["model_state_dict"].items()}
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != 12288:
        raise RuntimeError("unexpected F1 trainable parameter count")
    identity = identity_check(model, base, loaders["valid"], device)
    if not identity["passed"]:
        raise RuntimeError(f"initial F1 is not B0: {identity}")
    baseline_rows = json.loads((METRICS / "b2_b0_inherent_detail.json").read_text(encoding="utf-8"))
    baseline_summary = summary(baseline_rows)
    cache_started = time.perf_counter()
    train_data = build_train_cache(model, datasets["train"], int(cfg["data"]["batch_size"]), device)
    train_cache_seconds = time.perf_counter() - cache_started
    cache_started = time.perf_counter()
    validation_cache = build_validation_cache(model, loaders["valid"], device)
    validation_cache_seconds = time.perf_counter() - cache_started
    initial_cached = evaluate_cached(model, validation_cache)
    full_initial = evaluate_benchmark(model, loaders["valid"], device)
    initial_equality = exact_rows(initial_cached, full_initial)
    baseline_equality = exact_rows(full_initial, baseline_rows)
    if not initial_equality["passed"] or not baseline_equality["passed"]:
        raise RuntimeError(f"initial cache/full/B0 mismatch: {initial_equality}, {baseline_equality}")
    counts = np.bincount(datasets["train"].cls_labels, minlength=3)
    if np.any(counts == 0):
        raise RuntimeError("train split lacks a class")
    weights = torch.as_tensor(len(datasets["train"]) / (3 * counts),
                              dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_data, batch_size=int(cfg["data"]["batch_size"]),
                              shuffle=True, num_workers=0, generator=generator)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=float(cfg["training"]["learning_rate"]),
                                  weight_decay=float(cfg["training"]["weight_decay"]))
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    clean_path = CHECKPOINTS / cfg["outputs"]["best_clean_checkpoint"]
    robust_path = CHECKPOINTS / cfg["outputs"]["best_robust_checkpoint"]
    best_clean = best_robust = -float("inf")
    clean_epoch = robust_epoch = stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        if model.modality_projection.training or model.fusion.training or any(
                d.training for d in model.modules() if isinstance(d, torch.nn.Dropout)):
            raise RuntimeError("frozen B0 Dropout became active")
        seen = 0
        loss_sum = 0.0
        for h_cpu, z_cpu, cls_cpu, reg_cpu in train_loader:
            h, z = h_cpu.to(device), z_cpu.to(device)
            labels = {"cls_label": cls_cpu.to(device), "reg_label": reg_cpu.to(device)}
            optimizer.zero_grad(set_to_none=True)
            output = model.predict_encoded(h, z)
            loss = multitask_loss(output, labels,
                                  lambda_reg=float(cfg["training"]["lambda_reg"]),
                                  class_weights=weights)["total"]
            loss.backward()
            if any(p.grad is not None for name, p in model.named_parameters()
                   if not name.startswith(("interaction_projection.", "interaction_output."))):
                raise RuntimeError("gradient reached frozen B0")
            optimizer.step()
            n = len(cls_cpu)
            seen += n
            loss_sum += float(loss.item()) * n
        valid_started = time.perf_counter()
        rows = evaluate_cached(model, validation_cache)
        valid_seconds = time.perf_counter() - valid_started
        metrics = summary(rows)
        clean_score, robust_score = metrics["clean"]["selection_score"], metrics["robust_score"]
        history.append({"epoch": epoch, "train_loss": loss_sum / seen,
                        "validation_seconds": valid_seconds, **metrics})
        checkpoint = {"model_state_dict": model.state_dict(), "config": cfg,
                      "epoch": epoch, "clean_score": clean_score,
                      "robust_score": robust_score, "benchmark_sha256": BENCHMARK_SHA256,
                      "train_class_counts": counts.tolist(), "class_weights": weights.cpu()}
        if clean_score > best_clean:
            best_clean, clean_epoch = clean_score, epoch
            torch.save(checkpoint, clean_path)
        if robust_score > best_robust:
            best_robust, robust_epoch, stale = robust_score, epoch, 0
            torch.save(checkpoint, robust_path)
        else:
            stale += 1
        print(json.dumps({"epoch": epoch, "train_loss": loss_sum / seen,
                          "clean_score": clean_score, "robust_score": robust_score,
                          "validation_seconds": valid_seconds}), flush=True)
        if stale >= int(cfg["training"]["patience"]):
            break
    training_seconds = time.perf_counter() - started
    final_frozen = frozen_equality(model, frozen_before)
    if not final_frozen["passed"]:
        raise RuntimeError(f"F1 changed frozen B0: {final_frozen}")
    best_robust_checkpoint = torch.load(robust_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_robust_checkpoint["model_state_dict"])
    model.eval()
    checkpoint_frozen = frozen_equality(model, frozen_before)
    if not checkpoint_frozen["passed"]:
        raise RuntimeError("robust checkpoint changed B0")
    restored = B5FusionInteraction(**cfg["model"]).to(device).eval()
    restored.load_state_dict(best_robust_checkpoint["model_state_dict"])
    with torch.no_grad():
        first = validation_cache[0]
        expected = model.predict_encoded(first.representations, first.z_b0)
        actual = restored.predict_encoded(first.representations, first.z_b0)
        roundtrip = all(torch.equal(expected[key], actual[key])
                        for key in ("classification_logits", "regression", "delta_z"))
    if not roundtrip:
        raise RuntimeError("checkpoint round trip changed predictions")
    final_rows = evaluate_cached(model, validation_cache)
    full_started = time.perf_counter()
    full_rows = evaluate_benchmark(model, loaders["valid"], device)
    full_seconds = time.perf_counter() - full_started
    final_equality = exact_rows(final_rows, full_rows)
    if not final_equality["passed"]:
        raise RuntimeError(f"trained cache/full mismatch: {final_equality}")
    metrics = summary(final_rows)
    if metrics["robust_score"] != best_robust:
        raise RuntimeError("restored checkpoint robust score changed")
    diagnostics = interaction_diagnostics(model, validation_cache)
    clean_checkpoint = torch.load(clean_path, map_location="cpu", weights_only=False)
    clean_model = B5FusionInteraction(**cfg["model"]).to(device).eval()
    clean_model.load_state_dict(clean_checkpoint["model_state_dict"])
    clean_selected_rows = evaluate_cached(clean_model, validation_cache)
    clean_selected = summary(clean_selected_rows)
    result = {"training_seed": 42, "benchmark_seed": 20260923,
              "benchmark_sha256": BENCHMARK_SHA256, "normalization": "none",
              "training_augmentation": "none", "test_or_attachment3_used": False,
              "F0": {"checkpoint": str(cfg["training"]["init_checkpoint"]),
                     "parameter_count": sum(p.numel() for p in base.parameters()),
                     "validation": baseline_summary,
                     "vision_all_zero": vision_subset(baseline_rows)},
              "F1": {"parameter_count": sum(p.numel() for p in model.parameters()),
                     "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                     "initial_identity": identity, "frozen_equality": final_frozen,
                     "checkpoint_frozen_equality": checkpoint_frozen,
                     "checkpoint_roundtrip": roundtrip,
                     "cached_full_equality": {"initial": initial_equality,
                                              "final": final_equality,
                                              "passed": initial_equality["passed"] and final_equality["passed"]},
                     "baseline_equivalence": baseline_equality,
                     "best_clean_epoch": clean_epoch, "best_robust_epoch": robust_epoch,
                     "epochs_run": len(history), "validation": metrics,
                     "clean_selected_validation": clean_selected,
                     "vision_all_zero": vision_subset(final_rows),
                     "interaction_diagnostics": diagnostics,
                     "scenario_details": final_rows,
                     "timing_seconds": {"train_cache_build": train_cache_seconds,
                                        "validation_cache_build": validation_cache_seconds,
                                        "training": training_seconds,
                                        "cached_validation_total": float(sum(h["validation_seconds"] for h in history)),
                                        "cached_validation_mean_per_epoch": float(np.mean([
                                            h["validation_seconds"] for h in history])),
                                        "full_validation_verification": full_seconds},
                     "checkpoints": {"best_clean": str(clean_path), "best_robust": str(robust_path)}},
              "robust_delta_vs_f0": metrics["robust_score"] - baseline_summary["robust_score"]}
    write_json(METRICS / cfg["outputs"]["history"], history)
    write_json(METRICS / cfg["outputs"]["metrics"], result)
    (METRICS / cfg["outputs"]["report"]).write_text(render_report(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "b5_fusion.yaml")
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({"F0_robust": result["F0"]["validation"]["robust_score"],
                      "F1_robust": result["F1"]["validation"]["robust_score"],
                      "delta": result["robust_delta_vs_f0"],
                      "best_clean_epoch": result["F1"]["best_clean_epoch"],
                      "best_robust_epoch": result["F1"]["best_robust_epoch"]}), flush=True)


if __name__ == "__main__":
    main()
