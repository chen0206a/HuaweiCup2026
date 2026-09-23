"""Train the B4' reliability router using the frozen B0/B2 validation protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.block_mask import augment_train_batch, predictor_inputs
from src.data.dataset import MODALITIES
from src.data.preprocess import build_datasets_and_loaders
from src.evaluation.missing_benchmark import evaluate_benchmark, mask_scenario, scenarios, summary
from src.models.baseline import B0Baseline, multitask_loss
from src.models.reliability_gate import B4ReliabilityGate
from src.training.train_b1 import seed_everything
from src.utils.metrics import compute_metrics, validation_selection_score

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "outputs" / "metrics"
CHECKPOINTS = ROOT / "outputs" / "checkpoints"
BENCHMARK_SHA256 = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"
SCORE_FIELDS = ("accuracy", "macro_f1", "mae", "pearson", "selection_score")


@dataclass
class EncodedChunk:
    scene_ids: tuple[str, ...]
    batch_size: int
    embeddings: torch.Tensor
    statistics: torch.Tensor
    cls_true: list[int]
    reg_true: list[float]
    vision_zero: list[bool]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_benchmark(cfg: dict) -> None:
    path = Path(cfg["benchmark"]["definition"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    definition = json.loads(path.read_text(encoding="utf-8"))
    if (digest != BENCHMARK_SHA256 or digest != cfg["benchmark"]["sha256"] or
            definition["seed"] != 20260923 or
            definition["scenarios"] != [scene.as_dict() for scene in scenarios()]):
        raise RuntimeError("frozen B2 benchmark definition changed")


def load_data(cfg: dict):
    if cfg["preprocessing"]["normalization"] != "none":
        raise ValueError("B4' must use normalization=none")
    return build_datasets_and_loaders(
        cfg["data"]["pkl_path"], normalization="none",
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        seed=int(cfg["training"]["seed"]), include_test=False,
    )


def initialize_from_b0(cfg: dict, device: torch.device) -> tuple[B4ReliabilityGate, B0Baseline, dict]:
    source = torch.load(cfg["training"]["init_checkpoint"], map_location="cpu", weights_only=False)
    if int(source["config"]["training"]["seed"]) != 42:
        raise RuntimeError("expected seed-42 B0-WCE checkpoint")
    b0_cfg = source["config"]["model"]
    if any(cfg["model"][key] != b0_cfg[key] for key in ("hidden_dim", "fusion_dim", "dropout")):
        raise RuntimeError("B4' B0 model configuration differs from the source checkpoint")
    b0 = B0Baseline(**b0_cfg).to(device).eval()
    b0.load_state_dict(source["model_state_dict"])
    model = B4ReliabilityGate(**cfg["model"]).to(device)
    mismatch = model.load_state_dict(source["model_state_dict"], strict=False)
    if mismatch.unexpected_keys or not mismatch.missing_keys or not all(
            key.startswith("router.") for key in mismatch.missing_keys):
        raise RuntimeError(f"B0 initialization mismatch: {mismatch}")
    model.freeze_b0()
    return model, b0, source


@torch.no_grad()
def identity_check(model: B4ReliabilityGate, b0: B0Baseline, valid_loader,
                   device: torch.device) -> dict:
    model.eval()
    b0.eval()
    maximum = {"classification_logits": 0.0, "regression": 0.0, "gates_minus_one": 0.0}
    exact = True
    count = 0
    for batch in valid_loader:
        moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        inputs = predictor_inputs(moved)
        expected = b0(inputs)
        actual = model(inputs)
        for key in ("classification_logits", "regression"):
            maximum[key] = max(maximum[key], float((expected[key] - actual[key]).abs().max().item()))
            exact = exact and torch.equal(expected[key], actual[key])
        maximum["gates_minus_one"] = max(maximum["gates_minus_one"],
                                           float((actual["gates"] - 1).abs().max().item()))
        count += len(moved["cls_label"])
    return {"passed": exact and maximum["gates_minus_one"] == 0.0,
            "exact_prediction_match": exact, "max_absolute_error": maximum,
            "valid_samples": count}


@torch.no_grad()
def make_validation_cache(model: B4ReliabilityGate, valid_loader,
                          device: torch.device) -> list[EncodedChunk]:
    """Encode frozen B0 pooled embeddings and observable statistics once."""
    model.eval()
    benchmark = scenarios()
    cache = []
    for batch in valid_loader:
        moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        batch_size = len(moved["cls_label"])
        truth_cls = moved["cls_label"].cpu().tolist()
        truth_reg = moved["reg_label"].cpu().tolist()
        zero_flags = moved["vision_all_zero"].cpu().tolist()
        embeddings, stats = model.encode_observation(predictor_inputs(moved))
        cache.append(EncodedChunk(("clean",), batch_size, embeddings, stats,
                                  truth_cls, truth_reg, zero_flags))
        for left in range(0, len(benchmark), 6):
            chunk = benchmark[left:left + 6]
            masked = [mask_scenario(moved, scene)[0] for scene in chunk]
            predictor = {key: torch.cat([part[key] for part in masked], dim=0)
                         for key in (*MODALITIES, "padding_mask")}
            embeddings, stats = model.encode_observation(predictor)
            cache.append(EncodedChunk(tuple(scene.scenario_id for scene in chunk),
                                      batch_size, embeddings, stats,
                                      truth_cls, truth_reg, zero_flags))
    return cache


@torch.no_grad()
def evaluate_cached(model: B4ReliabilityGate, cache: list[EncodedChunk],
                    *, collect_gates: bool = False) -> tuple[list[dict], dict[str, np.ndarray] | None]:
    """Use only router and frozen B0 fusion/heads during epoch validation."""
    model.eval()
    keys = ("clean",) + tuple(scene.scenario_id for scene in scenarios())
    collected = {key: {"cls_true": [], "cls_pred": [], "reg_true": [],
                       "reg_pred": [], "vision_zero": []} for key in keys}
    gate_parts = {key: [] for key in keys} if collect_gates else None
    for chunk in cache:
        outputs = model.predict_encoded(chunk.embeddings, chunk.statistics)
        for index, scene_id in enumerate(chunk.scene_ids):
            left, right = index * chunk.batch_size, (index + 1) * chunk.batch_size
            target = collected[scene_id]
            target["cls_true"].extend(chunk.cls_true)
            target["cls_pred"].extend(outputs["classification_logits"][left:right].argmax(-1).cpu().tolist())
            target["reg_true"].extend(chunk.reg_true)
            target["reg_pred"].extend(outputs["regression"][left:right].cpu().tolist())
            target["vision_zero"].extend(chunk.vision_zero)
            if gate_parts is not None:
                gate_parts[scene_id].append(outputs["gates"][left:right].cpu().numpy())
    rows = []
    for scene in (None, *scenarios()):
        key = "clean" if scene is None else scene.scenario_id
        data = collected[key]
        metrics = compute_metrics(data["cls_true"], data["cls_pred"],
                                  data["reg_true"], data["reg_pred"])
        indices = np.flatnonzero(data["vision_zero"])
        subset = compute_metrics(
            np.asarray(data["cls_true"])[indices], np.asarray(data["cls_pred"])[indices],
            np.asarray(data["reg_true"])[indices], np.asarray(data["reg_pred"])[indices],
        ) if indices.size else None
        rows.append({"scenario_id": key,
                     "modalities": "none" if scene is None else "+".join(scene.modalities),
                     "rho": 0.0 if scene is None else scene.rho,
                     "location": "clean" if scene is None else scene.location,
                     "sample_count": len(data["cls_true"]),
                     "selection_score": validation_selection_score(metrics),
                     "vision_all_zero_count": int(indices.size),
                     "vision_all_zero_metrics": subset, **metrics})
    clean = rows[0]
    for row in rows:
        for metric in SCORE_FIELDS:
            row[f"delta_{metric}"] = row[metric] - clean[metric]
    gate_arrays = ({key: np.concatenate(parts, axis=0) for key, parts in gate_parts.items()}
                   if gate_parts is not None else None)
    return rows, gate_arrays


def compare_cached_normal(cached: list[dict], normal: list[dict]) -> dict:
    if len(cached) != 55 or len(normal) != 55:
        raise RuntimeError("validation did not contain clean plus 54 scenes")
    fields = ("scenario_id", "modalities", "rho", "location", "sample_count",
              "selection_score", "vision_all_zero_count", "vision_all_zero_metrics",
              "accuracy", "macro_f1", "mae", "pearson", "per_class", "confusion_matrix",
              *(f"delta_{key}" for key in SCORE_FIELDS))
    mismatches = [(i, key) for i in range(55) for key in fields
                  if cached[i][key] != normal[i][key]]
    return {"passed": not mismatches, "compared_scenarios": 55,
            "compared_fields": list(fields), "mismatch_count": len(mismatches),
            "mismatches": mismatches[:20]}


def gate_distribution(values: np.ndarray) -> dict:
    result = {}
    for index, modality in enumerate(MODALITIES):
        column = values[:, index]
        result[modality] = {
            "mean": float(column.mean()),
            "sample_std": float(column.std(ddof=1)),
            "quantiles": {str(q): float(np.quantile(column, q))
                          for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "count": int(len(column)),
        }
    return result


def gate_analysis(gates: dict[str, np.ndarray], zero_flags: np.ndarray) -> dict:
    scenes = scenarios()
    groups = {"clean": ("clean",),
              "mean_missing": tuple(s.scenario_id for s in scenes)}
    for modality in MODALITIES:
        groups[f"{modality}_missing"] = tuple(
            s.scenario_id for s in scenes if s.modalities == (modality,))
    for pair in (("text", "audio"), ("text", "vision"), ("audio", "vision")):
        groups["+".join(pair)] = tuple(
            s.scenario_id for s in scenes if s.modalities == pair)
    distributions = {name: gate_distribution(np.concatenate([gates[key] for key in keys]))
                     for name, keys in groups.items()}
    by_rho = {}
    for modality in MODALITIES:
        by_rho[modality] = {}
        for rho in (0.1, 0.2, 0.3, 0.4, 0.5):
            selected = [s.scenario_id for s in scenes
                        if s.modalities == (modality,) and s.rho == rho]
            values = np.concatenate([gates[key] for key in selected])[:, MODALITIES.index(modality)]
            by_rho[modality][str(rho)] = {"mean_damaged_modality_gate": float(values.mean()),
                                          "sample_std": float(values.std(ddof=1))}
    subset = {"count": int(zero_flags.sum()),
              "clean": gate_distribution(gates["clean"][zero_flags]),
              "mean_missing": gate_distribution(np.concatenate([
                  gates[s.scenario_id][zero_flags] for s in scenes]))}
    return {"distributions": distributions, "damaged_gate_by_rho": by_rho,
            "vision_all_zero": subset}


def frozen_snapshot(model: B4ReliabilityGate) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if not key.startswith("router.")}


def frozen_equality(model: B4ReliabilityGate, reference: dict[str, torch.Tensor]) -> dict:
    current = model.state_dict()
    changed = [key for key, value in reference.items()
               if not torch.equal(current[key].detach().cpu(), value)]
    return {"passed": not changed, "tensor_count": len(reference), "changed_tensors": changed}


def render_report(result: dict) -> str:
    b0, current = result["b0_reference"], result["validation"]
    lines = ["# Q2 B4' reliability-aware dynamic fusion", "",
             "Validation uses the unchanged B2 benchmark (54 missing scenarios + clean); seed 42 only.",
             "All B0 prediction parameters are frozen. Gate input uses only current observations and padding.", "",
             "## Main comparison", "",
             "| Model | Condition | Accuracy | Macro-F1 | MAE | Pearson | Selection score | Robust score |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label, block in (("B0-WCE", b0), ("B4'", current)):
        for condition in ("clean", "mean_missing"):
            m = block[condition]
            lines.append(f"| {label} | {condition} | " + " | ".join(
                f"{m[key]:.4f}" for key in SCORE_FIELDS) +
                f" | {block['robust_score']:.4f} |")
    lines += ["", "## Missing scene groups"]
    for section in ("by_modality", "by_ratio", "by_location", "double_stress"):
        lines += ["", f"### {section}", "",
                  "| Group | Accuracy | Macro-F1 | MAE | Pearson | Score | Score Δ vs B0 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for group, m in current[section].items():
            delta = m["selection_score"] - b0[section][group]["selection_score"]
            lines.append(f"| {group} | " + " | ".join(f"{m[key]:.4f}" for key in SCORE_FIELDS) +
                         f" | {delta:+.4f} |")
    lines += ["", "## Gate distributions", "",
              "| Group | Gate | Mean | Sample std | P05 | P25 | P50 | P75 | P95 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group, modalities in result["gate_analysis"]["distributions"].items():
        for modality, stats in modalities.items():
            q = stats["quantiles"]
            lines.append(f"| {group} | {modality} | {stats['mean']:.4f} | {stats['sample_std']:.4f} | " +
                         " | ".join(f"{q[str(p)]:.4f}" for p in (0.05, 0.25, 0.5, 0.75, 0.95)) + " |")
    lines += ["", "## Damaged modality gate by rho", "",
              "| Modality | Rho | Mean gate | Sample std |", "|---|---:|---:|---:|"]
    for modality, rhos in result["gate_analysis"]["damaged_gate_by_rho"].items():
        for rho, stats in rhos.items():
            lines.append(f"| {modality} | {rho} | {stats['mean_damaged_modality_gate']:.4f} | "
                         f"{stats['sample_std']:.4f} |")
    lines += ["", "## vision_all_zero subset", "",
              f"Count: {result['vision_all_zero']['count']}", "",
              "| Model | Condition | Accuracy | Macro-F1 | MAE | Pearson |",
              "|---|---|---:|---:|---:|---:|"]
    for label in ("B0-WCE", "B4'"):
        for condition in ("clean", "mean_missing"):
            m = result["vision_all_zero"][label][condition]
            lines.append(f"| {label} | {condition} | " +
                         " | ".join(f"{m[key]:.4f}" for key in SCORE_FIELDS[:4]) + " |")
    for condition in ("clean", "mean_missing"):
        lines += ["", f"### vision_all_zero gates: {condition}", "",
                  "| Gate | Mean | Sample std | P05 | P25 | P50 | P75 | P95 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for modality, stats in result["gate_analysis"]["vision_all_zero"][condition].items():
            q = stats["quantiles"]
            lines.append(f"| {modality} | {stats['mean']:.4f} | {stats['sample_std']:.4f} | " +
                         " | ".join(f"{q[str(p)]:.4f}" for p in (0.05, 0.25, 0.5, 0.75, 0.95)) + " |")
    timing = result["timing_seconds"]
    lines += ["", "## Verification and timing", "",
              f"Identity with B0: {result['identity_check']['passed']}; frozen tensor equality: "
              f"{result['frozen_equality']['passed']}; cached/full validation: "
              f"{result['cached_validation_equivalence']['passed']}.",
              f"Parameters: {result['parameter_count_total']:,} total, "
              f"{result['parameter_count_trainable']:,} trainable.",
              f"Epoch: {result['best_robust_epoch']} best robust, "
              f"{result['best_clean_epoch']} best clean, {result['epochs_run']} run.",
              f"Training {timing['training']:.2f}s; validation cache setup "
              f"{timing['cache_build']:.2f}s; total cached validation "
              f"{timing['cached_validation_total']:.2f}s; mean/epoch "
              f"{timing['cached_validation_mean_per_epoch']:.2f}s; normal verification "
              f"{timing['normal_validation_verification']:.2f}s."]
    return "\n".join(lines) + "\n"


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verify_benchmark(cfg)
    seed = int(cfg["training"]["seed"])
    if seed != 42 or int(cfg["training"]["max_epochs"]) != 50 or int(cfg["training"]["patience"]) != 10:
        raise ValueError("B4' screening requires seed42, max_epochs50, patience10")
    seed_everything(seed)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, loaders, _ = load_data(cfg)
    model, b0, source = initialize_from_b0(cfg, device)
    if any(p.requires_grad for name, p in model.named_parameters() if not name.startswith("router.")):
        raise RuntimeError("B0 parameter is trainable")
    if not all(p.requires_grad for p in model.router.parameters()):
        raise RuntimeError("router parameter is frozen")
    frozen_before = frozen_snapshot(model)
    if any(not torch.equal(source["model_state_dict"][key].cpu(), value)
           for key, value in frozen_before.items()):
        raise RuntimeError("initialized B0 tensors do not match source checkpoint")
    identity = identity_check(model, b0, loaders["valid"], device)
    if not identity["passed"]:
        raise RuntimeError(f"fresh B4' router is not B0 identity: {identity}")
    cache_started = time.perf_counter()
    cache = make_validation_cache(model, loaders["valid"], device)
    cache_seconds = time.perf_counter() - cache_started
    initial_cached, _ = evaluate_cached(model, cache)
    normal_initial = evaluate_benchmark(model, loaders["valid"], device)
    initial_equivalence = compare_cached_normal(initial_cached, normal_initial)
    if not initial_equivalence["passed"]:
        raise RuntimeError(f"initial cached validation differs from full forward: {initial_equivalence}")

    counts = np.bincount(datasets["train"].cls_labels, minlength=3)
    if np.any(counts == 0) or cfg["training"]["class_weighting"] != "balanced_train":
        raise ValueError("balanced WCE requires all three train classes")
    weights = torch.as_tensor(len(datasets["train"]) / (3.0 * counts),
                              dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.router.parameters(),
                                  lr=float(cfg["training"]["learning_rate"]),
                                  weight_decay=float(cfg["training"]["weight_decay"]))
    best_clean = best_robust = -float("inf")
    best_clean_epoch = best_robust_epoch = stale = 0
    history = []
    train_started = time.perf_counter()
    clean_path = CHECKPOINTS / cfg["outputs"]["best_clean_checkpoint"]
    robust_path = CHECKPOINTS / cfg["outputs"]["best_robust_checkpoint"]
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(cfg["training"]["max_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        masked_blocks = 0
        for batch in loaders["train"]:
            moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            masked, metadata = augment_train_batch(moved, rng)
            masked_blocks += len(metadata)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(predictor_inputs(masked))
            loss = multitask_loss(outputs, masked,
                                  lambda_reg=float(cfg["training"]["lambda_reg"]),
                                  class_weights=weights)["total"]
            loss.backward()
            if any(p.grad is not None for name, p in model.named_parameters()
                   if not name.startswith("router.")):
                raise RuntimeError("gradient reached the frozen B0 backbone")
            optimizer.step()
            batch_size = len(masked["cls_label"])
            seen += batch_size
            loss_sum += float(loss.item()) * batch_size
        valid_started = time.perf_counter()
        rows, _ = evaluate_cached(model, cache)
        validation_seconds = time.perf_counter() - valid_started
        metrics = summary(rows)
        clean_score, robust_score = metrics["clean"]["selection_score"], metrics["robust_score"]
        history.append({"epoch": epoch, "train_loss": loss_sum / seen,
                        "masked_blocks": masked_blocks, "validation_seconds": validation_seconds,
                        **metrics})
        checkpoint = {"model_state_dict": model.state_dict(), "config": cfg,
                      "epoch": epoch, "benchmark_seed": 20260923,
                      "benchmark_sha256": BENCHMARK_SHA256,
                      "train_class_counts": counts.tolist(), "class_weights": weights.cpu(),
                      "clean_score": clean_score, "robust_score": robust_score}
        if clean_score > best_clean:
            best_clean, best_clean_epoch = clean_score, epoch
            torch.save(checkpoint, clean_path)
        if robust_score > best_robust:
            best_robust, best_robust_epoch, stale = robust_score, epoch, 0
            torch.save(checkpoint, robust_path)
        else:
            stale += 1
        print(json.dumps({"epoch": epoch, "train_loss": loss_sum / seen,
                          "clean_score": clean_score, "missing_score": metrics["mean_missing"]["selection_score"],
                          "robust_score": robust_score, "validation_seconds": validation_seconds}), flush=True)
        if stale >= int(cfg["training"]["patience"]):
            break
    training_seconds = time.perf_counter() - train_started
    equality = frozen_equality(model, frozen_before)
    if not equality["passed"]:
        raise RuntimeError(f"B0 tensors changed during training: {equality}")
    checkpoint = torch.load(robust_path, map_location="cpu", weights_only=False)
    if checkpoint["benchmark_sha256"] != BENCHMARK_SHA256:
        raise RuntimeError("best robust checkpoint benchmark mismatch")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    checkpoint_equality = frozen_equality(model, frozen_before)
    if not checkpoint_equality["passed"]:
        raise RuntimeError("best robust checkpoint changed B0 tensors")
    # Round-trip into a fresh instance, using the same cached B0 embeddings.
    restored = B4ReliabilityGate(**cfg["model"]).to(device).eval()
    restored.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        first = cache[0]
        original = model.predict_encoded(first.embeddings, first.statistics)
        reloaded = restored.predict_encoded(first.embeddings, first.statistics)
        roundtrip = all(torch.equal(original[key], reloaded[key])
                        for key in ("classification_logits", "regression", "gates"))
    if not roundtrip:
        raise RuntimeError("checkpoint round trip changed predictions")
    rows, gates = evaluate_cached(model, cache, collect_gates=True)
    normal_started = time.perf_counter()
    normal_rows = evaluate_benchmark(model, loaders["valid"], device)
    normal_seconds = time.perf_counter() - normal_started
    final_equivalence = compare_cached_normal(rows, normal_rows)
    if not final_equivalence["passed"]:
        raise RuntimeError(f"trained cached validation differs from full forward: {final_equivalence}")
    metrics = summary(rows)
    baseline = json.loads((METRICS / "b2_inherent_summary.json").read_text(encoding="utf-8"))["B0-WCE"]["summary"]
    baseline_rows = json.loads((METRICS / "b2_b0_inherent_detail.json").read_text(encoding="utf-8"))
    zero_flags = np.concatenate([np.asarray(chunk.vision_zero, dtype=bool)
                                 for chunk in cache if chunk.scene_ids == ("clean",)])
    gate_info = gate_analysis(gates, zero_flags)
    result = {"model": cfg["run_name"], "training_seed": seed,
              "benchmark_seed": 20260923, "benchmark_sha256": BENCHMARK_SHA256,
              "validation": metrics, "b0_reference": baseline,
              "delta_vs_b0": {condition: {key: metrics[condition][key] - baseline[condition][key]
                                           for key in SCORE_FIELDS}
                              for condition in ("clean", "mean_missing")},
              "robust_delta_vs_b0": metrics["robust_score"] - baseline["robust_score"],
              "scenario_details": rows, "gate_analysis": gate_info,
              "vision_all_zero": {"count": int(zero_flags.sum()),
                                  "B0-WCE": {"clean": baseline_rows[0]["vision_all_zero_metrics"],
                                             "mean_missing": {key: float(np.mean([
                                                 scene["vision_all_zero_metrics"][key]
                                                 for scene in baseline_rows[1:]]))
                                                 for key in SCORE_FIELDS[:4]}},
                                  "B4'": {"clean": rows[0]["vision_all_zero_metrics"],
                                          "mean_missing": {key: float(np.mean([
                                              scene["vision_all_zero_metrics"][key] for scene in rows[1:]]))
                                              for key in SCORE_FIELDS[:4]}}},
              "identity_check": identity, "frozen_equality": equality,
              "checkpoint_frozen_equality": checkpoint_equality,
              "cached_validation_equivalence": {"initial": initial_equivalence,
                                                "final": final_equivalence,
                                                "passed": initial_equivalence["passed"] and final_equivalence["passed"]},
              "checkpoint_roundtrip_passed": roundtrip,
              "parameter_count_total": sum(p.numel() for p in model.parameters()),
              "parameter_count_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "best_clean_epoch": best_clean_epoch, "best_robust_epoch": best_robust_epoch,
              "epochs_run": len(history),
              "timing_seconds": {"training": training_seconds, "cache_build": cache_seconds,
                                 "cached_validation_total": float(sum(row["validation_seconds"] for row in history)),
                                 "cached_validation_mean_per_epoch": float(np.mean([
                                     row["validation_seconds"] for row in history])),
                                 "normal_validation_verification": normal_seconds},
              "checkpoints": {"best_clean": str(clean_path), "best_robust": str(robust_path)},
              "test_or_attachment3_used": False}
    write_json(METRICS / cfg["outputs"]["history"], history)
    write_json(METRICS / cfg["outputs"]["metrics"], result)
    (METRICS / cfg["outputs"]["report"]).write_text(render_report(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "b4_reliability_gate.yaml")
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({"best_robust_epoch": result["best_robust_epoch"],
                      "clean": result["validation"]["clean"],
                      "mean_missing": result["validation"]["mean_missing"],
                      "robust_score": result["validation"]["robust_score"],
                      "robust_delta_vs_b0": result["robust_delta_vs_b0"]}), flush=True)


if __name__ == "__main__":
    main()
