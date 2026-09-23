"""B3 B0 latent reconstruction: equivalence gate, training, fixed validation."""
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

from src.data.block_mask import augment_train_batch, predictor_inputs
from src.data.dataset import MODALITIES
from src.data.preprocess import build_datasets_and_loaders
from src.evaluation.missing_benchmark import evaluate_benchmark, scenarios, summary
from src.models.baseline import B0Baseline, multitask_loss
from src.models.reconstruction import B3LatentReconstruction, reconstruction_losses
from src.training.run_b2 import flat_rows, write_csv, write_json
from src.training.train_b1 import seed_everything
from src.utils.metrics import validation_selection_score
from src.utils.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "outputs" / "metrics"
CHECKPOINTS = ROOT / "outputs" / "checkpoints"
BENCHMARK_SHA256 = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"


def verify_frozen(cfg: dict) -> None:
    path = Path(cfg["benchmark"]["definition"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    definition = json.loads(path.read_text(encoding="utf-8"))
    if (digest != BENCHMARK_SHA256 or digest != cfg["benchmark"]["sha256"] or
            definition["seed"] != 20260923 or
            definition["scenarios"] != [scenario.as_dict() for scenario in scenarios()]):
        raise RuntimeError("B2 validation benchmark changed")


def build_data(cfg: dict):
    if cfg["preprocessing"]["normalization"] != "none":
        raise ValueError("B3 requires normalization=none")
    return build_datasets_and_loaders(
        cfg["data"]["pkl_path"], normalization="none",
        batch_size=int(cfg["data"]["batch_size"]), num_workers=0,
        seed=int(cfg["training"]["seed"]), include_test=False,
    )


def initialize(cfg: dict, device: torch.device) -> tuple[B0Baseline, B3LatentReconstruction, dict]:
    checkpoint = torch.load(cfg["training"]["init_checkpoint"],
                            map_location="cpu", weights_only=False)
    if int(checkpoint["config"]["training"]["seed"]) != 42:
        raise RuntimeError("B3 must initialize from B0-WCE training seed 42")
    b0_cfg = checkpoint["config"]["model"]
    for key in ("hidden_dim", "fusion_dim", "dropout"):
        if cfg["model"][key] != b0_cfg[key]:
            raise RuntimeError(f"B0 backbone config changed: {key}")
    baseline = B0Baseline(**b0_cfg).to(device).eval()
    baseline.load_state_dict(checkpoint["model_state_dict"])
    model = B3LatentReconstruction(**cfg["model"]).to(device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatible.unexpected_keys or not incompatible.missing_keys or not all(
            key.startswith("reconstructors.") for key in incompatible.missing_keys):
        raise RuntimeError(f"B0 weight initialization mismatch: {incompatible}")
    return baseline, model, checkpoint


@torch.no_grad()
def equivalence(cfg: dict, *, max_batches: int | None = None) -> dict:
    """Actual seed-42 B0 checkpoint on complete valid inputs, no reconstruction."""
    verify_frozen(cfg)
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, loaders, _ = build_data(cfg)
    baseline, model, checkpoint = initialize(cfg, device)
    model.eval()
    maximum = {"classification_logits": 0.0, "regression": 0.0}
    count = 0
    for index, batch in enumerate(loaders["valid"]):
        if max_batches is not None and index >= max_batches:
            break
        moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        inputs = predictor_inputs(moved)
        original = baseline(inputs)
        adapted = model(inputs, reconstruction_enabled=False)
        for key in maximum:
            maximum[key] = max(maximum[key], float((original[key] - adapted[key]).abs().max().item()))
        count += int(moved["padding_mask"].shape[0])
    tolerance = 1e-5
    result = {"checkpoint": cfg["training"]["init_checkpoint"],
              "checkpoint_seed": int(checkpoint["config"]["training"]["seed"]),
              "valid_samples_checked": count, "max_absolute_error": maximum,
              "tolerance": tolerance, "passed": max(maximum.values()) <= tolerance}
    if not result["passed"]:
        raise RuntimeError(f"B0 affine/mean equivalence failed: {result}")
    write_json(METRICS / "b3_b0_equivalence.json", result)
    return result


def train(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verify_frozen(cfg)
    gate = equivalence(cfg)  # full validation; abort before training if not equivalent
    seed = int(cfg["training"]["seed"])
    if seed != 42:
        raise ValueError("B3 first run is frozen at training seed 42")
    seed_everything(seed)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, loaders, _ = build_data(cfg)
    _, model, _ = initialize(cfg, device)
    counts = np.bincount(datasets["train"].cls_labels, minlength=3)
    if np.any(counts == 0):
        raise ValueError("missing class in train split")
    weights = torch.as_tensor(len(datasets["train"]) / (3.0 * counts),
                              dtype=torch.float32, device=device)
    tcfg = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["learning_rate"]),
                                  weight_decay=float(tcfg["weight_decay"]))
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    clean_path = CHECKPOINTS / cfg["outputs"]["best_clean_checkpoint"]
    robust_path = CHECKPOINTS / cfg["outputs"]["best_robust_checkpoint"]
    best_clean = best_robust = -float("inf")
    best_clean_epoch = best_robust_epoch = stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(tcfg["epochs"]) + 1):
        model.train()
        totals = {key: 0.0 for key in ("total", "task", "reconstruction", "identity")}
        seen = 0
        for batch in loaders["train"]:
            full = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()}
            masked, metadata = augment_train_batch(full, rng)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(predictor_inputs(masked), return_aux=True)
            task = multitask_loss(outputs, masked, lambda_reg=float(tcfg["lambda_reg"]),
                                  class_weights=weights)["total"]
            rec = reconstruction_losses(model, outputs, full, masked)
            total = task + float(tcfg["lambda_rec"]) * rec["reconstruction"] + float(tcfg["lambda_id"]) * rec["identity"]
            total.backward()
            optimizer.step()
            n = int(masked["cls_label"].shape[0])
            seen += n
            for key, value in (("total", total), ("task", task),
                               ("reconstruction", rec["reconstruction"]),
                               ("identity", rec["identity"])):
                totals[key] += float(value.item()) * n
        rows = evaluate_benchmark(model, loaders["valid"], device)
        valid = summary(rows)
        clean_score = valid["clean"]["selection_score"]
        robust_score = valid["robust_score"]
        item = {"epoch": epoch,
                "train_loss": {key: value / seen for key, value in totals.items()},
                "clean": valid["clean"], "mean_missing": valid["mean_missing"],
                "robust_score": robust_score}
        history.append(item)
        checkpoint = {"model_state_dict": model.state_dict(), "config": cfg,
                      "init_checkpoint": cfg["training"]["init_checkpoint"],
                      "benchmark_sha256": BENCHMARK_SHA256,
                      "train_class_counts": counts.tolist(), "class_weights": weights.cpu(),
                      "epoch": epoch, "clean_score": clean_score, "robust_score": robust_score}
        if clean_score > best_clean:
            best_clean, best_clean_epoch = clean_score, epoch
            torch.save(checkpoint, clean_path)
        if robust_score > best_robust:
            best_robust, best_robust_epoch, stale = robust_score, epoch, 0
            torch.save(checkpoint, robust_path)
        else:
            stale += 1
        print(json.dumps({"model": cfg["run_name"], **item}), flush=True)
        if stale >= int(tcfg["patience"]):
            break
    seconds = time.perf_counter() - started
    write_json(METRICS / cfg["outputs"]["history_filename"], history)
    result = {"model": cfg["run_name"], "training_seed": seed,
              "benchmark_seed": 20260923, "benchmark_sha256": BENCHMARK_SHA256,
              "b0_equivalence": gate, "normalization": "none",
              "train_class_counts": counts.tolist(), "class_weights": weights.cpu().tolist(),
              "best_clean_epoch": best_clean_epoch, "best_robust_epoch": best_robust_epoch,
              "best_clean_score": best_clean, "best_robust_score": best_robust,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "training_seconds": seconds, "train_count": len(datasets["train"]),
              "valid_count": len(datasets["valid"]), "test_role": "never constructed/evaluated"}
    write_json(METRICS / cfg["outputs"]["metrics_filename"], result)
    return result


def compare(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verify_frozen(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, loaders, _ = build_data(cfg)
    state = torch.load(CHECKPOINTS / cfg["outputs"]["best_robust_checkpoint"],
                       map_location="cpu", weights_only=False)
    model = B3LatentReconstruction(**cfg["model"]).to(device).eval()
    model.load_state_dict(state["model_state_dict"])
    rows = evaluate_benchmark(model, loaders["valid"], device)
    own = summary(rows)
    previous = json.loads((METRICS / "b2_summary.json").read_text(encoding="utf-8"))
    selected = {name: previous[name]["summary"] for name in
                ("B0-WCE", "B2-B0-BlockMask")}
    selected[cfg["run_name"]] = own
    diagnostics = diagnose_reconstruction(model, loaders["valid"], device)
    clean_ablation = clean_reconstruction_ablation(model, loaders["valid"], device)
    write_json(METRICS / "b3_b0_detail.json", rows)
    write_json(METRICS / "b3_reconstruction_diagnostics.json", diagnostics)
    write_csv(METRICS / "b3_missing_scenarios.csv", flat_rows(cfg["run_name"], rows))
    base_details = {
        "B0-WCE": json.loads((METRICS / "b2_b0_inherent_detail.json").read_text()),
        "B2-B0-BlockMask": json.loads((METRICS / "b2_b0_blockmask_detail.json").read_text()),
        cfg["run_name"]: rows,
    }
    zero_subset = {}
    for name, detail in base_details.items():
        subset = [row["vision_all_zero_metrics"] for row in detail]
        zero_subset[name] = {
            "count": detail[0]["vision_all_zero_count"],
            "clean": subset[0],
            "mean_missing": {metric: float(np.mean([x[metric] for x in subset[1:]]))
                             for metric in ("accuracy", "macro_f1", "mae", "pearson")},
        }
    result = {"checkpoint": cfg["outputs"]["best_robust_checkpoint"],
              "benchmark_sha256": BENCHMARK_SHA256,
              "comparison": selected, "vision_all_zero": zero_subset,
              "diagnostics": diagnostics,
              "clean_reconstruction_ablation": clean_ablation}
    write_json(METRICS / "b3_summary.json", result)
    make_report(result)
    return result


@torch.no_grad()
def clean_reconstruction_ablation(model, loader, device) -> dict:
    """Same trained checkpoint, clean inputs, reconstruction on versus off."""
    model.eval()
    true_cls, true_reg, flags = [], [], []
    predictions = {status: {"cls": [], "reg": []} for status in ("enabled", "disabled")}
    for batch in loader:
        moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        true_cls.extend(moved["cls_label"].cpu().tolist())
        true_reg.extend(moved["reg_label"].cpu().tolist())
        flags.extend(moved["vision_all_zero"].cpu().tolist())
        for status, enabled in (("enabled", True), ("disabled", False)):
            outputs = model(predictor_inputs(moved), reconstruction_enabled=enabled)
            predictions[status]["cls"].extend(outputs["classification_logits"].argmax(-1).cpu().tolist())
            predictions[status]["reg"].extend(outputs["regression"].cpu().tolist())
    subset = np.flatnonzero(flags)
    result = {}
    for status, prediction in predictions.items():
        result[status] = {
            "all": compute_metrics(true_cls, prediction["cls"], true_reg, prediction["reg"]),
            "vision_all_zero": compute_metrics(
                np.asarray(true_cls)[subset], np.asarray(prediction["cls"])[subset],
                np.asarray(true_reg)[subset], np.asarray(prediction["reg"])[subset]),
        }
        result[status]["all"]["selection_score"] = validation_selection_score(result[status]["all"])
    result["classification_changes"] = int(np.sum(
        np.asarray(predictions["enabled"]["cls"]) != np.asarray(predictions["disabled"]["cls"])))
    result["mean_absolute_regression_prediction_change"] = float(np.mean(np.abs(
        np.asarray(predictions["enabled"]["reg"]) - np.asarray(predictions["disabled"]["reg"]))))
    return result


@torch.no_grad()
def diagnose_reconstruction(model, loader, device) -> dict:
    """Reconstruction quality and trigger accounting on frozen validation scenes."""
    model.eval()
    all_scenarios = (None, *scenarios())
    scene_results = []
    for scenario in all_scenarios:
        counts = {m: {"valid": 0, "zero_trigger": 0, "synthetic_missing": 0,
                      "trigger_missing_overlap": 0, "native_extra_trigger": 0,
                      "rec_error_sum": 0.0, "cosine_sum": 0.0, "rec_count": 0}
                  for m in MODALITIES}
        for batch in loader:
            full = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()}
            if scenario is None:
                masked = full
            else:
                from src.evaluation.missing_benchmark import mask_scenario
                masked, _ = mask_scenario(full, scenario)
            outputs = model(predictor_inputs(masked), return_aux=True)
            targets = model.full_targets(full)
            pad = full["padding_mask"]
            for index, modality in enumerate(MODALITIES):
                stats = counts[modality]
                trigger = outputs["zero_trigger"][modality]
                missing = pad & ~masked["availability_mask"][:, index]
                native = full["native_zero_mask"][:, index]
                stats["valid"] += int(pad.sum().item())
                stats["zero_trigger"] += int(trigger.sum().item())
                stats["synthetic_missing"] += int(missing.sum().item())
                stats["trigger_missing_overlap"] += int((trigger & missing).sum().item())
                stats["native_extra_trigger"] += int((trigger & native & ~missing).sum().item())
                if missing.any():
                    from torch.nn import functional as F
                    estimate = outputs["reconstructed"][modality][missing]
                    target = targets[modality][missing]
                    error = F.smooth_l1_loss(estimate, target, reduction="none").mean(-1)
                    cosine = F.cosine_similarity(estimate, target, dim=-1, eps=1e-8)
                    stats["rec_error_sum"] += float(error.sum().item())
                    stats["cosine_sum"] += float(cosine.sum().item())
                    stats["rec_count"] += int(missing.sum().item())
        for modality in MODALITIES:
            stats = counts[modality]
            stats["zero_trigger_ratio"] = stats["zero_trigger"] / stats["valid"]
            stats["reconstruction_smooth_l1"] = (stats["rec_error_sum"] / stats["rec_count"]
                                                  if stats["rec_count"] else None)
            stats["reconstruction_cosine"] = (stats["cosine_sum"] / stats["rec_count"]
                                               if stats["rec_count"] else None)
        scene_results.append({"scenario_id": "clean" if scenario is None else scenario.scenario_id,
                              "rho": 0.0 if scenario is None else scenario.rho,
                              "modalities": "none" if scenario is None else "+".join(scenario.modalities),
                              "location": "clean" if scenario is None else scenario.location,
                              "per_modality": counts})

    def group(records: list[dict], modality: str) -> dict:
        gathered = [record["per_modality"][modality] for record in records]
        n = sum(record["rec_count"] for record in gathered)
        return {"masked_position_count": n,
                "smooth_l1": sum(record["rec_error_sum"] for record in gathered) / n if n else None,
                "cosine": sum(record["cosine_sum"] for record in gathered) / n if n else None}

    missing_scenes = scene_results[1:]
    totals = {m: group(missing_scenes, m) for m in MODALITIES}
    by_rho = {m: {str(rho): group([r for r in missing_scenes if
                                  r["modalities"] == m and r["rho"] == rho], m)
                  for rho in (0.1, 0.2, 0.3, 0.4, 0.5)} for m in MODALITIES}
    aggregate = {}
    for label, records in (("clean", scene_results[:1]), ("missing_benchmark", missing_scenes)):
        valid = sum(r["per_modality"][m]["valid"] for r in records for m in MODALITIES)
        trigger = sum(r["per_modality"][m]["zero_trigger"] for r in records for m in MODALITIES)
        synthetic = sum(r["per_modality"][m]["synthetic_missing"] for r in records for m in MODALITIES)
        overlap = sum(r["per_modality"][m]["trigger_missing_overlap"] for r in records for m in MODALITIES)
        extra = sum(r["per_modality"][m]["native_extra_trigger"] for r in records for m in MODALITIES)
        aggregate[label] = {"valid_modality_timesteps": valid, "zero_trigger_count": trigger,
                            "zero_trigger_ratio": trigger / valid,
                            "synthetic_missing_count": synthetic,
                            "trigger_missing_overlap_count": overlap,
                            "trigger_missing_recall": overlap / synthetic if synthetic else None,
                            "native_zero_extra_trigger_count": extra}
    return {"aggregate": aggregate, "by_modality": totals,
            "error_vs_rho_single_modality": by_rho,
            "per_scenario": scene_results}


def make_report(result: dict) -> None:
    comparison = result["comparison"]
    lines = ["# Q2 B3 frozen validation comparison", "",
             "B0-WCE, B2-B0 and B3-B0 all use training seed 42. Only B3 best robust checkpoint is compared.",
             "Benchmark definition and SHA-256 are unchanged; test and attachment3 unused.", "",
             "| Model | Clean Acc | Clean F1 | Clean MAE | Clean r | Missing Acc | Missing F1 | Missing MAE | Missing r | Robust score |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, data in comparison.items():
        clean, missing = data["clean"], data["mean_missing"]
        values = [clean[x] for x in ("accuracy", "macro_f1", "mae", "pearson")]
        values += [missing[x] for x in ("accuracy", "macro_f1", "mae", "pearson")]
        values += [data["robust_score"]]
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    for section in ("by_ratio", "by_location", "by_modality", "double_stress"):
        lines += ["", f"## {section}", "", "| Model | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for name, data in comparison.items():
            for group, values in data[section].items():
                lines.append(f"| {name} | {group} | " + " | ".join(
                    f"{values[x]:.4f}" for x in ("accuracy", "macro_f1", "mae", "pearson", "selection_score")) + " |")
    lines += ["", "## Reconstruction diagnosis", "",
              "| Modality | Masked positions | SmoothL1 | Cosine |",
              "|---|---:|---:|---:|"]
    for modality, values in result["diagnostics"]["by_modality"].items():
        lines.append(f"| {modality} | {values['masked_position_count']} | {values['smooth_l1']:.4f} | {values['cosine']:.4f} |")
    lines += ["", "Detailed trigger counts and error by rho are in `b3_reconstruction_diagnostics.json`.",
              "The 15 vision_all_zero validation samples are reported separately in `b3_summary.json`; no selection uses this subset.",
              "", "## Same-checkpoint clean reconstruction ablation", "",
              "This diagnostic disables reconstruction only at inference; it is not a checkpoint selection or training variant.",
              "| State | Accuracy | Macro-F1 | MAE | Pearson | Score |",
              "|---|---:|---:|---:|---:|---:|"]
    for status, value in result["clean_reconstruction_ablation"].items():
        if status not in ("enabled", "disabled"):
            continue
        metric = value["all"]
        lines.append(f"| {status} | " + " | ".join(
            f"{metric[key]:.4f}" for key in ("accuracy", "macro_f1", "mae", "pearson", "selection_score")) + " |")
    (METRICS / "b3_model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("equivalence", "train", "compare"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "b3_b0_reconstruction.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.phase == "equivalence":
        print(json.dumps(equivalence(cfg)), flush=True)
    elif args.phase == "train":
        print(json.dumps(train(args.config)), flush=True)
    else:
        result = compare(args.config)
        print(json.dumps({"comparison": result["comparison"],
                          "trigger": result["diagnostics"]["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()
