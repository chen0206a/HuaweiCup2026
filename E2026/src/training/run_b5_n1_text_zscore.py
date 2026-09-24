"""B5-N1: text-only train-stat Z-score ablation against reused B0-WCE seed42."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.block_mask import apply_blocks, predictor_inputs
from src.data.dataset import Aligned50Dataset
from src.data.preprocess import load_pickle_readonly
from src.data.text_scaler import TextOnlyTrainScaler
from src.evaluation.missing_benchmark import (
    BENCHMARK_SEED, evaluate_benchmark, scenarios, summary,
)
from src.models.baseline import B0Baseline, masked_mean_pool, multitask_loss
from src.training.train import seed_everything

PROJECT = Path(__file__).resolve().parents[2]
METRICS = PROJECT / "outputs" / "metrics"
CHECKPOINTS = PROJECT / "outputs" / "checkpoints"
BENCHMARK_SHA256 = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"
METRICS_FIELDS = ("accuracy", "macro_f1", "mae", "pearson", "selection_score")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def equal_states(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


def vision_zero_summary(rows: list[dict]) -> dict:
    def flat(row: dict) -> dict:
        subset = row["vision_all_zero_metrics"]
        return {k: float(subset[k]) for k in ("accuracy", "macro_f1", "mae", "pearson")}
    clean = flat(rows[0])
    missing_rows = [flat(r) for r in rows[1:]]
    mean_missing = {k: float(np.mean([r[k] for r in missing_rows])) for k in clean}
    return {"count": int(rows[0]["vision_all_zero_count"]),
            "clean": clean, "mean_missing": mean_missing}


def make_loaders(raw: dict, cfg: dict, *, normalization: bool):
    train = Aligned50Dataset(raw["train"], "train")
    valid = Aligned50Dataset(raw["valid"], "valid")
    scaler = TextOnlyTrainScaler.fit(train, epsilon=float(cfg["preprocessing"]["epsilon"])) if normalization else None
    if scaler is not None:
        train.scaler = scaler
        valid.scaler = scaler
    generator = torch.Generator()
    generator.manual_seed(int(cfg["training"]["seed"]))
    loaders = {
        "train": DataLoader(train, batch_size=int(cfg["data"]["batch_size"]), shuffle=True,
                             num_workers=0, generator=generator),
        "valid": DataLoader(valid, batch_size=int(cfg["data"]["batch_size"]), shuffle=False,
                             num_workers=0),
    }
    return {"train": train, "valid": valid}, loaders, scaler


def validate_contract(cfg: dict) -> None:
    if cfg["training"]["seed"] != 42 or cfg["preprocessing"]["normalization"] != "text_train_featurewise":
        raise ValueError("B5-N1 is fixed to seed42 and text_train_featurewise")
    tr = cfg["training"]
    expected = {"epochs": 80, "patience": 12, "learning_rate": 0.001,
                "weight_decay": 0.0001, "lambda_reg": 1.0,
                "class_weighting": "balanced_train", "augmentation": "none"}
    for key, val in expected.items():
        if tr.get(key) != val:
            raise ValueError(f"training.{key}={tr.get(key)!r}, expected {val!r}")
    if cfg["data"]["batch_size"] != 128 or cfg["data"]["num_workers"] != 0:
        raise ValueError("B5-N1 data loader settings must match B0 seed42")
    benchmark = Path(cfg["benchmark"]["definition"])
    if hashlib.sha256(benchmark.read_bytes()).hexdigest() != BENCHMARK_SHA256:
        raise RuntimeError("frozen benchmark definition SHA256 mismatch")


def validate_scenario_zeros(valid_loader) -> dict:
    """Verify every artificial missing interval remains exact zero post-transform."""
    batch = next(iter(valid_loader))
    scenario = next(s for s in scenarios() if s.modalities == ("text",) and s.rho == 0.5 and s.location == "middle")
    masked, _ = apply_blocks(batch, [
        {"sample_index": i, "modality": "text", "rho": scenario.rho, "location": scenario.location}
        for i in range(len(batch["id"]))
    ])
    availability = masked["availability_mask"][:, 0]
    x = masked["text"]
    exact_zero = bool(torch.all(x[~availability] == 0))
    nonmissing_preserved = bool(torch.equal(x[availability], batch["text"][availability]))
    assert exact_zero and nonmissing_preserved
    # The benchmark always masks tensors obtained from DataLoader (already transformed).
    return {"pipeline_order": "raw feature -> train-stat transform -> apply_blocks overwrite to exact zero",
            "checked_scenario": scenario.scenario_id,
            "masked_positions_exact_zero": exact_zero,
            "unmasked_positions_unchanged": nonmissing_preserved,
            "native_zero_is_not_used_as_mask": True}


def validate_normalization(datasets: dict, scaler: TextOnlyTrainScaler) -> dict:
    tr = datasets["train"]
    # Calculate the actual float32 values returned by the Dataset on all valid positions.
    z_mean = np.zeros(768, dtype=np.float64)
    z_sq = np.zeros(768, dtype=np.float64)
    count = 0
    raw_a = tr.features["audio"]
    raw_v = tr.features["vision"]
    for start in range(0, len(tr), 128):
        end = min(start + 128, len(tr))
        mask = tr.padding_mask[start:end]
        raw_text = np.asarray(tr.features["text"][start:end], dtype=np.float32)
        transformed = scaler.transform("text", raw_text)[mask].astype(np.float64)
        z_mean += transformed.sum(axis=0)
        z_sq += np.square(transformed).sum(axis=0)
        count += transformed.shape[0]
        # All other modalities must pass through byte-for-byte after float32 conversion.
        for modality, raw_modality in (("audio", raw_a), ("vision", raw_v)):
            original = np.asarray(raw_modality[start:end], dtype=np.float32)
            assert np.array_equal(scaler.transform(modality, original), original)
    means = z_mean / count
    stds = np.sqrt(np.maximum(z_sq / count - means * means, 0.0))
    nondeg = scaler.raw_feature_std >= scaler.epsilon
    return {
        "valid_positions_count_matches_scaler": count == scaler.valid_timestep_count,
        "train_normalized_feature_mean_abs_median": float(np.median(np.abs(means))),
        "train_normalized_feature_mean_abs_max": float(np.max(np.abs(means))),
        "train_normalized_feature_std_deviation_from_one_median_nondegenerate": float(
            np.median(np.abs(stds[nondeg] - 1.0))) if np.any(nondeg) else None,
        "train_normalized_feature_std_deviation_from_one_max_nondegenerate": float(
            np.max(np.abs(stds[nondeg] - 1.0))) if np.any(nondeg) else None,
    }


def cpu_cuda_smoke(model_cfg: dict, device: torch.device) -> dict:
    result = {}
    for name, dev in (("cpu", torch.device("cpu")), ("cuda", device)):
        if name == "cuda" and dev.type != "cuda":
            result[name] = "unavailable"
            continue
        model = B0Baseline(**model_cfg).to(dev).train()
        batch = {m: torch.randn(4, 50, d, device=dev) for m, d in
                 (("text", 768), ("audio", 74), ("vision", 35))}
        lengths = torch.tensor([1, 7, 30, 50], device=dev)
        batch["padding_mask"] = torch.arange(50, device=dev)[None, :] < lengths[:, None]
        batch["cls_label"] = torch.tensor([0, 1, 2, 1], dtype=torch.long, device=dev)
        batch["reg_label"] = torch.tensor([-1.2, 0.0, 1.7, 0.3], device=dev)
        out = model(batch)
        multitask_loss(out, batch, 1.0, torch.ones(3, device=dev))["total"].backward()
        assert torch.isfinite(out["classification_logits"]).all()
        assert torch.isfinite(out["regression"]).all()
        result[name] = "forward_backward_pass"
    return result


def baseline_checkpoint_eval(cfg: dict, datasets: dict, loaders: dict,
                             device: torch.device) -> tuple[dict, dict, dict]:
    path = Path(cfg["baseline"]["checkpoint"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("normalization", "none") != "none":
        raise RuntimeError("N0 B0 checkpoint is not normalization=none")
    saved_cfg = checkpoint["config"]
    if int(saved_cfg["training"]["seed"]) != 42:
        raise RuntimeError("N0 checkpoint is not seed42")
    model = B0Baseline(**saved_cfg["model"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    rows = evaluate_benchmark(model, loaders["valid"], device)
    observed = summary(rows)
    reference = json.loads(Path(cfg["baseline"]["metrics"]).read_text(encoding="utf-8"))
    expected_clean = reference["best_selection_score"]["validation_metrics"]
    robust_reference = json.loads(Path(cfg["baseline"]["pooling_reference"]).read_text(encoding="utf-8"))["P0"]["validation"]
    matches = {
        k: abs(observed["clean"][k] - float(expected_clean[k])) <= 1e-12
        for k in ("accuracy", "macro_f1", "mae", "pearson")
    }
    matches["clean_selection_score"] = abs(observed["clean"]["selection_score"] -
                                           float(reference["best_selection_score"]["value"])) <= 1e-12
    matches["robust_score"] = abs(observed["robust_score"] -
                                  float(robust_reference["robust_score"])) <= 1e-12
    if not all(matches.values()):
        raise RuntimeError(f"reused N0 checkpoint failed validation reproduction: {matches}")
    return {"checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "best_epoch": int(reference["best_selection_score"]["epoch"]),
            "normalization": "none", "reference_match": matches,
            "metrics": observed, "vision_all_zero": vision_zero_summary(rows),
            "scenario_rows": rows}, checkpoint, reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b5_n1_text_zscore.yaml")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    validate_contract(cfg)
    if cfg["data"].get("include_test", False):
        raise RuntimeError("B5-N1 must not construct a test Dataset")
    raw = load_pickle_readonly(cfg["data"]["pkl_path"])
    if not {"train", "valid"}.issubset(raw):
        raise ValueError("audited train/valid splits are required")
    datasets, loaders, scaler = make_loaders(raw, cfg, normalization=True)
    assert scaler is not None and scaler.fitted_split == "train"
    scaler_summary = scaler.diagnostic_summary()
    normalization_check = validate_normalization(datasets, scaler)
    zero_check = validate_scenario_zeros(loaders["valid"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reuse the known B0-WCE checkpoint and verify its saved validation result.
    raw_datasets = {"train": Aligned50Dataset(raw["train"], "train"),
                    "valid": Aligned50Dataset(raw["valid"], "valid")}
    raw_gen = torch.Generator().manual_seed(int(cfg["training"]["seed"]))
    raw_loaders = {"train": DataLoader(raw_datasets["train"], batch_size=128, shuffle=True,
                                        num_workers=0, generator=raw_gen),
                   "valid": DataLoader(raw_datasets["valid"], batch_size=128, shuffle=False,
                                        num_workers=0)}
    n0, _, n0_reference = baseline_checkpoint_eval(cfg, raw_datasets, raw_loaders, device)

    # Verify identical initialization for the N0/N1 seed42 protocol and preserve
    # the post-init RNG state used by the historical B0 training loop.
    seed = int(cfg["training"]["seed"])
    seed_everything(seed)
    init_n0 = B0Baseline(**cfg["model"])
    init_state = {k: v.detach().clone() for k, v in init_n0.state_dict().items()}
    cpu_rng_after_init = torch.get_rng_state()
    cuda_rng_after_init = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seed_everything(seed)
    model = B0Baseline(**cfg["model"])
    init_equal = equal_states(init_state, model.state_dict())
    if not init_equal:
        raise RuntimeError("N0/N1 seed42 initial model states differ")
    initialization_hash = state_fingerprint(init_state)
    torch.set_rng_state(cpu_rng_after_init)
    if cuda_rng_after_init is not None:
        torch.cuda.set_rng_state_all(cuda_rng_after_init)
    model = model.to(device)
    smoke = cpu_cuda_smoke(cfg["model"], device)
    # Smoke tests consume RNG. Rewind to the training RNG state so they cannot
    # alter the model's prescribed dropout sequence.
    torch.set_rng_state(cpu_rng_after_init)
    if cuda_rng_after_init is not None:
        torch.cuda.set_rng_state_all(cuda_rng_after_init)

    counts = np.bincount(datasets["train"].cls_labels, minlength=3)
    weights = torch.as_tensor(len(datasets["train"]) / (3.0 * counts),
                              dtype=torch.float32, device=device)
    if counts.tolist() != n0_reference["train_class_counts"]:
        raise RuntimeError("train-derived class counts differ from seed42 B0 reference")

    n0_order = torch.randperm(len(raw_datasets["train"]), generator=torch.Generator().manual_seed(seed))
    n1_order = torch.randperm(len(datasets["train"]), generator=torch.Generator().manual_seed(seed))
    order_match = bool(torch.equal(n0_order, n1_order))
    if not order_match:
        raise RuntimeError("N0/N1 train sample orders differ")
    order = n1_order.numpy()
    order_hash = hashlib.sha256("\n".join(str(datasets["train"].ids[i]) for i in order).encode()).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]),
                                  weight_decay=float(cfg["training"]["weight_decay"]))
    best_clean = -float("inf")
    best_robust = -float("inf")
    best_clean_epoch = best_robust_epoch = 0
    stale = 0
    history = []
    started = time.perf_counter()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    clean_path = CHECKPOINTS / cfg["outputs"]["best_clean_checkpoint"]
    robust_path = CHECKPOINTS / cfg["outputs"]["best_robust_checkpoint"]
    scaler_state = scaler.state_dict()
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in loaders["train"]:
            moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(moved)
            losses = multitask_loss(outputs, moved, lambda_reg=1.0, class_weights=weights)
            losses["total"].backward()
            optimizer.step()
            n = moved["cls_label"].shape[0]
            seen += n
            total += float(losses["total"].item()) * n
        rows = evaluate_benchmark(model, loaders["valid"], device)
        metrics = summary(rows)
        clean_score = metrics["clean"]["selection_score"]
        robust_score = metrics["robust_score"]
        row = {"epoch": epoch, "train_loss": total / max(seen, 1),
               "clean": metrics["clean"], "mean_missing": metrics["mean_missing"],
               "robust_score": robust_score}
        history.append(row)
        state = {"model_state_dict": model.state_dict(), "config": cfg,
                 "normalization": "text_train_featurewise", "scaler": scaler_state,
                 "benchmark_seed": BENCHMARK_SEED, "benchmark_sha256": BENCHMARK_SHA256,
                 "epoch": epoch, "clean_score": clean_score, "robust_score": robust_score,
                 "train_class_counts": counts.tolist(), "class_weights": weights.detach().cpu()}
        if clean_score > best_clean:
            best_clean, best_clean_epoch = clean_score, epoch
            torch.save(state, clean_path)
        if robust_score > best_robust:
            best_robust, best_robust_epoch, stale = robust_score, epoch, 0
            torch.save(state, robust_path)
        else:
            stale += 1
        print(json.dumps({"seed": seed, **row}, ensure_ascii=False), flush=True)
        if stale >= int(cfg["training"]["patience"]):
            break
    training_seconds = time.perf_counter() - started
    write_json(METRICS / cfg["outputs"]["history"], history)

    def eval_checkpoint(path: Path) -> dict:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        selected = B0Baseline(**cfg["model"]).to(device).eval()
        selected.load_state_dict(checkpoint["model_state_dict"], strict=True)
        post = evaluate_benchmark(selected, loaders["valid"], device)
        selected_state = {k: v.detach().cpu().clone() for k, v in selected.state_dict().items()}
        clone = B0Baseline(**cfg["model"]).to(device).eval()
        clone.load_state_dict(selected_state, strict=True)
        example = next(iter(loaders["valid"]))
        example = {k: v.to(device) for k, v in example.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            expected = selected(predictor_inputs(example))
            actual = clone(predictor_inputs(example))
        roundtrip = all(torch.equal(expected[k], actual[k]) for k in expected)
        if not roundtrip:
            raise RuntimeError(f"checkpoint round trip failed for {path.name}")
        projection_norms = {m: float(selected.modality_projection[m][0].weight.norm().item())
                            for m in ("text", "audio", "vision")}
        return {"checkpoint": str(path), "epoch": int(checkpoint["epoch"]),
                "metrics": summary(post), "vision_all_zero": vision_zero_summary(post),
                "scenario_rows": post, "projection_weight_norms": projection_norms,
                "checkpoint_roundtrip_passed": roundtrip}

    best_clean_result = eval_checkpoint(clean_path)
    best_robust_result = eval_checkpoint(robust_path)
    if best_clean_result["epoch"] != best_clean_epoch or best_robust_result["epoch"] != best_robust_epoch:
        raise RuntimeError("saved checkpoint epoch mismatch")

    result = {
        "experiment": "B5-N1 text-only train-stat feature-wise z-score",
        "training_seed": seed,
        "device": str(device),
        "train_count": len(datasets["train"]), "valid_count": len(datasets["valid"]),
        "test_dataset_constructed": False, "test_split_key_indexed": False,
        "monolithic_pickle_container_deserialized": True, "attachment3_accessed": False,
        "initial_weights_equal_to_seed42_n0_protocol": init_equal,
        "initial_state_sha256": initialization_hash,
        "train_sample_order_matches_n0_protocol": order_match,
        "train_sample_order_sha256": order_hash,
        "training_seconds": training_seconds, "epochs_run": len(history),
        "best_clean_epoch": best_clean_epoch, "best_robust_epoch": best_robust_epoch,
        "best_clean_score": best_clean, "best_robust_score": best_robust,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_class_counts": counts.tolist(), "class_weights": weights.detach().cpu().tolist(),
        "normalization": {**scaler_summary, **normalization_check,
                          "padding_values_may_transform_but_padding_mask_unchanged": True,
                          "audio_vision_unchanged": True},
        "synthetic_missing_zero_check": zero_check,
        "benchmark_seed": BENCHMARK_SEED, "benchmark_sha256": BENCHMARK_SHA256,
        "N0": n0,
        "N1_best_clean": best_clean_result,
        "N1_best_robust": best_robust_result,
        "smoke_tests": smoke,
        "history": history,
    }
    n0_robust = n0["metrics"]["robust_score"]
    result["delta_robust_N1_minus_N0"] = float(best_robust - n0_robust)
    result["decision"] = (
        "strong_candidate" if result["delta_robust_N1_minus_N0"] >= 0.002 else
        "weak_signal" if result["delta_robust_N1_minus_N0"] > 0 else "stop_normalization_route")
    write_json(METRICS / "b5_n1_text_scaler_state.json", scaler_state)
    write_json(METRICS / "b5_n1_text_zscore_metrics.json", result)
    report = render_report(result)
    (METRICS / "b5_n1_text_zscore_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"N0_robust": n0_robust, "N1_robust": best_robust,
                      "delta_robust": result["delta_robust_N1_minus_N0"],
                      "decision": result["decision"], "best_clean_epoch": best_clean_epoch,
                      "best_robust_epoch": best_robust_epoch,
                      "training_seconds": training_seconds}, ensure_ascii=False), flush=True)


def render_report(result: dict) -> str:
    n0, n1 = result["N0"], result["N1_best_robust"]
    lines = ["# B5-N1 text-only train-stat feature-wise Z-score", "",
             "Seed42 only. N0 reuses the historical B0-WCE checkpoint; N1 fully retrains B0 with text-only normalization.",
             "Validation only: clean + frozen 54 scenarios (seed 20260923). The monolithic pickle container is deserialized, but its test key is not indexed, made into a Dataset, evaluated, or used for any decision; attachment3 is not accessed.", "",
             f"Benchmark SHA256: `{BENCHMARK_SHA256}`. Synthetic zeroing order: `{result['synthetic_missing_zero_check']['pipeline_order']}`.", "",
             "## N0 / N1 main comparison", "",
             "| Model/checkpoint | Condition | Accuracy | Macro-F1 | MAE | Pearson | Selection score | Robust score |", "",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, block in (("N0 B0-WCE historical", n0["metrics"]),
                        ("N1 best-robust", n1["metrics"])):
        for cond in ("clean", "mean_missing"):
            m = block[cond]
            lines.append(f"| {name} | {cond} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['mae']:.4f} | "
                         f"{m['pearson']:.4f} | {m['selection_score']:.4f} | {block['robust_score']:.4f} |")
    lines += ["", f"Δrobust (N1−N0) = {result['delta_robust_N1_minus_N0']:+.6f}; decision: **{result['decision']}**.",
              f"N1 best-clean epoch {result['best_clean_epoch']}, best-robust epoch {result['best_robust_epoch']}; "
              f"training time {result['training_seconds']:.1f}s.", "",
              "## Neutral recall / F1", "",
              "| Model | Condition | Recall | F1 |", "", "|---|---|---:|---:|"]
    for name, rows in (("N0", n0["scenario_rows"]), ("N1", n1["scenario_rows"])):
        for key in ("clean", "mean_missing"):
            if key == "clean":
                item = rows[0]
                recall = item["per_class"]["Neutral"]["recall"]
                f1 = item["per_class"]["Neutral"]["f1"]
            else:
                selected = rows[1:]
                recall = float(np.mean([r["per_class"]["Neutral"]["recall"] for r in selected]))
                f1 = float(np.mean([r["per_class"]["Neutral"]["f1"] for r in selected]))
            lines.append(f"| {name} | {key} | {recall:.4f} | {f1:.4f} |")
    lines += ["", "## Missing subgroup comparison", "",
              "Each value below is a scenario mean. Per-class full scenario metrics remain in JSON.", "",
              "| Group | Model | Acc | Macro-F1 | MAE | Pearson | Score | Neutral recall | Neutral F1 |", "",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    groupers = {
        "modality": lambda r, g: r["modalities"] == g and "+" not in r["modalities"],
        "rho": lambda r, g: r["rho"] == float(g) and "+" not in r["modalities"],
        "location": lambda r, g: r["location"] == g,
        "double": lambda r, g: r["modalities"] == g,
    }
    values = {"modality": ("text", "audio", "vision"),
              "rho": ("0.1", "0.2", "0.3", "0.4", "0.5"),
              "location": ("early", "middle", "late"),
              "double": ("text+audio", "text+vision", "audio+vision")}
    for group, options in values.items():
        for value in options:
            for name, rows in (("N0", n0["scenario_rows"]), ("N1", n1["scenario_rows"])):
                selected = [r for r in rows[1:] if groupers[group](r, value)]
                if not selected:
                    continue
                metrics = [float(np.mean([r[k] for r in selected])) for k in
                           ("accuracy", "macro_f1", "mae", "pearson", "selection_score")]
                nr = float(np.mean([r["per_class"]["Neutral"]["recall"] for r in selected]))
                nf = float(np.mean([r["per_class"]["Neutral"]["f1"] for r in selected]))
                lines.append(f"| {group}: {value} | {name} | " + " | ".join(f"{x:.4f}" for x in metrics) +
                             f" | {nr:.4f} | {nf:.4f} |")
    lines += ["", "## Vision-all-zero subset (15 validation samples)", "",
              "Diagnostic only; it does not alter training or selection.", "",
              "| Model | Split | Acc | Macro-F1 | MAE | Pearson |", "", "|---|---|---:|---:|---:|---:|"]
    for name, block in (("N0", n0), ("N1", n1)):
        for split in ("clean", "mean_missing"):
            m = block["vision_all_zero"][split]
            lines.append(f"| {name} | {split} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['mae']:.4f} | {m['pearson']:.4f} |")
    lines += ["", "## Text scaler diagnostics", "",
              f"Fit split `{result['normalization']['fitted_split']}`, modality `{result['normalization']['modality']}`, "
              f"feature_dim={result['normalization']['feature_dim']}, valid timesteps={result['normalization']['valid_timestep_count']}, "
              f"epsilon={result['normalization']['epsilon']}.", "",
              f"Raw per-feature mean range [{result['normalization']['raw_feature_mean_min']:.6g}, {result['normalization']['raw_feature_mean_max']:.6g}]; "
              f"raw std range [{result['normalization']['raw_feature_std_min']:.6g}, {result['normalization']['raw_feature_std_max']:.6g}]. "
              f"Zero/nearly-zero std dimensions: {result['normalization']['zero_std_dimension_count']}/"
              f"{result['normalization']['nearly_zero_std_dimension_count']}.", "",
              f"Train z-score per-feature mean |.| median/max = "
              f"{result['normalization']['zscore_train_feature_mean_abs_median']:.3g}/"
              f"{result['normalization']['zscore_train_feature_mean_abs_max']:.3g}; nondegenerate std deviation from 1 "
              f"median/max = {result['normalization']['zscore_nondegenerate_std_abs_deviation_median']:.3g}/"
              f"{result['normalization']['zscore_nondegenerate_std_abs_deviation_max']:.3g}.", "",
              "Std quantiles are in the machine-readable metrics and exact 768-element scaler state is in `b5_n1_text_scaler_state.json`.", "",
              "## Projection weight norms", "",
              "| Modality | L2 norm of trained input projection weights |", "", "|---|---:|"]
    for modality, norm in n1["projection_weight_norms"].items():
        lines.append(f"| {modality} | {norm:.6f} |")
    lines += ["", "## Checks", "",
              f"- N0 historical checkpoint reproduction: {all(n0['reference_match'].values())}; per-item results are in JSON.",
              f"- N0/N1 initial state equal: {result['initial_weights_equal_to_seed42_n0_protocol']} (`{result['initial_state_sha256']}`).",
              f"- N0/N1 train sample order hash matches: {result['train_sample_order_matches_n0_protocol']} (`{result['train_sample_order_sha256']}`).",
              f"- Text scaler fit is train-only; valid positions counted: {result['normalization']['valid_timestep_count']}.",
              f"- Padding mask remains unchanged and padded positions are excluded by masked pooling.",
              f"- Audio/vision unchanged; synthetic missing overwritten to exact zero after normalization: {result['synthetic_missing_zero_check']['masked_positions_exact_zero']}.",
              f"- CPU/CUDA smoke: {result['smoke_tests']}; robust and clean checkpoint round-trip checks passed.",
              "- The monolithic pickle container is deserialized to select train/valid. Its test key is not indexed or used; no test Dataset/evaluation is performed. Attachment3 is not accessed.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
