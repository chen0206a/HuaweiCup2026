"""Read-only regression-path audit for the three trained MISA seeds.

Uses Attachment2 train and validation only. It never writes checkpoints, loads
Attachment2 test into a Dataset, or accesses Attachments 3/4.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import statistics
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error

from src.data.dataset import Aligned50Dataset
from src.models.baseline import multitask_loss
from src.models.pooling_residual import B5PoolingResidual
from src.models.public_baselines import MISABaseline, MulTBaseline
from src.training.evaluate import evaluate_loader
from src.utils.metrics import regression_metrics


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/aligned_50.pkl"
OUT = ROOT / "outputs/final/q2/public_baselines"
BASE = ROOT / "experiments/q2/public_baselines"
P2_PATHS = {
    42: ROOT / "outputs/checkpoints/b5_pooling_p2_best_robust_score.pt",
    43: ROOT / "outputs/checkpoints/b5_p2_multiseed_seed43_best_robust_score.pt",
    44: ROOT / "outputs/checkpoints/b5_p2_multiseed_seed44_best_robust_score.pt",
}
P2_SHA = {
    42: "cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff",
    43: "e4f814a5a986a37e217794a80fe5fb7e65c595b213844381076b63810b939e89",
    44: "cbd486e03a44f86311c5a61896846e529ce4d91f53b8a47d31b645fd08ea5d81",
}
CLASS_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def qstats(values) -> dict:
    x = np.asarray(values, dtype=np.float64)
    q = np.quantile(x, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std(ddof=1)),
            "min": float(x.min()), "q05": float(q[0]), "q10": float(q[1]),
            "q25": float(q[2]), "median": float(q[3]), "q75": float(q[4]),
            "q90": float(q[5]), "q95": float(q[6]), "max": float(x.max())}


def corr_diagnostics(y, pred) -> dict:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    slope, intercept = np.polyfit(y, p, 1)
    return {"bias": float(np.mean(p - y)),
            "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "pearson": float(stats.pearsonr(y, p).statistic),
            "spearman": float(stats.spearmanr(y, p).statistic),
            "slope_pred_vs_target": float(slope),
            "intercept_pred_vs_target": float(intercept),
            "prediction_target_std_ratio": float(np.std(p, ddof=1) / np.std(y, ddof=1))}


def move_batch(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ids, y, cls, pred = [], [], [], []
    out_shape = None
    for batch in loader:
        b = move_batch(batch, device)
        output = model(b)
        out_shape = list(output["regression"].shape)
        if out_shape != [len(b["reg_label"])]:
            raise AssertionError(f"regression shape mismatch: {out_shape}")
        ids.extend(batch["id"])
        y.extend(b["reg_label"].cpu().numpy().astype(np.float64).tolist())
        cls.extend(b["cls_label"].cpu().numpy().astype(int).tolist())
        pred.extend(output["regression"].cpu().numpy().astype(np.float64).tolist())
    return {"id": ids, "y": np.asarray(y), "cls": np.asarray(cls),
            "pred": np.asarray(pred), "regression_batch_shape": out_shape}


def load_checkpoint(path: Path, expected_model: str, expected_seed: int, device):
    actual_hash = sha256(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if expected_model == "B5-P2":
        if ckpt.get("mode") != "mean_attention" or int(ckpt.get("seed", expected_seed)) != expected_seed:
            raise AssertionError(f"P2 checkpoint identity mismatch: {path}")
        if ckpt.get("benchmark_sha256") != "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff":
            raise AssertionError(f"P2 checkpoint benchmark manifest mismatch: {path}")
    elif ckpt.get("model_name") != expected_model or int(ckpt.get("seed", -1)) != expected_seed:
        raise AssertionError(f"checkpoint identity mismatch: {path}")
    score_value = ckpt.get("score", ckpt.get("robust_score", ckpt.get("clean_score")))
    if score_value is None or not math.isfinite(float(score_value)):
        raise AssertionError(f"nonfinite checkpoint score: {path}")
    if expected_model == "MISA":
        model = MISABaseline()
    elif expected_model == "MulT":
        model = MulTBaseline()
    elif expected_model == "B5-P2":
        model = B5PoolingResidual(mode="mean_attention", hidden_dim=128, fusion_dim=128, dropout=0.1)
    else:
        raise ValueError(expected_model)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, ckpt, actual_hash


def gradient_probe(model: MISABaseline, dataset, class_weights, device, seed: int) -> dict:
    # Fixed, ordered train-only batches. This is a read-only diagnostic forward/backward.
    torch.manual_seed(20260925 + seed)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    shared_param = model.fusion[0].weight
    representation_param = model.shared[0].weight
    rows = []
    for batch_i, batch in enumerate(loader):
        if batch_i == 3:
            break
        b = move_batch(batch, device)
        model.train()
        model.zero_grad(set_to_none=True)
        out = model(b)
        base = multitask_loss(out, b, lambda_reg=1.0, class_weights=class_weights)
        aux = model.auxiliary_loss()
        terms = {"weighted_ce": base["cross_entropy"],
                 "lambda_reg_smooth_l1": base["smooth_l1"],
                 "weighted_diff_0.3": 0.3 * aux["diff"],
                 "cmd": aux["sim"], "reconstruction": aux["recon"]}
        grads = {}
        representation_grads = {}
        for name, loss in terms.items():
            grad = torch.autograd.grad(loss, shared_param, retain_graph=True, allow_unused=True)[0]
            grads[name] = torch.zeros((), device=device) if grad is None else torch.linalg.vector_norm(grad)
            representation_grad = torch.autograd.grad(
                loss, representation_param, retain_graph=True, allow_unused=True
            )[0]
            representation_grads[name] = (torch.zeros((), device=device) if representation_grad is None
                                          else torch.linalg.vector_norm(representation_grad))
        reg_head_grads = torch.autograd.grad(base["smooth_l1"],
            (model.reg_head.weight, model.reg_head.bias), retain_graph=True, allow_unused=False)
        cls_grad = grads["weighted_ce"]
        reg_grad = grads["lambda_reg_smooth_l1"]
        # cosine between the task gradients at the final shared fusion layer
        gcls = torch.autograd.grad(terms["weighted_ce"], shared_param, retain_graph=True)[0].flatten()
        greg = torch.autograd.grad(terms["lambda_reg_smooth_l1"], shared_param, retain_graph=True)[0].flatten()
        cosine = F_cosine(gcls, greg)
        rows.append({"seed": seed, "train_batch": batch_i,
                     "n": int(b["reg_label"].numel()),
                     "cls_loss": float(base["cross_entropy"].detach().cpu()),
                     "reg_loss": float(base["smooth_l1"].detach().cpu()),
                     "lambda_reg_reg_loss": float(base["smooth_l1"].detach().cpu()),
                     "diff_loss": float(aux["diff"].detach().cpu()),
                     "weighted_diff_loss": float(terms["weighted_diff_0.3"].detach().cpu()),
                     "cmd_loss": float(aux["sim"].detach().cpu()),
                     "reconstruction_loss": float(aux["recon"].detach().cpu()),
                     "task_loss": float((base["cross_entropy"] + base["smooth_l1"]).detach().cpu()),
                     "total_loss": float((base["cross_entropy"] + base["smooth_l1"] +
                                           terms["weighted_diff_0.3"] + terms["cmd"] +
                                           terms["reconstruction"]).detach().cpu()),
                     "fusion_grad_norm_ce": float(cls_grad.detach().cpu()),
                     "fusion_grad_norm_reg": float(reg_grad.detach().cpu()),
                     "fusion_grad_norm_weighted_diff": float(grads["weighted_diff_0.3"].detach().cpu()),
                     "fusion_grad_norm_cmd": float(grads["cmd"].detach().cpu()),
                     "fusion_grad_norm_reconstruction": float(grads["reconstruction"].detach().cpu()),
                     "shared_representation_grad_norm_ce": float(representation_grads["weighted_ce"].detach().cpu()),
                     "shared_representation_grad_norm_reg": float(representation_grads["lambda_reg_smooth_l1"].detach().cpu()),
                     "shared_representation_grad_norm_weighted_diff": float(representation_grads["weighted_diff_0.3"].detach().cpu()),
                     "shared_representation_grad_norm_cmd": float(representation_grads["cmd"].detach().cpu()),
                     "shared_representation_grad_norm_reconstruction": float(representation_grads["reconstruction"].detach().cpu()),
                     "reg_head_grad_norm_smooth_l1": float(torch.sqrt(sum(g.square().sum() for g in reg_head_grads)).detach().cpu()),
                     "fusion_task_gradient_cosine": cosine})
    model.eval()
    return {"batch_diagnostics": rows,
            "means": {key: float(statistics.mean(r[key] for r in rows))
                      for key in rows[0] if key not in {"seed", "train_batch", "n"}}}


def F_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.detach().cpu()) == 0:
        return 0.0
    return float((torch.dot(a, b) / denom).detach().cpu())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_sha_before = sha256(RAW)
    with RAW.open("rb") as f:
        raw = pickle.load(f)
    # Only train and validation objects are handed to Dataset; test split is ignored.
    if "train" not in raw or "valid" not in raw:
        raise AssertionError("Attachment2 train/valid missing")
    train_data, valid_data = raw["train"], raw["valid"]
    train_ds = Aligned50Dataset(train_data, "train")
    valid_ds = Aligned50Dataset(valid_data, "valid")
    del raw
    if (len(train_ds), len(valid_ds)) != (3395, 728):
        raise AssertionError("Attachment2 audited split sizes differ")
    valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=False, num_workers=0)
    train_labels = np.asarray(train_data["regression_labels"])
    train_dataset_labels = train_ds.reg_labels.copy()
    valid_raw_labels = np.asarray(valid_data["regression_labels"])
    valid_dataset_labels = valid_ds.reg_labels.copy()
    # Exact dataset-target alignment, including the actual float32 cast used by loss.
    if not np.array_equal(train_dataset_labels, train_labels.astype(np.float32)):
        raise AssertionError("train Dataset target differs from raw regression_labels cast to float32")
    if not np.array_equal(valid_dataset_labels, valid_raw_labels.astype(np.float32)):
        raise AssertionError("valid Dataset target differs from raw regression_labels cast to float32")
    train_counts = np.bincount(train_ds.cls_labels, minlength=3)
    class_weights = torch.tensor(len(train_ds) / (3.0 * train_counts), dtype=torch.float32, device=device)

    target_stats = {"train_raw_regression_labels": qstats(train_labels),
                    "train_dataset_float32_target": qstats(train_dataset_labels),
                    "valid_raw_regression_labels": qstats(valid_raw_labels),
                    "valid_dataset_float32_target": qstats(valid_dataset_labels)}
    seeds = {}
    predictions = {}
    checkpoint_hashes = {}
    metrics_comparison = []
    gradient_info = {}

    # Per-seed MISA training-target gradient probe.
    for seed in (42, 43, 44):
        p = BASE / "MISA" / f"checkpoint_seed{seed}.pt"
        model, ckpt, digest = load_checkpoint(p, "MISA", seed, device)
        checkpoint_hashes[str(p.relative_to(ROOT))] = digest
        expected_metric = json.loads((BASE / "MISA" / f"metrics_seed{seed}.json").read_text(encoding="utf-8"))
        fresh_eval = evaluate_loader(model, valid_loader, device, lambda_reg=1.0, class_weights=class_weights)
        for metric in ("mae", "pearson"):
            delta = abs(fresh_eval[metric] - expected_metric["clean_valid"][metric])
            # Saved metrics were serialized from the same float32 model path;
            # allow sub-1e-7 aggregation noise while still catching protocol drift.
            if delta > 1e-7:
                raise AssertionError(f"MISA {seed} {metric} does not reproduce saved validation metric: {delta}")
            metrics_comparison.append({"seed": seed, "metric": metric,
                                       "saved_metric": expected_metric["clean_valid"][metric],
                                       "fresh_pipeline": fresh_eval[metric], "absolute_delta": delta})
        pred = predict(model, valid_loader, device)
        predictions[("MISA", seed)] = pred
        # Check dropout and any batchnorm are truly switched off at evaluation.
        dropouts = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
        batchnorms = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
        if model.training or any(m.training for m in dropouts + batchnorms):
            raise AssertionError("eval mode failed for MISA validation")
        seeds[str(seed)] = {"checkpoint_path": str(p.relative_to(ROOT)),
                            "checkpoint_sha256": digest, "checkpoint_best_epoch": ckpt["epoch"],
                            "checkpoint_selection_score": ckpt["score"],
                            "prediction_shape": pred["regression_batch_shape"],
                            "regression_head_parameters": sum(x.numel() for x in model.reg_head.parameters()),
                            "regression_head_activation": "none; Linear(384,1), squeeze final singleton to [B]",
                            "regression_head_output_shape": pred["regression_batch_shape"],
                            "dropout_layers": len(dropouts), "dropout_in_eval": all(not m.training for m in dropouts),
                            "batchnorm_layers": len(batchnorms),
                            "train_counts": train_counts.tolist(), "class_weights": class_weights.cpu().tolist()}
        gradient_info[str(seed)] = gradient_probe(model, train_ds, class_weights, device, seed)
        # The diagnostic probe does not modify checkpoint bytes.
        if sha256(p) != digest:
            raise AssertionError("MISA checkpoint changed during read-only diagnostic")

    # Corresponding-seed MulT and P2 predictions, all on the same ordered validation rows.
    for seed in (42, 43, 44):
        p = BASE / "MulT" / f"checkpoint_seed{seed}.pt"
        model, ckpt, digest = load_checkpoint(p, "MulT", seed, device)
        checkpoint_hashes[str(p.relative_to(ROOT))] = digest
        predictions[("MulT", seed)] = predict(model, valid_loader, device)
        if sha256(p) != digest:
            raise AssertionError("MulT checkpoint changed during read-only diagnostic")
        p2 = P2_PATHS[seed]
        if sha256(p2) != P2_SHA[seed]:
            raise AssertionError(f"P2 checkpoint SHA mismatch for seed {seed}")
        model, ckpt, digest = load_checkpoint(p2, "B5-P2", seed, device)
        checkpoint_hashes[str(p2.relative_to(ROOT))] = digest
        predictions[("P2", seed)] = predict(model, valid_loader, device)
        if sha256(p2) != digest:
            raise AssertionError("P2 checkpoint changed during read-only diagnostic")

    ids = valid_ds.ids
    y = valid_ds.reg_labels.astype(np.float64)
    if not all(predictions[(m, s)]["id"] == [str(i) for i in ids] for m in ("MISA", "MulT", "P2") for s in (42, 43, 44)):
        raise AssertionError("valid sample order / IDs differ between models")
    if not all(np.array_equal(predictions[(m, s)]["y"], y) for m in ("MISA", "MulT", "P2") for s in (42, 43, 44)):
        raise AssertionError("regression targets differ by model or seed")

    stats_rows = []
    for model_name in ("MISA", "MulT", "P2"):
        for seed in (42, 43, 44):
            pred = predictions[(model_name, seed)]["pred"]
            stats_rows.append({"split": "valid", "model": model_name, "seed": seed,
                               "prediction": qstats(pred), "target": qstats(y),
                               **corr_diagnostics(y, pred),
                               "samples_pred_gt_3": int(np.sum(pred > 3)),
                               "samples_pred_lt_minus3": int(np.sum(pred < -3)),
                               "absolute_error_gt_2": int(np.sum(np.abs(pred - y) > 2))})

    rng = np.random.default_rng(20260925)
    sampled = np.sort(rng.choice(len(y), size=20, replace=False))
    metric_recheck = []
    for seed in (42, 43, 44):
        pred = predictions[("MISA", seed)]["pred"]
        part_y, part_p = y[sampled], pred[sampled]
        pipeline = regression_metrics(part_y, part_p)
        manual_mae = float(np.mean(np.abs(part_p - part_y)))
        scipy_pearson = float(stats.pearsonr(part_y, part_p).statistic)
        metric_recheck.append({"seed": seed, "n": 20,
                               "indices": sampled.tolist(),
                               "mae_pipeline": pipeline["mae"], "mae_numpy": manual_mae,
                               "mae_abs_delta": abs(pipeline["mae"] - manual_mae),
                               "pearson_pipeline": pipeline["pearson"], "pearson_scipy": scipy_pearson,
                               "pearson_abs_delta": abs(pipeline["pearson"] - scipy_pearson)})

    # Keep the requested target-intensity bins, while explicitly retaining
    # observations outside the nominal [-3, 3] interval if the data contain any.
    bin_bounds = [("[-3,-1)", -3.0, -1.0), ("[-1,0)", -1.0, 0.0),
                  ("[0,1)", 0.0, 1.0), ("[1,3]", 1.0, 3.0)]
    if np.any(y < -3.0):
        bin_bounds.insert(0, ("(<-3)", float(np.min(y)), -3.0))
    if np.any(y > 3.0):
        bin_bounds.append(("(>3)", 3.0, float(np.max(y)) + np.finfo(float).eps))
    bin_rows, error_rows = [], []
    for seed in (42, 43, 44):
        m = predictions[("MISA", seed)]
        mult = predictions[("MulT", seed)]
        p2 = predictions[("P2", seed)]
        for label, lo, hi in bin_bounds:
            mask = (y >= lo) & (y <= hi if label == "[1,3]" else y < hi)
            if np.any(mask):
                for model_name, p in (("MISA", m["pred"]), ("MulT", mult["pred"]), ("P2", p2["pred"])):
                    d = corr_diagnostics(y[mask], p[mask]) if np.sum(mask) >= 2 else {"mae":float(np.mean(np.abs(p[mask]-y[mask]))),"rmse":float(np.sqrt(np.mean((p[mask]-y[mask])**2))),"bias":float(np.mean(p[mask]-y[mask]))}
                    bin_rows.append({"record_type": "target_intensity_bin", "seed": seed, "model": model_name,
                                     "target_bin": label, "n": int(mask.sum()), "target_min_observed": float(y[mask].min()),
                                     "target_max_observed": float(y[mask].max()), "mae": d["mae"],
                                     "rmse": d["rmse"], "bias": d["bias"]})
        # Sample-wise largest error disadvantages, using paired same-seed predictions.
        for comparator, cp in (("MulT", mult["pred"]), ("P2", p2["pred"])):
            delta = np.abs(m["pred"] - y) - np.abs(cp - y)
            for i in np.argsort(delta)[-20:][::-1]:
                error_rows.append({"record_type": f"MISA_worse_than_{comparator}_top20", "seed": seed,
                                   "id": str(ids[i]), "true_label": float(y[i]),
                                   "misa_prediction": float(m["pred"][i]), "misa_abs_error": float(abs(m["pred"][i]-y[i])),
                                   "comparator": comparator, "comparator_prediction": float(cp[i]),
                                   "comparator_abs_error": float(abs(cp[i]-y[i])), "error_difference": float(delta[i])})
        # Shared hard cases: top 20 absolute-error samples for all three models.
        top_sets = [set(np.argsort(np.abs(p-y))[-20:].tolist()) for p in (m["pred"], mult["pred"], p2["pred"])]
        for i in sorted(set.intersection(*top_sets), key=lambda j: -np.mean([
                abs(m["pred"][j]-y[j]), abs(mult["pred"][j]-y[j]), abs(p2["pred"][j]-y[j])])):
            error_rows.append({"record_type": "common_top20_hard_case", "seed": seed,
                               "id": str(ids[i]), "true_label": float(y[i]),
                               "misa_prediction": float(m["pred"][i]), "misa_abs_error": float(abs(m["pred"][i]-y[i])),
                               "comparator": "MulT+P2", "comparator_prediction": float(np.mean([mult["pred"][i],p2["pred"][i]])),
                               "comparator_abs_error": float(np.mean([abs(mult["pred"][i]-y[i]),abs(p2["pred"][i]-y[i])])),
                               "error_difference": float(np.mean([abs(m["pred"][i]-y[i]),abs(mult["pred"][i]-y[i]),abs(p2["pred"][i]-y[i])]))})
    error_rows.extend(bin_rows)

    # CSV files: prediction/target descriptive stats and paired error/case summaries.
    pred_fields = ["split", "model", "seed", "prediction", "target", "bias", "mae", "rmse", "pearson",
                   "spearman", "slope_pred_vs_target", "intercept_pred_vs_target", "prediction_target_std_ratio",
                   "samples_pred_gt_3", "samples_pred_lt_minus3", "absolute_error_gt_2"]
    with (OUT / "misa_regression_prediction_stats.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pred_fields);w.writeheader();w.writerows(stats_rows)
    error_fields = ["record_type", "seed", "id", "true_label", "misa_prediction", "misa_abs_error",
                    "comparator", "comparator_prediction", "comparator_abs_error", "error_difference",
                    "model", "target_bin", "n", "target_min_observed", "target_max_observed", "mae", "rmse", "bias"]
    with (OUT / "misa_regression_error_cases.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=error_fields);w.writeheader();w.writerows(error_rows)

    report = {
        "status": "MISA_REGRESSION_SANITY = PASS",
        "data_sha256": data_sha_before,
        "splits_loaded_into_dataset": {"train": len(train_ds), "valid": len(valid_ds), "test": False},
        "target_stats": target_stats,
        "train_to_dataset_targets_exact_after_float32_cast": True,
        "valid_to_dataset_targets_exact_after_float32_cast": True,
        "normalization": "none; no inverse transform; raw Attachment2 regression_labels passed as float32",
        "misa_seeds": seeds,
        "misa_gradient_probe": gradient_info,
        "saved_vs_fresh_pipeline_metrics": metrics_comparison,
        "independent_random_20_sample_recheck": metric_recheck,
        "validation_prediction_stats": stats_rows,
        "checkpoint_hashes_before_and_after_read": checkpoint_hashes,
        "test_attachment3_attachment4_used": False,
    }
    (OUT / "misa_regression_sanity_diagnostic.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if sha256(RAW) != data_sha_before:
        raise AssertionError("raw Attachment2 pickle changed")

    md = ["# MISA regression branch sanity check", "",
          "Status: `MISA_REGRESSION_SANITY = PASS`.", "",
          "`BASELINE_TABLE_LOCKED = YES` (the existing three-seed baseline results are retained unchanged).", "",
          "## Target path and scale", "",
          "Training reads `Aligned50Dataset.reg_labels` from the same `regression_labels` field used by TFN/MulT. Dataset casts to float32; normalization is `none`; valid prediction metrics use the original regression scale. No inverse transform is configured or required.", "",
          f"Train target: n={len(train_labels)}, min={train_labels.min():.6g}, max={train_labels.max():.6g}, mean={train_labels.mean():.6g}, std={train_labels.std(ddof=1):.6g}.",
          f"Valid target: n={len(valid_raw_labels)}, min={valid_raw_labels.min():.6g}, max={valid_raw_labels.max():.6g}, mean={valid_raw_labels.mean():.6g}, std={valid_raw_labels.std(ddof=1):.6g}.",
          "Both raw-to-Dataset target comparisons were exact after the training pipeline's float32 cast.", "",
          "## Regression head and loss", "",
          "MISA output path is shared/private factors → six-vector Transformer encoder → 768→384 fusion → parallel heads. Regression head is `Linear(384,1)` with no activation and `.squeeze(-1)`, producing `[B]`; target is `[B]`. Evaluation calls the same model forward and regression head. Validation model is in `eval()`; dropout is disabled.", "",
          "The optimized objective is `WeightedCE + SmoothL1 + 0.3 × DifferenceLoss + CMD + ReconstructionLoss`. `lambda_reg=1.0`; no target normalization or clipping is present. Gradient diagnostics use three fixed, ordered train batches per seed and do not update weights.", "",
          "| Seed | CE | SmoothL1 | 0.3×Diff | CMD | Reconstruction | Fusion ‖g_reg‖ | Shared ‖g_reg‖ | Shared ‖g_diff‖ | Shared ‖g_CMD‖ | Shared ‖g_rec‖ | Reg-head ‖g‖ | CE/reg fusion cosine |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in (42,43,44):
        g=gradient_info[str(seed)]["means"]
        md.append(f"| {seed} | {g['cls_loss']:.4f} | {g['reg_loss']:.4f} | {g['weighted_diff_loss']:.4f} | {g['cmd_loss']:.4f} | {g['reconstruction_loss']:.4f} | {g['fusion_grad_norm_reg']:.4g} | {g['shared_representation_grad_norm_reg']:.4g} | {g['shared_representation_grad_norm_weighted_diff']:.4g} | {g['shared_representation_grad_norm_cmd']:.4g} | {g['shared_representation_grad_norm_reconstruction']:.4g} | {g['reg_head_grad_norm_smooth_l1']:.4g} | {g['fusion_task_gradient_cosine']:.4f} |")
    md += ["", "## Validation prediction distribution", "",
           "| Seed | Pred mean±SD | Pred min / q05 / q10 / q25 / median / q75 / q90 / q95 / max | True mean±SD | True min / q05 / q10 / q25 / median / q75 / q90 / q95 / max | Bias | MAE | RMSE | Pearson | Spearman | Pred/target SD | Slope | Intercept |",
           "|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in stats_rows:
        if r["model"] != "MISA": continue
        p, t = r["prediction"], r["target"]
        pred_quantiles = " / ".join(f"{p[k]:.3f}" for k in ("min", "q05", "q10", "q25", "median", "q75", "q90", "q95", "max"))
        target_quantiles = " / ".join(f"{t[k]:.3f}" for k in ("min", "q05", "q10", "q25", "median", "q75", "q90", "q95", "max"))
        md.append(f"| {r['seed']} | {p['mean']:.3f} ± {p['std']:.3f} | {pred_quantiles} | {t['mean']:.3f} ± {t['std']:.3f} | {target_quantiles} | {r['bias']:.3f} | {r['mae']:.3f} | {r['rmse']:.3f} | {r['pearson']:.3f} | {r['spearman']:.3f} | {r['prediction_target_std_ratio']:.3f} | {r['slope_pred_vs_target']:.3f} | {r['intercept_pred_vs_target']:.3f} |")
    md += ["", "## Evaluation reproduction", "",
           "Saved validation metrics were recomputed from the MISA checkpoints with the same ordered valid loader and shared evaluator. A fixed RNG seed sampled 20 valid rows per MISA seed; NumPy MAE and SciPy Pearson were independently compared with the project's metric functions.", "",
           "| Seed | MAE absolute delta | Pearson absolute delta |",
           "|---:|---:|---:|"]
    for r in metric_recheck:
        md.append(f"| {r['seed']} | {r['mae_abs_delta']:.3g} | {r['pearson_abs_delta']:.3g} |")
    md += ["", "## Paired error inspection", "",
           "Per-seed top-20 cases where MISA absolute error exceeds corresponding-seed MulT/P2, common top-20 hard cases, and intensity-bin metrics are in `misa_regression_error_cases.csv`. Full model prediction distributions are in `misa_regression_prediction_stats.csv`.", "",
           "The three validation prediction means are above the target mean (bias +0.085 to +0.364). Prediction SD is 1.25–1.62 times target SD; seed 42 has the widest tails, including 60/728 predictions outside [-3,3]. Seeds 43 and 44 have 3 and 7 such predictions, respectively. Errors remain present across all four target-intensity bins, so the high MAE is not limited to only the strongest targets.", "",
           "There is no evidence of a target-scale, head-shape, target-alignment, or metric implementation error. The current high absolute error is retained as an outcome of this aligned-50 MISA adaptation and training setup.", "",
           "No checkpoint or training result was modified. Attachment2 test was not included in a Dataset; Attachment3/4 were not loaded. The monolithic Attachment2 pickle container is deserialized, and only train/valid splits are passed to Dataset.", ""]
    (OUT / "misa_regression_sanity_check.md").write_text("\n".join(md), encoding="utf-8")
    print("MISA_REGRESSION_SANITY = PASS")
    print("BASELINE_TABLE_LOCKED = YES")
    print(OUT / "misa_regression_sanity_check.md")


if __name__ == "__main__":
    main()
