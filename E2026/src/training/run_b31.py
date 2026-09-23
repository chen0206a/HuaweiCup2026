"""B3.1 frozen-backbone conservative latent reconstruction diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

from src.data.block_mask import augment_train_batch, predictor_inputs
from src.data.dataset import MODALITIES
from src.data.preprocess import build_datasets_and_loaders
from src.evaluation.missing_benchmark import evaluate_benchmark, scenarios, summary
from src.models.baseline import B0Baseline, multitask_loss
from src.models.reconstruction import B31FrozenBackbone, B3LatentReconstruction
from src.training.run_b2 import flat_rows, write_csv, write_json
from src.training.train_b1 import seed_everything
from src.utils.metrics import compute_metrics, validation_selection_score

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "outputs" / "metrics"
CHECKPOINTS = ROOT / "outputs" / "checkpoints"
BENCHMARK_SHA256 = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def verify_benchmark(cfg: dict) -> None:
    path = Path(cfg["benchmark"]["definition"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    definition = json.loads(path.read_text(encoding="utf-8"))
    if (digest != BENCHMARK_SHA256 or digest != cfg["benchmark"]["sha256"] or
            definition["seed"] != 20260923 or
            definition["scenarios"] != [scenario.as_dict() for scenario in scenarios()]):
        raise RuntimeError("frozen B2 benchmark changed")


def loaders(cfg: dict):
    if cfg["preprocessing"]["normalization"] != "none":
        raise ValueError("B3.1 requires normalization=none")
    return build_datasets_and_loaders(
        cfg["data"]["pkl_path"], normalization="none",
        batch_size=int(cfg["data"]["batch_size"]), num_workers=0,
        seed=42, include_test=False,
    )


def load_b0(path: str, model_cfg: dict, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if int(ckpt["config"]["training"]["seed"]) != 42:
        raise RuntimeError("expected seed-42 B0-WCE checkpoint")
    for key in ("hidden_dim", "fusion_dim", "dropout"):
        if model_cfg[key] != ckpt["config"]["model"][key]:
            raise RuntimeError(f"B0 model config mismatch: {key}")
    model = B0Baseline(**ckpt["config"]["model"]).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


class AlphaView(nn.Module):
    """Inference-only wrapper; alpha is global and has no learned parameters."""

    def __init__(self, model: B3LatentReconstruction, alpha: float):
        super().__init__()
        self.model = model
        self.alpha = float(alpha)

    def forward(self, batch):
        return self.model(batch, alpha=self.alpha)


def summarize_diagnostic(rows: list[dict]) -> dict:
    grouped = summary(rows)
    return {"clean": grouped["clean"], "mean_missing": grouped["mean_missing"],
            "robust_score_diagnostic": grouped["robust_score"],
            "by_modality": grouped["by_modality"], "by_rho": grouped["by_ratio"],
            "by_location": grouped["by_location"], "double_modality": grouped["double_stress"]}


def diagnostic(config_path: Path) -> dict:
    """Evaluate existing joint-finetuned B3 checkpoint at alpha 1 vs 0 only."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verify_benchmark(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, valid_loaders, _ = loaders(cfg)
    state = torch.load(CHECKPOINTS / cfg["outputs"]["diagnostic_checkpoint"],
                       map_location="cpu", weights_only=False)
    old = B3LatentReconstruction(**cfg["model"]).to(device).eval()
    old.load_state_dict(state["model_state_dict"])
    result = {}
    detail = {}
    for key, alpha in (("enabled", 1.0), ("disabled", 0.0)):
        rows = evaluate_benchmark(AlphaView(old, alpha), valid_loaders["valid"], device)
        result[key] = summarize_diagnostic(rows)
        detail[key] = rows
    result["paired_delta_enabled_minus_disabled"] = {
        metric: result["enabled"]["mean_missing"][metric] - result["disabled"]["mean_missing"][metric]
        for metric in ("accuracy", "macro_f1", "mae", "pearson", "selection_score")
    }
    result["checkpoint"] = cfg["outputs"]["diagnostic_checkpoint"]
    result["checkpoint_epoch"] = int(state["epoch"])
    result["is_checkpoint_selection"] = False
    write_json(METRICS / "b31_b3_checkpoint_enabled_disabled.json", result)
    write_json(METRICS / "b31_b3_checkpoint_enabled_disabled_detail.json", detail)
    return result


def _rec_loss(model: B31FrozenBackbone, outputs: dict, full: dict, masked: dict) -> torch.Tensor:
    pad = masked["padding_mask"].bool()
    availability = masked["availability_mask"].bool()
    targets = model.full_targets(full)
    per_modality = []
    for index, modality in enumerate(MODALITIES):
        missing = pad & ~availability[:, index]
        estimate = outputs["reconstructed"][modality]
        target = targets[modality]
        if missing.any():
            per_modality.append(nn.functional.smooth_l1_loss(estimate[missing], target[missing]))
        else:
            per_modality.append(estimate.sum() * 0.0)
    return torch.stack(per_modality).mean()


def _frozen_snapshot(model: B31FrozenBackbone) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if not k.startswith("reconstructors.")}


def _same_frozen(model: B31FrozenBackbone, snapshot: dict[str, torch.Tensor]) -> dict:
    current = model.state_dict()
    changed = [key for key, value in snapshot.items()
               if not torch.equal(current[key].detach().cpu(), value)]
    return {"passed": not changed, "tensor_count": len(snapshot), "changed_tensors": changed}


def train(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verify_benchmark(cfg)
    diagnostic_path = METRICS / "b31_b3_checkpoint_enabled_disabled.json"
    if not diagnostic_path.exists():
        raise RuntimeError("run the required zero-training B3 checkpoint diagnostic first")
    seed = int(cfg["training"]["seed"])
    if seed != 42:
        raise ValueError("B3.1 uses training seed 42 only")
    seed_everything(seed)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, data_loaders, _ = loaders(cfg)
    b0, b0_state = load_b0(cfg["training"]["init_checkpoint"], cfg["model"], device)
    model = B31FrozenBackbone(**cfg["model"]).to(device)
    incompatible = model.load_state_dict(b0_state["model_state_dict"], strict=False)
    if incompatible.unexpected_keys or not incompatible.missing_keys or not all(
            name.startswith("reconstructors.") for name in incompatible.missing_keys):
        raise RuntimeError(f"B0 init mismatch: {incompatible}")
    model.freeze_backbone()
    if any(parameter.requires_grad for name, parameter in model.named_parameters()
           if not name.startswith("reconstructors.")):
        raise RuntimeError("a B0 backbone parameter remains trainable")
    if not all(parameter.requires_grad for parameter in model.reconstructors.parameters()):
        raise RuntimeError("some reconstructor parameters are frozen")
    frozen_before = _frozen_snapshot(model)
    counts = np.bincount(datasets["train"].cls_labels, minlength=3)
    if np.any(counts == 0):
        raise ValueError("empty training label class")
    weights = torch.as_tensor(len(datasets["train"]) / (3.0 * counts),
                              dtype=torch.float32, device=device)
    tcfg = cfg["training"]
    optimizer = torch.optim.AdamW(model.reconstructors.parameters(),
                                  lr=float(tcfg["learning_rate"]),
                                  weight_decay=float(tcfg["weight_decay"]))
    history = []
    started = time.perf_counter()
    epochs = int(tcfg["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        sums = {k: 0.0 for k in ("total", "task", "reconstruction")}
        count = 0
        for batch in data_loaders["train"]:
            full = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()}
            masked, _ = augment_train_batch(full, rng)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(predictor_inputs(masked), alpha=1.0, return_aux=True)
            task = multitask_loss(outputs, masked, lambda_reg=float(tcfg["lambda_reg"]),
                                  class_weights=weights)["total"]
            rec = _rec_loss(model, outputs, full, masked)
            total = task + float(tcfg["lambda_rec"]) * rec
            total.backward()
            if any(parameter.grad is not None for name, parameter in model.named_parameters()
                   if not name.startswith("reconstructors.")):
                raise RuntimeError("gradient escaped into frozen B0 backbone")
            optimizer.step()
            n = int(masked["cls_label"].shape[0])
            count += n
            for key, value in (("total", total), ("task", task), ("reconstruction", rec)):
                sums[key] += float(value.item()) * n
        row = {"epoch": epoch, **{k: v / count for k, v in sums.items()}}
        history.append(row)
        print(json.dumps({"model": cfg["run_name"], **row}), flush=True)
    seconds = time.perf_counter() - started
    frozen_check = _same_frozen(model, frozen_before)
    if not frozen_check["passed"]:
        raise RuntimeError(f"frozen B0 tensors changed: {frozen_check}")
    # An independent comparison to every tensor in the source B0 checkpoint.
    state_now = model.state_dict()
    source = b0_state["model_state_dict"]
    source_mismatches = [key for key, value in source.items()
                         if not torch.equal(state_now[key].detach().cpu(), value.detach().cpu())]
    frozen_check["matches_source_checkpoint"] = not source_mismatches
    frozen_check["source_mismatch_tensors"] = source_mismatches
    if source_mismatches:
        raise RuntimeError(f"B0 tensors no longer match source checkpoint: {source_mismatches}")
    path = CHECKPOINTS / cfg["outputs"]["checkpoint"]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"reconstructor_state_dict": model.reconstructors.state_dict(),
                "config": cfg, "training_seed": seed, "epochs": epochs,
                "b0_checkpoint": cfg["training"]["init_checkpoint"],
                "benchmark_sha256": BENCHMARK_SHA256,
                "frozen_backbone_check": frozen_check,
                "train_class_counts": counts.tolist(), "class_weights": weights.cpu()}, path)
    write_json(METRICS / cfg["outputs"]["history"], history)
    result = {"model": cfg["run_name"], "training_seed": seed,
              "epochs": epochs, "training_seconds": seconds,
              "parameter_count_total": sum(p.numel() for p in model.parameters()),
              "parameter_count_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "frozen_backbone_check": frozen_check,
              "loss": {"lambda_rec": tcfg["lambda_rec"], "lambda_id": 0.0,
                       "lambda_reg": tcfg["lambda_reg"]},
              "batch_size": cfg["data"]["batch_size"], "checkpoint": str(path),
              "test_role": "never constructed/evaluated"}
    write_json(METRICS / cfg["outputs"]["training_metrics"], result)
    return result


@torch.no_grad()
def alpha_zero_equivalence(model: B31FrozenBackbone, b0: B0Baseline,
                           loader, device: torch.device) -> dict:
    model.eval()
    b0.eval()
    maximum = {"classification_logits": 0.0, "regression": 0.0}
    exact = True
    count = 0
    for batch in loader:
        moved = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        inputs = predictor_inputs(moved)
        expected = b0(inputs)
        actual = model(inputs, alpha=0.0)
        for key in maximum:
            diff = (expected[key] - actual[key]).abs()
            maximum[key] = max(maximum[key], float(diff.max().item()))
            exact = exact and torch.equal(expected[key], actual[key])
        count += len(moved["cls_label"])
    return {"passed": exact, "exact_tensor_match": exact,
            "max_absolute_error": maximum, "valid_samples": count}


def _vision_subset_summary(rows: list[dict]) -> dict:
    values = [row["vision_all_zero_metrics"] for row in rows]
    clean = values[0]
    missing = {metric: float(np.mean([x[metric] for x in values[1:]]))
               for metric in ("accuracy", "macro_f1", "mae", "pearson")}
    clean["selection_score"] = validation_selection_score(clean)
    missing["selection_score"] = validation_selection_score(missing)
    return {"count": rows[0]["vision_all_zero_count"], "clean": clean,
            "mean_missing": missing}


def sweep(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    verify_benchmark(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, data_loaders, _ = loaders(cfg)
    source_b0, b0_state = load_b0(cfg["training"]["init_checkpoint"], cfg["model"], device)
    model = B31FrozenBackbone(**cfg["model"]).to(device).eval()
    model.load_state_dict(b0_state["model_state_dict"], strict=False)
    state = torch.load(CHECKPOINTS / cfg["outputs"]["checkpoint"],
                       map_location="cpu", weights_only=False)
    if state["benchmark_sha256"] != BENCHMARK_SHA256 or state["training_seed"] != 42:
        raise RuntimeError("B3.1 checkpoint provenance mismatch")
    model.reconstructors.load_state_dict(state["reconstructor_state_dict"])
    model.freeze_backbone()
    frozen_check = state["frozen_backbone_check"]
    if not frozen_check["passed"] or not frozen_check["matches_source_checkpoint"]:
        raise RuntimeError("saved checkpoint did not pass frozen tensor equality")
    alpha0_check = alpha_zero_equivalence(model, source_b0, data_loaders["valid"], device)
    if not alpha0_check["passed"]:
        raise RuntimeError(f"alpha=0 failed exact B0 reproduction: {alpha0_check}")
    write_json(METRICS / "b31_alpha0_equivalence.json", alpha0_check)
    previous_b0 = json.loads((METRICS / "b2_b0_inherent_detail.json").read_text(encoding="utf-8"))
    alpha_details = {}
    alpha_summaries = {}
    for alpha in ALPHAS:
        view = AlphaView(model, alpha)
        rows = evaluate_benchmark(view, data_loaders["valid"], device)
        alpha_details[str(alpha)] = rows
        alpha_summaries[str(alpha)] = summary(rows)
    # Verify every scenario metric at alpha zero matches the previous direct B0 eval.
    alpha0_rows = alpha_details["0.0"]
    compare_fields = ("accuracy", "macro_f1", "mae", "pearson", "confusion_matrix", "per_class")
    alpha0_metric_match = all(alpha0_rows[i][field] == previous_b0[i][field]
                              for i in range(55) for field in compare_fields)
    if not alpha0_metric_match:
        raise RuntimeError("alpha=0 scenario metrics diverge from stored direct B0 benchmark")
    summaries = {}
    for alpha in ALPHAS:
        rows = alpha_details[str(alpha)]
        current = alpha_summaries[str(alpha)]
        summaries[str(alpha)] = {
            "clean": current["clean"], "mean_missing": current["mean_missing"],
            "robust_score": current["robust_score"],
            "by_modality": current["by_modality"], "by_rho": current["by_ratio"],
            "by_location": current["by_location"],
            "double_modality": current["double_stress"],
            "vision_all_zero": _vision_subset_summary(rows),
        }
    best_alpha = max(ALPHAS, key=lambda a: summaries[str(a)]["robust_score"])
    result = {"training_seed": 42, "benchmark_seed": 20260923,
              "benchmark_sha256": BENCHMARK_SHA256,
              "checkpoint": cfg["outputs"]["checkpoint"],
              "alpha0_equivalence": alpha0_check,
              "alpha0_metrics_match_previous_B0_scenarios": alpha0_metric_match,
              "alphas": summaries, "best_alpha_by_robust_score": best_alpha,
              "above_alpha0_consistent_improvement": all(
                  summaries[str(alpha)]["robust_score"] > summaries["0.0"]["robust_score"]
                  for alpha in ALPHAS[1:]),
              "alpha_details": alpha_details}
    write_json(METRICS / "b31_alpha_sweep.json", result)
    rows_csv = []
    for alpha in ALPHAS:
        rows_csv.extend(flat_rows(f"B3.1-alpha{alpha:g}", alpha_details[str(alpha)]))
    write_csv(METRICS / "b31_alpha_scenarios.csv", rows_csv)
    make_report(result)
    return result


def make_report(result: dict) -> None:
    diagnostic_path = METRICS / "b31_b3_checkpoint_enabled_disabled.json"
    training_path = METRICS / "b31_training_metrics.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    lines = ["# Q2 B3.1 frozen-backbone conservative reconstruction", "",
             "All validation metrics use the unchanged B2 benchmark; no test or attachment3 data used.",
             "B0 prediction parameters are frozen; alpha is a global inference blend.",
             f"Training seed {training['training_seed']}, epochs {training['epochs']}, "
             f"training time {training['training_seconds']:.2f} s; "
             f"trainable parameters {training['parameter_count_trainable']:,}.",
             f"Frozen B0 equality: {training['frozen_backbone_check']['passed']}; "
             f"matches source checkpoint: {training['frozen_backbone_check']['matches_source_checkpoint']} "
             f"({training['frozen_backbone_check']['tensor_count']} tensors checked).", "",
             "## Existing B3 checkpoint: replacement enabled vs disabled", "",
             f"Checkpoint epoch {diagnostic['checkpoint_epoch']}; diagnostic only, not used for selection.", "",
             "| Mode | Mean missing Acc | Macro-F1 | MAE | Pearson | Score | Robust diagnostic |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, label in (("enabled", "Enabled"), ("disabled", "Disabled")):
        data = diagnostic[key]
        m = data["mean_missing"]
        lines.append(f"| {label} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | "
                     f"{m['mae']:.4f} | {m['pearson']:.4f} | {m['selection_score']:.4f} | "
                     f"{data['robust_score_diagnostic']:.4f} |")
    lines += ["", "### B3 diagnostic by modality", "",
              "| Mode | Modality | Accuracy | Macro-F1 | MAE | Pearson | Score |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for key, label in (("enabled", "Enabled"), ("disabled", "Disabled")):
        for group, values in diagnostic[key]["by_modality"].items():
            lines.append(f"| {label} | {group} | " + " | ".join(
                f"{values[k]:.4f}" for k in ("accuracy", "macro_f1", "mae", "pearson", "selection_score")) + " |")
    lines += ["", "### B3 diagnostic by rho", "",
              "| Mode | Rho | Accuracy | Macro-F1 | MAE | Pearson | Score |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for key, label in (("enabled", "Enabled"), ("disabled", "Disabled")):
        for group, values in diagnostic[key]["by_rho"].items():
            lines.append(f"| {label} | {group} | " + " | ".join(
                f"{values[k]:.4f}" for k in ("accuracy", "macro_f1", "mae", "pearson", "selection_score")) + " |")
    lines += ["", "## Conservative alpha sweep", "",
             "| Alpha | Clean Acc | Clean F1 | Clean MAE | Clean r | Clean score | Missing Acc | Missing F1 | Missing MAE | Missing r | Missing score | Robust score |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for alpha, data in result["alphas"].items():
        clean, missing = data["clean"], data["mean_missing"]
        values = [clean[k] for k in ("accuracy", "macro_f1", "mae", "pearson", "selection_score")]
        values += [missing[k] for k in ("accuracy", "macro_f1", "mae", "pearson", "selection_score")]
        values += [data["robust_score"]]
        lines.append(f"| {alpha} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    for section in ("by_modality", "by_rho", "by_location", "double_modality"):
        lines += ["", f"## {section}", "", "| Alpha | Group | Accuracy | Macro-F1 | MAE | Pearson | Score |",
                  "|---:|---|---:|---:|---:|---:|---:|"]
        for alpha, data in result["alphas"].items():
            for group, values in data[section].items():
                lines.append(f"| {alpha} | {group} | " + " | ".join(
                    f"{values[key]:.4f}" for key in
                    ("accuracy", "macro_f1", "mae", "pearson", "selection_score")) + " |")
    lines += ["", "## vision_all_zero (15 validation samples)", "",
              "| Alpha | Clean Acc | Clean F1 | Clean MAE | Clean r | Clean score | Missing Acc | Missing F1 | Missing MAE | Missing r | Missing score |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for alpha, data in result["alphas"].items():
        clean, missing = data["vision_all_zero"]["clean"], data["vision_all_zero"]["mean_missing"]
        clean_score = clean.get("selection_score", validation_selection_score(clean))
        missing_score = missing.get("selection_score", validation_selection_score(missing))
        lines.append(f"| {alpha} | " + " | ".join(f"{x:.4f}" for x in
                     (clean["accuracy"], clean["macro_f1"], clean["mae"], clean["pearson"],
                      clean_score, missing["accuracy"], missing["macro_f1"],
                      missing["mae"], missing["pearson"], missing_score)) + " |")
    (METRICS / "b31_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("diagnostic", "train", "sweep"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "b31_frozen_reconstruction.yaml")
    args = parser.parse_args()
    if args.phase == "diagnostic":
        print(json.dumps(diagnostic(args.config), ensure_ascii=False), flush=True)
    elif args.phase == "train":
        print(json.dumps(train(args.config), ensure_ascii=False), flush=True)
    else:
        result = sweep(args.config)
        print(json.dumps({"best_alpha": result["best_alpha_by_robust_score"],
                          "robust_scores": {a: v["robust_score"] for a, v in result["alphas"].items()},
                          "alpha0_equivalence": result["alpha0_equivalence"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
