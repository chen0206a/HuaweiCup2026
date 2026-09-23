"""Train B0, select by validation loss, and evaluate the test holdout once."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.preprocess import build_datasets_and_loaders
from src.models.baseline import B0Baseline, multitask_loss
from src.training.evaluate import evaluate_loader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def train(config_path: str | Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["training"]
    seed_everything(int(tcfg["seed"]))
    requested_device = tcfg.get("device", "auto")
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else
                          "cpu" if requested_device == "auto" else requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    evaluate_test = bool(tcfg.get("evaluate_test", True))
    datasets, loaders, scaler = build_datasets_and_loaders(
        cfg["data"]["pkl_path"],
        normalization=cfg["preprocessing"]["normalization"],
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        seed=int(tcfg["seed"]),
        include_test=evaluate_test,
    )
    class_weighting = tcfg.get("class_weighting", "none")
    if class_weighting == "balanced_train":
        train_counts = np.bincount(datasets["train"].cls_labels, minlength=3)
        if np.any(train_counts == 0):
            raise ValueError(f"cannot compute balanced weights from empty train class: {train_counts.tolist()}")
        class_weights = torch.tensor(
            len(datasets["train"]) / (3.0 * train_counts), dtype=torch.float32, device=device
        )
    elif class_weighting == "none":
        train_counts = np.bincount(datasets["train"].cls_labels, minlength=3)
        class_weights = None
    else:
        raise ValueError("training.class_weighting must be 'none' or 'balanced_train'")
    model = B0Baseline(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tcfg["learning_rate"]), weight_decay=float(tcfg["weight_decay"])
    )
    lambda_reg = float(tcfg["lambda_reg"])
    metrics_dir = Path(cfg["outputs"]["metrics_dir"])
    checkpoints_dir = Path(cfg["outputs"]["checkpoints_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / cfg["outputs"].get("checkpoint_name", "b0_best.pt")

    best_valid = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    start_time = time.perf_counter()
    for epoch in range(1, int(tcfg["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in loaders["train"]:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss_parts = multitask_loss(
                output, batch, lambda_reg=lambda_reg, class_weights=class_weights
            )
            loss_parts["total"].backward()
            optimizer.step()
            n = len(batch["cls_label"])
            running_loss += loss_parts["total"].item() * n
            seen += n

        valid = evaluate_loader(
            model, loaders["valid"], device, lambda_reg=lambda_reg, class_weights=class_weights
        )
        train_loss = running_loss / max(seen, 1)
        valid_loss = float(valid["loss"]["total"])
        row = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss,
               "valid_accuracy": valid["accuracy"], "valid_macro_f1": valid["macro_f1"],
               "valid_mae": valid["mae"], "valid_pearson": valid["pearson"]}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "best_epoch": best_epoch,
                "valid_loss": best_valid,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "normalization": cfg["preprocessing"]["normalization"],
                "class_weights": class_weights.detach().cpu() if class_weights is not None else None,
                "train_class_counts": train_counts.tolist(),
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= int(tcfg["patience"]):
                break

    elapsed = time.perf_counter() - start_time
    history_path = metrics_dir / cfg["outputs"].get("history_filename", "b0_training_history.json")
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    # These are final reporting evaluations. Test is never used for selection.
    train_metrics = evaluate_loader(
        model, loaders["train"], device, lambda_reg=lambda_reg, class_weights=class_weights
    )
    valid_metrics = evaluate_loader(
        model, loaders["valid"], device, lambda_reg=lambda_reg, class_weights=class_weights
    )
    test_metrics = (
        evaluate_loader(model, loaders["test"], device, lambda_reg=lambda_reg)
        if evaluate_test else None
    )
    result = {
        "model": "B0Baseline",
        "normalization": cfg["preprocessing"]["normalization"],
        "device": str(device),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_seconds": elapsed,
        "best_epoch": best_epoch,
        "best_valid_selection_loss": best_valid,
        "class_weighting": class_weighting,
        "train_class_counts": train_counts.tolist(),
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        "split_sizes": {name: len(ds) for name, ds in datasets.items()},
        "metrics": {"train": train_metrics, "valid": valid_metrics, "test": test_metrics},
        "test_role": (
            "final_holdout_only; evaluated after model selection" if evaluate_test
            else "not loaded into Dataset, evaluated, or used for any decision"
        ),
        "checkpoint": str(checkpoint_path),
    }
    metrics_path = metrics_dir / cfg["outputs"].get("metrics_filename", "b0_metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"training_seconds": elapsed, "best_epoch": best_epoch,
                      "parameter_count": result["parameter_count"],
                      "metrics_path": str(metrics_path)}, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/b0.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
