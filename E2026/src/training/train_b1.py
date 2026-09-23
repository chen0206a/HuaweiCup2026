"""Train B1 temporal encoders with train/validation only and dual checkpoints."""
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
from src.models.baseline import multitask_loss
from src.models.temporal import B1TemporalEncoder
from src.training.evaluate import evaluate_loader
from src.utils.metrics import validation_selection_score


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: (value.to(device) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()}


def train(config_path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    train_cfg = cfg["training"]
    seed_everything(int(train_cfg["seed"]))
    request_device = train_cfg.get("device", "auto")
    device = torch.device(
        "cuda" if request_device == "auto" and torch.cuda.is_available()
        else "cpu" if request_device == "auto" else request_device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # B1 ablations only construct train/valid datasets and loaders.
    datasets, loaders, scaler = build_datasets_and_loaders(
        cfg["data"]["pkl_path"], normalization=cfg["preprocessing"]["normalization"],
        batch_size=int(cfg["data"]["batch_size"]), num_workers=int(cfg["data"]["num_workers"]),
        seed=int(train_cfg["seed"]), include_test=False,
    )
    counts = np.bincount(datasets["train"].cls_labels, minlength=3)
    class_weighting = train_cfg["class_weighting"]
    if class_weighting == "balanced_train":
        if np.any(counts == 0):
            raise ValueError(f"cannot compute balanced train weights: {counts.tolist()}")
        weights = torch.as_tensor(len(datasets["train"]) / (3.0 * counts), dtype=torch.float32, device=device)
    elif class_weighting == "none":
        weights = None
    else:
        raise ValueError("class_weighting must be none or balanced_train")

    model = B1TemporalEncoder(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    lambda_reg = float(train_cfg["lambda_reg"])
    metrics_dir = Path(cfg["outputs"]["metrics_dir"])
    checkpoints_dir = Path(cfg["outputs"]["checkpoints_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    loss_path = checkpoints_dir / cfg["outputs"]["best_loss_checkpoint"]
    score_path = checkpoints_dir / cfg["outputs"]["best_score_checkpoint"]

    best_loss = float("inf")
    best_loss_epoch = 0
    best_score = -float("inf")
    best_score_epoch = 0
    stale_loss_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        sum_loss = 0.0
        num_seen = 0
        for batch in loaders["train"]:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            parts = multitask_loss(output, batch, lambda_reg=lambda_reg, class_weights=weights)
            parts["total"].backward()
            optimizer.step()
            size = len(batch["cls_label"])
            sum_loss += float(parts["total"].item()) * size
            num_seen += size

        valid = evaluate_loader(model, loaders["valid"], device,
                                lambda_reg=lambda_reg, class_weights=weights)
        score = validation_selection_score(valid)
        valid_loss = float(valid["loss"]["total"])
        row = {
            "epoch": epoch, "train_loss": sum_loss / max(num_seen, 1),
            "valid_loss": valid_loss, "selection_score": score,
            "accuracy": valid["accuracy"], "macro_f1": valid["macro_f1"],
            "mae": valid["mae"], "pearson": valid["pearson"],
            "per_class": {name: {k: value[k] for k in ("recall", "f1")}
                          for name, value in valid["per_class"].items()},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        checkpoint_core = {
            "model_state_dict": model.state_dict(), "config": cfg,
            "class_weights": weights.detach().cpu() if weights is not None else None,
            "train_class_counts": counts.tolist(), "normalization": "none",
        }
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_loss_epoch = epoch
            stale_loss_epochs = 0
            torch.save({**checkpoint_core, "best_loss_epoch": epoch, "valid_loss": best_loss}, loss_path)
        else:
            stale_loss_epochs += 1
        if score > best_score:
            best_score = score
            best_score_epoch = epoch
            torch.save({**checkpoint_core, "best_score_epoch": epoch, "selection_score": best_score}, score_path)
        if stale_loss_epochs >= int(train_cfg["patience"]):
            break

    training_seconds = time.perf_counter() - started
    history_path = metrics_dir / cfg["outputs"]["history_filename"]
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    def evaluate_checkpoint(path: Path) -> dict:
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        return evaluate_loader(model, loaders["valid"], device,
                              lambda_reg=lambda_reg, class_weights=weights)

    valid_at_loss = evaluate_checkpoint(loss_path)
    valid_at_score = evaluate_checkpoint(score_path)
    result = {
        "model": cfg["run_name"], "architecture": "B1TemporalEncoder",
        "normalization": cfg["preprocessing"]["normalization"], "device": str(device),
        "parameter_count": sum(param.numel() for param in model.parameters()),
        "training_seconds": training_seconds, "train_class_counts": counts.tolist(),
        "class_weights": weights.detach().cpu().tolist() if weights is not None else None,
        "class_weighting": class_weighting,
        "best_valid_loss": {"epoch": best_loss_epoch, "value": best_loss,
                            "validation_metrics": valid_at_loss},
        "best_selection_score": {"epoch": best_score_epoch, "value": best_score,
                                  "validation_metrics": valid_at_score},
        "metrics": {"valid": valid_at_score},
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "test_role": "not constructed as Dataset/Loader, evaluated, or used for any decision",
        "best_loss_checkpoint": str(loss_path), "best_score_checkpoint": str(score_path),
    }
    metrics_path = metrics_dir / cfg["outputs"]["metrics_filename"]
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"training_seconds": training_seconds, "best_loss_epoch": best_loss_epoch,
                      "best_score_epoch": best_score_epoch, "metrics_path": str(metrics_path)}, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
