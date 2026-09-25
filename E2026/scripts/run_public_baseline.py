"""Train public-architecture baselines on Attachment2 train/valid only.

Run from E2026/. No Attachment2 test, Attachment3, or Attachment4 loader is used.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.preprocess import build_datasets_and_loaders
from src.models.baseline import multitask_loss
from src.models.public_baselines import MODELS
from src.training.evaluate import evaluate_loader
from src.training.train import seed_everything
from src.utils.metrics import validation_selection_score


def run(model_name: str, seed: int, *, max_epochs: int = 80, smoke: bool = False,
        missing: bool = False) -> dict:
    root = Path(__file__).resolve().parents[1]
    pkl = root / "data/raw/aligned_50.pkl"
    definition = root / "outputs/metrics/b2_benchmark_definition.json"
    import hashlib
    digest = hashlib.sha256(definition.read_bytes()).hexdigest()
    expected = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"
    if digest != expected:
        raise RuntimeError(f"frozen benchmark definition changed: {digest}")
    torch.set_num_threads(6)
    seed_everything(seed)
    datasets, loaders, _ = build_datasets_and_loaders(
        pkl, normalization="none", batch_size=128, num_workers=0,
        seed=seed, include_test=False)
    if (len(datasets["train"]), len(datasets["valid"])) != (3395, 728):
        raise RuntimeError("Attachment2 split size differs from frozen audit")
    if smoke:
        # Same train/valid protocol, but a short diagnostic run with no formal result.
        max_epochs = min(max_epochs, 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODELS[model_name]().to(device)
    counts = np.bincount(datasets["train"].cls_labels, minlength=3)
    weights = torch.tensor(len(datasets["train"]) / (3.0 * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    out = root / "experiments/q2/public_baselines" / model_name
    out.mkdir(parents=True, exist_ok=True)
    name = f"seed{seed}" + ("_smoke" if smoke else "")
    checkpoint = out / f"checkpoint_{name}.pt"
    history, best_score, best_epoch, stale = [], -float("inf"), 0, 0
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total, n_seen = 0.0, 0
        for batch in loaders["train"]:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            parts = multitask_loss(prediction, batch, lambda_reg=1.0, class_weights=weights)
            loss = parts["total"]
            aux_values = {}
            if model_name == "MISA":
                aux_values = model.auxiliary_loss()
                loss = loss + 0.3 * aux_values["diff"] + aux_values["sim"] + aux_values["recon"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: {model_name} seed={seed} epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            size = batch["cls_label"].numel()
            total += float(loss.item()) * size
            n_seen += size
        valid = evaluate_loader(model, loaders["valid"], device,
                                lambda_reg=1.0, class_weights=weights)
        score = validation_selection_score(valid)
        record = {"epoch": epoch, "train_loss": total / n_seen,
                  "valid_accuracy": valid["accuracy"], "valid_macro_f1": valid["macro_f1"],
                  "valid_mae": valid["mae"], "valid_pearson": valid["pearson"],
                  "valid_selection_score": score}
        history.append(record)
        print(json.dumps({"model": model_name, "seed": seed, **record}), flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "model_name": model_name,
                        "seed": seed, "epoch": epoch, "score": score,
                        "train_counts": counts.tolist(), "train_weights": weights.cpu().tolist(),
                        "protocol": "Attachment2 train/valid; none; batch128; AdamW 1e-3 1e-4; WCE+SmoothL1; clean score"}, checkpoint)
        else:
            stale += 1
            if stale >= 12 and not smoke:
                break
    elapsed = time.perf_counter() - start
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    valid = evaluate_loader(model, loaders["valid"], device,
                            lambda_reg=1.0, class_weights=weights)
    result = {"model": model_name, "seed": seed, "smoke": smoke,
              "benchmark_definition_sha256": digest,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "training_seconds": elapsed, "epochs_run": len(history),
              "best_clean_epoch": best_epoch, "clean_selection_score": best_score,
              "clean_valid": valid, "history": history,
              "checkpoint": str(checkpoint.relative_to(root)), "missing": None}
    if missing and not smoke:
        from src.evaluation.missing_benchmark import evaluate_benchmark, summary
        rows = evaluate_benchmark(model, loaders["valid"], device, scenario_chunk_size=2)
        result["missing"] = {"rows": rows, "summary": summary(rows)}
    metrics = out / f"metrics_{name}.json"
    metrics.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"completed": str(metrics.relative_to(root)), "best_epoch": best_epoch,
                      "clean_score": best_score, "seconds": elapsed}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--missing", action="store_true")
    args = parser.parse_args()
    run(args.model, args.seed, max_epochs=args.epochs, smoke=args.smoke, missing=args.missing)
