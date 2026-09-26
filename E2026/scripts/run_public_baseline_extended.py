"""Aligned-50 architecture adaptations, clean-selected train/valid protocol.

No test or specialist attachment is loaded. Original paper training pipelines are
not claimed; see adaptation notes in the experiment output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

from src.data.block_mask import augment_train_batch
from src.data.preprocess import build_datasets_and_loaders
from src.evaluation.missing_benchmark import evaluate_benchmark, summary
from src.models.baseline import multitask_loss
from src.models.public_baselines_extended import EXTENDED_MODELS
from src.training.evaluate import evaluate_loader
from src.training.train import seed_everything
from src.utils.metrics import validation_selection_score

ROOT = Path(__file__).resolve().parents[1]
DEFINITION_HASH = "3dda8bcef01bb5eba005e4ac7b729cccec5c1943177066d7c82a15ad99205bff"
MISSING_AWARE = {"TFR-Net", "MissModal", "M3S", "MMIN"}


def data(seed: int):
    digest = hashlib.sha256((ROOT / "outputs/metrics/b2_benchmark_definition.json").read_bytes()).hexdigest()
    if digest != DEFINITION_HASH:
        raise RuntimeError(f"Frozen definition mismatch: {digest}")
    datasets, loaders, _ = build_datasets_and_loaders(
        ROOT / "data/raw/aligned_50.pkl", normalization="none", batch_size=128,
        num_workers=0, seed=seed, include_test=False)
    if (len(datasets["train"]), len(datasets["valid"])) != (3395, 728):
        raise RuntimeError("Unexpected train/valid sizes")
    return datasets, loaders


def move(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def mask_full_modality(batch: dict, rng: random.Random) -> dict:
    out = dict(batch)
    out["clean_features"] = {m: batch[m] for m in ("text", "audio", "vision")}
    for m in ("text", "audio", "vision"):
        out[m] = batch[m].clone()
    for i in range(batch["cls_label"].size(0)):
        if rng.random() < 0.5:
            choices = rng.sample(("text", "audio", "vision"), 1 if rng.random() < 0.75 else 2)
            for m in choices:
                out[m][i] = 0
    return out


def augment(model_name: str, batch: dict, rng: random.Random) -> dict:
    if model_name not in MISSING_AWARE:
        return batch
    if model_name == "MMIN":
        return mask_full_modality(batch, rng)
    masked, _ = augment_train_batch(batch, rng)
    masked["clean_features"] = {m: batch[m] for m in ("text", "audio", "vision")}
    return masked


def loss_for(model, name: str, batch: dict, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred = model(batch)
    primary = multitask_loss(pred, batch, lambda_reg=1.0, class_weights=weights)["total"]
    auxiliary = model.auxiliary_loss(batch, pred) if hasattr(model, "auxiliary_loss") else primary.new_zeros(())
    return primary + auxiliary, auxiliary


def m3s_loss(model, batch: dict, weights: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """One-step first-order meta-sampling: clean support, masked query."""
    clean_loss = multitask_loss(model(batch), batch, lambda_reg=1.0, class_weights=weights)["total"]
    names, params = zip(*model.named_parameters())
    grads = torch.autograd.grad(clean_loss, params, create_graph=False, retain_graph=True)
    adapted = {n: p - 0.001 * g.detach() for n, p, g in zip(names, params, grads)}
    query = augment("M3S", batch, rng)
    prediction = functional_call(model, adapted, (query,))
    query_loss = multitask_loss(prediction, query, lambda_reg=1.0, class_weights=weights)["total"]
    return 0.5 * clean_loss + 0.5 * query_loss


def preflight(name: str, seed: int, datasets, loaders, device: torch.device, weights: torch.Tensor) -> dict:
    seed_everything(seed)
    a = EXTENDED_MODELS[name]().to(device)
    initial = {k: v.detach().cpu().clone() for k, v in a.state_dict().items()}
    seed_everything(seed)
    b = EXTENDED_MODELS[name]().to(device)
    if not all(torch.equal(initial[k], v.detach().cpu()) for k, v in b.state_dict().items()):
        raise RuntimeError("Initial state is not seed reproducible")
    del b
    batch = move(next(iter(loaders["train"])), device)
    for m, d in (("text", 768), ("audio", 74), ("vision", 35)):
        if tuple(batch[m].shape[1:]) != (50, d):
            raise RuntimeError(f"Input shape {m}")
    rng = random.Random(seed)
    a.train()
    opt = torch.optim.AdamW(a.parameters(), lr=0.001, weight_decay=0.0001)
    losses = []
    for _ in range(10):
        use = augment(name, batch, rng)
        opt.zero_grad(set_to_none=True)
        if name == "M3S":
            loss = m3s_loss(a, batch, weights, rng)
        else:
            loss, _ = loss_for(a, name, use, weights)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite preflight loss: {name}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in a.parameters()):
            raise RuntimeError(f"Nonfinite preflight gradient: {name}")
        torch.nn.utils.clip_grad_norm_(a.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.item()))
    a.eval()
    with torch.no_grad():
        valid = move(next(iter(loaders["valid"])), device)
        out = a(valid)
        if out["classification_logits"].shape != (valid["cls_label"].size(0), 3):
            raise RuntimeError("Classification output shape")
        if out["regression"].shape != valid["reg_label"].shape:
            raise RuntimeError("Regression output shape")
        if not all(torch.isfinite(x).all() for x in out.values()):
            raise RuntimeError("Nonfinite valid output")
    return {"model": name, "seed": seed, "parameters": sum(p.numel() for p in a.parameters()),
            "ten_step_loss_first": losses[0], "ten_step_loss_last": losses[-1],
            "ten_step_losses": losses, "valid_forward": "PASS", "finite_gradient": "PASS",
            "initial_seed_reproducibility": "PASS"}


def run(name: str, seed: int, *, preflight_only: bool = False) -> dict:
    torch.set_num_threads(6)
    seed_everything(seed)
    datasets, loaders = data(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    counts = np.bincount(datasets["train"].cls_labels, minlength=3)
    weights = torch.tensor(len(datasets["train"]) / (3 * counts), dtype=torch.float32, device=device)
    outdir = ROOT / "experiments/q2/public_baselines" / name
    outdir.mkdir(parents=True, exist_ok=True)
    pre = preflight(name, seed, datasets, loaders, device, weights)
    (outdir / f"preflight_seed{seed}.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")
    print(json.dumps({"preflight": pre}), flush=True)
    if preflight_only:
        return pre

    # Preflight is isolated: all formal runs reset RNG before model init and loader iteration.
    seed_everything(seed)
    loaders["train"].generator.manual_seed(seed)
    model = EXTENDED_MODELS[name]().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    rng = random.Random(seed)
    best, best_epoch, stale, history = -float("inf"), 0, 0, []
    checkpoint = outdir / f"checkpoint_seed{seed}.pt"
    start = time.perf_counter()
    for epoch in range(1, 81):
        model.train()
        total, aux_total, n = 0.0, 0.0, 0
        for batch in loaders["train"]:
            batch = move(batch, device)
            use = augment(name, batch, rng)
            optimizer.zero_grad(set_to_none=True)
            if name == "M3S":
                loss = m3s_loss(model, batch, weights, rng)
                aux = loss.new_zeros(())
            else:
                loss, aux = loss_for(model, name, use, weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss {name}/{seed}/{epoch}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"Nonfinite gradient {name}/{seed}/{epoch}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            size = batch["cls_label"].numel()
            total += float(loss.item()) * size
            aux_total += float(aux.item()) * size
            n += size
        valid = evaluate_loader(model, loaders["valid"], device, lambda_reg=1.0, class_weights=weights)
        score = validation_selection_score(valid)
        record = {"epoch": epoch, "train_loss": total / n, "aux_loss": aux_total / n,
                  "clean_accuracy": valid["accuracy"], "clean_macro_f1": valid["macro_f1"],
                  "clean_mae": valid["mae"], "clean_pearson": valid["pearson"], "clean_score": score}
        history.append(record)
        print(json.dumps({"model": name, "seed": seed, **record}), flush=True)
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "model_name": name,
                        "seed": seed, "epoch": epoch, "clean_score": score,
                        "training_regime": "missing-aware training" if name in MISSING_AWARE else "standard clean training",
                        "class_counts": counts.tolist(), "class_weights": weights.cpu().tolist()}, checkpoint)
        else:
            stale += 1
            if stale >= 12:
                break
    train_seconds = time.perf_counter() - start
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    valid = evaluate_loader(model, loaders["valid"], device, lambda_reg=1.0, class_weights=weights)
    eval_start = time.perf_counter()
    rows = evaluate_benchmark(model, loaders["valid"], device, scenario_chunk_size=2)
    benchmark_seconds = time.perf_counter() - eval_start
    result = {"model": name, "seed": seed, "parameter_count": pre["parameters"],
              "training_regime": "missing-aware training" if name in MISSING_AWARE else "standard clean training",
              "benchmark_definition_sha256": DEFINITION_HASH, "best_clean_epoch": best_epoch,
              "training_seconds": train_seconds, "benchmark_seconds": benchmark_seconds,
              "checkpoint": str(checkpoint.relative_to(ROOT)), "clean_valid": valid,
              "missing": {"rows": rows, "summary": summary(rows)}, "history": history}
    path = outdir / f"metrics_seed{seed}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"complete": str(path.relative_to(ROOT)), "best_epoch": best_epoch,
                      "clean_score": best, "training_seconds": train_seconds,
                      "benchmark_seconds": benchmark_seconds}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=EXTENDED_MODELS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.model, args.seed, preflight_only=args.preflight_only)
