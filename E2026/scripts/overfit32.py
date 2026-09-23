"""Fixed 32-sample implementation check; does not use validation/test data."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import build_datasets_and_loaders
from src.models.baseline import B0Baseline, multitask_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="/root/workspace/E2026/data/raw/attachment2/aligned_50.pkl")
    parser.add_argument("--output", default="outputs/metrics/b0_overfit32.json")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    seed = 2026
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    datasets, _, _ = build_datasets_and_loaders(args.pkl, normalization="none", batch_size=32, num_workers=0, seed=seed)
    # Fixed first 32 training examples; no split other than train is touched.
    sample_ids = list(range(32))
    samples = [datasets["train"][i] for i in sample_ids]
    batch = {k: ([x[k] for x in samples] if k == "id" else torch.stack([x[k] for x in samples])) for k in samples[0]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    model = B0Baseline(dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        out = model(batch); init_parts = multitask_loss(out, batch)
        initial_loss = float(init_parts["total"].item())
        initial_accuracy = float((out["classification_logits"].argmax(-1) == batch["cls_label"]).float().mean().item())
    losses = []
    for step in range(1, args.steps + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        out = model(batch); parts = multitask_loss(out, batch)
        parts["total"].backward(); optimizer.step()
        losses.append(float(parts["total"].item()))
        if step % 50 == 0: print(f"step={step} loss={losses[-1]:.5f}", flush=True)
    model.eval()
    with torch.no_grad():
        out = model(batch)
        final_loss = float(multitask_loss(out, batch)["total"].item())
        final_accuracy = float((out["classification_logits"].argmax(-1) == batch["cls_label"]).float().mean().item())
    result = {
        "train_indices": sample_ids, "device": str(device), "steps": args.steps,
        "initial_loss": initial_loss, "final_loss": final_loss,
        "initial_accuracy": initial_accuracy, "final_accuracy": final_accuracy,
        "loss_reduction_ratio": final_loss / max(initial_loss, 1e-12),
        "elapsed_seconds": time.perf_counter() - started,
        "passed": final_loss < initial_loss * 0.5 and final_accuracy >= max(initial_accuracy, 0.75),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]: raise SystemExit("32-sample overfit check did not pass; full training must not start")


if __name__ == "__main__":
    main()
