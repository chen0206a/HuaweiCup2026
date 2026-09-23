"""Fixed first 32 training examples, checking B1 implementation can overfit."""
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
from src.models.baseline import multitask_loss
from src.models.temporal import B1TemporalEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="/root/workspace/E2026/data/raw/attachment2/aligned_50.pkl")
    parser.add_argument("--output", default="outputs/metrics/b1_overfit32.json")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    datasets, _, _ = build_datasets_and_loaders(
        args.pkl, normalization="none", batch_size=32, seed=seed, include_test=False
    )
    samples = [datasets["train"][i] for i in range(32)]
    batch = {key: ([sample[key] for sample in samples] if key == "id"
                   else torch.stack([sample[key] for sample in samples]))
             for key in samples[0]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    model = B1TemporalEncoder(dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0)
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        output = model(batch)
        initial_loss = float(multitask_loss(output, batch)["total"].item())
        initial_accuracy = float((output["classification_logits"].argmax(-1) == batch["cls_label"]).float().mean().item())
    for step in range(args.steps):
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = multitask_loss(output, batch)["total"]
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        output = model(batch)
        final_loss = float(multitask_loss(output, batch)["total"].item())
        final_accuracy = float((output["classification_logits"].argmax(-1) == batch["cls_label"]).float().mean().item())
    result = {
        "indices": list(range(32)), "device": str(device), "steps": args.steps,
        "initial_loss": initial_loss, "final_loss": final_loss,
        "initial_accuracy": initial_accuracy, "final_accuracy": final_accuracy,
        "elapsed_seconds": time.perf_counter() - started,
        "passed": final_loss < initial_loss * 0.5 and final_accuracy >= max(initial_accuracy, 0.75),
        "test_used": False,
    }
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("B1 32-sample overfit failed; formal training must not start")


if __name__ == "__main__":
    main()
