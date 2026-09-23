"""B1 CPU/CUDA smoke using actual aligned-50 train data only."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import build_datasets_and_loaders
from src.models.temporal import B1TemporalEncoder


def one_device(batch, device):
    model = B1TemporalEncoder().to(device)
    moved = {key: value.to(device) if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    output = model(moved)
    loss = output["classification_logits"].square().mean() + output["regression"].square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "b1.pt"
        torch.save(model.state_dict(), path)
        B1TemporalEncoder().load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return {"device": str(device), "loss": float(loss.item()), "forward_backward": True,
            "checkpoint_roundtrip": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="/root/workspace/E2026/data/raw/attachment2/aligned_50.pkl")
    parser.add_argument("--output", default="outputs/metrics/b1_smoke.json")
    args = parser.parse_args()
    datasets, loaders, _ = build_datasets_and_loaders(
        args.pkl, normalization="none", batch_size=4, include_test=False
    )
    batch = next(iter(loaders["train"]))
    results = [one_device(batch, torch.device("cpu"))]
    if torch.cuda.is_available():
        results.append(one_device(batch, torch.device("cuda")))
    result = {"split_sizes": {key: len(value) for key, value in datasets.items()},
              "batch_shapes": {key: list(value.shape) for key, value in batch.items()
                               if isinstance(value, torch.Tensor)}, "devices": results,
              "test_used": False}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
