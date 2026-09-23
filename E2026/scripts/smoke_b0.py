"""Exercise the audited data contract and B0 forward/backward on CPU and CUDA."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import build_datasets_and_loaders
from src.models.baseline import B0Baseline, multitask_loss


def run_device(batch, device: torch.device) -> dict:
    model = B0Baseline().to(device)
    moved = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    output = model(moved)
    assert output["classification_logits"].shape == (len(batch["id"]), 3)
    assert output["regression"].shape == (len(batch["id"]),)
    loss = multitask_loss(output, moved)["total"]
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.pt"
        torch.save(model.state_dict(), path)
        clone = B0Baseline().to(device)
        clone.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return {"device": str(device), "loss": float(loss.item()), "forward_backward": True,
            "checkpoint_roundtrip": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", default="/root/workspace/E2026/data/raw/attachment2/aligned_50.pkl")
    parser.add_argument("--output", default="outputs/metrics/b0_smoke.json")
    args = parser.parse_args()
    _, loaders, _ = build_datasets_and_loaders(args.pkl, batch_size=8, normalization="none", num_workers=0)
    batch = next(iter(loaders["train"]))
    assert batch["text"].shape == (8, 50, 768)
    assert batch["audio"].shape == (8, 50, 74) and batch["audio"].dtype == torch.float32
    assert batch["vision"].shape == (8, 50, 35) and batch["vision"].dtype == torch.float32
    assert batch["padding_mask"].shape == (8, 50) and batch["padding_mask"].dtype == torch.bool
    assert batch["availability_mask"].shape == (8, 3, 50)
    assert batch["native_zero_mask"].shape == (8, 3, 50)
    results = [run_device(batch, torch.device("cpu"))]
    if torch.cuda.is_available():
        results.append(run_device(batch, torch.device("cuda")))
    output = {"schema": {k: (list(v.shape), str(v.dtype)) for k, v in batch.items() if isinstance(v, torch.Tensor)},
              "devices": results, "vision_all_zero_in_batch": int(batch["vision_all_zero"].sum())}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
