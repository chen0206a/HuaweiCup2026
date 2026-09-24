"""Q3-1 validation entry point; only Attachment2 valid is indexed."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import Aligned50Dataset
from src.q3.frozen_predictor import FrozenP2Predictor
from src.utils.metrics import compute_metrics, validation_selection_score


ROOT = Path(__file__).resolve().parents[2]
SHA42 = "cc4cf890a857042c9c3af1313abb16c73f109a18cb7706a939f401930e9efaff"


def valid_dataset() -> Aligned50Dataset:
    # The official pickle is one monolithic container. Deserialization necessarily
    # materializes it, but only its valid entry is selected/indexed/evaluated.
    with (ROOT / "data/raw/aligned_50.pkl").open("rb") as stream:
        container = pickle.load(stream)
    valid = container["valid"]
    del container
    return Aligned50Dataset(valid, "valid")


def clean_reproduction(predictor: FrozenP2Predictor, dataset: Aligned50Dataset) -> dict:
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    cls_true, cls_pred, reg_true, reg_pred = [], [], [], []
    for batch in loader:
        output = predictor.predict(batch)
        cls_true.extend(batch["cls_label"].tolist())
        cls_pred.extend(output["classification_logits"].argmax(dim=1).cpu().tolist())
        reg_true.extend(batch["reg_label"].tolist())
        reg_pred.extend(output["regression"].cpu().tolist())
    metrics = compute_metrics(cls_true, cls_pred, reg_true, reg_pred)
    metrics["selection_score"] = validation_selection_score(metrics)
    manifest = json.loads((ROOT / "outputs/final/q2/q2_checkpoint_manifest.json").read_text(encoding="utf-8"))
    locked = next(item for item in manifest["checkpoints"] if item["model"] == "B5-P2" and item["seed"] == 42)
    if locked["checkpoint"]["sha256"] != predictor.sha256:
        raise RuntimeError("Q2 manifest and loaded checkpoint hashes differ")
    reference = locked["validation_metrics"]["clean"]
    delta = {key: metrics[key] - reference[key] for key in ("accuracy", "macro_f1", "mae", "pearson", "selection_score")}
    # Predictions should reproduce the frozen Q2 evaluator in float32. A one-class
    # change is far above this threshold, and CPU/GPU reduction noise is below it.
    passed = all(abs(value) <= 1e-6 for value in delta.values())
    return {"passed": passed, "n_valid": len(dataset), "actual": metrics,
            "locked": reference, "delta": delta, "absolute_tolerance": 1e-6}


if __name__ == "__main__":
    torch.set_num_threads(min(8, torch.get_num_threads()))
    predictor = FrozenP2Predictor(
        ROOT / "outputs/checkpoints/b5_pooling_p2_best_robust_score.pt", SHA42, 42)
    print("checkpoint_verified", predictor.sha256, flush=True)
    dataset = valid_dataset()
    print("valid_loaded", len(dataset), flush=True)
    result = clean_reproduction(predictor, dataset)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result["passed"]:
        raise SystemExit("STOP: frozen P2 clean validation reproduction failed")
