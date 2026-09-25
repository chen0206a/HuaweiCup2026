"""Final label-free Attachment3 inference after the text-interface gate passes."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.attachment3_inference import Attachment3InferenceDataset
from src.models.pooling_residual import B5PoolingResidual
from src.q3.text_feature_reconstruction import (
    MODEL_ID, REVISION, WEIGHTS_SHA256, load_pinned_bert,
)


OUT = ROOT / "outputs/final/q2/attachment3"
LOCK_PATH = ROOT / "outputs/final/q2/q2_model_lock.json"
MANIFEST_PATH = ROOT / "data/manifests/attachment3_sealed_inventory.json"
GATE_PATH = OUT / "attachment3_text_interface_audit.json"
DELIVERY_PATH = OUT / "attachment3_predictions.csv"
AUDIT_PATH = OUT / "attachment3_predictions_audit.csv"
SUMMARY_PATH = OUT / "attachment3_summary.json"
CLASS_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def describe(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "min": float(values.min()),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def main() -> None:
    torch.set_num_threads(1)
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("known_sample_count") != 20:
        raise RuntimeError("BLOCKED_BY_TEXT_INTERFACE: regression gate did not pass")
    if (gate["adapter"]["model"] != MODEL_ID
            or gate["adapter"]["model_revision"] != REVISION
            or gate["adapter"]["weights_sha256"] != WEIGHTS_SHA256):
        raise RuntimeError("Text adapter does not match the validated revision")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    checkpoint_info = lock["checkpoint_paths"]["B5-P2:42"]
    checkpoint_path = ROOT / checkpoint_info["relative_path"]
    hash_before = sha256(checkpoint_path)
    if hash_before != checkpoint_info["sha256"]:
        raise RuntimeError("locked Q2 checkpoint SHA256 mismatch")
    if lock["class_mapping"] != {str(i): CLASS_NAMES[i] for i in range(3)}:
        raise RuntimeError("locked class mapping changed")
    if lock["training_protocol"]["normalization"] != "none":
        raise RuntimeError("locked Q2 normalization changed")

    bert_model = load_pinned_bert(local_files_only=True)
    dataset = Attachment3InferenceDataset(MANIFEST_PATH, bert_model)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("mode") != "mean_attention":
        raise RuntimeError("locked Q2 checkpoint mode mismatch")
    predictor = B5PoolingResidual("mean_attention", **checkpoint["config"]["model"])
    predictor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    predictor.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            output = predictor(batch)
            logits = output["classification_logits"]
            regression = output["regression"]
            probabilities = torch.softmax(logits, dim=1)
            for i, sample_id in enumerate(batch["sample_id"]):
                cls = int(torch.argmax(probabilities[i]).item())
                prob = probabilities[i].cpu().numpy().astype(np.float64)
                logit = logits[i].cpu().numpy().astype(np.float64)
                rows.append({
                    "sample_id": sample_id,
                    "predicted_class_id": cls,
                    "predicted_class_name": CLASS_NAMES[cls],
                    "predicted_intensity": float(regression[i].item()),
                    "confidence": float(prob.max()),
                    "prob_negative": float(prob[0]),
                    "prob_neutral": float(prob[1]),
                    "prob_positive": float(prob[2]),
                    "logit_negative": float(logit[0]),
                    "logit_neutral": float(logit[1]),
                    "logit_positive": float(logit[2]),
                    "valid_length": int(batch["valid_length"][i].item()),
                    "source_file": batch["source_file"][i],
                    "source_sha256": batch["source_sha256"][i],
                })
    expected_ids = [record["sample_id"] for record in dataset.records]
    ids = [row["sample_id"] for row in rows]
    if len(rows) != 30 or len(set(ids)) != 30 or ids != expected_ids:
        raise RuntimeError("Attachment3 prediction coverage, uniqueness or order failed")
    max_probability_sum_error = 0.0
    for row in rows:
        if row["predicted_class_id"] not in CLASS_NAMES:
            raise RuntimeError("illegal predicted class")
        if row["predicted_class_name"] != CLASS_NAMES[row["predicted_class_id"]]:
            raise RuntimeError("class ID/name mapping mismatch")
        numbers = [row[key] for key in (
            "predicted_intensity", "confidence", "prob_negative", "prob_neutral",
            "prob_positive", "logit_negative", "logit_neutral", "logit_positive")]
        if not all(math.isfinite(value) for value in numbers):
            raise RuntimeError("nonfinite model output")
        p_sum = row["prob_negative"] + row["prob_neutral"] + row["prob_positive"]
        max_probability_sum_error = max(max_probability_sum_error, abs(p_sum - 1.0))
    if max_probability_sum_error > 1e-6:
        raise RuntimeError("softmax probability sums differ from one")
    hash_after = sha256(checkpoint_path)
    if hash_after != hash_before:
        raise RuntimeError("locked checkpoint changed during inference")

    OUT.mkdir(parents=True, exist_ok=True)
    delivery_columns = ["sample_id", "predicted_class_id", "predicted_class_name",
                        "predicted_intensity"]
    audit_columns = delivery_columns + ["confidence", "prob_negative", "prob_neutral",
        "prob_positive", "logit_negative", "logit_neutral", "logit_positive",
        "valid_length", "source_file", "source_sha256"]
    write_csv(DELIVERY_PATH, rows, delivery_columns)
    write_csv(AUDIT_PATH, rows, audit_columns)
    with DELIVERY_PATH.open(newline="", encoding="utf-8-sig") as stream:
        reread = list(csv.DictReader(stream))
    if len(reread) != 30 or [r["sample_id"] for r in reread] != ids:
        raise RuntimeError("delivery CSV re-read failed")

    counts = {CLASS_NAMES[i]: sum(r["predicted_class_id"] == i for r in rows) for i in range(3)}
    intensity = np.asarray([r["predicted_intensity"] for r in rows], dtype=np.float64)
    confidence = np.asarray([r["confidence"] for r in rows], dtype=np.float64)
    summary = {
        "status": "ATTACHMENT3_FINAL_INFERENCE_COMPLETE",
        "label_status": "unlabeled; distribution only, no accuracy/F1/MAE/Pearson",
        "input_version": "Attachment3 aligned-50, one file per sample",
        "sample_count": len(rows), "coverage_fraction": len(rows) / len(dataset),
        "sample_id_source": "pickle filename stem; no id field exists inside Attachment3 files",
        "checkpoint": checkpoint_info["relative_path"],
        "checkpoint_sha256_before": hash_before,
        "checkpoint_sha256_after": hash_after,
        "text_adapter_model": MODEL_ID,
        "text_adapter_revision": REVISION,
        "text_adapter_weight_sha256": WEIGHTS_SHA256,
        "text_interface_gate": "PASS; outputs/final/q2/attachment3/attachment3_text_interface_audit.json",
        "normalization": "none", "regression_postprocessing": "raw model output; no clipping",
        "class_mapping": {str(i): CLASS_NAMES[i] for i in range(3)},
        "class_counts": counts,
        "class_percent": {name: 100 * count / len(rows) for name, count in counts.items()},
        "intensity": describe(intensity),
        "confidence": describe(confidence),
        "qa": {
            "unique_sample_ids": len(set(ids)) == len(rows),
            "no_missing_samples": ids == expected_ids,
            "valid_classes": True,
            "all_finite_outputs": True,
            "max_probability_sum_error": max_probability_sum_error,
            "delivery_csv_reread": True,
            "checkpoint_unchanged": hash_before == hash_after,
        },
        "outputs": {
            "delivery_csv": str(DELIVERY_PATH.relative_to(ROOT)),
            "audit_csv": str(AUDIT_PATH.relative_to(ROOT)),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "sample_count": len(rows),
                      "class_counts": counts, "intensity": summary["intensity"],
                      "max_probability_sum_error": max_probability_sum_error},
                     ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
