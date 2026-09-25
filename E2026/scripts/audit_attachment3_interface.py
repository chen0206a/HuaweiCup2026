"""Read-only preflight for the locked Q2 model and local Attachment3 aligned files.

This script does not run inference or read Attachment2 test data. In particular,
it never treats text_bert token IDs as the model's 768-dimensional text feature.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/attachment3_sealed_inventory.json"
LOCK = ROOT / "outputs/final/q2/q2_model_lock.json"
OUTPUT = ROOT / "outputs/final/q2/attachment3/attachment3_input_audit.json"
EXPECTED = {"text": (1, 50, 768), "audio": (1, 50, 74), "vision": (1, 50, 35)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    inventory = json.loads(MANIFEST.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    checkpoint = lock["checkpoint_paths"]["B5-P2:42"]
    checkpoint_path = ROOT / checkpoint["relative_path"]
    checkpoint_observed_hash = sha256(checkpoint_path)
    aligned = sorted(
        (item for item in inventory["file_metadata_only"]
         if "/对齐版本/" in item["absolute_path"]
         and "/未对齐版本/" not in item["absolute_path"]),
        key=lambda item: item["absolute_path"],
    )
    files = []
    for item in aligned:
        path = Path(item["absolute_path"])
        observed_hash = sha256(path)
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or set(payload) != {"test"}:
            fields = {}
            top_level_keys = list(payload) if isinstance(payload, dict) else None
        else:
            sample = payload["test"]
            top_level_keys = list(payload)
            fields = {
                name: {"shape": list(np.asarray(value).shape),
                       "dtype": str(np.asarray(value).dtype),
                       "finite": (bool(np.isfinite(value).all())
                                  if np.issubdtype(np.asarray(value).dtype, np.number) else None)}
                for name, value in sample.items()
            }
        files.append({
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": observed_hash,
            "manifest_sha256_match": observed_hash == item["sha256"],
            "top_level_keys": top_level_keys,
            "fields": fields,
            "missing_model_fields": sorted(set(EXPECTED) - set(fields)),
            "model_feature_shapes_match": all(
                fields.get(name, {}).get("shape") == list(shape)
                for name, shape in EXPECTED.items()
            ),
        })
    report = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "local official Attachment3 aligned version; no labels read",
        "model_checkpoint": checkpoint["relative_path"],
        "model_checkpoint_sha256_expected": checkpoint["sha256"],
        "model_checkpoint_sha256_observed": checkpoint_observed_hash,
        "model_checkpoint_sha256_match": checkpoint_observed_hash == checkpoint["sha256"],
        "expected_model_features": {k: list(v) for k, v in EXPECTED.items()},
        "aligned_file_count": len(files),
        "all_manifest_hashes_match": all(f["manifest_sha256_match"] for f in files),
        "all_model_feature_shapes_match": all(f["model_feature_shapes_match"] for f in files),
        "all_files_lack_precomputed_text": all("text" in f["missing_model_fields"] for f in files),
        "inference_status": "BLOCKED_INPUT_INTERFACE_MISMATCH",
        "files": files,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "aligned_file_count", "all_manifest_hashes_match",
        "model_checkpoint_sha256_match", "all_model_feature_shapes_match",
        "all_files_lack_precomputed_text", "inference_status")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
