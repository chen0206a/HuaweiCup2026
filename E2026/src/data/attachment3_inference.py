"""Strict label-free Attachment3 aligned-50 adapter for the locked Q2 model."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.q3.text_feature_reconstruction import reconstruct_aligned_text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Attachment3InferenceDataset(Dataset):
    """30 per-file samples; no labels, no zero-derived padding, no normalization."""

    def __init__(self, manifest_path: str | Path, bert_model) -> None:
        inventory = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        entries = sorted(
            (row for row in inventory["file_metadata_only"]
             if "/对齐版本/" in row["absolute_path"]
             and "/未对齐版本/" not in row["absolute_path"]),
            key=lambda row: row["absolute_path"],
        )
        if len(entries) != 30:
            raise ValueError(f"expected 30 aligned Attachment3 files, got {len(entries)}")
        self.bert_model = bert_model.eval()
        self.records: list[dict] = []
        seen_ids = set()
        for entry in entries:
            path = Path(entry["absolute_path"])
            if not path.is_file() or _sha256(path) != entry["sha256"]:
                raise ValueError(f"Attachment3 file missing or SHA256 mismatch: {path}")
            with path.open("rb") as stream:
                payload = pickle.load(stream)
            if not isinstance(payload, dict) or set(payload) != {"test"}:
                raise ValueError(f"{path}: expected only top-level test key")
            fields = payload["test"]
            if not isinstance(fields, dict) or set(fields) != {"text_bert", "audio", "vision"}:
                raise ValueError(f"{path}: unexpected fields; labels must not be accessed")
            bert = np.asarray(fields["text_bert"])
            audio = np.asarray(fields["audio"])
            vision = np.asarray(fields["vision"])
            if bert.shape != (1, 3, 50) or audio.shape != (1, 50, 74) or vision.shape != (1, 50, 35):
                raise ValueError(f"{path}: aligned shape mismatch")
            if not all(np.issubdtype(x.dtype, np.number) and np.isfinite(x).all()
                       for x in (bert, audio, vision)):
                raise ValueError(f"{path}: nonnumeric or nonfinite inputs")
            sample_bert = bert[0]
            mask_values = sample_bert[1]
            if not np.isin(mask_values, (0, 1)).all() or not np.all(np.diff(mask_values) <= 0):
                raise ValueError(f"{path}: text_bert[1] is not a binary valid prefix")
            padding_mask = mask_values.astype(np.bool_, copy=True)
            if not padding_mask.any():
                raise ValueError(f"{path}: no valid timestep")
            if np.any(audio[0, ~padding_mask] != 0) or np.any(vision[0, ~padding_mask] != 0):
                raise ValueError(f"{path}: audio/vision nonzero in padding suffix")
            sample_id = path.stem
            if sample_id in seen_ids:
                raise ValueError(f"duplicate file-derived sample ID {sample_id}")
            seen_ids.add(sample_id)
            self.records.append({
                "sample_id": sample_id, "source_file": str(path),
                "source_sha256": entry["sha256"],
                "text_bert": sample_bert.copy(),
                "audio": audio[0].astype(np.float32, copy=True),
                "vision": vision[0].astype(np.float32, copy=True),
                "padding_mask": padding_mask,
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        # Reconstruct only the text input. Audio, vision and the mask are exact
        # copies of the audited source; native zeros remain ordinary values.
        text = reconstruct_aligned_text(self.bert_model, record["text_bert"])
        return {
            "sample_id": record["sample_id"],
            "source_file": record["source_file"],
            "source_sha256": record["source_sha256"],
            "text": torch.from_numpy(text),
            "audio": torch.from_numpy(record["audio"].copy()),
            "vision": torch.from_numpy(record["vision"].copy()),
            "padding_mask": torch.from_numpy(record["padding_mask"].copy()),
            "valid_length": int(record["padding_mask"].sum()),
        }
