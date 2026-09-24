"""Strict label-free aligned-50 adapter for Attachment4 interface auditing."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import EXPECTED_DIMS, MODALITIES, TIME_STEPS


REQUIRED_FIELDS = frozenset(("id", "raw_text", "text_bert", *MODALITIES))


def inspect_aligned_sample(sample: dict[str, Any], source: str | Path) -> dict:
    """Validate model inputs without requiring labels or inferring padding from zeros."""
    if not isinstance(sample, dict):
        raise TypeError(f"{source}: expected a sample dictionary")
    missing = sorted(REQUIRED_FIELDS - sample.keys())
    if missing:
        raise ValueError(f"{source}: missing required fields {missing}")
    sample_id = str(sample["id"])
    if not sample_id or sample_id.lower() in ("none", "nan"):
        raise ValueError(f"{source}: invalid sample ID")
    if not isinstance(sample["raw_text"], (str, np.str_)):
        raise TypeError(f"{source}: raw_text must be a string")
    bert = np.asarray(sample["text_bert"])
    if bert.shape != (3, TIME_STEPS) or not np.issubdtype(bert.dtype, np.integer):
        raise ValueError(f"{source}: text_bert must be integer [3,50]")
    mask_row = bert[1]
    if not np.isin(mask_row, (0, 1)).all() or not np.all(np.diff(mask_row) <= 0):
        raise ValueError(f"{source}: text_bert[1] must be a binary true prefix")
    padding = mask_row.astype(np.bool_, copy=True)
    valid_length = int(padding.sum())
    if valid_length < 1:
        raise ValueError(f"{source}: empty valid sequence")
    features = {}
    native_zero = {}
    suffix_nonzero = {}
    for modality, dim in EXPECTED_DIMS.items():
        value = np.asarray(sample[modality])
        if value.shape != (TIME_STEPS, dim) or not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"{source}: {modality} must be numeric [50,{dim}]")
        if not np.isfinite(value).all():
            raise ValueError(f"{source}: {modality} contains NaN or Inf")
        features[modality] = value
        native_zero[modality] = np.all(value == 0, axis=1) & padding
        suffix_nonzero[modality] = int(np.count_nonzero(value[~padding]))
    # This matches the audited Attachment2 aligned interface. Text embeddings
    # may be nonzero outside the mask; audio/vision suffixes must be zero.
    if suffix_nonzero["audio"] or suffix_nonzero["vision"]:
        raise ValueError(f"{source}: audio/vision nonzero inside text_bert padding")
    return {"id": sample_id, "raw_text": str(sample["raw_text"]),
            "text_bert": bert, "features": features, "padding_mask": padding,
            "valid_length": valid_length, "native_zero": native_zero,
            "suffix_nonzero": suffix_nonzero, "source": str(source),
            "keys": sorted(sample.keys())}


class UnlabeledAligned50Dataset(Dataset):
    """One-pkl-per-sample data interface, without labels or inferred masks."""

    def __init__(self, feature_paths: list[str | Path]) -> None:
        self.records = []
        seen = set()
        for path_like in feature_paths:
            path = Path(path_like)
            with path.open("rb") as stream:
                sample = pickle.load(stream)
            record = inspect_aligned_sample(sample, path)
            if record["id"] in seen:
                raise ValueError(f"duplicate sample ID: {record['id']}")
            seen.add(record["id"])
            self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        features = {modality: torch.from_numpy(np.asarray(
            record["features"][modality], dtype=np.float32).copy()) for modality in MODALITIES}
        return {"id": record["id"], "raw_text": record["raw_text"], **features,
                "padding_mask": torch.from_numpy(record["padding_mask"].copy()),
                "native_zero_mask": torch.stack([
                    torch.from_numpy(record["native_zero"][modality].copy())
                    for modality in MODALITIES], dim=0)}
