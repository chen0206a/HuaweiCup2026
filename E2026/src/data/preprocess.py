"""Read-only loading and optional train-only feature-wise normalization."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import Aligned50Dataset, EXPECTED_DIMS, TIME_STEPS


@dataclass
class TrainFeaturewiseScaler:
    """Feature-wise mean/std computed only on valid train timesteps.

    Native zero vectors inside valid regions remain part of the train statistics.
    Padding coordinates never contribute. The source pkl is never changed.
    """

    mean: dict[str, np.ndarray]
    scale: dict[str, np.ndarray]

    @classmethod
    def fit(cls, train_dataset: Aligned50Dataset, chunk_size: int = 128) -> "TrainFeaturewiseScaler":
        means: dict[str, np.ndarray] = {}
        scales: dict[str, np.ndarray] = {}
        pad = train_dataset.padding_mask
        for modality, dim in EXPECTED_DIMS.items():
            total = np.zeros(dim, dtype=np.float64)
            total_sq = np.zeros(dim, dtype=np.float64)
            count = 0
            arr = train_dataset.features[modality]
            for start in range(0, len(train_dataset), chunk_size):
                end = min(start + chunk_size, len(train_dataset))
                x = np.asarray(arr[start:end], dtype=np.float64)
                valid = pad[start:end]
                total += np.where(valid[:, :, None], x, 0.0).sum(axis=(0, 1))
                total_sq += np.where(valid[:, :, None], x * x, 0.0).sum(axis=(0, 1))
                count += int(valid.sum())
            if count == 0:
                raise ValueError(f"cannot fit scaler: no valid training positions for {modality}")
            mu = total / count
            variance = np.maximum(total_sq / count - mu * mu, 0.0)
            sigma = np.sqrt(variance)
            sigma[sigma == 0] = 1.0
            means[modality] = mu.astype(np.float32)
            scales[modality] = sigma.astype(np.float32)
        return cls(means, scales)

    def transform(self, modality: str, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean[modality]) / self.scale[modality]).astype(np.float32, copy=False)

    def state_dict(self) -> dict[str, dict[str, list[float]]]:
        return {
            "mean": {k: v.tolist() for k, v in self.mean.items()},
            "scale": {k: v.tolist() for k, v in self.scale.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "TrainFeaturewiseScaler":
        return cls(
            {k: np.asarray(v, dtype=np.float32) for k, v in state["mean"].items()},
            {k: np.asarray(v, dtype=np.float32) for k, v in state["scale"].items()},
        )


def load_pickle_readonly(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def build_datasets_and_loaders(
    pkl_path: str | Path,
    *,
    normalization: str = "none",
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[dict[str, Aligned50Dataset], dict[str, DataLoader], TrainFeaturewiseScaler | None]:
    if normalization not in {"none", "train_featurewise"}:
        raise ValueError("normalization must be 'none' or 'train_featurewise'")
    if num_workers < 0 or num_workers > 2:
        raise ValueError("num_workers must be 0, 1, or 2 for this approximately 1 GB pickle")
    data = load_pickle_readonly(pkl_path)
    required_splits = ("train", "valid", "test")
    if any(split not in data for split in required_splits):
        raise ValueError(f"expected audited splits {required_splits}, got {list(data)}")
    datasets = {split: Aligned50Dataset(data[split], split) for split in required_splits}
    scaler = TrainFeaturewiseScaler.fit(datasets["train"]) if normalization == "train_featurewise" else None
    if scaler is not None:
        for dataset in datasets.values():
            dataset.scaler = scaler

    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator),
        "valid": DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return datasets, loaders, scaler
