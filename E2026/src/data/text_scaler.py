"""Train-only feature-wise scaler for the B5-N1 text ablation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.dataset import Aligned50Dataset


@dataclass
class TextOnlyTrainScaler:
    """Z-score text dimensions using only valid positions from the train split.

    Padding values are transformed for pipeline consistency but remain excluded
    from pooling by ``padding_mask``. Synthetic block masking is applied later,
    in the transformed input space, so missing positions are overwritten to 0.
    Audio and vision are converted to float32 by the Dataset but not normalized.
    """

    mean: np.ndarray
    std: np.ndarray
    epsilon: float
    fitted_split: str
    valid_timestep_count: int
    raw_feature_std: np.ndarray
    z_feature_mean: np.ndarray
    z_feature_std: np.ndarray

    @classmethod
    def fit(cls, train_dataset: Aligned50Dataset, epsilon: float = 1e-6,
            chunk_size: int = 128) -> "TextOnlyTrainScaler":
        if train_dataset.split_name != "train":
            raise ValueError("text scaler may only be fit on split='train'")
        if epsilon <= 0 or chunk_size < 1:
            raise ValueError("epsilon and chunk_size must be positive")
        arr = train_dataset.features["text"]
        mask = train_dataset.padding_mask
        total = np.zeros(arr.shape[-1], dtype=np.float64)
        count = int(mask.sum())
        if count == 0:
            raise ValueError("cannot fit text scaler without valid train timesteps")
        for start in range(0, len(train_dataset), chunk_size):
            end = min(start + chunk_size, len(train_dataset))
            valid = np.asarray(arr[start:end], dtype=np.float64)[mask[start:end]]
            total += valid.sum(axis=0)
        mean = total / count

        squared = np.zeros_like(mean)
        for start in range(0, len(train_dataset), chunk_size):
            end = min(start + chunk_size, len(train_dataset))
            valid = np.asarray(arr[start:end], dtype=np.float64)[mask[start:end]]
            squared += np.square(valid - mean).sum(axis=0)
        raw_std = np.sqrt(squared / count)
        scale = np.maximum(raw_std, epsilon)

        z_total = np.zeros_like(mean)
        z_squared = np.zeros_like(mean)
        for start in range(0, len(train_dataset), chunk_size):
            end = min(start + chunk_size, len(train_dataset))
            valid = np.asarray(arr[start:end], dtype=np.float64)[mask[start:end]]
            z = (valid - mean) / scale
            z_total += z.sum(axis=0)
            z_squared += np.square(z).sum(axis=0)
        z_mean = z_total / count
        z_std = np.sqrt(np.maximum(z_squared / count - np.square(z_mean), 0.0))
        return cls(mean.astype(np.float32), scale.astype(np.float32), float(epsilon),
                   train_dataset.split_name, count, raw_std.astype(np.float64),
                   z_mean.astype(np.float64), z_std.astype(np.float64))

    def transform(self, modality: str, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if modality != "text":
            return x
        return ((x - self.mean) / self.std).astype(np.float32, copy=False)

    def state_dict(self) -> dict[str, Any]:
        return {"fitted_split": self.fitted_split, "modality": "text",
                "feature_dim": int(self.mean.size), "valid_timestep_count": self.valid_timestep_count,
                "epsilon": self.epsilon, "mean": self.mean.tolist(), "scale": self.std.tolist()}

    def diagnostic_summary(self) -> dict[str, Any]:
        qs = (0, 1, 5, 25, 50, 75, 95, 99, 100)
        quantiles = lambda v: {str(q): float(np.percentile(v, q)) for q in qs}
        nondeg = self.raw_feature_std >= self.epsilon
        return {
            "fitted_split": self.fitted_split,
            "modality": "text",
            "feature_dim": int(self.mean.size),
            "valid_timestep_count": int(self.valid_timestep_count),
            "epsilon": self.epsilon,
            "raw_feature_mean_min": float(self.mean.min()),
            "raw_feature_mean_max": float(self.mean.max()),
            "raw_feature_std_min": float(self.raw_feature_std.min()),
            "raw_feature_std_max": float(self.raw_feature_std.max()),
            "raw_feature_std_quantiles": quantiles(self.raw_feature_std),
            "zero_std_dimension_count": int(np.sum(self.raw_feature_std == 0)),
            "nearly_zero_std_dimension_count": int(np.sum(self.raw_feature_std < self.epsilon)),
            "zscore_train_feature_mean_min": float(self.z_feature_mean.min()),
            "zscore_train_feature_mean_max": float(self.z_feature_mean.max()),
            "zscore_train_feature_mean_abs_median": float(np.median(np.abs(self.z_feature_mean))),
            "zscore_train_feature_mean_abs_max": float(np.max(np.abs(self.z_feature_mean))),
            "zscore_train_feature_std_min": float(self.z_feature_std.min()),
            "zscore_train_feature_std_max": float(self.z_feature_std.max()),
            "zscore_train_feature_std_quantiles": quantiles(self.z_feature_std),
            "zscore_nondegenerate_std_abs_deviation_median": (
                float(np.median(np.abs(self.z_feature_std[nondeg] - 1.0)))
                if np.any(nondeg) else None),
            "zscore_nondegenerate_std_abs_deviation_max": (
                float(np.max(np.abs(self.z_feature_std[nondeg] - 1.0)))
                if np.any(nondeg) else None),
        }
