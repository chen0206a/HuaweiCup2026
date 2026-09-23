"""Dataset adapter for the audited CMU-MOSEI aligned-50 pickle."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


MODALITIES = ("text", "audio", "vision")
EXPECTED_DIMS = {"text": 768, "audio": 74, "vision": 35}
TIME_STEPS = 50


class Aligned50Dataset(Dataset):
    """Read one split without modifying or serializing the source pickle.

    Mask meanings are deliberately separate:
    * padding_mask: true at positions that belong to the aligned utterance;
    * availability_mask: true where an observation has not been deliberately
      hidden. It starts all true; future block masking changes selected entries
      to false. Padding is handled separately by padding_mask.
    * native_zero_mask: true when an original modality vector is all zero at a
      valid position. It is a quality/content record, never an implicit mask.
    """

    def __init__(
        self,
        split_data: dict[str, Any],
        split_name: str,
        *,
        scaler=None,
        verify_padding_alignment: bool = True,
    ) -> None:
        self.split_name = split_name
        self.split_data = split_data
        self.scaler = scaler
        required = {"id", "text", "audio", "vision", "text_bert", "classification_labels", "regression_labels"}
        missing = sorted(required - set(split_data))
        if missing:
            raise ValueError(f"{split_name}: missing audited fields {missing}")

        self.ids = split_data["id"]
        self.features: dict[str, np.ndarray] = {}
        n = len(self.ids)
        for modality, dim in EXPECTED_DIMS.items():
            arr = np.asarray(split_data[modality])
            if arr.shape != (n, TIME_STEPS, dim):
                raise ValueError(
                    f"{split_name}.{modality}: expected {(n, TIME_STEPS, dim)} from audit, got {arr.shape}"
                )
            if not np.issubdtype(arr.dtype, np.number):
                raise TypeError(f"{split_name}.{modality}: expected numeric features, got {arr.dtype}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{split_name}.{modality}: found NaN or Inf after audit")
            self.features[modality] = arr

        bert = np.asarray(split_data["text_bert"])
        if bert.shape != (n, 3, TIME_STEPS):
            raise ValueError(f"{split_name}.text_bert: expected {(n, 3, TIME_STEPS)}, got {bert.shape}")
        # Audit established that component 1 is binary, valid-prefix / zero-suffix.
        observed_mask = bert[:, 1, :]
        if not np.isin(observed_mask, (0, 1)).all():
            raise ValueError(f"{split_name}.text_bert[:, 1, :] is no longer binary")
        if not np.all(np.diff(observed_mask, axis=1) <= 0):
            raise ValueError(f"{split_name}: text_bert mask no longer has a valid prefix and zero suffix")
        self.padding_mask = observed_mask.astype(np.bool_, copy=True)
        if not np.all(self.padding_mask.any(axis=1)):
            raise ValueError(f"{split_name}: found a sample with no valid timestep")

        # The audit found no nonzero audio/vision vectors in the text_bert zero
        # suffix. Recheck in bounded chunks so the verification does not allocate
        # a second full-size feature tensor.
        if verify_padding_alignment:
            for modality in ("audio", "vision"):
                arr = self.features[modality]
                for start in range(0, n, 128):
                    end = min(start + 128, n)
                    pad = ~self.padding_mask[start:end]
                    if np.any(arr[start:end][pad] != 0):
                        raise ValueError(
                            f"{split_name}.{modality}: nonzero features occur inside text_bert padding suffix"
                        )

        self.native_zero: dict[str, np.ndarray] = {
            m: np.all(self.features[m] == 0, axis=2) & self.padding_mask for m in MODALITIES
        }
        self.vision_all_zero = np.all(
            self.native_zero["vision"] | ~self.padding_mask, axis=1
        )

        cls = np.asarray(split_data["classification_labels"])
        reg = np.asarray(split_data["regression_labels"])
        if cls.shape != (n,) or reg.shape != (n,):
            raise ValueError(f"{split_name}: label arrays must each have shape ({n},)")
        if not np.isfinite(cls).all() or not np.isfinite(reg).all():
            raise ValueError(f"{split_name}: labels contain NaN or Inf")
        if not np.isin(cls, (0, 1, 2)).all():
            raise ValueError(f"{split_name}: observed class encoding changed from audited values 0, 1, 2")
        self.cls_labels = cls.astype(np.int64, copy=True)
        self.reg_labels = reg.astype(np.float32, copy=True)
        if "raw_text" in split_data:
            self.raw_text = split_data["raw_text"]
        else:
            self.raw_text = None

    @classmethod
    def from_pickle(cls, pkl_path: str | Path, split_name: str, **kwargs) -> "Aligned50Dataset":
        with Path(pkl_path).open("rb") as f:
            data = pickle.load(f)
        if split_name not in data:
            raise KeyError(f"split {split_name!r} not found; keys={list(data)}")
        return cls(data[split_name], split_name, **kwargs)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pad = self.padding_mask[index]
        features: dict[str, torch.Tensor] = {}
        for modality in MODALITIES:
            # Explicit conversion unifies audio/vision float64 and text float32.
            x = np.asarray(self.features[modality][index], dtype=np.float32)
            if self.scaler is not None:
                x = self.scaler.transform(modality, x)
            features[modality] = torch.from_numpy(np.array(x, dtype=np.float32, copy=True))

        padding_mask = torch.from_numpy(pad.copy())
        # Availability indicates deliberate observation hiding only. Original
        # complete samples start fully available, including padded coordinates;
        # consumers combine it with padding_mask when pooling or attending.
        availability_mask = torch.ones((len(MODALITIES), TIME_STEPS), dtype=torch.bool)
        native_zero_mask = torch.stack(
            [torch.from_numpy(self.native_zero[m][index].copy()) for m in MODALITIES], dim=0
        )
        return {
            "id": str(self.ids[index]),
            **features,
            "padding_mask": padding_mask,
            "availability_mask": availability_mask,
            "native_zero_mask": native_zero_mask,
            "cls_label": torch.tensor(self.cls_labels[index], dtype=torch.long),
            "reg_label": torch.tensor(self.reg_labels[index], dtype=torch.float32),
            "vision_all_zero": torch.tensor(self.vision_all_zero[index], dtype=torch.bool),
        }
