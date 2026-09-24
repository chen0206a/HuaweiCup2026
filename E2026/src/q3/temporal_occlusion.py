"""Continuous valid-prefix occlusion for a frozen full-input prediction."""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from src.data.dataset import MODALITIES
from src.q3.frozen_predictor import FrozenP2Predictor


def stable_rng(sample_id: str, purpose: str, seed: int = 20260924) -> np.random.Generator:
    token = f"{seed}|{sample_id}|{purpose}".encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "big"))


def window_length(valid_length: int, rho: float) -> int:
    if not 1 <= valid_length <= 50 or rho not in (0.10, 0.20, 0.30):
        raise ValueError("invalid valid length or candidate rho")
    return max(1, round(rho * valid_length))


def fixed_target_outputs(logits: np.ndarray, regression: np.ndarray,
                         fixed_class: int) -> dict[str, np.ndarray]:
    z = np.asarray(logits, dtype=np.float64)
    r = np.asarray(regression, dtype=np.float64)
    other = np.delete(z, fixed_class, axis=1)
    mx = other.max(axis=1)
    margin = z[:, fixed_class] - (mx + np.log(np.exp(other - mx[:, None]).sum(axis=1)))
    max_logits = z.max(axis=1)
    probability = np.exp(z[:, fixed_class] - max_logits) / np.exp(z - max_logits[:, None]).sum(axis=1)
    return {"class_logit": z[:, fixed_class], "class_margin": margin,
            "confidence": probability, "regression": r}


def _repeat_batch(sample: dict, count: int) -> dict:
    if sample["padding_mask"].shape[0] != 1:
        raise ValueError("one sample expected")
    return {key: sample[key].repeat((count,) + (1,) * (sample[key].ndim - 1))
            for key in (*MODALITIES, "padding_mask")}


def evaluate_position_sets(predictor: FrozenP2Predictor, sample: dict,
                           modality: str, position_sets: list[np.ndarray],
                           fixed_class: int, full_logits: np.ndarray,
                           full_regression: float, chunk_size: int = 128) -> dict[str, np.ndarray]:
    """Delete arbitrary valid positions, then return full-minus-deleted outputs."""
    if modality not in MODALITIES or not position_sets:
        raise ValueError("invalid modality or empty position sets")
    length = int(sample["padding_mask"][0].sum())
    if any(len(positions) == 0 or np.any(np.asarray(positions) < 0) or
           np.any(np.asarray(positions) >= length) for positions in position_sets):
        raise ValueError("deletion outside the valid prefix")
    logits_chunks, reg_chunks = [], []
    for offset in range(0, len(position_sets), chunk_size):
        subset = position_sets[offset:offset + chunk_size]
        batch = _repeat_batch(sample, len(subset))
        # _repeat_batch allocates independent tensors; the source remains untouched.
        for row, positions in enumerate(subset):
            batch[modality][row, np.asarray(positions, dtype=np.int64), :] = 0
        output = predictor.predict(batch)
        logits_chunks.append(output["classification_logits"].cpu().numpy())
        reg_chunks.append(output["regression"].cpu().numpy())
    deleted = fixed_target_outputs(np.concatenate(logits_chunks), np.concatenate(reg_chunks), fixed_class)
    full = fixed_target_outputs(np.asarray(full_logits, dtype=np.float64)[None, :],
                                np.asarray([full_regression], dtype=np.float64), fixed_class)
    return {key: full[key][0] - deleted[key] for key in deleted}


def evaluate_windows(predictor: FrozenP2Predictor, sample: dict, modality: str,
                     rho: float, fixed_class: int, full_logits: np.ndarray,
                     full_regression: float, sample_id: str,
                     random_draws: int = 32) -> dict:
    length = int(sample["padding_mask"][0].sum())
    width = window_length(length, rho)
    starts = np.arange(length - width + 1, dtype=np.int64)
    windows = [np.arange(start, start + width, dtype=np.int64) for start in starts]
    drops = evaluate_position_sets(predictor, sample, modality, windows,
                                   fixed_class, full_logits, full_regression)
    margin = drops["class_margin"]
    if np.any(margin > 0):
        top_index = int(np.argmax(margin))
        supports = True
    else:
        top_index = int(np.argmax(np.abs(margin)))
        supports = False
    rng = stable_rng(sample_id, f"interval:{rho:.2f}")
    random_indices = rng.integers(0, len(windows), size=random_draws)
    random_mean = {key: float(values[random_indices].mean()) for key, values in drops.items()}
    top = {key: float(values[top_index]) for key, values in drops.items()}
    top["regression_absolute"] = abs(top["regression"])
    random_mean["regression_absolute"] = float(np.abs(drops["regression"][random_indices]).mean())
    curve = np.full(50, np.nan, dtype=np.float64)
    regression_curve = np.full(50, np.nan, dtype=np.float64)
    for t in range(length):
        covering = (starts <= t) & (t < starts + width)
        curve[t] = float(np.max(margin[covering]))
        regression_curve[t] = float(np.max(np.abs(drops["regression"][covering])))
    return {"modality": modality, "rho": rho, "valid_length": length,
            "window_length": width, "starts": starts, "drops": drops,
            "top_index": top_index, "top_start": int(starts[top_index]),
            "top_end": int(starts[top_index] + width), "top": top,
            "random_mean": random_mean, "random_indices": random_indices,
            "supports_predicted_class": supports, "curve": curve,
            "regression_curve": regression_curve}
