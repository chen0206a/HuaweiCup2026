"""Exact three-modality coalitions, Shapley values and pair interactions."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from src.data.dataset import MODALITIES
from src.q3.frozen_predictor import FrozenP2Predictor


COALITIONS = ((), ("text",), ("audio",), ("vision",),
              ("text", "audio"), ("text", "vision"), ("audio", "vision"),
              ("text", "audio", "vision"))
INDEX = {coalition: i for i, coalition in enumerate(COALITIONS)}
PAIR_NAMES = ("text_audio", "text_vision", "audio_vision")


def masked_coalition(batch: dict, included: tuple[str, ...]) -> dict:
    """Zero excluded *valid* features without changing padding/native zeros/source."""
    if any(modality not in MODALITIES for modality in included):
        raise ValueError("unknown modality")
    pad = batch["padding_mask"]
    if pad.dtype != torch.bool:
        raise ValueError("padding_mask must be bool")
    result = {"padding_mask": pad}
    for modality in MODALITIES:
        original = batch[modality]
        if modality in included:
            result[modality] = original
        else:
            masked = original.clone()
            masked[pad] = 0
            result[modality] = masked
    if "native_zero_mask" in batch:
        result["native_zero_mask"] = batch["native_zero_mask"]
    return result


def class_margin(logits: np.ndarray, fixed_class: np.ndarray) -> np.ndarray:
    """Log-odds margin for the full-input predicted class across all coalitions."""
    values = np.asarray(logits, dtype=np.float64)
    cls = np.asarray(fixed_class, dtype=np.int64)
    own = np.take_along_axis(values, np.broadcast_to(cls[:, None, None], (*values.shape[:2], 1)), axis=2)[..., 0]
    others = values.copy()
    rows = np.arange(len(cls))[:, None]
    columns = np.arange(values.shape[1])[None, :]
    others[rows, columns, cls[:, None]] = -np.inf
    maximum = np.max(others, axis=2)
    lse = maximum + np.log(np.exp(others - maximum[..., None]).sum(axis=2))
    return own - lse


def exact_shapley(values: np.ndarray) -> np.ndarray:
    """Return [B,3] Shapley values from [B,8] ordered coalition payoffs."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError("expected [B,8] coalition values")
    result = np.zeros((len(values), 3), dtype=np.float64)
    for m, modality in enumerate(MODALITIES):
        rest = [other for other in MODALITIES if other != modality]
        result[:, m] = (values[:, INDEX[(modality,)]] - values[:, INDEX[()]]) / 3
        for other in rest:
            pair = tuple(item for item in MODALITIES if item in (modality, other))
            result[:, m] += (values[:, INDEX[pair]] - values[:, INDEX[(other,)]]) / 6
        pair_rest = tuple(rest)
        result[:, m] += (values[:, INDEX[COALITIONS[-1]]] - values[:, INDEX[pair_rest]]) / 3
    return result


def pair_interactions(values: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    values = np.asarray(values, dtype=np.float64)
    result = {}
    for (i, j), name in zip(combinations(MODALITIES, 2), PAIR_NAMES):
        k = next(modality for modality in MODALITIES if modality not in (i, j))
        ij = tuple(item for item in MODALITIES if item in (i, j))
        ik = tuple(item for item in MODALITIES if item in (i, k))
        jk = tuple(item for item in MODALITIES if item in (j, k))
        without = values[:, INDEX[ij]] - values[:, INDEX[(i,)]] - values[:, INDEX[(j,)]] + values[:, INDEX[()]]
        with_third = values[:, INDEX[COALITIONS[-1]]] - values[:, INDEX[ik]] - values[:, INDEX[jk]] + values[:, INDEX[(k,)]]
        result[name] = {"value": (without + with_third) / 2,
                        "delta_without_third": without, "delta_with_third": with_third}
    return result


def primary_modality(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positive-support maximum; otherwise largest absolute negative/zero effect."""
    values = np.asarray(phi, dtype=np.float64)
    positives = values > 0
    has_positive = positives.any(axis=1)
    positive_index = np.argmax(np.where(positives, values, -np.inf), axis=1)
    fallback_index = np.argmax(np.abs(values), axis=1)
    return np.where(has_positive, positive_index, fallback_index), has_positive


def explain_batch(predictor: FrozenP2Predictor, batch: dict) -> dict:
    logits = []
    regression = []
    for coalition in COALITIONS:
        output = predictor.predict(masked_coalition(batch, coalition))
        logits.append(output["classification_logits"].detach().cpu().numpy())
        regression.append(output["regression"].detach().cpu().numpy())
    z = np.stack(logits, axis=1).astype(np.float64)
    r = np.stack(regression, axis=1).astype(np.float64)
    if not np.isfinite(z).all() or not np.isfinite(r).all():
        raise RuntimeError("coalition produced nonfinite output")
    cls = z[:, -1, :].argmax(axis=1)
    margins = class_margin(z, cls)
    phi_class = exact_shapley(margins)
    phi_reg = exact_shapley(r)
    err_class = phi_class.sum(axis=1) - (margins[:, -1] - margins[:, 0])
    err_reg = phi_reg.sum(axis=1) - (r[:, -1] - r[:, 0])
    if not np.allclose(err_class, 0, atol=1e-5, rtol=0) or not np.allclose(err_reg, 0, atol=1e-5, rtol=0):
        raise RuntimeError("Shapley efficiency failed")
    primary_class, supports_class = primary_modality(phi_class)
    primary_reg = np.argmax(np.abs(phi_reg), axis=1)
    return {"logits": z, "regression": r, "fixed_class": cls,
            "class_margin": margins, "phi_class": phi_class, "phi_reg": phi_reg,
            "interaction_class": pair_interactions(margins),
            "interaction_reg": pair_interactions(r),
            "primary_class": primary_class, "primary_reg": primary_reg,
            "supports_class": supports_class,
            "efficiency_error_class": err_class, "efficiency_error_reg": err_reg}
