"""Same-budget deletion checks and video-grouped uncertainty summaries."""
from __future__ import annotations

import numpy as np

from src.q3.temporal_occlusion import evaluate_position_sets, stable_rng


FRACTIONS = (0.10, 0.20, 0.30, 0.40)
FIELDS = ("class_margin", "confidence", "class_logit", "regression", "regression_absolute")


def deletion_curve(predictor, sample: dict, window_result: dict,
                   fixed_class: int, full_logits: np.ndarray,
                   full_regression: float, sample_id: str,
                   random_draws: int = 32) -> list[dict]:
    length = window_result["valid_length"]
    rank = np.argsort(-window_result["curve"][:length], kind="stable")
    position_sets: list[np.ndarray] = []
    slices: list[tuple[int, slice]] = []
    for fraction in FRACTIONS:
        k = max(1, round(fraction * length))
        top_index = len(position_sets)
        position_sets.append(rank[:k])
        random_start = len(position_sets)
        rng = stable_rng(sample_id, f"deletion:{fraction:.2f}")
        for _ in range(random_draws):
            position_sets.append(rng.choice(length, size=k, replace=False))
        slices.append((top_index, slice(random_start, random_start + random_draws)))
    drops = evaluate_position_sets(predictor, sample, window_result["modality"],
                                   position_sets, fixed_class, full_logits,
                                   full_regression)
    result = []
    for fraction, (top_index, random_slice) in zip(FRACTIONS, slices):
        top = {key: float(drops[key][top_index]) for key in drops}
        random_mean = {key: float(np.mean(drops[key][random_slice])) for key in drops}
        top["regression_absolute"] = abs(top["regression"])
        random_mean["regression_absolute"] = float(np.abs(drops["regression"][random_slice]).mean())
        result.append({"fraction": fraction, "deleted_positions": max(1, round(fraction * length)),
                       "top": top, "random_mean": random_mean})
    return result


def grouped_bootstrap_mean(values: np.ndarray, groups: list[str],
                           reps: int = 1000, seed: int = 20260924) -> list[float]:
    """Percentile CI for the clip-weighted mean with video IDs resampled."""
    values = np.asarray(values, dtype=np.float64)
    unique = sorted(set(groups))
    labels = np.asarray(groups)
    group_sums = np.asarray([values[labels == group].sum() for group in unique])
    group_counts = np.asarray([np.count_nonzero(labels == group) for group in unique])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(unique), size=(reps, len(unique)))
    means = group_sums[picks].sum(axis=1) / group_counts[picks].sum(axis=1)
    return [float(x) for x in np.quantile(means, (0.025, 0.975))]


def summarize_top_random(rows: list[dict], section: str,
                         bootstrap_reps: int = 1000) -> dict:
    """rows hold video_id and section={top,random_mean} fields."""
    if not rows:
        raise ValueError("no rows")
    groups = [row["video_id"] for row in rows]
    result = {"n_samples": len(rows), "n_video_ids": len(set(groups)),
              "bootstrap_unit": "video_id", "bootstrap_reps": bootstrap_reps}
    for field in FIELDS:
        top = np.asarray([row[section]["top"][field] for row in rows], dtype=np.float64)
        random = np.asarray([row[section]["random_mean"][field] for row in rows], dtype=np.float64)
        delta = top - random
        result[field] = {
            "top_mean": float(top.mean()), "top_median": float(np.median(top)),
            "top_grouped_bootstrap_95ci": grouped_bootstrap_mean(top, groups, bootstrap_reps),
            "random_mean": float(random.mean()), "random_median": float(np.median(random)),
            "random_grouped_bootstrap_95ci": grouped_bootstrap_mean(random, groups, bootstrap_reps),
            "top_minus_random_mean": float(delta.mean()),
            "top_minus_random_median": float(np.median(delta)),
            "top_minus_random_grouped_bootstrap_95ci": grouped_bootstrap_mean(
                delta, groups, bootstrap_reps),
            "fraction_top_gt_random": float(np.mean(delta > 0)),
            "negative_effect_fraction": float(np.mean(top < 0)),
        }
    return result


def summarize_curve(rows: list[dict], bootstrap_reps: int = 1000) -> dict:
    result = {}
    for index, fraction in enumerate(FRACTIONS):
        flattened = [{"video_id": row["video_id"], "point": row["deletion_curve"][index]}
                     for row in rows]
        result[f"{fraction:.2f}"] = summarize_top_random(flattened, "point", bootstrap_reps)
    return result
