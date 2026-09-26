def window_length(valid_length: int, rho: float) -> int:
    if not 1 <= valid_length <= 50 or rho not in (0.10, 0.20, 0.30):
        raise ValueError("invalid valid length or candidate rho")
    return max(1, round(rho * valid_length))

def evaluate_windows(predictor: FrozenP2Predictor, sample: dict, modality:
    str,
                     rho: float, fixed_class: int, full_logits: np.ndarray,
                     full_regression: float, sample_id: str,
                     random_draws: int = 32) -> dict:
    length = int(sample["padding_mask"][0].sum())
    width = window_length(length, rho)
    starts = np.arange(length - width + 1, dtype=np.int64)
    windows = [np.arange(start, start + width, dtype=np.int64) for start in
        starts]
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
    random_mean = {key: float(values[random_indices].mean()) for key, values
        in drops.items()}
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
