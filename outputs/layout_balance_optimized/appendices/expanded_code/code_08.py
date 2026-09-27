def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def compute_metrics(y_true_cls, y_pred_cls, y_true_reg, y_pred_reg) -> dict:
    return {
        **classification_metrics(y_true_cls, y_pred_cls),
        **regression_metrics(y_true_reg, y_pred_reg),
    }

def validation_selection_score(metrics: dict) -> float:
    return float(
        0.25 * metrics["accuracy"]
        + 0.25 * metrics["macro_f1"]
        + 0.25 * (1.0 - metrics["mae"] / 6.0)
        + 0.25 * (metrics["pearson"] + 1.0) / 2.0
    )
