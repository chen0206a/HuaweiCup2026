"""Evaluation metrics shared by validation and final holdout evaluation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, mean_absolute_error


CLASS_NAMES = ("Negative", "Neutral", "Positive")


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def classification_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(3)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson": safe_pearson(y_true, y_pred),
    }


def compute_metrics(y_true_cls, y_pred_cls, y_true_reg, y_pred_reg) -> dict:
    return {
        **classification_metrics(y_true_cls, y_pred_cls),
        **regression_metrics(y_true_reg, y_pred_reg),
    }


def validation_selection_score(metrics: dict) -> float:
    """Project-only validation score; it is not an official competition score."""
    return float(
        0.25 * metrics["accuracy"]
        + 0.25 * metrics["macro_f1"]
        + 0.25 * (1.0 - metrics["mae"] / 6.0)
        + 0.25 * (metrics["pearson"] + 1.0) / 2.0
    )
