import numpy as np

from src.utils.metrics import compute_metrics


def test_metric_values_and_confusion_matrix():
    got = compute_metrics([0, 1, 2, 2], [0, 2, 2, 1], [0.0, 1.0, 2.0], [0.0, 1.0, 1.0])
    assert got["accuracy"] == 0.5
    assert got["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [0, 1, 1]]
    assert got["per_class"]["Neutral"]["recall"] == 0.0
    assert np.isclose(got["mae"], 1 / 3)
    assert np.isclose(got["pearson"], 0.8660254037844387)
