import numpy as np
import pytest

torch = pytest.importorskip("torch")

from baf.evaluation import select_threshold, threshold_metrics


@pytest.mark.parametrize("cap", [0.0, 0.01, 0.05, 0.5, 1.0])
def test_threshold_is_optimal_with_ties_and_inclusive_boundary(cap):
    rng = np.random.default_rng(8)
    for _ in range(20):
        labels = np.array([0, 1] * 12)
        scores = rng.integers(-3, 4, size=len(labels)).astype(float)
        result = select_threshold(labels, scores, cap)
        candidates = [np.nextafter(scores.max(), np.inf), *np.unique(scores)]
        feasible = [threshold_metrics(labels, scores, t) for t in candidates]
        feasible = [r for r in feasible if r["fpr"] <= cap]
        expected = max(feasible, key=lambda r: (r["recall"], -r["fpr"], r["threshold"]))
        assert result == expected
        assert result["confusion_matrix"] == [[result["tn"], result["fp"]], [result["fn"], result["tp"]]]


def test_tied_negatives_cannot_be_split_to_meet_fpr_cap():
    result = select_threshold(np.array([0, 0, 1]), np.array([0.5, 0.5, 0.5]), 0.05)
    assert result["tp"] == result["fp"] == 0
    assert result["precision"] == result["recall"] == result["f1"] == 0
    assert result["threshold"] > 0.5


@pytest.mark.parametrize("scores", [[np.nan, 0], [np.inf, 0], [-np.inf, 0]])
def test_threshold_rejects_nonfinite_scores(scores):
    with pytest.raises(ValueError, match="finite"):
        select_threshold(np.array([0, 1]), np.array(scores), 0.01)


@pytest.mark.parametrize("labels", [[0, 0], [1, 1], [0, 2]])
def test_threshold_requires_both_binary_classes(labels):
    with pytest.raises(ValueError, match="classes"):
        select_threshold(np.array(labels), np.array([0.2, 0.8]), 0.01)
