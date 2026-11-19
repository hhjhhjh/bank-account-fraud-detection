import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("captum")

from baf.xai_cohorts import sample_fraud_cohorts, summarize_cohorts


def test_fraud_samples_are_deterministic_disjoint_and_keep_random_order():
    labels = np.r_[np.ones(300), np.zeros(200)]
    scores = np.linspace(-2, 2, len(labels))
    groups = sample_fraud_cohorts(labels, scores, -0.5, 100, 42)
    repeated = sample_fraud_cohorts(labels, scores, -0.5, 100, 42)
    for name, rows in groups.items():
        np.testing.assert_array_equal(rows, repeated[name])
        assert len(rows) == len(np.unique(rows)) == 100
        assert np.all(labels[rows] == 1)
        assert np.all((scores[rows] >= -0.5) == (name == "TP"))
    assert not np.intersect1d(groups["TP"], groups["FN"]).size


def test_fraud_sampling_fails_if_requested_cohort_is_unavailable():
    with pytest.raises(ValueError, match="FN has only"):
        sample_fraud_cohorts(np.ones(5), np.ones(5), 0.5, 2, 42)


def test_cohort_statistics_preserve_sign_and_report_overlap():
    values = np.array([[1, -2], [3, -4], [5, -6], [7, -8]], dtype=float)
    groups = {"TP": values, "FN": -values}
    table, stability = summarize_cohorts(groups, ["a", "b"], 2)
    tp = table[(table["group"] == "TP") & (table["sample_size"] == 4)].set_index("feature")
    assert tp.loc["a", "mean_signed_attribution"] == 4
    assert tp.loc["b", "mean_signed_attribution"] == -5
    assert tp.loc["b", "mean_absolute_attribution"] == 5
    assert tp.loc["b", "positive_fraction"] == 0
    assert tp.loc["b", "negative_fraction"] == 1
    assert stability["TP"]["subset_vs_full"]["overlapping_rows"] == 2
    assert stability["TP"]["split_half"]["overlapping_rows"] == 0
    assert stability["TP"]["subset_vs_full"]["top_k_overlap"] == 2
    assert stability["TP"]["subset_vs_full"]["spearman_rank_correlation"] == pytest.approx(1)
