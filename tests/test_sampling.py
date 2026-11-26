import numpy as np
import pytest

from baf.sampling import oversample_train_rows


def test_oversampling_preserves_selected_rows_and_duplicates_only_selected_fraud():
    labels = np.r_[np.zeros(100, dtype=np.int64), np.ones(20, dtype=np.int64)]
    selected = np.r_[np.arange(50), np.arange(100, 104)]
    rows = oversample_train_rows(labels, selected, 0.5, seed=42)
    repeated = oversample_train_rows(labels, selected, 0.5, seed=42)
    np.testing.assert_array_equal(rows, repeated)
    np.testing.assert_array_equal(rows[:len(selected)], selected)
    assert len(rows) == 75 and int(labels[rows].sum()) == 25
    assert np.isin(rows, selected).all()
    assert (labels[rows[len(selected):]] == 1).all()
    np.testing.assert_array_equal(labels[rows[:50]], np.zeros(50))


def test_oversampling_can_balance_classes_and_rounds_target_up():
    labels = np.r_[np.zeros(11, dtype=np.int64), np.ones(1, dtype=np.int64)]
    selected = np.arange(len(labels))
    rounded = oversample_train_rows(labels, selected, 0.25, seed=42)
    assert len(rounded) == 14 and int(labels[rounded].sum()) == 3
    balanced = oversample_train_rows(labels, selected, 1.0, seed=42)
    assert int(labels[balanced].sum()) == 11 and len(balanced) == 22


@pytest.mark.parametrize("ratio", [0, -0.1, 1.1, float("nan"), float("inf"), True])
def test_oversampling_rejects_invalid_ratios(ratio):
    with pytest.raises(ValueError, match="oversampling_ratio"):
        oversample_train_rows(np.array([0, 0, 1]), np.arange(3), ratio, seed=42)


def test_oversampling_rejects_ratio_that_would_remove_fraud_rows():
    with pytest.raises(ValueError, match="below the original"):
        oversample_train_rows(np.array([0, 0, 1]), np.arange(3), 0.1, seed=42)
