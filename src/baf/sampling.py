import math

import numpy as np


def oversample_train_rows(labels, rows, target_ratio: float, *, seed: int):
    """Retain selected Train rows and duplicate fraud rows to reach fraud/normal ratio."""
    if type(target_ratio) not in (int, float) or not math.isfinite(target_ratio) or not 0 < target_ratio <= 1:
        raise ValueError("oversampling_ratio must be finite and in (0, 1]")
    selected_labels = labels[rows]
    if not np.array_equal(np.unique(selected_labels), [0, 1]):
        raise ValueError("Training selection must contain both binary classes")
    fraud_rows = rows[selected_labels == 1]
    normal_count = len(rows) - len(fraud_rows)
    if target_ratio < len(fraud_rows) / normal_count:
        raise ValueError("oversampling_ratio is below the original fraud/normal ratio")
    target_fraud_count = math.ceil(normal_count * target_ratio)
    extra_rows = np.random.default_rng(seed).choice(
        fraud_rows, size=target_fraud_count - len(fraud_rows), replace=True
    )
    return np.concatenate((rows, extra_rows))
