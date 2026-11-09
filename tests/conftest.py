import numpy as np
import pandas as pd
import pytest

from baf.schema import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


@pytest.fixture
def frame():
    """Synthetic contract fixture, never a BAF observation or result."""
    months = np.repeat(np.arange(8), 6)
    n = len(months)
    values = {name: np.arange(n, dtype=float) + 1 for name in NUMERIC_FEATURES}
    values.update({name: np.arange(n) % 2 for name in BINARY_FEATURES})
    values.update({name: np.where(np.arange(n) % 2, "A", "B") for name in CATEGORICAL_FEATURES})
    return pd.DataFrame({"fraud_bool": np.arange(n) % 2, **values, "month": months})
