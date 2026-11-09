import numpy as np
import pandas as pd
import pytest

from baf.data import iter_base_csv, sample_by_month, split_time
from baf.schema import normalize_missing, validate_frame


def test_time_split_is_disjoint_exhaustive_and_in_order(frame):
    parts = split_time(frame)
    assert [set(p.month) for p in parts.values()] == [set(range(6)), {6}, {7}]
    indices = [set(p.index) for p in parts.values()]
    assert not (indices[0] & indices[1] or indices[0] & indices[2] or indices[1] & indices[2])
    assert set.union(*indices) == set(frame.index)


@pytest.mark.parametrize("column,value", [("month", 8), ("month", 0.5), ("month", np.nan), ("fraud_bool", 2), ("fraud_bool", np.nan), ("income", np.inf), ("phone_home_valid", 3)])
def test_invalid_values_fail_loudly(frame, column, value):
    if isinstance(value, float):
        frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        validate_frame(frame)


def test_missing_extra_or_non_numeric_columns_fail(frame):
    with pytest.raises(ValueError, match="schema"):
        validate_frame(frame.drop(columns="income"))
    with pytest.raises(ValueError, match="schema"):
        validate_frame(frame.assign(accidental_target=frame.fraud_bool))
    frame["income"] = "bad"
    with pytest.raises(ValueError, match="income"):
        validate_frame(frame)


def test_only_documented_negative_values_become_missing(frame):
    for name in ["prev_address_months_count", "current_address_months_count", "bank_months_count", "session_length_in_minutes", "device_distinct_emails"]:
        frame.loc[0, name] = -1
    frame.loc[0, "intended_balcon_amount"] = -3.4
    frame.loc[0, ["velocity_6h", "credit_risk_score"]] = -1
    clean = normalize_missing(frame)
    assert clean.loc[0].isna().sum() == 6
    assert clean.loc[0, "velocity_6h"] == -1
    assert clean.loc[0, "credit_risk_score"] == -1
    assert frame.loc[0, "intended_balcon_amount"] == -3.4


@pytest.mark.parametrize("with_index", [False, True])
def test_csv_preserves_target_and_checks_exported_index(frame, tmp_path, with_index):
    path = tmp_path / "Base.csv"
    frame.to_csv(path, index=with_index)
    loaded = pd.concat(iter_base_csv(path, chunksize=7), ignore_index=True)
    pd.testing.assert_frame_equal(loaded, frame, check_dtype=False)
    if with_index:
        bad = path.read_text().replace("\n0,", "\n999,", 1)
        path.write_text(bad)
        with pytest.raises(ValueError, match="index"):
            list(iter_base_csv(path, chunksize=7))


def test_duplicate_csv_header_is_rejected(frame, tmp_path):
    path = tmp_path / "Base.csv"
    frame.to_csv(path, index=False)
    path.write_text(path.read_text().replace("income,", "fraud_bool,", 1))
    with pytest.raises(ValueError, match="header"):
        list(iter_base_csv(path))


def test_base_header_alias_is_mapped_without_losing_values(frame, tmp_path):
    path = tmp_path / "Base.csv"
    raw = frame.rename(columns={"device_distinct_emails": "device_distinct_emails_8w"})
    raw.to_csv(path, index=False)
    loaded = pd.concat(iter_base_csv(path, chunksize=7), ignore_index=True)
    pd.testing.assert_frame_equal(loaded, frame, check_dtype=False)

    raw["device_distinct_emails"] = frame.device_distinct_emails
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="header"):
        list(iter_base_csv(path))


def test_sample_is_deterministic_and_covers_every_month(frame):
    a = sample_by_month([frame.iloc[:15], frame.iloc[15:]], 3, 42)
    b = sample_by_month([frame], 3, 42)
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("month").size().tolist() == [3] * 8
