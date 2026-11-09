import joblib
import numpy as np
import pytest

from baf.data import split_time
from baf.preprocessing import TrainPreprocessor


def test_validation_cannot_fit_preprocessor(frame):
    with pytest.raises(ValueError, match="Train"):
        TrainPreprocessor().fit(frame)
    with pytest.raises(ValueError, match="Train"):
        TrainPreprocessor().fit(frame[frame.month == 6])


def test_scaling_imputation_and_categories_only_learn_train(frame):
    parts = split_time(frame)
    train = parts["train"].copy()
    train.loc[0, "income"] = np.nan
    pre = TrainPreprocessor().fit(train)
    numeric = pre.transformer.named_transformers_["numeric"]
    assert numeric.named_steps["impute"].statistics_[0] == train.income.median()
    filled = train.income.fillna(train.income.median())
    assert numeric.named_steps["scale"].mean_[0] == pytest.approx(filled.mean())
    before = pre.transform(train).copy()
    future = parts["validation"].copy()
    future["income"] = 1e9
    future["payment_type"] = "FUTURE_ONLY"
    assert np.isfinite(pre.transform(future)).all()
    np.testing.assert_array_equal(before, pre.transform(train))
    cats = pre.transformer.named_transformers_["categorical"].named_steps["encode"].categories_
    assert all("FUTURE_ONLY" not in c for c in cats)


def test_target_and_month_never_enter_features_and_mapping_roundtrips(frame, tmp_path):
    parts = split_time(frame)
    pre = TrainPreprocessor().fit(parts["train"])
    original = pre.transform(parts["test"])
    altered = parts["test"].assign(fraud_bool=999, month=999)
    np.testing.assert_array_equal(original, pre.transform(altered))
    mapping = pre.feature_mapping()
    assert len(mapping) == original.shape[1]
    assert {r["original_feature"] for r in mapping}.isdisjoint({"month", "fraud_bool"})
    assert [r["index"] for r in mapping] == list(range(original.shape[1]))
    path = tmp_path / "pre.joblib"
    joblib.dump(pre, path)
    np.testing.assert_array_equal(original, joblib.load(path).transform(parts["test"]))
    assert original.dtype == np.float32


def test_all_missing_numeric_and_categorical_columns_remain_finite(frame):
    train = split_time(frame)["train"].copy()
    train["bank_months_count"] = -1
    train["payment_type"] = np.nan
    pre = TrainPreprocessor().fit(train)
    assert np.isfinite(pre.transform(frame)).all()
    mapping = pre.feature_mapping()
    assert any(r["kind"] == "missing_indicator" and r["original_feature"] == "bank_months_count" for r in mapping)


def test_output_names_width_and_binary_values_match_across_splits(frame):
    parts = split_time(frame)
    train = parts["train"]
    train.loc[0, "income"] = np.nan
    train["email_is_free"] = train.email_is_free.astype(float)
    train.loc[0, "email_is_free"] = np.nan
    pre = TrainPreprocessor().fit(train)
    mapping = pre.feature_mapping()

    assert all(part.shape[1] == 32 for part in parts.values())
    names = [row["transformed_feature"] for row in mapping]
    assert len(set(names)) == len(names) == 43
    assert names == pre.transformer.get_feature_names_out().tolist()
    assert {row["original_feature"] for row in mapping} == set(frame) - {"fraud_bool", "month"}
    indicators = {row["original_feature"]: row["index"] for row in mapping if row["kind"] == "missing_indicator"}
    assert set(indicators) == {
        "prev_address_months_count", "current_address_months_count", "bank_months_count",
        "session_length_in_minutes", "device_distinct_emails", "intended_balcon_amount",
        "income", "email_is_free",
    }
    binary_indices = [row["index"] for row in mapping if row["kind"] in {"binary", "one_hot", "missing_indicator"}]
    for part in parts.values():
        values = pre.transform(part)
        assert values.shape == (len(part), len(names))
        assert np.isfinite(values).all()
        assert np.isin(values[:, binary_indices], [0, 1]).all()

    values = pre.transform(train)
    assert values[0, indicators["income"]] == 1
    assert values[1, indicators["income"]] == 0
    assert not values[:, indicators["current_address_months_count"]].any()


@pytest.mark.parametrize("split", ["validation", "test"])
def test_first_missing_after_train_keeps_width_and_logs_warning(frame, split, caplog):
    parts = split_time(frame)
    pre = TrainPreprocessor().fit(parts["train"])
    future = parts[split].copy()
    row = future.index[0]
    future.loc[row, "current_address_months_count"] = -1
    future.loc[row, "income"] = np.nan
    future["phone_home_valid"] = future.phone_home_valid.astype(float)
    future.loc[row, "phone_home_valid"] = np.nan
    future.loc[row, "payment_type"] = np.nan

    mapping = pre.feature_mapping()
    values = pre.transform(future, split=split)
    assert values.shape == (len(future), len(mapping))
    assert np.isfinite(values).all()
    assert mapping == pre.feature_mapping()
    indicator = next(r["index"] for r in mapping if r["kind"] == "missing_indicator" and r["original_feature"] == "current_address_months_count")
    np.testing.assert_array_equal(values[:, indicator], [1, 0, 0, 0, 0, 0])
    assert not any(r["kind"] == "missing_indicator" and r["original_feature"] == "income" for r in mapping)
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert caplog.records[0].levelname == "WARNING"
    for expected in [split, "current_address_months_count", "income", "phone_home_valid", "payment_type"]:
        assert expected in message
