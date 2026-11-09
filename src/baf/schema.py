"""BAF Base contract from the official datasheet, Q7 (not measured statistics)."""

import numpy as np
import pandas as pd

TARGET = "fraud_bool"
TIME = "month"
CATEGORICAL_FEATURES = ("payment_type", "employment_status", "housing_status", "source", "device_os")
BINARY_FEATURES = ("email_is_free", "phone_home_valid", "phone_mobile_valid", "has_other_cards", "foreign_request", "keep_alive_session")
NUMERIC_FEATURES = (
    "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "zip_count_4w", "velocity_6h", "velocity_24h",
    "velocity_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
    "credit_risk_score", "bank_months_count", "proposed_credit_limit",
    "session_length_in_minutes", "device_distinct_emails", "device_fraud_count",
)
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
EXPECTED_COLUMNS = (TARGET, *FEATURES, TIME)
MINUS_ONE_MISSING = (
    "prev_address_months_count", "current_address_months_count", "bank_months_count",
    "session_length_in_minutes", "device_distinct_emails",
)
NEGATIVE_MISSING_FEATURE = "intended_balcon_amount"
DOCUMENTED_MISSING_FEATURES = (*MINUS_ONE_MISSING, NEGATIVE_MISSING_FEATURE)
SPLIT_MONTHS = {"train": tuple(range(6)), "validation": (6,), "test": (7,)}


def validate_frame(frame: pd.DataFrame) -> None:
    missing = set(EXPECTED_COLUMNS) - set(frame.columns)
    extra = set(frame.columns) - set(EXPECTED_COLUMNS)
    if missing or extra or frame.columns.duplicated().any():
        raise ValueError(f"BAF Base schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    for name in (TARGET, TIME, *NUMERIC_FEATURES, *BINARY_FEATURES):
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError(f"{name}: expected numeric dtype, got {frame[name].dtype}")
        if np.isinf(frame[name].to_numpy(dtype=float)).any():
            raise ValueError(f"{name}: infinite values are invalid")
    for name, allowed in [(TARGET, (0, 1)), (TIME, tuple(range(8))), *[(n, (0, 1)) for n in BINARY_FEATURES]]:
        values = frame[name]
        if name in (TARGET, TIME) and values.isna().any():
            raise ValueError(f"{name}: missing values are invalid")
        if not values.dropna().isin(allowed).all():
            raise ValueError(f"{name}: expected values {allowed}")
    for name in CATEGORICAL_FEATURES:
        values = frame[name].dropna()
        if not values.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError(f"{name}: expected nonblank categorical strings")


def normalize_missing(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    for name in MINUS_ONE_MISSING:
        clean[name] = clean[name].mask(clean[name].eq(-1))
    name = NEGATIVE_MISSING_FEATURE
    clean[name] = clean[name].mask(clean[name].lt(0))
    return clean
