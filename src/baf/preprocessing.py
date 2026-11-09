import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from baf.schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    DOCUMENTED_MISSING_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    normalize_missing,
    validate_frame,
)

logger = logging.getLogger(__name__)


class TrainPreprocessor:
    def fit(self, train: pd.DataFrame):
        validate_frame(train)
        if train.empty or not train.month.isin(range(6)).all():
            raise ValueError("Preprocessor must fit nonempty Train (month 0..5) only")
        clean = normalize_missing(train.loc[:, FEATURES])
        if any(clean[name].nunique(dropna=False) > 100 for name in CATEGORICAL_FEATURES):
            raise ValueError("Categorical cardinality exceeds the BAF contract safety limit (100)")
        self.train_missing_features_ = frozenset(clean.columns[clean.isna().any()])
        self.numeric_indicator_features_ = [
            name for name in NUMERIC_FEATURES
            if name in DOCUMENTED_MISSING_FEATURES or name in self.train_missing_features_
        ]
        self.transformer = ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), list(NUMERIC_FEATURES)),
            ("numeric_missing", MissingIndicator(features="all"), self.numeric_indicator_features_),
            ("binary", SimpleImputer(strategy="most_frequent", add_indicator=True, keep_empty_features=True), list(BINARY_FEATURES)),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
            ]), list(CATEGORICAL_FEATURES)),
        ], remainder="drop", verbose_feature_names_out=True)
        self.transformer.fit(clean)
        return self

    def transform(self, frame: pd.DataFrame, *, split: str | None = None) -> np.ndarray:
        # Explicit projection also accepts features-only inference inputs.
        clean = normalize_missing(frame.loc[:, FEATURES])
        missing = clean.isna().sum()
        first_seen = missing[(missing > 0) & ~missing.index.isin(self.train_missing_features_)]
        if not first_seen.empty:
            logger.warning(
                "Missing values not seen in Train (%s): %s",
                split or "transform",
                ", ".join(f"{name}={count}" for name, count in first_seen.items()),
            )
        result = self.transformer.transform(clean).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Preprocessing produced non-finite features")
        return result

    def feature_mapping(self) -> list[dict]:
        rows = [{"original_feature": name, "kind": "numeric"} for name in NUMERIC_FEATURES]
        rows.extend({"original_feature": name, "kind": "missing_indicator"} for name in self.numeric_indicator_features_)
        rows.extend({"original_feature": name, "kind": "binary"} for name in BINARY_FEATURES)
        binary = self.transformer.named_transformers_["binary"]
        rows.extend(
            {"original_feature": BINARY_FEATURES[index], "kind": "missing_indicator"}
            for index in binary.indicator_.features_
        )
        encoder = self.transformer.named_transformers_["categorical"].named_steps["encode"]
        for name, categories in zip(CATEGORICAL_FEATURES, encoder.categories_, strict=True):
            for category in categories:
                rows.append({"original_feature": name, "kind": "one_hot", "category": str(category)})
        names = self.transformer.get_feature_names_out()
        if len(rows) != len(names):
            raise ValueError("Feature mapping and transformed feature width differ")
        return [{"index": i, "transformed_feature": str(name), **row} for i, (name, row) in enumerate(zip(names, rows, strict=True))]
