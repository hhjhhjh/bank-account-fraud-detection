from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from baf.schema import BINARY_FEATURES, CATEGORICAL_FEATURES, EXPECTED_COLUMNS, NUMERIC_FEATURES, SPLIT_MONTHS, normalize_missing


def audit_chunks(chunks, output: Path) -> dict:
    """Aggregate observed values only; never materialize all raw rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = 0
    dtypes = defaultdict(set)
    raw_missing = Counter()
    special_missing = Counter()
    effective_missing = Counter()
    monthly = defaultdict(Counter)
    numeric = {}
    category = Counter()
    for frame in chunks:
        rows += len(frame)
        clean = normalize_missing(frame)
        for name in frame.columns:
            dtypes[name].add(str(frame[name].dtype))
            raw_missing[name] += int(frame[name].isna().sum())
            effective_missing[name] += int(clean[name].isna().sum())
            special_missing[name] += int((clean[name].isna() & frame[name].notna()).sum())
        for month, group in clean.groupby("month"):
            month = int(month)
            monthly[month].update(rows=len(group), fraud=int(group.fraud_bool.sum()))
            for name in (*NUMERIC_FEATURES, *BINARY_FEATURES):
                values = group[name].dropna().to_numpy(dtype=np.float64)
                stats = numeric.setdefault((month, name), {"count": 0, "sum": 0., "sum_squares": 0., "min": np.inf, "max": -np.inf})
                if len(values):
                    stats["count"] += len(values)
                    stats["sum"] += float(values.sum())
                    stats["sum_squares"] += float(np.square(values).sum())
                    stats["min"] = min(stats["min"], float(values.min()))
                    stats["max"] = max(stats["max"], float(values.max()))
            for name in CATEGORICAL_FEATURES:
                counts = group[name].fillna("__MISSING__").value_counts()
                category.update({(month, name, str(value)): int(count) for value, count in counts.items()})
        if any(len({key[2] for key in category if key[1] == name}) > 100 for name in CATEGORICAL_FEATURES):
            raise ValueError("Unexpected categorical cardinality above 100; check Base version")
    if set(monthly) != set(range(8)):
        raise ValueError(f"Expected month 0..7; observed {sorted(monthly)}")
    month_rows = [{"month": month, "rows": monthly[month]["rows"], "fraud_count": monthly[month]["fraud"], "fraud_rate": monthly[month]["fraud"] / monthly[month]["rows"]} for month in range(8)]
    monthly_df = pd.DataFrame(month_rows)
    split_rows = []
    for split, months in SPLIT_MONTHS.items():
        selected = monthly_df[monthly_df.month.isin(months)]
        total, positives = int(selected.rows.sum()), int(selected.fraud_count.sum())
        if positives == 0 or positives == total:
            raise ValueError(f"{split}: both target classes are required for future evaluation")
        split_rows.append({"split": split, "months": ",".join(map(str, months)), "rows": total, "fraud_count": positives, "fraud_rate": positives / total})
    schema_df = pd.DataFrame([{"column": name, "observed_dtypes": ",".join(sorted(dtypes[name])), "role": "target" if name == "fraud_bool" else "time" if name == "month" else "categorical" if name in CATEGORICAL_FEATURES else "binary" if name in BINARY_FEATURES else "numeric", "raw_missing": raw_missing[name], "special_missing": special_missing[name], "effective_missing": effective_missing[name], "effective_missing_rate": effective_missing[name] / rows} for name in EXPECTED_COLUMNS])
    numeric_rows = []
    for (month, name), stats in sorted(numeric.items()):
        count = stats["count"]
        mean = stats["sum"] / count if count else np.nan
        std = np.sqrt(max(0., stats["sum_squares"] / count - mean ** 2)) if count else np.nan
        numeric_rows.append({"month": month, "feature": name, "count": count, "mean": mean, "std": std, "min": stats["min"] if count else np.nan, "max": stats["max"] if count else np.nan})
    output.mkdir(parents=True, exist_ok=True)
    monthly_df.to_csv(output / "monthly.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output / "splits.csv", index=False)
    schema_df.to_csv(output / "schema.csv", index=False)
    pd.DataFrame(numeric_rows).to_csv(output / "numeric_by_month.csv", index=False)
    pd.DataFrame([{"month": m, "feature": f, "category": c, "count": n, "proportion": n / monthly[m]["rows"]} for (m, f, c), n in sorted(category.items())]).to_csv(output / "categories_by_month.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].bar(monthly_df.month, monthly_df.rows)
    axes[0].set_ylabel("Applications")
    axes[1].plot(monthly_df.month, monthly_df.fraud_rate * 100, marker="o")
    axes[1].set(xlabel="Month", ylabel="Fraud (%)", xticks=range(8))
    fig.tight_layout()
    fig.savefig(output / "monthly.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 7))
    shown = schema_df.sort_values("effective_missing_rate").tail(15)
    ax.barh(shown.column, shown.effective_missing_rate * 100)
    ax.set_xlabel("Missing after documented sentinel conversion (%)")
    fig.tight_layout()
    fig.savefig(output / "missingness.png", dpi=160)
    plt.close(fig)
    # Standardized monthly means show drift without changing preprocessing.
    means = pd.DataFrame(numeric_rows).pivot(index="feature", columns="month", values="mean")
    standardized = means.sub(means.mean(axis=1), axis=0).div(means.std(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(9, 9))
    plot = ax.imshow(np.ma.masked_invalid(standardized.to_numpy()), aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set(xticks=range(8), xticklabels=range(8), yticks=range(len(means)), yticklabels=means.index, xlabel="Month", title="Monthly means: z-score across months (EDA only)")
    fig.colorbar(plot, ax=ax, label="Across-month z-score")
    fig.tight_layout()
    fig.savefig(output / "monthly_mean_shift.png", dpi=160)
    plt.close(fig)
    fraud = sum(v["fraud"] for v in monthly.values())
    return {"rows": rows, "columns": len(EXPECTED_COLUMNS), "fraud_count": fraud, "fraud_rate": fraud / rows, "splits": split_rows}
