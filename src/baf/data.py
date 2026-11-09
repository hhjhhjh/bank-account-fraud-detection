import csv
from pathlib import Path
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd

from baf.schema import EXPECTED_COLUMNS, SPLIT_MONTHS, validate_frame

DOWNLOAD_URL = "https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022"
CSV_COLUMN_ALIASES = {"device_distinct_emails_8w": "device_distinct_emails"}


def read_csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle), [])


def iter_base_csv(path: Path | str, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"BAF Base CSV not found: {path}. Download Base.csv from Kaggle: {DOWNLOAD_URL}")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    header = read_csv_header(path)
    has_index = bool(header) and header[0] in ("", "Unnamed: 0")
    raw_columns = header[1:] if has_index else header
    columns = [CSV_COLUMN_ALIASES.get(name, name) for name in raw_columns]
    if len(columns) != len(set(columns)) or set(columns) != set(EXPECTED_COLUMNS):
        raise ValueError("BAF Base CSV header does not match the 32-column schema; inspect missing/extra/duplicate names")
    offset = 0
    # Never unconditionally set index_col=0: some distributions have no index.
    for chunk in pd.read_csv(path, chunksize=chunksize, encoding="utf-8-sig"):
        if has_index:
            exported = chunk.pop(chunk.columns[0]).to_numpy()
            if not np.array_equal(exported, np.arange(offset, offset + len(chunk))):
                raise ValueError("CSV exported index must be a contiguous zero-based row index")
        chunk.rename(columns=CSV_COLUMN_ALIASES, inplace=True)
        chunk.index = pd.RangeIndex(offset, offset + len(chunk))
        validate_frame(chunk)
        offset += len(chunk)
        yield chunk
    if offset == 0:
        raise ValueError("BAF Base CSV is empty")


def split_time(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    validate_frame(frame)
    return {name: frame.loc[frame.month.isin(months)].copy() for name, months in SPLIT_MONTHS.items()}


def sample_by_month(chunks: Iterable[pd.DataFrame], per_month: int, seed: int) -> pd.DataFrame:
    """Uniform priority sampling, bounded by eight reservoirs and one input chunk."""
    if per_month <= 0:
        raise ValueError("sample_per_month must be positive")
    rng = np.random.default_rng(seed)
    reservoirs = {}
    for chunk in chunks:
        priorities = pd.Series(rng.random(len(chunk)), index=chunk.index)
        for month, group in chunk.groupby("month"):
            group = group.assign(_priority=priorities.loc[group.index])
            combined = pd.concat([reservoirs[month], group]) if month in reservoirs else group
            reservoirs[month] = combined.nsmallest(per_month, "_priority")
    if set(reservoirs) != set(range(8)):
        raise ValueError("Smoke sampling requires every month 0..7")
    return pd.concat(reservoirs.values()).drop(columns="_priority").sort_index()
