import argparse
import hashlib
import json
import shutil
import tempfile
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from baf.audit import audit_chunks
from baf.data import CSV_COLUMN_ALIASES, DOWNLOAD_URL, iter_base_csv, read_csv_header, sample_by_month, split_time
from baf.preprocessing import TrainPreprocessor
from baf.runtime import environment_report
from baf.schema import DOCUMENTED_MISSING_FEATURES, SPLIT_MONTHS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare_arrays(chunks, preprocessor, counts, output):
    width = len(preprocessor.feature_mapping())
    arrays = {}
    offsets = {name: 0 for name in SPLIT_MONTHS}
    for name, count in counts.items():
        for suffix, dtype, shape in [("X", np.float32, (count, width)), ("y", np.int64, (count,)), ("row_ids", np.int64, (count,))]:
            arrays[name, suffix] = np.lib.format.open_memmap(output / f"{name}_{suffix}.npy", mode="w+", dtype=dtype, shape=shape)
    for chunk in chunks:
        for name, months in SPLIT_MONTHS.items():
            part = chunk.loc[chunk.month.isin(months)]
            if part.empty:
                continue
            start = offsets[name]
            end = start + len(part)
            arrays[name, "X"][start:end] = preprocessor.transform(part, split=name)
            arrays[name, "y"][start:end] = part.fraud_bool.to_numpy(dtype=np.int64)
            arrays[name, "row_ids"][start:end] = part.index.to_numpy(dtype=np.int64)
            offsets[name] = end
    if offsets != counts:
        raise ValueError(f"Prepared row counts do not match audited counts: {offsets} != {counts}")
    for array in arrays.values():
        array.flush()


def run_phase1(config: dict, sample_per_month: int | None = None, audit_only=False):
    started = time.perf_counter()
    source = Path(config["data"]["path"])
    if not source.is_file():
        raise FileNotFoundError(f"BAF Base CSV not found: {source}. Download Base.csv from Kaggle: {DOWNLOAD_URL}")
    chunksize = config["data"]["chunksize"]
    seed = config["data"]["seed"]
    if type(chunksize) is not int or chunksize <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("chunksize must be a positive integer and seed a nonnegative integer")
    if sample_per_month is not None and (type(sample_per_month) is not int or sample_per_month <= 0):
        raise ValueError("sample_per_month must be a positive integer")
    if audit_only and sample_per_month is not None:
        raise ValueError("audit-only scans the complete file; do not combine with sampling")
    report_dir = Path(config["output"]["report_dir"]).resolve()
    processed_dir = Path(config["output"]["processed_dir"]).resolve()
    if report_dir.is_relative_to(processed_dir) or processed_dir.is_relative_to(report_dir):
        raise ValueError("Report and processed directories must be separate, non-nested paths")
    destinations = [report_dir] if audit_only else [report_dir, processed_dir]
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}. Keep it or select a fresh output directory; no files overwritten")
    environment = environment_report(config.get("runtime", {}).get("device", "auto"))
    original_hash = sha256_file(source)
    source_columns = read_csv_header(source)
    stages = []
    published = []
    try:
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            stages.append(Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)))
        report_stage = stages[0]
        print("Auditing complete source CSV ...", flush=True)
        audit = audit_chunks(iter_base_csv(source, chunksize), report_stage)
        counts = {}
        if not audit_only:
            processed_stage = stages[1]
            sampled = None
            if sample_per_month is not None:
                sampled = sample_by_month(iter_base_csv(source, chunksize), sample_per_month, seed)
                train = split_time(sampled)["train"]
                counts = {name: int(sampled.month.isin(months).sum()) for name, months in SPLIT_MONTHS.items()}
            else:
                # Only the raw Train partition is materialized for fitting.
                train_parts = [chunk.loc[chunk.month.isin(SPLIT_MONTHS["train"])] for chunk in iter_base_csv(source, chunksize)]
                train = pd.concat(train_parts)
                del train_parts
                counts = {part["split"]: part["rows"] for part in audit["splits"]}
            print(f"Fitting Train only: {len(train):,} rows ...", flush=True)
            preprocessor = TrainPreprocessor().fit(train)
            del train
            joblib.dump(preprocessor, processed_stage / "preprocessor.joblib")
            write_json(report_stage / "feature_mapping.json", preprocessor.feature_mapping())
            write_json(processed_stage / "feature_mapping.json", preprocessor.feature_mapping())
            chunks = [sampled] if sampled is not None else iter_base_csv(source, chunksize)
            prepare_arrays(chunks, preprocessor, counts, processed_stage)
        if original_hash != sha256_file(source):
            raise ValueError("Source file changed during processing; discard this run and retry on a stable copy")
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "audit_only" if audit_only else "smoke" if sample_per_month is not None else "full",
            "source_filename": source.name,
            "source_sha256": original_hash,
            "source_bytes": source.stat().st_size,
            "source_columns": source_columns,
            "source_column_aliases": {name: CSV_COLUMN_ALIASES[name] for name in source_columns if name in CSV_COLUMN_ALIASES},
            "config": config,
            "sample_per_month": sample_per_month,
            "audit_scope": "complete_source_file",
            "audit": audit,
            "prepared_rows": counts,
            "preprocessor_fit_split": None if audit_only else "train",
            "preprocessing": {
                "numeric_missing": "train_median; all-missing Train column falls back to 0",
                "binary_missing": "train_mode; all-missing Train column falls back to 0",
                "indicators": "0/1 for documented numeric columns and numeric/binary columns missing in Train",
                "fixed_numeric_indicators": list(DOCUMENTED_MISSING_FEATURES),
                "categorical": "constant __MISSING__ then train-fitted 0/1 one-hot; unseen categories map to all zeros",
                "numeric_scaling": "Train StandardScaler on numeric values only",
                "excluded": ["fraud_bool", "month"],
            },
            "environment": environment,
            "elapsed_seconds": time.perf_counter() - started,
            "training_executed": False,
        }
        write_json(report_stage / "manifest.json", manifest)
        if not audit_only:
            write_json(stages[1] / "manifest.json", manifest)
        for stage, destination in zip(stages, destinations, strict=True):
            stage.rename(destination)
            published.append(destination)
        print(f"Saved verified Phase 1 artifacts: {report_dir}", flush=True)
        return manifest
    except BaseException:
        # Roll back only the directories created by this invocation.
        for path in published:
            shutil.rmtree(path)
        raise
    finally:
        for stage in stages:
            if stage.exists():
                shutil.rmtree(stage)


def load_config(path: Path) -> dict:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    # Paths are relative to the project root (the parent of configs/), never cwd.
    root = path.resolve().parent.parent
    for section, key in [("data", "path"), ("data", "processed_dir"), ("output", "processed_dir"), ("output", "report_dir"), ("training", "checkpoint_dir")]:
        if section in config and key in config[section]:
            value = Path(config[section][key]).expanduser()
            config[section][key] = str(value if value.is_absolute() else root / value)
    return config


def main():
    parser = argparse.ArgumentParser(description="Validate BAF Base, audit time splits, fit Train-only preprocessing. No training.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase1.toml"))
    parser.add_argument("--data", type=Path, help="Override CSV path (relative to cwd)")
    parser.add_argument("--processed-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--sample-per-month", type=int, help="Uniform sample per month for smoke preprocessing; audit still reads the full CSV")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        for value, section, key in [(args.data, "data", "path"), (args.processed_dir, "output", "processed_dir"), (args.report_dir, "output", "report_dir")]:
            if value is not None:
                config[section][key] = str(value.expanduser().resolve())
        run_phase1(config, args.sample_per_month, args.audit_only)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
