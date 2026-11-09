import json
from pathlib import Path

import numpy as np
import pytest

from baf.cli import run_phase1


def config(tmp_path):
    return {"data": {"path": str(tmp_path / "Base.csv"), "chunksize": 7, "seed": 42}, "output": {"processed_dir": str(tmp_path / "processed"), "report_dir": str(tmp_path / "reports")}, "runtime": {"device": "auto"}}


def test_missing_real_data_does_not_create_reports(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        run_phase1(cfg)
    assert not Path(cfg["output"]["report_dir"]).exists()


def test_full_csv_pipeline_and_no_overwrite(frame, tmp_path):
    cfg = config(tmp_path)
    frame.to_csv(cfg["data"]["path"], index=False)
    run_phase1(cfg)
    report = Path(cfg["output"]["report_dir"])
    manifest = json.loads((report / "manifest.json").read_text())
    assert manifest["audit"]["rows"] == len(frame)
    assert manifest["mode"] == "full"
    assert len(manifest["source_sha256"]) == 64
    processed = Path(cfg["output"]["processed_dir"])
    assert np.load(processed / "train_X.npy", mmap_mode="r").shape[0] == 36
    for split, expected in [("train", set(range(6))), ("validation", {6}), ("test", {7})]:
        ids = np.load(processed / f"{split}_row_ids.npy")
        assert set(frame.iloc[ids].month) == expected
        np.testing.assert_array_equal(np.load(processed / f"{split}_y.npy"), frame.iloc[ids].fraud_bool)
    assert (report / "monthly.png").stat().st_size > 0
    assert (report / "schema.csv").stat().st_size > 0
    with pytest.raises(FileExistsError):
        run_phase1(cfg)


def test_smoke_sample_is_explicit(frame, tmp_path):
    cfg = config(tmp_path)
    frame.to_csv(cfg["data"]["path"], index=False)
    run_phase1(cfg, sample_per_month=2)
    manifest = json.loads((Path(cfg["output"]["report_dir"]) / "manifest.json").read_text())
    assert manifest["mode"] == "smoke"
    assert manifest["audit"]["rows"] == 48
    assert sum(manifest["prepared_rows"].values()) == 16


def test_missing_month_stops_without_publishing_outputs(frame, tmp_path):
    cfg = config(tmp_path)
    frame[frame.month != 7].to_csv(cfg["data"]["path"], index=False)
    with pytest.raises(ValueError, match="month"):
        run_phase1(cfg)
    assert not Path(cfg["output"]["processed_dir"]).exists()
    assert not Path(cfg["output"]["report_dir"]).exists()


def test_audit_only_does_not_fit_or_write_preprocessor(frame, tmp_path):
    cfg = config(tmp_path)
    frame.to_csv(cfg["data"]["path"], index=False)
    result = run_phase1(cfg, audit_only=True)
    assert result["mode"] == "audit_only"
    assert result["preprocessor_fit_split"] is None
    assert not Path(cfg["output"]["processed_dir"]).exists()


@pytest.mark.parametrize("sample_per_month", [0, -1])
def test_bad_sample_size_creates_no_outputs(frame, tmp_path, sample_per_month):
    cfg = config(tmp_path)
    frame.to_csv(cfg["data"]["path"], index=False)
    with pytest.raises(ValueError, match="positive"):
        run_phase1(cfg, sample_per_month=sample_per_month)
    assert not Path(cfg["output"]["report_dir"]).exists()
