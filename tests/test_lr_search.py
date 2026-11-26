import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from baf import lr_search
from baf.cli import load_config
from baf.train import train_mlp


@pytest.fixture
def search(tmp_path, monkeypatch):
    data = tmp_path / "processed"
    configs = tmp_path / "configs"
    data.mkdir()
    configs.mkdir()
    rng = np.random.default_rng(7)
    for split, rows in (("train", 60), ("validation", 30)):
        x = rng.normal(size=(rows, 4)).astype(np.float32)
        np.save(data / f"{split}_X.npy", x)
        np.save(data / f"{split}_y.npy", (x[:, 0] > 0.4).astype(np.int64))
    (data / "manifest.json").write_text(json.dumps({"source_sha256": "a" * 64, "prepared_rows": {"train": 60, "validation": 30}}))
    (data / "feature_mapping.json").write_text(json.dumps([
        {"index": i, "transformed_feature": f"x{i}", "original_feature": f"x{i}"} for i in range(4)
    ]))
    (configs / "base.toml").write_text('''[data]
processed_dir = "processed"
[model]
hidden_dims = [8, 4]
dropout = 0.1
[runtime]
device = "cpu"
[training]
seed = 42
epochs = 2
batch_size = 16
learning_rate = 0.001
weight_decay = 0.0001
patience = 5
mixed_precision = "off"
checkpoint_dir = "checkpoints/b0"
''')
    for seed in (42, 43, 44):
        config = load_config(configs / "base.toml")
        config["training"]["seed"] = seed
        train_mlp(config, tmp_path / f"checkpoints/b0/old-seed{seed}")
    (configs / "search.toml").write_text('''processed_dir = "processed"
checkpoint_dir = "checkpoints"
output_dir = "results/search"
baseline_output_dir = "results/fixed"
device = "cpu"
batch_size = 16
epochs = 2
selection_seed = 42
seeds = [42, 43, 44]
[models.b0]
config = "configs/base.toml"
baseline = "b0/old-seed{seed}"
learning_rates = [0.0003, 0.001, 0.003]
''')
    original_load = np.load

    def forbid_test(path, *args, **kwargs):
        assert not Path(path).name.startswith("test_")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", forbid_test)
    return lr_search.prepare_search(lr_search.load_search_config(configs / "search.toml"))


def test_candidates_reuse_baseline_and_repeat_execution_is_idempotent(search, monkeypatch):
    jobs = lr_search.candidate_jobs(search)
    assert len(jobs) == 3
    baseline = next(j for j in jobs if j["learning_rate"] == 0.001)
    assert "old-seed42" in str(baseline["run_dir"])
    calls = []

    def spy(config, run_dir, **kwargs):
        calls.append(config["training"]["learning_rate"])
        return train_mlp(config, run_dir, **kwargs)

    monkeypatch.setattr(lr_search, "train_mlp", spy)
    first = lr_search.execute_jobs(search, jobs)
    assert calls == [0.0003, 0.003]
    before = {Path(j["run_dir"]): (Path(j["run_dir"]) / "metrics.json").stat().st_mtime_ns for j in jobs}
    second = lr_search.execute_jobs(search, jobs)
    assert len(first) == len(second) == 3
    assert calls == [0.0003, 0.003]
    assert all((p / "metrics.json").stat().st_mtime_ns == stamp for p, stamp in before.items())


def test_selection_requires_all_candidates_and_repeats_selected_lr(search):
    with pytest.raises(ValueError, match="Incomplete|Missing"):
        lr_search.select_learning_rates(search)
    lr_search.execute_jobs(search, lr_search.candidate_jobs(search))
    selection = lr_search.select_learning_rates(search)
    winner = selection["selected"]["b0"]
    jobs = lr_search.repetition_jobs(search)
    assert [j["seed"] for j in jobs] == [42, 43, 44]
    assert all(j["learning_rate"] == winner for j in jobs)
    result = lr_search.execute_jobs(search, jobs)
    assert len(result) == 3


def test_reuse_rejects_changed_data_fingerprint(search):
    job = next(j for j in lr_search.candidate_jobs(search) if j["learning_rate"] == 0.001)
    path = Path(job["run_dir"]) / "metrics.json"
    payload = json.loads(path.read_text())
    payload["signature"]["data"]["source_sha256"] = "b" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="configuration or data"):
        lr_search.execute_jobs(search, [job])


def test_prepare_rejects_changed_baseline_training_conditions(search, tmp_path):
    path = tmp_path / "configs/base.toml"
    path.write_text(path.read_text().replace("weight_decay = 0.0001", "weight_decay = 0.0002"))
    with pytest.raises(ValueError, match="configuration or data"):
        lr_search.prepare_search(lr_search.load_search_config(tmp_path / "configs/search.toml"))


def test_legacy_metrics_parameter_count_is_derived_without_modifying_original(search, tmp_path):
    job = next(j for j in lr_search.candidate_jobs(search) if j["learning_rate"] == 0.001)
    path = Path(job["run_dir"]) / "metrics.json"
    legacy = json.loads(path.read_text())
    legacy.pop("trainable_parameters")
    path.write_text(json.dumps(legacy))
    before = path.read_bytes()
    records = lr_search.execute_jobs(search, [job])
    lr_search.write_run_tables(records, tmp_path / "summary")
    assert records[0]["trainable_parameters"] == 81
    assert path.read_bytes() == before


@pytest.mark.parametrize("damage", ["missing", "different_best", "nonfinite_last"])
def test_reuse_rejects_damaged_checkpoint_weights(search, damage):
    job = next(j for j in lr_search.candidate_jobs(search) if j["learning_rate"] == 0.001)
    path = Path(job["run_dir"]) / ("last.pt" if damage == "nonfinite_last" else "best.pt")
    state = torch.load(path, weights_only=True)
    if damage == "missing":
        state.pop("model_state")
    else:
        key = next(iter(state["model_state"]))
        if damage == "different_best":
            state["model_state"][key] += 1
        else:
            state["model_state"][key].fill_(float("inf"))
    torch.save(state, path)
    with pytest.raises(ValueError, match="weights"):
        lr_search.execute_jobs(search, [job])


def test_cli_model_batches_keep_all_completed_records(search, tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    base = (configs / "base.toml").read_text()
    (configs / "e1.toml").write_text(base.replace("[training]", '[training]\nloss = "weighted_bce"'))
    for seed in (42, 43, 44):
        config = load_config(configs / "e1.toml")
        config["training"]["seed"] = seed
        train_mlp(config, tmp_path / f"checkpoints/e1/old-seed{seed}")
    path = configs / "search.toml"
    path.write_text(path.read_text() + '''
[models.e1]
config = "configs/e1.toml"
baseline = "e1/old-seed{seed}"
learning_rates = [0.0003, 0.001, 0.003]
''')
    for model in ("b0", "e1"):
        monkeypatch.setattr("sys.argv", ["lr_search", "run", "--config", str(path), "--models", model])
        lr_search.main()
    result = json.loads((tmp_path / "results/search/candidates/training_runs.json").read_text())
    assert len(result["runs"]) == 6
    assert {r["model"] for r in result["runs"]} == {"b0", "e1"}


def test_repetition_rejects_changed_protocol_or_candidate_record(search):
    lr_search.execute_jobs(search, lr_search.candidate_jobs(search))
    lr_search.select_learning_rates(search)
    altered = copy.deepcopy(search)
    altered["protocol_sha256"] = "changed"
    with pytest.raises(ValueError, match="protocol"):
        lr_search.repetition_jobs(altered)
    job = lr_search.candidate_jobs(search)[0]
    path = Path(job["run_dir"]) / "metrics.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="candidate"):
        lr_search.repetition_jobs(search)


def test_unrounded_ap_and_exact_tie_policy():
    choose = lr_search.choose_learning_rate
    assert choose({0.0003: 0.17, 0.001: 0.17, 0.003: 0.17}) == 0.001
    assert choose({0.0003: 0.18, 0.001: 0.17, 0.003: 0.18}) == 0.0003
    assert choose({0.0003: 0.17, 0.001: 0.17, 0.003: 0.170000001}) == 0.003
    with pytest.raises(ValueError, match="finite"):
        choose({0.001: float("nan")})


def test_selection_is_portable_and_cannot_change_the_winner(search, tmp_path):
    lr_search.execute_jobs(search, lr_search.candidate_jobs(search))
    lr_search.select_learning_rates(search)
    moved = copy.deepcopy(search)
    destination = tmp_path / "second-machine"
    shutil.copytree(search["checkpoint_dir"], destination / "checkpoints")
    shutil.copytree(search["output_dir"], destination / "results")
    moved["checkpoint_dir"] = str(destination / "checkpoints")
    moved["output_dir"] = str(destination / "results")
    assert len(lr_search.repetition_jobs(moved)) == 3
    path = destination / "results/selection.json"
    altered = json.loads(path.read_text())
    chosen = altered["selected"]["b0"]
    altered["selected"]["b0"] = next(r for r in (0.0003, 0.001, 0.003) if r != chosen)
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="selected|winner"):
        lr_search.repetition_jobs(moved)


def test_incomplete_run_is_preserved_and_requires_explicit_resume(search):
    job = lr_search.candidate_jobs(search)[0]
    cfg = copy.deepcopy(search["models"]["b0"]["config"])
    cfg["training"].update(learning_rate=job["learning_rate"], seed=42, epochs=1)
    train_mlp(cfg, Path(job["run_dir"]))
    (Path(job["run_dir"]) / "metrics.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        lr_search.execute_jobs(search, [job])
    result = lr_search.execute_jobs(search, [job], resume_incomplete=True)
    assert len(result[0]["metrics"]["history"]) == 2
    assert list(Path(job["run_dir"]).parent.glob(Path(job["run_dir"]).name + ".interrupted-*"))
