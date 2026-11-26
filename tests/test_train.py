import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from baf import train
from baf.train import train_b0


@pytest.fixture
def b0_config(tmp_path):
    data = tmp_path / "processed"
    data.mkdir()
    rng = np.random.default_rng(7)
    prepared_rows = {"train": 60, "validation": 30}
    for split, count in prepared_rows.items():
        features = rng.normal(size=(count, 4)).astype(np.float32)
        labels = (features[:, 0] > 0.4).astype(np.int64)
        np.save(data / f"{split}_X.npy", features)
        np.save(data / f"{split}_y.npy", labels)
    mapping = [{"index": i, "transformed_feature": f"x{i}", "original_feature": f"x{i}"} for i in range(4)]
    (data / "feature_mapping.json").write_text(json.dumps(mapping))
    (data / "manifest.json").write_text(json.dumps({"source_sha256": "a" * 64, "prepared_rows": prepared_rows}))
    return {
        "data": {"processed_dir": str(data)},
        "model": {"hidden_dims": [8, 4], "dropout": 0.1},
        "runtime": {"device": "cpu"},
        "training": {
            "seed": 42, "epochs": 3, "batch_size": 16, "learning_rate": 0.003,
            "weight_decay": 0.0001, "patience": 5, "mixed_precision": "off",
        },
        "smoke": {"train_rows": 40, "validation_rows": 20},
    }


def test_resume_matches_continuous_training_and_restores_best_checkpoint(b0_config, tmp_path):
    continuous = train_b0(b0_config, tmp_path / "continuous", smoke=True)
    first = {**b0_config, "training": {**b0_config["training"], "epochs": 1}}
    train_b0(first, tmp_path / "first", smoke=True)
    checkpoint = tmp_path / "first" / "last.pt"
    legacy = torch.load(checkpoint, weights_only=True)
    for row in legacy["history"]:
        for field in ("training_rows", "fraud_presentations", "optimizer_steps"):
            row.pop(field)
    torch.save(legacy, checkpoint)
    resumed = train_b0(b0_config, tmp_path / "resumed", smoke=True, resume=checkpoint)

    assert resumed["mode"] == "smoke"
    assert resumed["splits"]["train"]["rows"] == 40
    assert resumed["splits"]["validation"]["rows"] == 20
    assert [row["epoch"] for row in resumed["history"]] == [1, 2, 3]
    keys = ("train_loss", "validation_loss", "validation_ap")
    for a, b in zip(continuous["history"], resumed["history"], strict=True):
        assert [a[k] for k in keys] == [b[k] for k in keys]

    best = torch.load(tmp_path / "resumed" / "best.pt", weights_only=True)
    expected = max(resumed["history"], key=lambda row: row["validation_ap"])
    assert best["epoch"] == expected["epoch"]
    assert best["validation_ap"] == expected["validation_ap"]
    reference = torch.load(tmp_path / "continuous" / "last.pt", weights_only=True)
    restored = torch.load(tmp_path / "resumed" / "last.pt", weights_only=True)
    for key in reference["model_state"]:
        torch.testing.assert_close(reference["model_state"][key], restored["model_state"][key], rtol=0, atol=0)

    changed = {**b0_config, "training": {**b0_config["training"], "learning_rate": 0.01}}
    with pytest.raises(ValueError, match="configuration or data"):
        train_b0(changed, tmp_path / "changed", smoke=True, resume=checkpoint)


def test_ap_ties_keep_earliest_checkpoint_and_trigger_early_stopping(b0_config, tmp_path, monkeypatch):
    monkeypatch.setattr("baf.train.average_precision_score", lambda labels, scores: 0.5)
    b0_config["training"].update(epochs=6, patience=2)
    result = train_b0(b0_config, tmp_path / "ties")
    assert result["stop_reason"] == "early_stopping"
    assert len(result["history"]) == 3
    assert result["best_epoch"] == 1
    best = torch.load(tmp_path / "ties" / "best.pt", weights_only=True)
    assert best["epoch"] == 1


def test_mismatched_validation_width_fails_before_training(b0_config, tmp_path):
    data = Path(b0_config["data"]["processed_dir"])
    np.save(data / "validation_X.npy", np.zeros((30, 3), dtype=np.float32))
    run_dir = tmp_path / "bad-width"
    with pytest.raises(ValueError, match="validation_X.npy"):
        train_b0(b0_config, run_dir)
    assert not run_dir.exists()


def test_weighted_bce_uses_only_selected_train_labels(b0_config, tmp_path, monkeypatch):
    data = Path(b0_config["data"]["processed_dir"])
    np.save(data / "train_y.npy", (np.arange(60) < 17).astype(np.int64))
    np.save(data / "validation_y.npy", (np.arange(30) < 20).astype(np.int64))
    b0_config["training"].update(loss="weighted_bce", epochs=1)

    train_epoch = train.train_epoch
    validate_epoch = train.validate_epoch

    def check_train(model, batches, loss_fn, optimizer, scaler):
        assert loss_fn.pos_weight.item() == pytest.approx(29 / 11)
        return train_epoch(model, batches, loss_fn, optimizer, scaler)

    def check_validation(model, batches, loss_fn):
        assert loss_fn.pos_weight is None
        return validate_epoch(model, batches, loss_fn)

    monkeypatch.setattr(train, "train_epoch", check_train)
    monkeypatch.setattr(train, "validate_epoch", check_validation)
    run_dir = tmp_path / "weighted"
    result = train.train_mlp(b0_config, run_dir, smoke=True)

    assert result["splits"]["train"] == {"rows": 40, "fraud_count": 11}
    assert result["signature"]["experiment"] == "E1"
    assert result["signature"]["pos_weight"] == pytest.approx(29 / 11)
    checkpoint = torch.load(run_dir / "last.pt", weights_only=True)
    assert checkpoint["signature"] == result["signature"]


def test_weighted_bce_resume_rejects_switch_to_unweighted_bce(b0_config, tmp_path):
    b0_config["training"].update(loss="weighted_bce", epochs=1)
    run_dir = tmp_path / "weighted"
    train.train_mlp(b0_config, run_dir)

    b0_config["training"]["epochs"] = 2
    resumed = train.train_mlp(b0_config, tmp_path / "resumed", resume=run_dir / "last.pt")
    assert [row["epoch"] for row in resumed["history"]] == [1, 2]

    b0_config["training"]["loss"] = "bce"
    with pytest.raises(ValueError, match="configuration or data"):
        train.train_mlp(b0_config, tmp_path / "unweighted", resume=run_dir / "last.pt")


def test_focal_gamma_is_saved_and_checked_on_resume(b0_config, tmp_path):
    b0_config["training"].update(loss="focal", focal_gamma=2.0, epochs=1)
    run_dir = tmp_path / "focal"
    result = train.train_mlp(b0_config, run_dir)
    checkpoint = torch.load(run_dir / "last.pt", weights_only=True)
    assert checkpoint["signature"] == result["signature"]
    assert result["signature"]["experiment"] == "E2"
    assert result["signature"]["loss"] == "BinaryFocalLoss"
    assert result["signature"]["focal_gamma"] == 2.0

    b0_config["training"]["epochs"] = 2
    resumed = train.train_mlp(b0_config, tmp_path / "resumed", resume=run_dir / "last.pt")
    assert [row["epoch"] for row in resumed["history"]] == [1, 2]

    b0_config["training"]["focal_gamma"] = 1.0
    with pytest.raises(ValueError, match="configuration or data"):
        train.train_mlp(b0_config, tmp_path / "different-gamma", resume=run_dir / "last.pt")


def test_cli_overrides_focal_gamma_in_saved_metrics(b0_config, tmp_path, monkeypatch):
    b0_config["training"].update(loss="focal", focal_gamma=2.0, epochs=1, checkpoint_dir=str(tmp_path))
    monkeypatch.setattr(train, "load_config", lambda path: b0_config)
    monkeypatch.setattr("sys.argv", ["baf-train", "--run-name", "gamma-check", "--focal-gamma", "3.5"])

    train.main()

    metrics = json.loads((tmp_path / "gamma-check" / "metrics.json").read_text())
    assert metrics["signature"]["focal_gamma"] == 3.5


def test_cli_rejects_focal_gamma_for_bce(b0_config, tmp_path, monkeypatch, capsys):
    b0_config["training"]["checkpoint_dir"] = str(tmp_path)
    monkeypatch.setattr(train, "load_config", lambda path: b0_config)
    monkeypatch.setattr("sys.argv", ["baf-train", "--run-name", "bce-check", "--focal-gamma", "3.5"])

    with pytest.raises(SystemExit) as error:
        train.main()

    assert error.value.code == 2
    assert "--focal-gamma requires a focal loss configuration" in capsys.readouterr().err
    assert not (tmp_path / "bce-check").exists()


def test_cli_records_seed_override_in_metrics_and_checkpoint(b0_config, tmp_path, monkeypatch):
    b0_config["training"].update(epochs=1, checkpoint_dir=str(tmp_path))
    monkeypatch.setattr(train, "load_config", lambda path: b0_config)
    monkeypatch.setattr("sys.argv", ["baf-train", "--run-name", "seed-check", "--seed", "43"])

    train.main()

    run_dir = tmp_path / "seed-check"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    checkpoint = torch.load(run_dir / "last.pt", weights_only=True)
    assert metrics["signature"]["seed"] == checkpoint["signature"]["seed"] == 43


def test_e3_config_matches_e1_training_conditions():
    configs = Path(__file__).resolve().parents[1] / "configs"
    e1 = train.load_config(configs / "e1.toml")
    e3 = train.load_config(configs / "e3.toml")
    assert {k: v for k, v in e3["training"].items() if k != "checkpoint_dir"} == {
        k: v for k, v in e1["training"].items() if k != "checkpoint_dir"
    }
    assert e3["data"] == e1["data"]
    assert e3["runtime"] == e1["runtime"]
    assert e3["smoke"] == e1["smoke"]
    assert e3["model"]["dropout"] == e1["model"]["dropout"]


def test_residual_mlp_uses_weighted_bce_and_resumes(b0_config, tmp_path):
    e1_config = {**b0_config, "training": {**b0_config["training"], "loss": "weighted_bce", "epochs": 1}}
    e3_config = {
        **e1_config,
        "model": {"architecture": "residual_mlp", "width": 8, "num_blocks": 1, "dropout": 0.1},
        "training": e1_config["training"].copy(),
    }
    e1 = train.train_mlp(e1_config, tmp_path / "e1")
    e3 = train.train_mlp(e3_config, tmp_path / "e3")
    signature = e3["signature"]
    assert signature["experiment"] == "E3"
    assert signature["architecture"] == "residual_mlp"
    assert signature["loss"] == e1["signature"]["loss"]
    assert signature["pos_weight"] == e1["signature"]["pos_weight"]
    assert e3["trainable_parameters"] > 0

    checkpoint = tmp_path / "e3" / "last.pt"
    e3_config["training"]["epochs"] = 2
    resumed = train.train_mlp(e3_config, tmp_path / "resumed", resume=checkpoint)
    assert resumed["signature"] == signature
    assert [row["epoch"] for row in resumed["history"]] == [1, 2]

    e1_config["training"]["epochs"] = 2
    with pytest.raises(ValueError, match="configuration or data"):
        train.train_mlp(e1_config, tmp_path / "wrong-architecture", resume=checkpoint)


def test_validation_evaluation_uses_saved_model_and_never_opens_test(b0_config, tmp_path, monkeypatch):
    from baf.evaluation import evaluate_validation

    b0_config["training"].update(loss="weighted_bce", epochs=1)
    checkpoint_root = tmp_path / "checkpoints"
    train.train_mlp(b0_config, checkpoint_root / "e1/school-e1-seed42")
    original_load = np.load

    def forbid_test(path, *args, **kwargs):
        assert not Path(path).name.startswith("test_")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", forbid_test)
    config = {
        "processed_dir": b0_config["data"]["processed_dir"],
        "checkpoint_dir": str(checkpoint_root), "output_dir": str(tmp_path / "evaluation"),
        "device": "cpu", "batch_size": 16, "models": ["e1"], "seeds": [42], "fpr_caps": [0.01, 0.05],
    }
    result = evaluate_validation(config)
    assert result["split"] == "validation" and result["rows"] == 30
    run = result["runs"][0]
    assert run["ap"] == pytest.approx(run["training_validation_ap"])
    assert all(row["fpr"] <= row["fpr_cap"] for row in run["operating_points"])
    data = Path(config["processed_dir"])
    features = original_load(data / "validation_X.npy")
    features[0, 0] += 1
    np.save(data / "validation_X.npy", features)
    with pytest.raises(ValueError, match="dataset mismatch"):
        evaluate_validation(config)


def test_e4_oversamples_train_only_and_records_work(b0_config, tmp_path, monkeypatch):
    data = Path(b0_config["data"]["processed_dir"])
    np.save(data / "train_y.npy", (np.arange(60) < 6).astype(np.int64))
    np.save(data / "validation_y.npy", (np.arange(30) < 20).astype(np.int64))
    b0_config["training"].update(loss="bce", oversampling_ratio=0.5, epochs=1)
    expected_labels = np.load(data / "validation_y.npy")
    expected_rows = train.select_rows(expected_labels, 20, 42)
    original_train, original_validate = train.train_epoch, train.validate_epoch

    def check_train(model, batches, loss_fn, optimizer, scaler):
        batches = list(batches)
        labels = torch.cat([y for _, y in batches])
        assert loss_fn.pos_weight is None
        assert len(labels) == 54 and int(labels.sum()) == 18
        return original_train(model, iter(batches), loss_fn, optimizer, scaler)

    def check_validation(model, batches, loss_fn):
        batches = list(batches)
        labels = torch.cat([y for _, y in batches])
        np.testing.assert_array_equal(labels.numpy(), expected_labels[expected_rows])
        assert loss_fn.pos_weight is None
        return original_validate(model, iter(batches), loss_fn)

    monkeypatch.setattr(train, "train_epoch", check_train)
    monkeypatch.setattr(train, "validate_epoch", check_validation)
    result = train.train_mlp(b0_config, tmp_path / "e4", smoke=True)
    assert result["signature"]["experiment"] == "E4"
    assert result["signature"]["oversampling_ratio"] == 0.5
    assert result["splits"]["train"] == {"rows": 40, "fraud_count": 4}
    assert result["splits"]["validation"] == {"rows": 20, "fraud_count": int(expected_labels[expected_rows].sum())}
    assert result["training_work"]["rows_per_epoch"] == 54
    assert result["training_work"]["optimizer_steps_per_epoch"] == 4
    assert result["training_work"]["total_optimizer_steps"] == 4


def test_e4_resume_is_exact_and_rejects_sampling_ratio_change(b0_config, tmp_path):
    data = Path(b0_config["data"]["processed_dir"])
    np.save(data / "train_y.npy", (np.arange(60) < 6).astype(np.int64))
    b0_config["training"].update(loss="bce", oversampling_ratio=0.5, epochs=2)
    continuous = train.train_mlp(b0_config, tmp_path / "continuous-e4")
    b0_config["training"]["epochs"] = 1
    train.train_mlp(b0_config, tmp_path / "first-e4")
    b0_config["training"]["epochs"] = 2
    checkpoint = tmp_path / "first-e4/last.pt"
    resumed = train.train_mlp(b0_config, tmp_path / "resumed-e4", resume=checkpoint)
    for field in ("train_loss", "validation_loss", "validation_ap"):
        assert [r[field] for r in continuous["history"]] == [r[field] for r in resumed["history"]]
    assert resumed["training_work"] == continuous["training_work"]
    a = torch.load(tmp_path / "continuous-e4/last.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed-e4/last.pt", weights_only=True)
    for key in a["model_state"]:
        torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
    b0_config["training"]["oversampling_ratio"] = 1.0
    with pytest.raises(ValueError, match="configuration or data"):
        train.train_mlp(b0_config, tmp_path / "different-ratio", resume=checkpoint)


def test_e4_rejects_combined_class_weighting(b0_config, tmp_path):
    b0_config["training"].update(loss="weighted_bce", oversampling_ratio=1.0)
    with pytest.raises(ValueError, match="MLP with BCE"):
        train.train_mlp(b0_config, tmp_path / "combined")


def test_cli_overrides_oversampling_ratio_and_preserves_e4_contract(b0_config, tmp_path, monkeypatch):
    data = Path(b0_config["data"]["processed_dir"])
    np.save(data / "train_y.npy", (np.arange(60) < 6).astype(np.int64))
    b0_config["training"].update(loss="bce", epochs=1, oversampling_ratio=0.1, checkpoint_dir=str(tmp_path))
    monkeypatch.setattr(train, "load_config", lambda path: b0_config)
    monkeypatch.setattr("sys.argv", ["baf-train", "--run-name", "ratio-check", "--oversampling-ratio", "0.5"])
    train.main()
    result = json.loads((tmp_path / "ratio-check/metrics.json").read_text())
    assert result["signature"]["experiment"] == "E4"
    assert result["signature"]["oversampling_ratio"] == 0.5
    assert result["training_work"]["fraud_presentations_per_epoch"] == 27


def test_e4_config_matches_b0_except_sampling_and_checkpoint_path():
    configs = Path(__file__).resolve().parents[1] / "configs"
    baseline = train.load_config(configs / "b0.toml")
    sampled = train.load_config(configs / "e4.toml")
    assert sampled["model"] == baseline["model"]
    assert sampled["data"] == baseline["data"]
    assert sampled["runtime"] == baseline["runtime"]
    assert sampled["smoke"] == baseline["smoke"]
    excluded = {"oversampling_ratio", "loss", "checkpoint_dir"}
    assert {k: v for k, v in sampled["training"].items() if k not in excluded} == {
        k: v for k, v in baseline["training"].items() if k not in excluded
    }
    assert sampled["training"]["loss"] == "bce"
