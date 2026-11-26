"""Validation operating points from frozen training checkpoints."""

import argparse
import json
import math
import platform
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from baf.cli import sha256_file, write_json
from baf.models import BaselineMLP, ResidualMLP
from baf.ft_transformer import FTTransformer
from baf.runtime import resolve_device
from baf.train import load_training_data


def validate_scores(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Labels and scores must be matching one-dimensional arrays")
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Both binary classes are required")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    return labels, scores


def threshold_metrics(labels, scores, threshold):
    positive = scores >= threshold
    tp = int(np.count_nonzero(positive & (labels == 1)))
    fp = int(np.count_nonzero(positive & (labels == 0)))
    fn = int(np.count_nonzero(~positive & (labels == 1)))
    tn = int(np.count_nonzero(~positive & (labels == 0)))
    return {
        "threshold": float(threshold),
        "fpr": fp / (fp + tn), "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def select_threshold(labels, scores, fpr_cap):
    labels, scores = validate_scores(labels, scores)
    if not math.isfinite(fpr_cap) or not 0 <= fpr_cap <= 1:
        raise ValueError("FPR cap must be finite and between 0 and 1")
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    # Each candidate includes the entire tied score group.
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(ordered) - 1]
    tp = np.r_[0, np.cumsum(labels[order] == 1)[ends]]
    fp = np.r_[0, (ends + 1) - tp[1:]]
    thresholds = np.r_[np.nextafter(ordered[0], np.inf), ordered[ends]]
    fpr = fp / np.count_nonzero(labels == 0)
    feasible = np.flatnonzero(fpr <= fpr_cap)
    # Maximize recall, then minimize false positives, then choose highest threshold.
    best = feasible[np.lexsort((-thresholds[feasible], fp[feasible], -tp[feasible]))[0]]
    return threshold_metrics(labels, scores, thresholds[best])


def load_analysis_config(path):
    path = Path(path)
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    root = path.resolve().parent.parent
    for key in ("processed_dir", "checkpoint_dir", "output_dir"):
        config[key] = str(root / config[key])
    return config


def load_frozen_model(checkpoint_path, fingerprints, device):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1 or state["signature"]["data"] != fingerprints:
        raise ValueError(f"Checkpoint format or dataset mismatch: {checkpoint_path}")
    signature = state["signature"]
    architecture = signature.get("architecture", "mlp")
    model_classes = {"mlp": BaselineMLP, "residual_mlp": ResidualMLP, "ft_transformer": FTTransformer}
    if architecture not in model_classes:
        raise ValueError(f"Unsupported architecture: {architecture}")
    model_class = model_classes[architecture]
    model = model_class(**signature["model"]).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, state


@torch.inference_mode()
def predict_logits(model, features, batch_size, device):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    outputs = []
    for start in range(0, len(features), batch_size):
        batch = np.array(features[start:start + batch_size], copy=True)
        if not np.isfinite(batch).all():
            raise ValueError("Input features must be finite")
        outputs.append(model(torch.from_numpy(batch).to(device)).cpu().numpy())
    return np.concatenate(outputs)


def evaluate_validation(config):
    device = resolve_device(config["device"], torch)
    # This reader only opens Train and Validation arrays, including for XAI reuse.
    arrays, fingerprints = load_training_data(Path(config["processed_dir"]))
    features, labels = arrays["validation"]
    output = Path(config["output_dir"])
    predictions = output / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    results, rows = [], []
    for model_id in config["models"]:
        for seed in config["seeds"]:
            name = f"school-e2-gamma0.5-seed{seed}" if model_id == "e2" else f"school-{model_id}-seed{seed}"
            checkpoint = Path(config["checkpoint_dir"]) / model_id / name / "best.pt"
            model, state = load_frozen_model(checkpoint, fingerprints, device)
            signature = state["signature"]
            if signature["experiment"] != model_id.upper() or signature["seed"] != seed or signature["mode"] != "full":
                raise ValueError(f"Unexpected experiment or seed in {checkpoint}")
            if model_id == "e2" and signature["focal_gamma"] != 0.5:
                raise ValueError("Midterm E2 requires the selected gamma 0.5")
            started = time.perf_counter()
            scores = predict_logits(model, features, config["batch_size"], device)
            labels_checked, scores = validate_scores(labels, scores)
            ap = float(average_precision_score(labels_checked, scores))
            if abs(ap - state["validation_ap"]) > 0.0001:
                raise ValueError(f"Validation AP differs from saved checkpoint: {name}")
            np.savez_compressed(predictions / f"{name}.npz", scores=scores, labels=labels)
            result = {
                "run_name": name, "model": model_id.upper(), "seed": seed,
                "checkpoint_sha256": sha256_file(checkpoint), "checkpoint_epoch": state["epoch"],
                "signature": signature, "ap": ap, "training_validation_ap": state["validation_ap"],
                "ap_difference_from_training": ap - state["validation_ap"],
                "roc_auc": float(roc_auc_score(labels, scores)),
                "inference_seconds": time.perf_counter() - started,
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "operating_points": [],
            }
            for cap in config["fpr_caps"]:
                metrics = select_threshold(labels, scores, cap)
                result["operating_points"].append({"fpr_cap": cap, **metrics})
                rows.append({
                    "model": model_id.upper(), "seed": seed, "fpr_cap": cap,
                    "ap": ap, "roc_auc": result["roc_auc"],
                    **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
                })
            results.append(result)
            print(f"{name}: AP={ap:.6f}, Recall={[(p['fpr_cap'], round(p['recall'], 6)) for p in result['operating_points']]}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "validation_by_seed.csv", index=False)
    metrics = ["ap", "roc_auc", "threshold", "fpr", "precision", "recall", "f1", "tp", "tn", "fp", "fn"]
    summary = table.groupby(["model", "fpr_cap"])[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "validation_summary.csv", index=False)
    report = {
        "split": "validation", "month": 6, "rows": len(labels), "fraud_count": int(labels.sum()),
        "score": "fraud_logit", "decision": "score >= threshold",
        "threshold_selection": "maximize recall, minimize FPR, then maximize threshold; whole score ties",
        "evaluation_environment": {"device": device, "torch": str(torch.__version__),
                                   "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                                   "python": platform.python_version()},
        "config": {k: v for k, v in config.items() if k not in ("processed_dir", "checkpoint_dir", "output_dir")},
        "data": fingerprints, "runs": results,
    }
    write_json(output / "validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen B0–E3 checkpoints on Validation month 6")
    parser.add_argument("--config", type=Path, default=Path("configs/midterm.toml"))
    args = parser.parse_args()
    evaluate_validation(load_analysis_config(args.config))


if __name__ == "__main__":
    main()
