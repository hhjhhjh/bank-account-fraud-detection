"""Initial Integrated Gradients analysis on Validation only."""

import argparse
import json
import warnings
from pathlib import Path

import captum
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from baf.cli import sha256_file, write_json
from baf.evaluation import load_analysis_config, load_frozen_model, predict_logits
from baf.midterm_plots import save_figure
from baf.runtime import resolve_device
from baf.train import load_training_data


def group_attributions(attributions, mapping):
    names = list(dict.fromkeys(row["original_feature"] for row in mapping))
    grouped = np.zeros((len(attributions), len(names)), dtype=np.float64)
    for index, name in enumerate(names):
        columns = [i for i, row in enumerate(mapping) if row["original_feature"] == name]
        grouped[:, index] = attributions[:, columns].sum(axis=1)
    return grouped, names


def select_local_cases(labels, scores, threshold):
    predicted = scores >= threshold
    masks = {"TP": (labels == 1) & predicted, "FN": (labels == 1) & ~predicted,
             "FP": (labels == 0) & predicted}
    selected = {}
    for outcome, mask in masks.items():
        rows = np.flatnonzero(mask)
        # Select a typical score within each outcome, with source order breaking ties.
        selected[outcome] = int(rows[np.argmin(np.abs(scores[rows] - np.median(scores[rows])))]) if len(rows) else None
    return selected


def explain_inputs(model, inputs, baseline, config):
    if not torch.isfinite(inputs).all() or not torch.isfinite(baseline).all():
        raise ValueError("IG inputs and baseline must be finite")
    initial, maximum = config["initial_steps"], config["max_steps"]
    if initial < 2 or maximum < initial or config["internal_batch_size"] < len(inputs):
        raise ValueError("Invalid IG step count or internal batch size")
    ig = IntegratedGradients(model)
    attrs = np.zeros(tuple(inputs.shape), dtype=np.float64)
    steps = np.zeros(len(inputs), dtype=np.int64)
    residual = np.full(len(inputs), np.nan)
    with torch.no_grad():
        difference = (model(inputs) - model(baseline)).cpu().numpy()
    pending = np.arange(len(inputs))
    n_steps = initial
    while len(pending):
        attr, delta = ig.attribute(
            inputs[pending], baselines=baseline, n_steps=n_steps, method="gausslegendre",
            internal_batch_size=config["internal_batch_size"], return_convergence_delta=True,
        )
        if not torch.isfinite(attr).all() or not torch.isfinite(delta).all():
            raise ValueError("Non-finite Integrated Gradients result")
        attrs[pending] = attr.detach().cpu().numpy()
        residual[pending] = attrs[pending].sum(axis=1) - difference[pending]
        steps[pending] = n_steps
        tolerance = config["completeness_atol"] + config["completeness_rtol"] * np.abs(difference[pending])
        pending = pending[np.abs(residual[pending]) > tolerance]
        if n_steps == maximum:
            break
        n_steps = min(n_steps * 2, maximum)
    if len(pending):
        warnings.warn(f"{len(pending)} IG samples exceed completeness tolerance at {maximum} steps", RuntimeWarning)
    return attrs, residual, steps


def run_xai(config):
    settings = config["xai"]
    if settings["model"] != "e1" or settings["seed"] != 42:
        raise ValueError("Initial midterm XAI uses E1 seed 42")
    device = resolve_device(config["device"], torch)
    arrays, fingerprints = load_training_data(Path(config["processed_dir"]))
    features, labels = arrays["validation"]
    output = Path(config["output_dir"])
    evaluation = json.loads((output / "validation.json").read_text())
    if evaluation["data"] != fingerprints or evaluation["split"] != "validation":
        raise ValueError("Validation evaluation dataset mismatch")
    run_name = "school-e1-seed42"
    evaluation_run = next(row for row in evaluation["runs"] if row["run_name"] == run_name)
    checkpoint = Path(config["checkpoint_dir"]) / "e1" / run_name / "best.pt"
    if sha256_file(checkpoint) != evaluation_run["checkpoint_sha256"]:
        raise ValueError("XAI checkpoint differs from evaluated checkpoint")
    model, state = load_frozen_model(checkpoint, fingerprints, device)
    with np.load(output / "predictions" / f"{run_name}.npz") as data:
        if not np.array_equal(data["labels"], labels):
            raise ValueError("Prediction labels mismatch")
        scores = data["scores"].copy()
    point = next(row for row in evaluation_run["operating_points"] if row["fpr_cap"] == settings["local_fpr_cap"])
    cases = select_local_cases(labels, scores, point["threshold"])
    global_rows, _ = train_test_split(
        np.arange(len(labels)), train_size=settings["global_sample_rows"],
        stratify=labels, random_state=settings["sampling_seed"],
    )
    global_rows.sort()
    local_rows = [row for row in cases.values() if row is not None]
    rows = np.unique(np.r_[global_rows, local_rows])

    # Accumulate the mean in float64 without materializing the full Train array.
    train = arrays["train"][0]
    total = np.zeros(train.shape[1], dtype=np.float64)
    for start in range(0, len(train), 16384):
        total += train[start:start + 16384].sum(axis=0, dtype=np.float64)
    baseline_values = (total / len(train)).astype(np.float32)
    baseline = torch.tensor(baseline_values[None, :], device=device)
    inputs = torch.tensor(np.array(features[rows]), device=device)
    checked_scores = predict_logits(model, np.array(features[rows]), config["batch_size"], device)
    if not np.allclose(checked_scores, scores[rows], atol=1e-5, rtol=1e-5):
        raise ValueError("Cached Validation scores differ from the evaluated checkpoint")
    attrs, residual, steps = explain_inputs(model, inputs, baseline, settings)
    mapping = json.loads((Path(config["processed_dir"]) / "feature_mapping.json").read_text())
    grouped, names = group_attributions(attrs, mapping)
    positions = np.searchsorted(rows, global_rows)
    importance = np.abs(grouped[positions]).mean(axis=0)
    global_table = pd.DataFrame({"feature": names, "mean_abs_grouped_logit_attribution": importance})
    xai_dir = output / "xai"
    xai_dir.mkdir(exist_ok=True)
    global_table.sort_values(global_table.columns[1], ascending=False).to_csv(xai_dir / "global_importance.csv", index=False)
    transformed = pd.DataFrame({
        "feature": [row["transformed_feature"] for row in mapping],
        "original_feature": [row["original_feature"] for row in mapping],
        "mean_absolute_attribution": np.abs(attrs[positions]).mean(axis=0),
    })
    transformed.to_csv(xai_dir / "transformed_importance.csv", index=False)
    np.savez_compressed(xai_dir / "attributions.npz", rows=rows, global_rows=global_rows,
                        attributions=attrs, baseline=baseline_values, completeness_residual=residual, steps=steps)
    fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    selected = np.argsort(importance)[-12:]
    ax.barh(np.array(names)[selected], importance[selected], color="#007c83")
    ax.set(xlabel="Mean absolute grouped attribution (logit)",
           title=f"E1 Integrated Gradients: {len(global_rows)} Validation samples")
    save_figure(fig, output / "figures", "ig_global")
    baseline_logit = float(predict_logits(model, baseline_values[None, :], 1, device)[0])
    source_ids = np.load(Path(config["processed_dir"]) / "validation_row_ids.npy", mmap_mode="r")
    case_records = {}
    for outcome, row in cases.items():
        if row is None:
            case_records[outcome] = None
            continue
        position = int(np.searchsorted(rows, row))
        values = grouped[position]
        top = np.argsort(np.abs(values))[-12:]
        fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
        ax.barh(np.array(names)[top], values[top],
                color=["#b44335" if value > 0 else "#2e6a9e" for value in values[top]])
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set(xlabel="Contribution to fraud logit (relative to Train mean)",
               title=f"E1 {outcome}, Validation row {row}, threshold FPR <= 5%")
        save_figure(fig, output / "figures", f"ig_local_{outcome.lower()}")
        case_records[outcome] = {
            "validation_row": row, "source_row_id": int(source_ids[row]), "label": int(labels[row]),
            "logit": float(scores[row]), "baseline_logit": baseline_logit,
            "grouped_attribution_sum": float(values.sum()), "completeness_residual": float(residual[position]),
            "steps": int(steps[position]), "attributions": dict(zip(names, values.tolist(), strict=True)),
        }
    logits = predict_logits(model, np.array(features[rows]), config["batch_size"], device)
    tolerance = settings["completeness_atol"] + settings["completeness_rtol"] * np.abs(logits - baseline_logit)
    report = {
        "split": "validation", "run_name": run_name, "checkpoint_sha256": sha256_file(checkpoint),
        "output_explained": "fraud_logit", "baseline": "mean of transformed Train features",
        "baseline_values": baseline_values.tolist(), "baseline_logit": baseline_logit,
        "device": device, "captum_version": captum.__version__, "torch_version": str(torch.__version__),
        "settings": settings, "method": "gausslegendre",
        "sample_selection": "stratified random sample of Validation; fixed sampling_seed",
        "global_sample_count": len(global_rows), "global_fraud_count": int(labels[global_rows].sum()),
        "global_grouping": "sum signed columns for each original feature, then mean absolute over samples",
        "local_selection": "closest to median logit within each outcome; earliest row breaks ties",
        "local_operating_point": point, "explained_sample_count": len(rows),
        "completeness": {"max_abs_residual": float(np.abs(residual).max()),
                         "mean_abs_residual": float(np.abs(residual).mean()),
                         "tolerance_atol": settings["completeness_atol"],
                         "tolerance_rtol": settings["completeness_rtol"],
                         "samples_exceeding_tolerance": int(np.count_nonzero(np.abs(residual) > tolerance)),
                         "max_steps_used": int(steps.max())},
        "cases": case_records,
    }
    write_json(xai_dir / "summary.json", report)
    print(f"IG completed for {len(rows)} Validation rows. Completeness: {report['completeness']}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Explain E1 seed 42 fraud logits on Validation with Captum")
    parser.add_argument("--config", type=Path, default=Path("configs/midterm.toml"))
    args = parser.parse_args()
    run_xai(load_analysis_config(args.config))


if __name__ == "__main__":
    main()
