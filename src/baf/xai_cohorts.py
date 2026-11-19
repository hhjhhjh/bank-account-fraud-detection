"""Fraud-only TP/FN Integrated Gradients comparison on Validation."""

import argparse
import json
from pathlib import Path

import captum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from baf.cli import sha256_file, write_json
from baf.evaluation import load_analysis_config, load_frozen_model, predict_logits
from baf.midterm_plots import save_figure
from baf.runtime import resolve_device
from baf.train import load_training_data
from baf.xai import explain_inputs, group_attributions, train_mean_baseline


def sample_fraud_cohorts(labels, scores, threshold, count, seed):
    if type(count) is not int or count < 2:
        raise ValueError("per_group_rows must be an integer of at least 2")
    if labels.shape != scores.shape or labels.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Labels and finite scores must have matching one-dimensional shapes")
    predicted = scores >= threshold
    rng = np.random.default_rng(seed)
    groups = {}
    for name, prediction in (("TP", True), ("FN", False)):
        eligible = np.flatnonzero((labels == 1) & (predicted == prediction))
        if len(eligible) < count:
            raise ValueError(f"{name} has only {len(eligible)} rows; requested {count}")
        groups[name] = rng.choice(eligible, size=count, replace=False)
    return groups


def importance_agreement(first, second):
    a, b = np.abs(first).mean(axis=0), np.abs(second).mean(axis=0)
    ranks_a, ranks_b = pd.Series(a).rank(), pd.Series(b).rank()
    correlation = float(ranks_a.corr(ranks_b)) if ranks_a.std() and ranks_b.std() else None
    k = min(10, first.shape[1])
    top_a = np.argsort(-a, kind="stable")[:k]
    top_b = np.argsort(-b, kind="stable")[:k]
    return {
        "spearman_rank_correlation": correlation, "top_k": k,
        "top_k_overlap": int(len(np.intersect1d(top_a, top_b))),
        "top_k_signed_mean_agreement": float(
            np.mean(np.sign(first.mean(axis=0)[top_b]) == np.sign(second.mean(axis=0)[top_b]))
        ),
    }


def summarize_cohorts(groups, names, subset_size):
    rows, stability = [], {}
    for group, values in groups.items():
        if type(subset_size) is not int or subset_size < 2 or 2 * subset_size != len(values):
            raise ValueError("subset_rows must be half of per_group_rows and at least 2")
        if not np.isfinite(values).all():
            raise ValueError("Attributions must be finite")
        for sample in (values[:subset_size], values):
            for column, name in enumerate(names):
                signed = sample[:, column]
                rows.append({
                    "group": group, "sample_size": len(sample), "feature": name,
                    "mean_signed_attribution": float(signed.mean()),
                    "std_signed_attribution": float(signed.std(ddof=1)),
                    "mean_absolute_attribution": float(np.abs(signed).mean()),
                    "std_absolute_attribution": float(np.abs(signed).std(ddof=1)),
                    "positive_fraction": float(np.mean(signed > 0)),
                    "negative_fraction": float(np.mean(signed < 0)),
                })
        stability[group] = {
            "subset_vs_full": {
                **importance_agreement(values[:subset_size], values),
                "sample_sizes": [subset_size, len(values)], "overlapping_rows": subset_size,
            },
            "split_half": {
                **importance_agreement(values[:subset_size], values[subset_size:]),
                "sample_sizes": [subset_size, subset_size], "overlapping_rows": 0,
            },
        }
    return pd.DataFrame(rows), stability


def cohort_figures(groups, names, output, stability, fpr_cap):
    colors = {"TP": "#007c83", "FN": "#b06a12"}
    relevance = sum(np.abs(values).mean(axis=0) for values in groups.values())
    for kind in ("absolute", "signed"):
        ranking = relevance if kind == "absolute" else np.abs(groups["TP"].mean(axis=0) - groups["FN"].mean(axis=0))
        top = np.argsort(ranking)[-12:]
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        positions = np.arange(len(top))
        for offset, (group, values) in zip((-0.18, 0.18), groups.items(), strict=True):
            estimates = np.abs(values).mean(axis=0) if kind == "absolute" else values.mean(axis=0)
            ax.barh(positions + offset, estimates[top], height=0.34, color=colors[group],
                    label=f"{group}, n={len(values)}")
        ax.set_yticks(positions, np.array(names)[top])
        ax.set_xlabel(f"Mean {'absolute ' if kind == 'absolute' else 'signed '}grouped logit attribution")
        title = f"Fraud-only Validation cohorts, E1 seed 42, FPR <= {fpr_cap:.0%}"
        if kind == "signed":
            title += "\nLargest differences between cohort mean attributions"
        ax.set_title(title)
        ax.axvline(0, color="#333333", linewidth=0.6)
        ax.legend()
        save_figure(fig, output, f"ig_tp_fn_{kind}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, (group, values) in zip(axes, groups.items(), strict=True):
        first = np.abs(values[:len(values) // 2]).mean(axis=0)
        full = np.abs(values).mean(axis=0)
        maximum = max(first.max(), full.max()) * 1.05
        ax.scatter(first, full, color=colors[group])
        ax.plot([0, maximum], [0, maximum], color="#777777", linestyle="--")
        rho = stability[group]["subset_vs_full"]["spearman_rank_correlation"]
        ax.set(xlabel=f"Mean absolute IG, first {len(values) // 2} samples",
               ylabel=f"Mean absolute IG, all {len(values)} samples",
               title=f"{group}, rank correlation={rho:.3f}" if rho is not None else group,
               xlim=(0, maximum), ylim=(0, maximum))
    save_figure(fig, output, "ig_tp_fn_stability")


def run_cohort_xai(config):
    settings = config["xai"]
    if settings["model"] != "e1" or settings["seed"] != 42:
        raise ValueError("Fraud comparison requires the evaluated E1 seed 42 checkpoint")
    cohort_settings = settings["cohorts"]
    count, subset = cohort_settings["per_group_rows"], cohort_settings["subset_rows"]
    if type(subset) is not int or subset < 2 or 2 * subset != count:
        raise ValueError("subset_rows must be half of per_group_rows and at least 2")
    device = resolve_device(config["device"], torch)
    arrays, fingerprints = load_training_data(Path(config["processed_dir"]))
    features, labels = arrays["validation"]
    root = Path(config["output_dir"])
    evaluation = json.loads((root / "validation.json").read_text())
    previous = json.loads((root / "xai/summary.json").read_text())
    if evaluation["split"] != "validation" or previous["split"] != "validation" or evaluation["data"] != fingerprints:
        raise ValueError("Validation analysis dataset mismatch")
    run = next(r for r in evaluation["runs"] if r["run_name"] == "school-e1-seed42")
    checkpoint = Path(config["checkpoint_dir"]) / "e1/school-e1-seed42/best.pt"
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != run["checkpoint_sha256"] or checkpoint_hash != previous["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from existing Validation and population XAI analyses")
    model, state = load_frozen_model(checkpoint, fingerprints, device)
    point = next(p for p in run["operating_points"] if p["fpr_cap"] == settings["local_fpr_cap"])
    scores = predict_logits(model, features, config["batch_size"], device)
    cohorts = sample_fraud_cohorts(labels, scores, point["threshold"], count, settings["sampling_seed"])
    rows = np.unique(np.concatenate(list(cohorts.values())))
    baseline_values = np.asarray(previous["baseline_values"], dtype=np.float32)
    train_mean = train_mean_baseline(arrays["train"][0])
    if not np.array_equal(train_mean, baseline_values):
        raise ValueError("Baseline differs from the mean of the verified Train data")
    baseline = torch.tensor(baseline_values[None, :], device=device)
    inputs = torch.tensor(np.array(features[rows]), device=device)
    attrs, residual, steps = explain_inputs(model, inputs, baseline, settings)
    mapping = json.loads((Path(config["processed_dir"]) / "feature_mapping.json").read_text())
    grouped, names = group_attributions(attrs, mapping)
    group_values = {group: grouped[np.searchsorted(rows, indices)] for group, indices in cohorts.items()}
    table, stability = summarize_cohorts(group_values, names, subset)
    output = root / "xai/cohorts"
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "feature_summary.csv", index=False)
    source_ids = np.load(Path(config["processed_dir"]) / "validation_row_ids.npy", mmap_mode="r")
    selection = []
    for group, indices in cohorts.items():
        for order, row in enumerate(indices):
            selection.append({"group": group, "sample_order": order, "in_first_subset": order < subset,
                              "validation_row": int(row), "source_row_id": int(source_ids[row]),
                              "label": int(labels[row]), "logit": float(scores[row])})
    pd.DataFrame(selection).to_csv(output / "sample_selection.csv", index=False)
    np.savez_compressed(output / "attributions.npz", rows=rows, attributions=attrs,
                        tp_rows=cohorts["TP"], fn_rows=cohorts["FN"], steps=steps,
                        completeness_residual=residual, baseline=baseline_values)
    baseline_logit = float(predict_logits(model, baseline_values[None, :], 1, device)[0])
    tolerance = settings["completeness_atol"] + settings["completeness_rtol"] * np.abs(scores[rows] - baseline_logit)
    report = {
        "split": "validation", "run_name": run["run_name"], "checkpoint_sha256": checkpoint_hash,
        "population_summary_sha256": sha256_file(root / "xai/summary.json"),
        "output_explained": "fraud_logit", "baseline": "same transformed Train mean as population analysis",
        "device": device, "operating_point": point, "sampling_seed": settings["sampling_seed"],
        "torch_version": str(torch.__version__), "captum_version": captum.__version__,
        "grouping": "sum signed columns per original feature before computing cohort statistics",
        "sampling": "uniform without replacement within TP and FN; random draw order preserved",
        "cohorts": {name: {"available_rows": point[name.lower()], "sampled_rows": len(indices)}
                    for name, indices in cohorts.items()},
        "subset_rows": subset, "explained_rows": len(rows),
        "settings": settings, "stability": stability,
        "completeness": {"max_abs_residual": float(np.abs(residual).max()),
                         "mean_abs_residual": float(np.abs(residual).mean()),
                         "samples_exceeding_tolerance": int(np.count_nonzero(np.abs(residual) > tolerance)),
                         "max_steps_used": int(steps.max())},
    }
    write_json(output / "summary.json", report)
    cohort_figures(group_values, names, root / "figures", stability, point["fpr_cap"])
    print(json.dumps({"cohorts": report["cohorts"], "stability": stability,
                      "completeness": report["completeness"]}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare TP/FN fraud-only Validation IG cohorts")
    parser.add_argument("--config", type=Path, default=Path("configs/midterm.toml"))
    args = parser.parse_args()
    run_cohort_xai(load_analysis_config(args.config))


if __name__ == "__main__":
    main()
