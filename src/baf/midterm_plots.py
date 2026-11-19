"""Presentation figures from frozen training logs and Validation predictions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from baf.evaluation import load_analysis_config


COLORS = {"B0": "#64748b", "E1": "#007c83", "E2": "#b06a12", "E3": "#7251a3"}


def save_figure(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    svg = directory / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def draw_model_comparison(ax, groups, field, label):
    for index, (model, values) in enumerate(groups):
        scores = values[field].to_numpy()
        ax.bar(index, scores.mean(), color=COLORS[model], alpha=0.72, width=0.58)
        ax.errorbar(index, scores.mean(), yerr=scores.std(ddof=1), color="#202020", capsize=5)
        ax.scatter(index + np.linspace(-0.12, 0.12, len(scores)), scores, color="#202020", s=25, zorder=3)
    ax.set_xticks(range(len(groups)), [model for model, _ in groups])
    ax.set_ylabel(label)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)


def create_figures(config):
    output = Path(config["output_dir"])
    figures = output / "figures"
    report = json.loads((output / "validation.json").read_text())
    per_seed = pd.read_csv(output / "validation_by_seed.csv")
    ap = per_seed.drop_duplicates(["model", "seed"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    draw_model_comparison(ax, list(ap.groupby("model")), "ap", "Validation AP")
    ax.set_ylim(0, 0.19)
    ax.set_title("Validation AP across seeds 42, 43, 44")
    save_figure(fig, figures, "validation_ap")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for cap, ax in zip(config["fpr_caps"], axes, strict=True):
        table = per_seed[per_seed["fpr_cap"] == cap]
        draw_model_comparison(ax, list(table.groupby("model")), "recall", "Validation recall")
        ax.set_title(f"Validation FPR <= {cap:.0%}")
        ax.set_ylim(0, 0.60)
    save_figure(fig, figures, "validation_recall")

    histories = []
    for run in report["runs"]:
        checkpoint_name = run["run_name"]
        metrics = json.loads((Path(config["checkpoint_dir"]) / run["model"].lower() /
                              checkpoint_name / "metrics.json").read_text())
        histories.append({"model": run["model"], "seed": run["seed"], **metrics})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for model, ax in zip(COLORS, axes.flat, strict=True):
        for run in (r for r in histories if r["model"] == model):
            epochs = [h["epoch"] for h in run["history"]]
            color = plt.get_cmap("viridis")((run["seed"] - 42) / 2)
            ax.plot(epochs, [h["train_loss"] for h in run["history"]], color=color,
                    label=f"seed {run['seed']} train")
            ax.plot(epochs, [h["validation_loss"] for h in run["history"]], color=color, linestyle="--",
                    label=f"seed {run['seed']} validation BCE")
        ax.set_title(f"{model} losses")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7, ncol=2)
    save_figure(fig, figures, "training_losses")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    draw_model_comparison(axes[0], list(ap[ap["model"].isin(["B0", "E1", "E2"])].groupby("model")),
                          "ap", "Validation AP")
    axes[0].set_ylim(0, 0.19)
    axes[0].set_title("Loss function comparison")
    source = Path(config["checkpoint_dir"]).parent / "results/reference/mlp-loss-comparison/runs.json"
    gamma_runs = [r for r in json.loads(source.read_text())["runs"]
                  if r["environment"] == "school_cuda" and r["purpose"] == "gamma_search"]
    gamma_runs.sort(key=lambda r: r["focal_gamma"])
    axes[1].plot([r["focal_gamma"] for r in gamma_runs],
                 [r["best_validation_ap"] for r in gamma_runs], marker="o", color=COLORS["E2"])
    axes[1].set_xlabel("Focal gamma")
    axes[1].set_ylabel("Best Validation AP")
    axes[1].set_title("Gamma search on seed 42")
    save_figure(fig, figures, "loss_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    draw_model_comparison(axes[0], list(ap[ap["model"].isin(["E1", "E3"])].groupby("model")),
                          "ap", "Validation AP")
    axes[0].set_ylim(0, 0.19)
    axes[0].set_title("E1 MLP vs E3 Residual MLP")
    time_table = pd.DataFrame([{"model": r["model"], "seconds": r["total_epoch_seconds"]}
                               for r in histories if r["model"] in ("E1", "E3")])
    draw_model_comparison(axes[1], list(time_table.groupby("model")), "seconds",
                          "Training + validation time (seconds)")
    axes[1].set_title("Total per run, early stopping")
    save_figure(fig, figures, "architecture_comparison")

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for run in (r for r in report["runs"] if r["seed"] == 42):
        with np.load(output / "predictions" / f"{run['run_name']}.npz") as data:
            precision, recall, _ = precision_recall_curve(data["labels"], data["scores"])
        ax.step(recall, precision, where="post", label=run["model"], color=COLORS[run["model"]])
    ax.axhline(report["fraud_count"] / report["rows"], color="#777777", linestyle=":", label="Fraud prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Validation PR curves, seed 42", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    save_figure(fig, figures, "validation_pr")
    print(f"Saved six PNG/SVG figure pairs to {figures}")


def main():
    parser = argparse.ArgumentParser(description="Create B0–E3 midterm figures from Validation results")
    parser.add_argument("--config", type=Path, default=Path("configs/midterm.toml"))
    args = parser.parse_args()
    create_figures(load_analysis_config(args.config))


if __name__ == "__main__":
    main()
