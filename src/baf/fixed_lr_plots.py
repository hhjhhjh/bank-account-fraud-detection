"""Plot the fixed-learning-rate Validation comparison from saved predictions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def make_figures(directory):
    directory = Path(directory)
    report = json.loads((directory / "validation.json").read_text())
    validation = pd.read_csv(directory / "validation_summary.csv")
    training = pd.read_csv(directory / "training_summary.csv")
    models = training["model"].tolist()
    colors = plt.get_cmap("tab10")(np.arange(len(models)))
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            path = figures / f"{name}.{extension}"
            fig.savefig(path, dpi=160)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for x, (model, color) in enumerate(zip(models, colors, strict=True)):
        row = validation[(validation.model == model) & (validation.fpr_cap == 0.01)].iloc[0]
        ax.errorbar(x, row.ap_mean, yerr=row.ap_std, fmt="o", color=color, capsize=5)
        values = [r["ap"] for r in report["runs"] if r["model"] == model]
        ax.scatter(x + np.linspace(-0.12, 0.12, len(values)), values, color=color, alpha=0.4, s=20)
    ax.set(xticks=range(len(models)), xticklabels=models, ylabel="Validation AP",
           title="Learning rate 0.001: mean and sample SD", ylim=(0.15, 0.19))
    ax.grid(axis="y", alpha=0.25)
    save(fig, "validation_ap")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, cap in zip(axes, (0.01, 0.05), strict=True):
        subset = validation[validation.fpr_cap == cap].set_index("model").loc[models]
        ax.bar(models, subset.recall_mean * 100, yerr=subset.recall_std * 100, color=colors, capsize=4)
        ax.set(title=f"Validation FPR <= {cap:.0%}", ylim=(0, 65))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Recall (%) — mean and sample SD")
    save(fig, "validation_recall")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, color in zip(models, colors, strict=True):
        run = next(r for r in report["runs"] if r["model"] == model and r["seed"] == 42)
        with np.load(directory / "predictions" / f"{run['run_name']}.npz", allow_pickle=False) as data:
            precision, recall, _ = precision_recall_curve(data["labels"], data["scores"])
        ax.step(recall, precision, where="post", color=color, label=f"{model}: AP {run['ap']:.4f}")
    ax.axhline(report["fraud_count"] / report["rows"], color="gray", linestyle=":", label="Fraud prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Validation PR curves — seed 42", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    save(fig, "validation_pr_seed42")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(models, training.seconds_mean, yerr=training.seconds_std, color=colors, capsize=4)
    ax.set(ylabel="Training + validation seconds", title="Time to stopping — mean and sample SD")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "training_time")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("results/reference/fixed-lr-comparison"))
    args = parser.parse_args()
    make_figures(args.directory)


if __name__ == "__main__":
    main()
