"""Limited Validation-only learning-rate search with verified checkpoint reuse."""

import argparse
import copy
import hashlib
import json
import math
import tomllib
from pathlib import Path

import pandas as pd
import torch

from baf.cli import load_config, sha256_file, write_json
from baf.evaluation import load_frozen_model
from baf.runtime import resolve_device
from baf.train import train_mlp


def load_search_config(path):
    path = Path(path).resolve()
    config = tomllib.loads(path.read_text())
    root = path.parent.parent
    config["config_sha256"] = sha256_file(path)
    for key in ("processed_dir", "checkpoint_dir", "output_dir", "baseline_output_dir", "environment_file"):
        if key in config:
            config[key] = str(root / config[key])
    for spec in config["models"].values():
        spec["config"] = str(root / spec["config"])
    return config


def prepare_search(config):
    context = {**config, "models": {}}
    if config["selection_seed"] != 42 or config["seeds"] != [42, 43, 44]:
        raise ValueError("This protocol uses selection seed 42 and repeat seeds 42, 43, 44")
    for model_id, spec in config["models"].items():
        if model_id not in ("b0", "e1", "e2", "e3", "e4", "e5"):
            raise ValueError(f"Unknown model: {model_id}")
        rates = spec["learning_rates"]
        if len(rates) != 3 or len(set(rates)) != 3 or 0.001 not in rates:
            raise ValueError("Each model requires three distinct learning rates including 0.001")
        if any(not math.isfinite(r) or r <= 0 for r in rates):
            raise ValueError("Learning rates must be finite and positive")
        template = Path(spec["baseline"])
        if template.is_absolute() or ".." in template.parts:
            raise ValueError("Baseline paths must be relative to checkpoint_dir")
        cfg = load_config(Path(spec["config"]))
        cfg["data"]["processed_dir"] = config["processed_dir"]
        cfg["runtime"]["device"] = config["device"]
        cfg["training"].update(batch_size=config["batch_size"], epochs=config["epochs"], seed=42, learning_rate=0.001)
        for key in ("focal_gamma", "oversampling_ratio"):
            if key in spec:
                cfg["training"][key] = spec[key]
        # Use the training function's actual signature contract without reserving a run
        # directory or requiring a local CUDA device during planning.
        preview = copy.deepcopy(cfg)
        preview["runtime"]["device"] = "cpu"
        signature = train_mlp(preview, Path(config["output_dir"]) / "unused-preview", dry_run=True)["signature"]
        if signature["experiment"] != model_id.upper():
            raise ValueError(f"Base configuration does not implement {model_id}")
        context["models"][model_id] = {**spec, "config": cfg, "signature": signature}
    sources = ["train.py", "models.py", "losses.py", "sampling.py", "ft_transformer.py", "lr_search.py", "evaluation.py", "cli.py"]
    code = {name: sha256_file(Path(__file__).parent / name) for name in sources}
    frozen = {"config_sha256": config["config_sha256"], "code_sha256": code,
              "signatures": {key: spec["signature"] for key, spec in context["models"].items()}}
    context["protocol_sha256"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    context["code_sha256"] = code
    for model_id in context["models"]:
        _read_run(_job(context, model_id, 0.001, 42), config["epochs"])
    return context


def _job(context, model_id, rate, seed):
    spec = context["models"][model_id]
    signature = {**spec["signature"], "learning_rate": rate, "seed": seed}
    if rate == 0.001:
        directory = Path(context["checkpoint_dir"]) / spec["baseline"].format(seed=seed)
    else:
        directory = Path(context["checkpoint_dir"]) / "lr-search" / model_id / f"lr{rate:g}-seed{seed}"
    return {"model": model_id, "learning_rate": rate, "seed": seed,
            "run_dir": str(directory), "signature": signature}


def candidate_jobs(context, models=None):
    chosen = list(context["models"]) if models is None else models
    if not chosen or len(set(chosen)) != len(chosen) or any(m not in context["models"] for m in chosen):
        raise ValueError("Choose distinct configured models")
    return [_job(context, model, rate, 42) for model in chosen for rate in context["models"][model]["learning_rates"]]


def _read_run(job, epochs):
    directory = Path(job["run_dir"])
    required = [directory / name for name in ("metrics.json", "best.pt", "last.pt")]
    if not all(path.is_file() for path in required):
        raise ValueError(f"Incomplete or Missing run: {directory}")
    metrics = json.loads(required[0].read_text())
    if metrics["signature"] != job["signature"] or metrics["requested_epochs"] != epochs or metrics["mode"] != "full":
        raise ValueError(f"Run configuration or data differs: {directory}")
    best = torch.load(required[1], map_location="cpu", weights_only=True)
    last = torch.load(required[2], map_location="cpu", weights_only=True)
    if any(s.get("format_version") != 1 or s["signature"] != job["signature"] for s in (best, last)):
        raise ValueError(f"Checkpoint configuration or data differs: {directory}")
    ap = metrics["best_validation_ap"]
    if (not math.isfinite(ap) or best["validation_ap"] != ap or best["epoch"] != metrics["best_epoch"]
            or last["history"] != metrics["history"] or last["epoch"] != len(metrics["history"])
            or max(h["validation_ap"] for h in metrics["history"]) != ap
            or "optimizer_state" not in last):
        raise ValueError(f"Checkpoint and metrics disagree: {directory}")
    if metrics["stop_reason"] == "epoch_limit" and len(metrics["history"]) != epochs:
        raise ValueError(f"Incomplete run at epoch limit: {directory}")
    try:
        saved_best = last["best"]
        if (saved_best["epoch"] != best["epoch"] or saved_best["validation_ap"] != ap
                or saved_best["model_state"].keys() != best["model_state"].keys()
                or any(not torch.equal(tensor, saved_best["model_state"][name]) for name, tensor in best["model_state"].items())):
            raise ValueError(f"Best checkpoint weights disagree: {directory}")
        model, _ = load_frozen_model(required[1], job["signature"]["data"], "cpu")
        for weights in (best["model_state"], last["model_state"]):
            model.load_state_dict(weights, strict=True)
            if not all(torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
                raise ValueError(f"Non-finite checkpoint weights: {directory}")
    except (KeyError, RuntimeError, TypeError) as error:
        raise ValueError(f"Invalid checkpoint weights: {directory}") from error
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if "trainable_parameters" in metrics and metrics["trainable_parameters"] != parameters:
        raise ValueError(f"Checkpoint parameter count disagrees: {directory}")
    return {**job, "metrics": metrics, "trainable_parameters": parameters,
            "files_sha256": {p.name: sha256_file(p) for p in required}}


def execute_jobs(context, jobs, *, resume_incomplete=False):
    completed, pending = {}, []
    for job in jobs:
        directory = Path(job["run_dir"])
        if (directory / "metrics.json").exists() or job["learning_rate"] == 0.001:
            completed[job["run_dir"]] = _read_run(job, context["epochs"])
        elif directory.exists() and not resume_incomplete:
            raise ValueError(f"Incomplete run: {directory}; inspect it or use --resume-incomplete")
        else:
            pending.append(job)
    if pending:
        resolve_device(context["device"], torch)
    for job in pending:
        directory = Path(job["run_dir"])
        resume = None
        if directory.exists():
            checkpoint = directory / "last.pt"
            if not checkpoint.is_file():
                raise ValueError(f"Incomplete run has no last.pt: {directory}")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (state.get("format_version") != 1 or state["signature"] != job["signature"]
                    or state["epoch"] >= context["epochs"] or state["stale_epochs"] >= job["signature"]["patience"]):
                raise ValueError(f"Incomplete checkpoint cannot be resumed under this protocol: {directory}")
            index = 1
            while directory.with_name(directory.name + f".interrupted-{index}").exists():
                index += 1
            archive = directory.with_name(directory.name + f".interrupted-{index}")
            directory.rename(archive)
            resume = archive / "last.pt"
        cfg = copy.deepcopy(context["models"][job["model"]]["config"])
        cfg["training"].update(learning_rate=job["learning_rate"], seed=job["seed"])
        print(f"TRAIN {job['model']} lr={job['learning_rate']:g} seed={job['seed']}", flush=True)
        train_mlp(cfg, directory, resume=resume)
        completed[job["run_dir"]] = _read_run(job, context["epochs"])
    return [completed[job["run_dir"]] for job in jobs]


def choose_learning_rate(scores):
    if not scores or any(not math.isfinite(ap) for ap in scores.values()):
        raise ValueError("Candidate AP values must be finite")
    best = max(scores.values())
    tied = [rate for rate, score in scores.items() if score == best]
    return 0.001 if 0.001 in tied else min(tied)


def select_learning_rates(context):
    records = [_read_run(job, context["epochs"]) for job in candidate_jobs(context)]
    selected = {}
    for model in context["models"]:
        scores = {r["learning_rate"]: r["metrics"]["best_validation_ap"] for r in records if r["model"] == model}
        selected[model] = choose_learning_rate(scores)
    result = {
        "protocol_sha256": context["protocol_sha256"], "selection_seed": 42,
        "criterion": "maximum unrounded Validation AP; exact ties prefer 0.001, then lower learning rate",
        "selected": selected,
        "candidates": [
            {**{k: r[k] for k in ("model", "learning_rate", "seed", "files_sha256")},
             "run_dir": str(Path(r["run_dir"]).relative_to(context["checkpoint_dir"]))}
            for r in records
        ],
        "limitation": "Seed 42 is used for selection and is included in the three-seed report; Validation scores may be optimistic.",
    }
    output = Path(context["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "selection.json", result)
    return result


def repetition_jobs(context, models=None):
    selection = json.loads((Path(context["output_dir"]) / "selection.json").read_text())
    if selection["protocol_sha256"] != context["protocol_sha256"]:
        raise ValueError("Selection protocol changed; do not reuse the selection")
    jobs = candidate_jobs(context)
    expected = {str(Path(j["run_dir"]).relative_to(context["checkpoint_dir"])) for j in jobs}
    if len(selection["candidates"]) != len(jobs) or {r["run_dir"] for r in selection["candidates"]} != expected:
        raise ValueError("Selection candidate list differs")
    for record in selection["candidates"]:
        for name, digest in record["files_sha256"].items():
            if sha256_file(Path(context["checkpoint_dir"]) / record["run_dir"] / name) != digest:
                raise ValueError("Selection candidate record changed")
    records = [_read_run(j, context["epochs"]) for j in jobs]
    winners = {
        model: choose_learning_rate({r["learning_rate"]: r["metrics"]["best_validation_ap"] for r in records if r["model"] == model})
        for model in context["models"]
    }
    if selection["selected"] != winners:
        raise ValueError("Stored selected learning rates do not match the candidate winners")
    chosen = list(context["models"]) if models is None else models
    if not chosen or any(m not in context["models"] for m in chosen):
        raise ValueError("Choose configured models")
    return [_job(context, m, selection["selected"][m], seed) for m in chosen for seed in context["seeds"]]


def write_run_tables(records, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        m = record["metrics"]
        work = m.get("training_work", {})
        rows.append({
            "model": record["model"].upper(), "seed": record["seed"], "learning_rate": record["learning_rate"],
            "validation_ap": m["best_validation_ap"], "best_epoch": m["best_epoch"],
            "completed_epochs": len(m["history"]), "stop_reason": m["stop_reason"],
            "best_at_epoch_limit": m["best_epoch"] == m["requested_epochs"],
            "seconds": m["total_epoch_seconds"], "trainable_parameters": record["trainable_parameters"],
            "rows_per_epoch": work.get("rows_per_epoch", m["splits"]["train"]["rows"]),
            "device": m["device"], "torch_version": m["torch_version"], "run_dir": record["run_dir"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(output / "training_by_seed.csv", index=False)
    summary = table.groupby(["model", "learning_rate"])[["validation_ap", "seconds", "completed_epochs", "trainable_parameters", "rows_per_epoch"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(output / "training_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "select", "repeat", "baseline"))
    parser.add_argument("--config", type=Path, default=Path("configs/learning-rate-search.toml"))
    parser.add_argument("--models", nargs="+", help="Limit plan/run/repeat to configured models")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"))
    parser.add_argument("--resume-incomplete", action="store_true", help="Archive incomplete runs and resume their last.pt")
    args = parser.parse_args()
    try:
        if args.resume_incomplete and args.action not in ("run", "repeat"):
            raise ValueError("--resume-incomplete applies only to run or repeat")
        cfg = load_search_config(args.config)
        if args.device is not None:
            cfg["device"] = args.device
        context = prepare_search(cfg)
        if args.action == "select":
            if args.models:
                raise ValueError("Selection requires all configured models")
            print(json.dumps(select_learning_rates(context), indent=2))
            return
        if args.action == "repeat":
            jobs = repetition_jobs(context, args.models)
        elif args.action == "baseline":
            if args.models:
                raise ValueError("Baseline recording requires all configured models")
            jobs = [_job(context, m, 0.001, seed) for m in context["models"] for seed in context["seeds"]]
        else:
            jobs = candidate_jobs(context, args.models)
        if args.action == "plan":
            preview = []
            for job in jobs:
                exists = Path(job["run_dir"]).exists()
                if exists:
                    _read_run(job, context["epochs"])
                preview.append({k: job[k] for k in ("model", "learning_rate", "seed", "run_dir") } | {"action": "reuse" if exists else "train"})
            print(json.dumps({"protocol_sha256": context["protocol_sha256"], "jobs": preview,
                              "new_training_runs": sum(j["action"] == "train" for j in preview)}, indent=2))
            return
        if args.action == "baseline":
            records = [_read_run(j, context["epochs"]) for j in jobs]
            output = Path(context["baseline_output_dir"])
        else:
            records = execute_jobs(context, jobs, resume_incomplete=args.resume_incomplete)
            output = Path(context["output_dir"]) / ("candidates" if args.action == "run" else "repeated")
            # Include earlier completed models when execution is split with --models.
            all_jobs = candidate_jobs(context) if args.action == "run" else repetition_jobs(context)
            records = [_read_run(j, context["epochs"]) for j in all_jobs if (Path(j["run_dir"]) / "metrics.json").is_file()]
        write_run_tables(records, output)
        code_label = "verification_code_sha256" if args.action == "baseline" else "protocol_code_sha256"
        payload = {"protocol_sha256": context["protocol_sha256"], code_label: context["code_sha256"], "runs": records}
        if args.action == "baseline" and "environment_file" in context:
            payload["school_environment"] = json.loads(Path(context["environment_file"]).read_text())
        write_json(output / "training_runs.json", payload)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
