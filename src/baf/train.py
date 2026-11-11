import argparse
import json
import math
import re
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from baf.cli import load_config, sha256_file, write_json
from baf.models import BaselineMLP
from baf.runtime import resolve_device
from baf.schema import TARGET, TIME


def load_training_data(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    mapping_path = directory / "feature_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    if not mapping or [row["index"] for row in mapping] != list(range(len(mapping))):
        raise ValueError("feature_mapping.json must list contiguous input indices")
    if len({row["transformed_feature"] for row in mapping}) != len(mapping):
        raise ValueError("feature_mapping.json contains duplicate input names")
    if any(row["original_feature"] in (TARGET, TIME) for row in mapping):
        raise ValueError("Target and month must be excluded from training features")

    arrays = {}
    fingerprints = {mapping_path.name: sha256_file(mapping_path)}
    for split in ("train", "validation"):
        x_path = directory / f"{split}_X.npy"
        y_path = directory / f"{split}_y.npy"
        features = np.load(x_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(y_path, mmap_mode="r", allow_pickle=False)
        rows = manifest["prepared_rows"][split]
        if features.dtype != np.float32 or features.shape != (rows, len(mapping)):
            raise ValueError(f"{x_path.name} must be float32 with shape ({rows}, {len(mapping)})")
        if labels.shape != (rows,) or labels.dtype != np.int64 or not np.array_equal(np.unique(labels), [0, 1]):
            raise ValueError(f"{y_path.name} must contain both classes as int64 labels with shape ({rows},)")
        arrays[split] = (features, labels)
        fingerprints[x_path.name] = sha256_file(x_path)
        fingerprints[y_path.name] = sha256_file(y_path)
    return arrays, {"source_sha256": manifest["source_sha256"], "files_sha256": fingerprints}


def select_rows(labels, limit: int, seed: int):
    rows = np.arange(len(labels))
    if limit < len(rows):
        rows, _ = train_test_split(rows, train_size=limit, random_state=seed, stratify=labels)
        rows.sort()
    if np.unique(labels[rows]).size != 2:
        raise ValueError("Smoke sample needs both classes; increase the sample size")
    return rows


def batches(arrays, rows, batch_size, device):
    features, labels = arrays
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        yield (
            torch.from_numpy(features[batch]).to(device),
            torch.from_numpy(labels[batch]).to(device=device, dtype=torch.float32),
        )


def train_epoch(model, batches_to_train, loss_fn, optimizer, scaler):
    model.train()
    loss_sum = count = 0
    for features, labels in batches_to_train:
        optimizer.zero_grad(set_to_none=True)
        autocast = torch.autocast("cuda", dtype=torch.float16) if scaler.is_enabled() else nullcontext()
        with autocast:
            loss = loss_fn(model(features), labels)
        value = loss.detach().item()
        if not math.isfinite(value):
            raise ValueError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += value * len(labels)
        count += len(labels)
    return loss_sum / count


@torch.inference_mode()
def validate_epoch(model, validation_batches, loss_fn):
    model.eval()
    loss_sum = count = 0
    labels_by_batch, scores = [], []
    for features, labels in validation_batches:
        logits = model(features)
        value = loss_fn(logits, labels).item()
        if not math.isfinite(value):
            raise ValueError("Non-finite validation loss")
        loss_sum += value * len(labels)
        count += len(labels)
        labels_by_batch.append(labels.cpu().numpy())
        scores.append(logits.cpu().numpy())
    ap = average_precision_score(np.concatenate(labels_by_batch), np.concatenate(scores))
    return loss_sum / count, float(ap)


def save_checkpoint(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(payload, handle)
            handle.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def train_b0(config: dict, run_dir: Path, *, smoke: bool = False, resume: Path | None = None) -> dict:
    training = config["training"]
    model_config = config["model"]
    batch_size = training.get("batch_size")
    epochs, patience, seed = (training[name] for name in ("epochs", "patience", "seed"))
    for name, value in [("batch_size", batch_size), ("epochs", epochs), ("patience", patience)]:
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer from 0 through 2**32 - 1")
    hidden_dims = model_config["hidden_dims"]
    if not hidden_dims or any(type(width) is not int or width < 1 for width in hidden_dims):
        raise ValueError("hidden_dims must contain positive integers")
    dropout = model_config["dropout"]
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    lr, weight_decay = training["learning_rate"], training["weight_decay"]
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative; both must be finite")

    device = resolve_device(config["runtime"]["device"], torch)
    precision = training["mixed_precision"]
    if precision not in ("off", "fp16"):
        raise ValueError("mixed_precision must be off or fp16")
    if precision == "fp16" and (device != "cuda" or not torch.amp.autocast_mode.is_autocast_available("cuda")):
        raise ValueError("fp16 requires CUDA with autocast support")

    arrays, fingerprints = load_training_data(Path(config["data"]["processed_dir"]))
    selections = {}
    for split, (_, labels) in arrays.items():
        limit = config["smoke"][f"{split}_rows"] if smoke else len(labels)
        if type(limit) is not int or limit < 2:
            raise ValueError(f"{split}_rows must be an integer of at least 2")
        selections[split] = select_rows(labels, limit, seed)

    model_args = {"input_dim": arrays["train"][0].shape[1], "hidden_dims": hidden_dims, "dropout": dropout}
    signature = {
        "experiment": "B0", "model": model_args, "loss": "BCEWithLogitsLoss", "optimizer": "AdamW",
        "seed": seed, "batch_size": batch_size, "learning_rate": lr, "weight_decay": weight_decay,
        "patience": patience, "mixed_precision": precision, "mode": "smoke" if smoke else "full",
        "rows": {split: len(rows) for split, rows in selections.items()},
        "data": fingerprints,
    }
    torch.manual_seed(seed)
    model = BaselineMLP(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    history, best, stale_epochs = [], None, 0
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state.get("format_version") != 1 or "optimizer_state" not in state:
            raise ValueError("Resume requires a B0 last.pt training checkpoint")
        if state["signature"] != signature:
            raise ValueError("Checkpoint training configuration or data differs from this run")
        history, best, stale_epochs = state["history"], state["best"], state["stale_epochs"]
        if epochs <= state["epoch"]:
            raise ValueError("epochs must exceed the completed checkpoint epoch")
        if stale_epochs >= patience:
            raise ValueError("The checkpoint has already reached early stopping")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])

    run_dir.mkdir(parents=True, exist_ok=False)
    stop_reason = "epoch_limit"
    for epoch in range(len(history) + 1, epochs + 1):
        started = time.perf_counter()
        # Derive shuffle and dropout seeds from the epoch so resume needs no RNG snapshot.
        torch.manual_seed(seed + epoch)
        order = np.random.default_rng(seed + epoch).permutation(selections["train"])
        train_loss = train_epoch(model, batches(arrays["train"], order, batch_size, device), loss_fn, optimizer, scaler)
        val_loss, val_ap = validate_epoch(model, batches(arrays["validation"], selections["validation"], batch_size, device), loss_fn)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
            "validation_ap": val_ap, "seconds": time.perf_counter() - started, "device": device,
        })
        if best is None or val_ap > best["validation_ap"]:
            best = {
                "epoch": epoch, "validation_ap": val_ap,
                "model_state": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        metadata = {"format_version": 1, "signature": signature}
        save_checkpoint(run_dir / "last.pt", {
            **metadata, "epoch": epoch, "history": history, "best": best, "stale_epochs": stale_epochs,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
        })
        save_checkpoint(run_dir / "best.pt", {**metadata, **best})
        print(f"epoch={epoch} train_loss={train_loss:.6f} validation_loss={val_loss:.6f} validation_ap={val_ap:.6f}", flush=True)
        if stale_epochs >= patience:
            stop_reason = "early_stopping"
            break

    result = {
        "mode": signature["mode"], "device": device, "torch_version": str(torch.__version__),
        "signature": signature, "requested_epochs": epochs, "stop_reason": stop_reason,
        "resumed_from": str(resume) if resume is not None else None,
        "best_epoch": best["epoch"], "best_validation_ap": best["validation_ap"],
        "total_epoch_seconds": sum(row["seconds"] for row in history),
        "splits": {
            split: {"rows": len(rows), "fraud_count": int(arrays[split][1][rows].sum())}
            for split, rows in selections.items()
        },
        "history": history,
    }
    write_json(run_dir / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Train the B0 MLP on prepared BAF Train and Validation arrays")
    parser.add_argument("--config", type=Path, default=Path("configs/b0.toml"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int, help="Total epoch limit, including completed epochs when resuming")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path, help="Resume a last.pt checkpoint into a new run directory")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", args.run_name):
        parser.error("run-name must contain only letters, digits, dots, underscores and hyphens")
    try:
        config = load_config(args.config)
        if args.batch_size is not None:
            config["training"]["batch_size"] = args.batch_size
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        if args.device is not None:
            config["runtime"]["device"] = args.device
        run_dir = Path(config["training"]["checkpoint_dir"]) / args.run_name
        result = train_b0(config, run_dir, smoke=args.smoke, resume=args.resume)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Saved B0 metrics and checkpoints to {run_dir}. Best validation AP: {result['best_validation_ap']:.6f}")


if __name__ == "__main__":
    main()
