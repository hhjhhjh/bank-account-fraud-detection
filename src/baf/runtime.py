import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def resolve_device(requested="auto", torch_module=None) -> str:
    if requested not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError(f"Invalid device: {requested}")
    if torch_module is None:
        import torch as torch_module
    available = {"cuda": torch_module.cuda.is_available(), "mps": torch_module.backends.mps.is_available(), "cpu": True}
    if requested == "auto":
        return next(name for name in ("cuda", "mps", "cpu") if available[name])
    if not available[requested]:
        raise ValueError(f"Requested device {requested} is unavailable")
    return requested


def environment_report(device="auto") -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "matplotlib", "joblib", "torch", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "current_host", "os": platform.platform(), "python": sys.version, "architecture": platform.machine(), "packages": packages, "disk_free_bytes": shutil.disk_usage(Path.cwd()).free, "requested_device": device, "training_executed": False}
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        if device != "auto":
            raise ValueError("PyTorch is required to verify the requested device; install the appropriate PyTorch build")
        report.update(selected_device=None, torch_status="not_installed")
        return report
    report.update(selected_device=resolve_device(device, torch), cuda_available=torch.cuda.is_available(), mps_available=torch.backends.mps.is_available(), pytorch_cuda_version=torch.version.cuda)
    if torch.cuda.is_available():
        report["gpus"] = [{"index": i, "name": torch.cuda.get_device_properties(i).name, "vram_bytes": torch.cuda.get_device_properties(i).total_memory, "capability": list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())]
        report["cuda_bf16_supported"] = torch.cuda.is_bf16_supported()
        report["cuda_autocast_available"] = torch.amp.autocast_mode.is_autocast_available("cuda")
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
            report["nvidia_smi"] = {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            report["nvidia_smi"] = {"error": str(error)}
    return report


def main():
    parser = argparse.ArgumentParser(description="Inspect this host only; never connects over SSH or starts training")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = environment_report(args.device)
    except ValueError as error:
        parser.exit(2, f"error: {error}\n")
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
