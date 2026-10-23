"""Read-only environment checks and server-local GPU isolation."""

from __future__ import annotations

from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from device_policy import validate_gpu_ids

PACKAGES = {"torch": "2.1.2+cu121", "torchvision": "0.16.2+cu121", "torchaudio": "2.1.2+cu121",
            "tensorflow": "2.11.1", "tensorflow-probability": "0.19.0", "numpy": "1.26.4",
            "scipy": "1.15.3", "pandas": "2.3.3", "scikit-learn": "1.3.2", "scikit-survival": "0.22.2",
            "lifelines": "0.29.0", "statsmodels": "0.14.0", "bnnsurv": "0.1.3",
            "pycox": "0.2.3", "torchtuples": "0.2.2", "rpy2": "3.5.1"}
METRICS = ("MAEFPTRef", "MAETrue", "MAEHinge", "MAEMargin", "MAEPseudo", "CIndexIPCW", "IBS", "DCalibPValue", "DCalib")


def environment_report():
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    mismatches = {name: {"expected": expected, "actual": versions[name]}
                  for name, expected in PACKAGES.items() if versions[name] != expected}
    if platform.python_version() != "3.10.20":
        mismatches["python"] = {"expected": "3.10.20", "actual": platform.python_version()}
    r = subprocess.check_output([str(Path(sys.prefix) / "bin/R"), "--version"], text=True)
    if not r.startswith("R version 4.5.3 "):
        mismatches["R"] = r.splitlines()[0]
    if mismatches:
        raise RuntimeError(f"surv environment differs from the paper; no packages were changed: {mismatches}")
    return {"executable": sys.executable, "prefix": sys.prefix, "python": platform.python_version(),
            "platform": platform.platform(), "R": r.splitlines()[0], "packages": versions,
            "installed_distributions": sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())}


def worker_environment(devices, threads):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(device["uuid"] for device in devices)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PYTHONHASHSEED"] = "0"
    env["R_HOME"] = str(Path(sys.prefix) / "lib/R")
    libraries = [Path(sys.prefix) / "lib/python3.10/site-packages/torch/lib", Path(sys.prefix) / "lib"]
    env["LD_LIBRARY_PATH"] = ":".join(map(str, libraries)) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["LC_ALL"] = "C"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(threads)
    return env


def validate_worker_devices(devices):
    try:
        if devices:
            validate_gpu_ids([device["physical_index"] for device in devices])
    except ValueError as exc:
        raise RuntimeError("worker received an unauthorized physical device") from exc
    if any(not device["uuid"].startswith("GPU-") for device in devices):
        raise RuntimeError("worker received an unauthorized physical device")
    expected = ",".join(device["uuid"] for device in devices)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected:
        raise RuntimeError("CUDA mask differs from the authorized worker assignment")


def authorized_idle_gpus(indices):
    """Allow the utilization sampling window to clear, never bypass it."""
    from fpt_detection.generation import authorized_gpus

    for attempt in range(16):
        try:
            return authorized_gpus(indices)
        except RuntimeError as exc:
            if not str(exc).startswith("authorized GPU is busy; no process will be stopped:") or attempt == 15:
                raise
            time.sleep(2)


def source_hashes(root):
    files = set()
    for directory in ("src/paper", "src/otsurv", "src/tools/Evaluations", "dsm", "fpt_detection"):
        files.update((root / directory).rglob("*.py"))
    for name in ("device_policy.py", "src/config.py", "src/run_cross_validation.py", "src/run_paper_cross_validation.py",
                 "src/tools/probability_metrics.py", "src/paper/reference.json",
                 "src/tools/evaluator.py", "src/tools/regressors.py",
                 "src/utility/bearing_splits.py", "src/utility/mtlr.py", "src/utility/survival.py", "src/utility/loss.py"):
        files.add(root / name)
    return {path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest() for path in sorted(files)}


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def json_safe(value):
    import math
    import numpy as np
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return str(value)
    return value
