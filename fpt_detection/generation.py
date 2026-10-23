"""Replay the frozen FPT source and enforce the shared GPU policy."""

from __future__ import annotations

from pathlib import Path
import subprocess

from device_policy import validate_gpu_ids
from src import config as cfg
from fpt_detection.data_loader import XJTUDataLoader
from fpt_detection.labels import (
    FPTLabelContractError,
    FPTReference,
    LABEL_COLUMNS,
    REFERENCE_LABEL_SHA256,
    compare_rows,
    detect_task,
    file_digest,
    fixed_tasks,
    json_digest,
    label_row,
    validate_raw_identity,
    write_artifact,
)


def raw_inventory(data_dir: Path, reference: FPTReference) -> dict:
    """Return and validate the canonical identity of the raw XJTU-SY files."""
    loader = XJTUDataLoader(data_dir)
    records = []
    for condition in cfg.CONDITIONS:
        for bearing in cfg.BEARING_IDS:
            _, paths = loader.raw_record_paths(condition, bearing)
            entry = reference.entries[f"fold1_c{condition}_b{bearing}"]
            if len(paths) != entry["n_samples"]:
                raise FPTLabelContractError(f"raw/reference length mismatch: C{condition + 1} B{bearing}")
            for path in paths:
                canonical = Path(cfg.CONDITION_NAMES[condition]) / f"Bearing{condition + 1}_{bearing}" / path.name
                records.append({"path": canonical.as_posix(),
                                "bytes": path.stat().st_size, "sha256": file_digest(path)})
    inventory = {"file_count": len(records), "sha256": json_digest(records)}
    validate_raw_identity(inventory)
    return inventory


def replay_labels(output_dir: Path, data_dir: Path) -> Path:
    """Replay the bundled scores and publish only the canonical label artifact."""
    if Path(output_dir).exists():
        raise FileExistsError(f"output already exists: {output_dir}")
    reference = FPTReference()
    rows, diagnostics = [], []
    for task in fixed_tasks():
        scores = reference.scores(task)
        decision = detect_task(task, scores)
        rows.append(label_row(task, scores, decision, reference.entries[task.key]))
        diagnostics.append({"task": task.key, "decision_source": decision["decision_source"],
                            "coverage_rescue_applied": bool(decision["coverage_rescue_applied"])})
    comparison = compare_rows(rows, reference.rows, LABEL_COLUMNS)
    if not comparison["passed"]:
        raise FPTLabelContractError(f"detector replay changed the main reference: {comparison}")
    print("[FPT] detector replay: 75/75 rows match; binding raw CSV inventory", flush=True)
    inventory = raw_inventory(Path(data_dir), reference)
    manifest = write_artifact(output_dir, rows, reference,
                              generation={"mode": "frozen_score_replay", "seed": 42,
                                          "device": "cpu", "groupvae_training_invoked": False,
                                          "raw_data": inventory, "comparison": comparison},
                              diagnostics=diagnostics)
    print(f"[FPT] official labels: {manifest}; CSV SHA-256={REFERENCE_LABEL_SHA256}", flush=True)
    return manifest


def authorized_gpus(indices: list[int]) -> list[dict]:
    """Return approved idle devices for the isolated survival workers."""
    validate_gpu_ids(indices)
    result = subprocess.run(
        ["nvidia-smi", "--id=" + ",".join(map(str, indices)),
         "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    devices = {}
    for line in result.stdout.splitlines():
        index, uuid, name, memory, utilization = [value.strip() for value in line.split(",")]
        devices[int(index)] = {"physical_index": int(index), "uuid": uuid, "name": name,
                               "memory_used_mib": int(memory), "utilization": int(utilization)}
    if set(devices) != set(indices):
        raise RuntimeError("GPU query did not return exactly the authorized devices")
    for device in devices.values():
        if not device["uuid"].startswith("GPU-"):
            raise RuntimeError("GPU query did not return a full UUID")
        if device["memory_used_mib"] > 512 or device["utilization"] > 0:
            raise RuntimeError(f"authorized GPU is busy; no process will be stopped: {device}")
    return [devices[index] for index in indices]
