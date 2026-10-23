#!/usr/bin/env python3
"""Run the fixed paper profile through device-isolated model workers."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from paper.runtime import environment_report, worker_environment, source_hashes, write_json, authorized_idle_gpus
from paper.protocol import MODELS, cpu_threads, task_gpu_ids, tasks_for
from device_policy import allowed_gpu_ids, default_gpu_ids, validate_gpu_ids


ALIASES = {"r14f": "otsurv", "raw-cnn": "rawcnn"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=["all", *MODELS, *ALIASES])
    parser.add_argument("--conditions", nargs="+", type=int, default=[1], choices=cfg.CONDITIONS)
    parser.add_argument("--outer-folds", nargs="+", type=int, default=list(cfg.BEARING_IDS), choices=cfg.BEARING_IDS)
    parser.add_argument("--censoring-levels", nargs="+", type=float, default=[0.5], choices=cfg.CENSORING_LEVELS)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=default_gpu_ids(),
                        help="physical GPU indices; z133-8 is restricted to 4 and 5")
    parser.add_argument("--seed", type=int, choices=[0], default=0)
    parser.add_argument("--fpt-label-manifest", type=Path, default=cfg.FPT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results-file", type=Path, help="Compatibility output location; artifacts are written beside it")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the task plan without data access, training or GPU queries")
    parser.add_argument("--continue-on-metric-mismatch", action="store_true",
                        help="Complete the flow audit, preserving mismatches and final exit code 2")
    return parser


def resolve_models(values):
    if "all" in values:
        if values != ["all"]:
            raise ValueError("all cannot be combined with model names")
        return list(MODELS)
    models = [ALIASES.get(value, value) for value in values]
    if len(models) != len(set(models)):
        raise ValueError("model names and aliases must not duplicate a task")
    return models


def aggregate(output, tasks, results_path):
    import numpy as np
    import pandas as pd
    from paper.runtime import METRICS

    rows = []
    for task in tasks:
        path = output / task / "result.json"
        if path.exists():
            record = json.loads(path.read_text())
            rows.append({k: v for k, v in record.items() if k not in ("metrics", "devices", "model_metadata")} | record["metrics"])
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(results_path, index=False)
    summary = []
    for key, group in frame.groupby(["condition", "censoring", "model"], sort=True):
        row = dict(zip(("condition", "censoring", "model"), key))
        row["FoldRows"] = len(group)
        row["HistoricalMetricsMatch"] = (bool(group.historical_metrics_match.all())
                                         if group.historical_metrics_match.notna().all() else None)
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            values = values[np.isfinite(values)]
            row.update({metric + "_N": len(values), metric + "_Mean": values.mean(),
                        metric + "_SD": values.std(ddof=1), metric + "_Median": values.median(),
                        metric + "_Min": values.min(), metric + "_Max": values.max()})
        summary.append(row)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)


def main(argv=None):
    args = build_parser().parse_args(argv)
    models = resolve_models(args.models)
    for values in (args.conditions, args.outer_folds, args.censoring_levels, args.gpu_ids):
        if len(values) != len(set(values)):
            raise ValueError("execution subsets must not contain duplicates")
    validate_gpu_ids(args.gpu_ids)
    if args.results_file and args.output_dir:
        raise ValueError("choose output-dir or results-file, not both")
    configuration = {"models": models, "conditions": args.conditions,
                     "outer_folds": args.outer_folds, "censoring_levels": args.censoring_levels,
                     "seed": 0, "device": args.device, "gpu_ids": args.gpu_ids,
                     "gpu_allowlist": list(allowed_gpu_ids()) if allowed_gpu_ids() is not None else None,
                     "continue_on_metric_mismatch": args.continue_on_metric_mismatch}
    tasks = tasks_for(configuration)
    if args.dry_run:
        print(json.dumps({"configuration": configuration, "task_count": len(tasks),
                          "tasks": [task[0] for task in tasks]}, indent=2))
        return 0
    environment = environment_report()
    # Imports remain CPU-only in the coordinator. Models load inside workers.
    from fpt_detection.labels import FPTLabelStore, FPTReference
    from fpt_detection.generation import raw_inventory
    from paper.data import prepare_features
    store = FPTLabelStore(args.fpt_label_manifest.resolve())
    inventory = raw_inventory(cfg.DATA_DIR, FPTReference())
    feature_paths = {condition: prepare_features(condition, store) for condition in args.conditions}
    configuration.update({"fpt_labels_sha256": store.csv_sha256,
                          "fpt_manifest_sha256": store.manifest_sha256, "raw_inventory": inventory})
    if args.preflight_only:
        print(json.dumps({"configuration": configuration, "environment": environment,
                          "feature_paths": {str(k): str(v) for k, v in feature_paths.items()}}, indent=2))
        return 0
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = (args.output_dir or (args.results_file.parent if args.results_file else cfg.OUTPUT_DIR / "paper_cv" / name)).resolve()
    results_path = args.results_file.resolve() if args.results_file else output / "results.csv"
    manifest_path = output / "run_manifest.json"
    sources = source_hashes(ROOT)
    if args.resume:
        old = json.loads(manifest_path.read_text())
        if old["configuration"] != configuration or old["source_hashes"] != sources or old["environment"]["packages"] != environment["packages"]:
            raise ValueError("cannot resume with changed data, configuration, source, or runtime packages")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(manifest_path, {"configuration": configuration, "environment": environment,
                                  "source_hashes": sources, "status": "created", "created_at": name})
    task_names = [task[0] for task in tasks]
    print(f"[paper] output={output}; model-fold tasks={len(tasks)}", flush=True)
    for task, condition, censoring, fold, model in tasks:
        target = output / task
        result_path = target / "result.json"
        if result_path.exists():
            record = json.loads(result_path.read_text())
            if record["status"] == "completed" and (record["historical_metrics_match"] is not False or args.continue_on_metric_mismatch):
                artifacts = json.loads((target / "artifact_hashes.json").read_text())
                if any(sha256((target / name).read_bytes()).hexdigest() != digest for name, digest in artifacts.items()):
                    raise RuntimeError(f"completed task artifacts changed: {target}")
                print(f"[paper] reuse {task}", flush=True)
                continue
            raise RuntimeError(f"inspect the existing numerical mismatch before retrying: {target}")
        if target.exists():
            raise RuntimeError(f"inspect the incomplete task before retrying: {target}")
        if source_hashes(ROOT) != sources:
            raise RuntimeError("runtime source changed after the run manifest was frozen")
        gpu_ids = task_gpu_ids(model, configuration)
        devices = authorized_idle_gpus(gpu_ids) if gpu_ids else []
        threads = cpu_threads(model)
        target.mkdir()
        spec = {"model": model, "condition": condition, "censoring": censoring, "fold": fold,
                "devices": devices, "cpu_workers": threads, "fpt_manifest": str(args.fpt_label_manifest.resolve()),
                "features": str(feature_paths[condition]), "output_dir": str(target)}
        write_json(target / "task.json", spec)
        env = worker_environment(devices, threads)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT)
        print(f"[paper] start {task}; physical_gpus={[d['physical_index'] for d in devices]}", flush=True)
        with (target / "worker.log").open("x") as log:
            process = subprocess.run([sys.executable, "-u", "-B", "-m", "paper.worker", str(target / "task.json")],
                                     env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        aggregate(output, task_names, results_path)
        print(f"[paper] finish {task}; exit={process.returncode}", flush=True)
        if process.returncode == 2 and args.continue_on_metric_mismatch:
            print(f"[paper] numerical mismatch retained: {target / 'metric_comparison.json'}", flush=True)
        elif process.returncode:
            print(f"[paper] stopped; inspect {target / 'worker.log'}", flush=True)
            return process.returncode
    aggregate(output, task_names, results_path)
    nonmatching = [task for task in task_names
                   if json.loads((output / task / "result.json").read_text())["historical_metrics_match"] is False]
    if not (output / "completion.json").exists():
        write_json(output / "completion.json", {"status": "completed", "flow_completed": True,
                                               "model_fold_results": len(tasks),
                                               "historical_metrics_match": not nonmatching,
                                               "nonmatching_tasks": nonmatching, "configuration": configuration})
    return 2 if nonmatching else 0


if __name__ == "__main__":
    raise SystemExit(main())
