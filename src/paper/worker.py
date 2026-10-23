"""An isolated model/fold worker. Device checks precede framework imports."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import sys
import time
import traceback

from paper.runtime import validate_worker_devices, environment_report, write_json, json_safe


def run(spec):
    validate_worker_devices(spec["devices"])
    environment = environment_report()
    import numpy as np
    import pandas as pd
    import torch
    import config as cfg
    from fpt_detection.labels import FPTLabelStore
    from paper.data import build_fold, outcome_hash, OUTCOMES
    from paper.models import OTSurv, PaperBaseline, seed_everything
    from paper.evaluation import evaluate, compare_metrics, reference_record
    from paper.config import REPORT_EPOCHS
    from otsurv.raw_signal_loader import RawSignalLoader
    from tools.probability_metrics import build_development_time_grid
    from utility.survival import make_event_times

    interop_threads = 1 if spec["model"] == "otsurv" else 4
    torch.set_num_threads(spec["cpu_workers"])
    torch.set_num_interop_threads(interop_threads)
    if spec["devices"]:
        if torch.cuda.device_count() != len(spec["devices"]) or torch.backends.cudnn.version() != 8902:
            raise RuntimeError("CUDA device count or cuDNN differs from the recorded paper runtime")
        device = "cuda:0"
    else:
        device = "cpu"
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(spec["cpu_workers"])
    tf.config.threading.set_inter_op_parallelism_threads(interop_threads)
    if spec["model"] == "bnnsurv" and spec["devices"]:
        physical = tf.config.list_physical_devices("GPU")
        if len(physical) != 1:
            raise RuntimeError("BNNSurv must see exactly its authorized GPU")
        tf.config.experimental.set_memory_growth(physical[0], True)
    else:
        tf.config.set_visible_devices([], "GPU")
    seed_everything()
    if spec["model"] == "bnnsurv":
        tf.keras.utils.set_random_seed(0)
    environment["numerical_runtime"] = {
        "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "devices": spec["devices"],
        "tf_intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "tf_inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
    }
    store = FPTLabelStore(Path(spec["fpt_manifest"]))
    development, test = build_fold(spec["condition"], spec["fold"], spec["censoring"], store, Path(spec["features"]))
    reference = reference_record(spec["model"], spec["condition"], spec["fold"], spec["censoring"])
    hashes = {"DevelopmentOutcomeSHA256": outcome_hash(development), "TestOutcomeSHA256": outcome_hash(test),
              "LabelArtifactSHA256": store.csv_sha256}
    for name, value in hashes.items():
        if value != reference[name]:
            raise RuntimeError(f"historical cohort mismatch before training: {name}: {value} != {reference[name]}")
    grid = build_development_time_grid(development.Survival_time, development.Event)
    output = Path(spec["output_dir"])
    started = time.monotonic()
    if spec["model"] == "otsurv":
        time_device = "cuda:1" if spec["devices"] else "cpu"
        model = OTSurv(spec["condition"], device=device, time_device=time_device)
        params = model.effective_parameters()
        for key in ("parameters", "rank_parameters", "time_parameters"):
            if params[key] != reference[key]:
                raise RuntimeError(f"OTSurv {key} differs from the historical task")
        write_json(output / "effective_parameters.json", params)
        loader = RawSignalLoader(spec["condition"], normalization="none")
        signals = loader.load_bearing_signals_from_dataframe(development)
        model.fit(signals, development.Survival_time, development.Event, bearing_ids=development.bearing_id,
                  time_steps=development.time_step, fpt_times=development.TrueTime, fpt_events=development.TrueEvent,
                  log_fn=lambda message: print(message, flush=True))
        fit_seconds = time.monotonic() - started
        model.save(output / "model.pt")
        for endpoint in ("rank", "time"):
            pd.DataFrame(getattr(model.model, f"r14f_{endpoint}_epoch_diagnostics_")).to_csv(output / f"{endpoint}_epochs.csv", index=False)
        evaluation_times = np.unique(np.concatenate(([0.0], development.Survival_time.to_numpy()[development.Survival_time.to_numpy() <= 1024], [1024.0])))
        test_signals = loader.load_bearing_signals_from_dataframe(test)
        meta = {"bearing_ids": test.bearing_id, "time_steps": test.time_step}
        risks = model.predict_risk(test_signals, **meta)
        curves = model.predict_survival(test_signals, evaluation_times, **meta)
        model_metadata = {"checkpoint_epoch": REPORT_EPOCHS[spec["condition"]], "train_epochs": 40}
    else:
        model = PaperBaseline(spec["model"], spec["condition"], device=device, cpu_workers=spec["cpu_workers"])
        if model.params != reference["parameters"]:
            raise RuntimeError(f"baseline parameters differ from main: {model.params} != {reference['parameters']}")
        write_json(output / "effective_parameters.json", model.params)
        model.fit(development, outer_fold=spec["fold"])
        fit_seconds = time.monotonic() - started
        evaluation_times = make_event_times(development.Survival_time.to_numpy(), development.Event.to_numpy()).astype(int)
        curves, risks = model.predict(test, evaluation_times)
        model_metadata = model.fit_metadata
        if spec["model"] == "rawcnn":
            pd.DataFrame(model.selection_diagnostics).to_csv(output / "selection_epochs.csv", index=False)
            pd.DataFrame(model.model.otsurv_epoch_diagnostics_).to_csv(output / "refit_epochs.csv", index=False)
    metrics, medians = evaluate(spec["model"], curves, risks, development, test, grid)
    comparison = compare_metrics(metrics, reference)
    predictions = test.loc[:, OUTCOMES].copy()
    predictions["risk"], predictions["median"] = np.asarray(risks), medians
    predictions.to_csv(output / "oof_predictions.csv", index=False)
    np.savez_compressed(output / "survival_curves.npz", survival=curves.to_numpy(), times=np.asarray(curves.columns),
                        bearing_id=test.bearing_id.to_numpy(), time_step=test.time_step.to_numpy())
    development.loc[:, OUTCOMES].to_csv(output / "development_outcomes.csv", index=False)
    write_json(output / "metric_protocol.json", {
        "development_only_ipcw_reference": True, "time_grid": grid.times.tolist(), "tau": grid.tau,
        "cindex": "uno_ipcw_sksurv", "ibs": "graf_ipcw_integrated_brier_fixed_horizon_v1",
        "median": "grid_crossing_only_no_extrapolation_v1", "seed": 0,
        "otsurv_checkpoint_origin": "post_hoc_outer_test_selection_not_nested_validation",
        "historical_reference_role": "verification_only_never_training_input",
    })
    write_json(output / "metric_comparison.json", json_safe(comparison))
    write_json(output / "environment.json", environment)
    result = {"model": spec["model"], "condition": spec["condition"], "outer_fold": spec["fold"],
              "censoring": spec["censoring"], "status": "completed", "historical_metrics_match": comparison["passed"],
              "fit_seconds": fit_seconds, "total_seconds": time.monotonic() - started,
              "n_train": len(development), "n_test": len(test), "metrics": metrics, "model_metadata": model_metadata,
              "devices": spec["devices"], **hashes}
    write_json(output / "result.json", json_safe(result))
    write_json(output / "artifact_hashes.json", {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir()) if path.is_file() and path.name != "worker.log"
    })
    print(json.dumps(json_safe(result)), flush=True)
    return 2 if comparison["passed"] is False else 0


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    try:
        return run(spec)
    except Exception:
        traceback.print_exc()
        path = Path(spec["output_dir"]) / "failure.json"
        if not path.exists():
            write_json(path, {"status": "failed", "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
