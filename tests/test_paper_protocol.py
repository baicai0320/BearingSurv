"""Focused paper-profile, endpoint, serialization and device contracts."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from paper import config as cfg
from paper.models import OTSurv, baseline_parameters, MODEL_NAMES
from paper.runtime import validate_worker_devices
from paper.training import R14BOTSurvRegressor, R14FOTSurvRegressor
from paper.data import build_fold, outcome_hash, FEATURES
from fpt_detection.labels import FPTReference


def references():
    return json.loads((ROOT / "src/paper/reference.json").read_text())["records"]


def test_parameters_match_every_historical_task():
    for row in references():
        if row["model"] == "otsurv":
            params = OTSurv(row["condition"]).effective_parameters()
            for key in ("parameters", "rank_parameters", "time_parameters"):
                assert params[key] == row[key]
            assert params["train_epochs"] == 40
            assert params["checkpoint_epoch"] == {0: 40, 1: 10, 2: 20}[row["condition"]]
            assert params["time_cnn_window_batch_size"] == {0: 16384, 1: 16384, 2: 8192}[row["condition"]]
        else:
            assert baseline_parameters(row["model"], row["condition"]) == row["parameters"]
    assert len(MODEL_NAMES) == 8


def test_historical_metric_comparison_covers_all_models():
    from paper.evaluation import compare_metrics, reference_record

    rows = references()
    assert len(rows) == 360
    assert {row["model"] for row in rows} == set(MODEL_NAMES)
    for row in rows:
        reference = reference_record(row["model"], row["condition"], row["fold"], row["censoring"])
        metrics = {**reference["metrics"], **reference["statuses"]}
        if "DCalibStatus" in metrics:
            metrics["DCalibPValueStatus"] = metrics["DCalibStatus"]
        assert compare_metrics(metrics, reference)["passed"]
        metrics["IBS"] = 10.0
        assert not compare_metrics(metrics, reference)["passed"]


def test_c2_cohort_hashes(tmp_path):
    import pandas as pd
    reference = FPTReference()
    labels = pd.read_csv(reference.directory / "labels.csv")
    store = SimpleNamespace(labels_for=lambda fold, condition: labels.loc[
        labels.outer_fold.eq(fold) & labels.condition_id.eq(condition)].sort_values("bearing_id"))
    frames = []
    for label in store.labels_for(1, 1).itertuples():
        frames.append(pd.DataFrame({"bearing_id": label.bearing_id, "time_step": np.arange(label.n_samples),
                                    **{"h_" + feature: np.ones(label.n_samples) for feature in FEATURES}}))
    path = tmp_path / "features.csv"
    pd.concat(frames).to_csv(path, index=False)
    for fold in range(1, 6):
        np.random.seed(0)
        development, test = build_fold(1, fold, 0.5, store, path)
        for row in references():
            if row["condition"] == 1 and row["fold"] == fold and row["censoring"] == 0.5:
                assert outcome_hash(development) == row["DevelopmentOutcomeSHA256"]
                assert outcome_hash(test) == row["TestOutcomeSHA256"]


def test_device_mask_is_checked_before_framework_use(monkeypatch):
    monkeypatch.setattr("device_policy.socket.gethostname", lambda: "z133-8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-approved")
    validate_worker_devices([{"physical_index": 4, "uuid": "GPU-approved"}])
    with pytest.raises(RuntimeError, match="unauthorized"):
        validate_worker_devices([{"physical_index": 3, "uuid": "GPU-approved"}])
    with pytest.raises(RuntimeError, match="mask"):
        validate_worker_devices([])


def test_gpu_release_wait_never_ignores_a_busy_card(monkeypatch):
    from paper.runtime import authorized_idle_gpus
    calls = []
    devices = [{"physical_index": 4, "uuid": "GPU-approved"}]
    def query(indices):
        assert indices == [4]
        calls.append(indices)
        if len(calls) < 3:
            raise RuntimeError("authorized GPU is busy; no process will be stopped: test")
        return devices
    monkeypatch.setattr("fpt_detection.generation.authorized_gpus", query)
    monkeypatch.setattr("paper.runtime.time.sleep", lambda _: None)
    assert authorized_idle_gpus([4]) == devices
    assert len(calls) == 3
    def permanently_busy(indices):
        raise RuntimeError("authorized GPU is busy; no process will be stopped: test")
    monkeypatch.setattr("fpt_detection.generation.authorized_gpus", permanently_busy)
    with pytest.raises(RuntimeError, match="busy"):
        authorized_idle_gpus([4])


def test_feature_cache_retains_both_historical_csv_passes(monkeypatch, tmp_path):
    import pandas as pd
    import paper.data as data

    monkeypatch.setattr(data.public_config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(data.public_config, "BEARING_IDS", (1,))
    monkeypatch.setattr(data, "resolve_bearing_directory", lambda *args: tmp_path)
    monkeypatch.setattr(data, "_read_signal", lambda path: None)
    monkeypatch.setattr(data, "extract_features", lambda *args: {"h_p2p": 3.6587119102478023})
    store = SimpleNamespace(manifest={"raw_data": {}}, labels_for=lambda *args: pd.DataFrame(
        {"bearing_id": [1], "n_samples": [1]}))
    path = data.prepare_features(1, store)
    # A single CSV pass produces 3.6587119102478023; main's second pass differs.
    assert pd.read_csv(path).h_p2p.item() == 3.6587119102478014
    assert data.prepare_features(1, store) == path


def test_two_epoch_endpoints_and_roundtrip(monkeypatch, tmp_path):
    rank = deepcopy(cfg.PARAMS_R13_OTSURV)
    rank.update(r13_reference_length=1, r13_transport_batch_size=2, ot_num_iters=2, ot_stoperr=0.0,
                r13_amplitude_init="random", r13_amplitude_anchor_weight=0.0,
                r13_amplitude_freeze_epochs=0, r13_shape_projection_freeze_epochs=0,
                r13_orthogonalize_trajectory_to_main=False, max_epochs=2,
                early_stopping_patience=1, min_epochs_before_stopping=1)
    time = deepcopy(cfg.PARAMS_R14B_OTSURV)
    time.update(signal_length=1024, r14b_reference_length_short=1, r14b_reference_length_medium=2,
                r14b_reference_length_long=3, r14b_transport_batch_size=2, ot_num_iters=2, ot_stoperr=0.0,
                r14b_state_hidden_dim=8, r14b_warmup_epochs=1, max_epochs=2,
                early_stopping_patience=1, min_epochs_before_stopping=1)
    joint = {**time, **{k: v for k, v in cfg.PARAMS_R14F_OTSURV.items() if k.startswith("r14f_")},
             "mode": "r14f", "loss_fn": cfg.PARAMS_R14F_OTSURV["loss_fn"], "r14f_fixed_epochs": 2}
    monkeypatch.setattr(cfg, "PARAMS_R13_OTSURV", rank)
    monkeypatch.setattr(cfg, "PARAMS_R14B_OTSURV", time)
    monkeypatch.setattr(cfg, "PARAMS_R14F_OTSURV", joint)
    monkeypatch.setattr(cfg, "TRAIN_EPOCHS", 2)
    monkeypatch.setattr(cfg, "CHECKPOINT_EPOCHS", (1, 2))
    monkeypatch.setattr(cfg, "REPORT_EPOCHS", {1: 1})
    monkeypatch.setattr(R14BOTSurvRegressor, "WARMUP_EPOCHS", 1)
    monkeypatch.setattr(R14FOTSurvRegressor, "FIXED_EPOCHS", 2)
    def forbidden(*args, **kwargs):
        raise AssertionError("the time endpoint must not train a separate teacher")
    monkeypatch.setattr(R14BOTSurvRegressor, "_fit_frozen_teacher", forbidden, raising=False)
    torch.set_num_threads(2)
    torch.manual_seed(7)
    signals = [torch.randn(1024, 2) + 0.05 * i for i in range(8)]
    meta = {"bearing_ids": np.repeat([1, 2], 4), "time_steps": np.tile(np.arange(4), 2)}
    model = OTSurv(1).fit(signals, [3, 4, 5, 6] * 2, [1, 1, 0, 1, 1, 0, 1, 1],
                         **meta, fpt_times=[2, 3, 4, 5] * 2, fpt_events=np.ones(8), log_fn=lambda _: None)
    assert model.model.r14f_rank_anchor_["frozen"] is False
    assert model.model.r14f_teacher_["gradient_path"] == "r14b_distillation_to_r13_none_detached"
    assert all(not row["teacher_gradient_to_rank"] for row in model.model.r14f_time_epoch_diagnostics_)
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(value, model.checkpoints[1][key], rtol=0, atol=0)
    assert any(not torch.equal(value, model.checkpoints[2][key]) for key, value in model.checkpoints[1].items())
    path = tmp_path / "model.pt"
    model.save(path)
    restored = OTSurv.load(path)
    assert model.model.time_expert.RAW_CNN_WINDOW_BATCH_SIZE == 16384
    assert restored.model.time_expert.RAW_CNN_WINDOW_BATCH_SIZE == 16384
    np.testing.assert_array_equal(model.predict_risk(signals, **meta), restored.predict_risk(signals, **meta))
    grid = np.asarray([0.0, 1.0, 10.0, 1024.0])
    a, b = model.predict_survival(signals, grid, **meta), restored.predict_survival(signals, grid, **meta)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all().all()
    assert (np.diff(a, axis=1) <= 0).all()
    with pytest.raises(FileExistsError):
        model.save(path)


def test_summary_preserves_metric_denominators(tmp_path):
    import pandas as pd
    from paper.runtime import METRICS, write_json
    from run_paper_cross_validation import aggregate
    tasks = ["fold1", "fold2"]
    for i, task in enumerate(tasks):
        directory = tmp_path / task
        directory.mkdir()
        write_json(directory / "result.json", {
            "condition": 1, "censoring": 0.5, "model": "otsurv", "historical_metrics_match": True,
            "metrics": {metric: float(i + 1) for metric in METRICS} | {"MAETrue": None if i == 0 else 2.0},
        })
    aggregate(tmp_path, tasks, tmp_path / "results.csv")
    summary = pd.read_csv(tmp_path / "summary.csv").iloc[0]
    assert summary.FoldRows == 2 and summary.MAETrue_N == 1
    assert summary.MAETrue_Mean == 2.0 and np.isnan(summary.MAETrue_SD)
    assert summary.IBS_Mean == 1.5
    assert np.isclose(summary.IBS_SD, np.sqrt(0.5))


def test_missingness_and_statuses_cannot_pass_as_numeric_matches():
    from paper.evaluation import compare_metrics, reference_record
    reference = reference_record("otsurv", 1, 1, 0.5)
    metrics = {**reference["metrics"], **reference["statuses"]}
    assert compare_metrics(metrics, reference)["passed"]
    metrics["MAEFPTRefStatus"] = "na:incorrect_status"
    assert not compare_metrics(metrics, reference)["passed"]
    metrics["MAEFPTRefStatus"] = reference["statuses"]["MAEFPTRefStatus"]
    metrics["IBS"] = None
    assert not compare_metrics(metrics, reference)["passed"]


@pytest.mark.parametrize("model", ["otsurv", "coxph"])
def test_core_evaluation_matches_uncensored_brier_definition(model):
    import pandas as pd
    from paper.evaluation import evaluate
    from tools.probability_metrics import build_development_time_grid

    development = pd.DataFrame({"Survival_time": [1.0, 5.0, 7.0, 12.0, 15.0, 20.0],
                                "Event": True})
    test = pd.DataFrame({"Survival_time": [1.0, 6.0, 14.0], "Event": True,
                         "TrueTime": [1.0, 6.0, 14.0], "TrueEvent": True})
    grid = build_development_time_grid(development.Survival_time, development.Event)
    times = np.unique(np.r_[0.0, grid.times, 20.0])
    rates = np.array([1.0, 0.5, 0.2])
    curves = pd.DataFrame(np.exp(-rates[:, None] * times), columns=times)
    metrics, medians = evaluate(model, curves, [3.0, 2.0, 1.0], development, test, grid)
    expected_survival = np.exp(-rates[:, None] * grid.times)
    observed = test.Survival_time.to_numpy()[:, None] > grid.times
    brier_scores = ((observed - expected_survival) ** 2).mean(axis=0)
    expected_ibs = np.trapz(brier_scores, grid.times) / (grid.times[-1] - grid.times[0])
    assert metrics["CIndexIPCWStatus"] == metrics["IBSStatus"] == "ok"
    assert metrics["CIndexIPCW"] == pytest.approx(1.0)
    assert metrics["IBS"] == pytest.approx(expected_ibs)
    assert np.isfinite(medians).all()


def test_flow_audit_requires_an_explicit_option():
    from run_paper_cross_validation import build_parser
    parser = build_parser()
    assert parser.parse_args([]).continue_on_metric_mismatch is False
    assert parser.parse_args(["--continue-on-metric-mismatch"]).continue_on_metric_mismatch is True


@pytest.mark.parametrize("continue_mismatch,codes,expected_exit,completed", [
    (False, (2,), 2, False),
    (True, (2, 0), 2, True),
    (True, (2, 1), 1, False),
    (False, (0, 0), 0, True),
])
def test_coordinator_distinguishes_mismatch_from_execution_failure(
        monkeypatch, tmp_path, continue_mismatch, codes, expected_exit, completed):
    from hashlib import sha256
    import run_paper_cross_validation as runner
    from paper.runtime import METRICS, write_json

    monkeypatch.setattr(runner, "environment_report", lambda: {"packages": {}})
    monkeypatch.setattr(runner, "source_hashes", lambda _: {"source": "frozen"})
    monkeypatch.setattr("fpt_detection.labels.FPTLabelStore", lambda _: SimpleNamespace(
        csv_sha256="labels", manifest_sha256="manifest"))
    monkeypatch.setattr("fpt_detection.generation.raw_inventory", lambda *args: {})
    monkeypatch.setattr("paper.data.prepare_features", lambda *args: tmp_path / "features.csv")
    calls = []

    def worker(command, *, env, cwd, stdout, stderr):
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        spec = json.loads(Path(command[-1]).read_text())
        assert spec["devices"] == []
        code = codes[len(calls)]
        calls.append(spec["model"])
        target = Path(spec["output_dir"])
        if code != 1:
            write_json(target / "result.json", {
                "condition": spec["condition"], "censoring": spec["censoring"],
                "outer_fold": spec["fold"], "model": spec["model"], "status": "completed",
                "historical_metrics_match": code == 0,
                "metrics": {metric: 1.0 for metric in METRICS},
            })
            write_json(target / "artifact_hashes.json", {
                "result.json": sha256((target / "result.json").read_bytes()).hexdigest(),
            })
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(runner.subprocess, "run", worker)
    output = tmp_path / "run"
    args = ["--models", "coxph", "coxboost", "--device", "cpu", "--outer-folds", "1",
            "--output-dir", str(output)]
    if continue_mismatch:
        args.append("--continue-on-metric-mismatch")
    assert runner.main(args) == expected_exit
    assert len(calls) == len(codes)
    assert (output / "completion.json").exists() is completed
    if completed:
        completion = json.loads((output / "completion.json").read_text())
        assert completion["flow_completed"] is True
        assert completion["historical_metrics_match"] is (2 not in codes)
        assert len(completion["nonmatching_tasks"]) == codes.count(2)
