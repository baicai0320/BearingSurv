"""Regression tests for the single downstream FPT replay source."""

from __future__ import annotations

from argparse import Namespace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from src import config as cfg
from fpt_detection import generation
from fpt_detection.data_loader import _read_signal, resolve_bearing_directory, XJTUDataLoader
from fpt_detection.generation import authorized_gpus
from fpt_detection.labels import (
    FPTLabelContractError,
    FPTLabelStore,
    FPTReference,
    FPTTask,
    LABEL_COLUMNS,
    REFERENCE_LABEL_SHA256,
    REFERENCE_RAW_SHA256,
    compare_rows,
    csv_bytes,
    detect_task,
    file_digest,
    fixed_tasks,
    label_row,
    write_artifact,
)


@pytest.mark.parametrize("header", [True, False])
@pytest.mark.parametrize("normalization", ["none", "per_signal_zscore"])
def test_shared_csv_parser_preserves_model_precision(tmp_path, header, normalization):
    from otsurv.raw_signal_loader import RawSignalLoader

    values = np.random.default_rng(7).normal(size=(32768, 2))
    path = tmp_path / "1.csv"
    pd.DataFrame(values, columns=["horizontal", "vertical"]).to_csv(path, index=False, header=header)
    expected = pd.read_csv(path, header=0 if header else None).to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(_read_signal(path), expected)
    expected = expected.astype(np.float32)
    if normalization == "per_signal_zscore":
        expected = (expected - expected.mean(axis=0)) / expected.std(axis=0)
    actual = RawSignalLoader._read_csv(path, normalization=normalization)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)


def test_invalid_csv_and_zero_variance_are_rejected(tmp_path):
    from otsurv.raw_signal_loader import RawSignalDataError, RawSignalLoader

    path = tmp_path / "1.csv"
    path.write_text("horizontal,vertical\n1,2\n")
    with pytest.raises(ValueError, match="finite"):
        _read_signal(path)
    with pytest.raises(RawSignalDataError):
        RawSignalLoader._read_csv(path)
    pd.DataFrame(np.zeros((32768, 2))).to_csv(path, index=False)
    with pytest.raises(RawSignalDataError, match="zero-variance"):
        RawSignalLoader._read_csv(path)


def test_raw_inventory_binds_paths_sizes_and_contents(tmp_path, monkeypatch):
    entries = {}
    expected = []
    for condition in cfg.CONDITIONS:
        for bearing in cfg.BEARING_IDS:
            relative = Path(cfg.CONDITION_NAMES[condition]) / f"Bearing{condition + 1}_{bearing}" / "1.csv"
            path = tmp_path / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"horizontal,vertical\n1,2\n")
            entries[f"fold1_c{condition}_b{bearing}"] = {"n_samples": 1}
            expected.append({"path": relative.as_posix(), "bytes": path.stat().st_size,
                             "sha256": file_digest(path)})
    seen = []
    monkeypatch.setattr(generation, "validate_raw_identity", seen.append)
    reference = SimpleNamespace(entries=entries)
    inventory = generation.raw_inventory(tmp_path, reference)
    digest = sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert inventory == {"file_count": 15, "sha256": digest}
    assert seen == [inventory]
    path.write_bytes(b"changed")
    assert generation.raw_inventory(tmp_path, reference)["sha256"] != digest
    entries["fold1_c0_b1"]["n_samples"] = 2
    with pytest.raises(FPTLabelContractError, match="length mismatch"):
        generation.raw_inventory(tmp_path, reference)


def test_raw_inventory_rejects_missing_numbered_records(tmp_path):
    directory = tmp_path / cfg.CONDITION_NAMES[0] / "Bearing1_1"
    directory.mkdir(parents=True)
    (directory / "1.csv").write_bytes(b"one")
    (directory / "3.csv").write_bytes(b"three")
    with pytest.raises(ValueError, match="contiguous"):
        XJTUDataLoader(tmp_path).raw_record_paths(0, 1)


@pytest.fixture(scope="module")
def replay():
    reference = FPTReference()
    rows, diagnostics = [], []
    for task in fixed_tasks():
        scores = reference.scores(task)
        decision = detect_task(task, scores)
        rows.append(label_row(task, scores, decision, reference.entries[task.key]))
        diagnostics.append({"task": task.key, "decision_source": decision["decision_source"],
                            "coverage_rescue_applied": bool(decision["coverage_rescue_applied"])})
    return reference, rows, diagnostics


def publish(path, replay):
    reference, rows, diagnostics = replay
    return write_artifact(path, rows, reference,
                          generation={"mode": "frozen_score_replay", "seed": 42,
                                      "device": "cpu", "groupvae_training_invoked": False,
                                      "comparison": compare_rows(rows, reference.rows, LABEL_COLUMNS),
                                      "raw_data": {"file_count": 9216, "sha256": REFERENCE_RAW_SHA256}},
                          diagnostics=diagnostics)


def test_replay_all_75_rows_and_original_csv_bytes(replay):
    reference, rows, diagnostics = replay
    assert compare_rows(rows, reference.rows, LABEL_COLUMNS) == {
        "passed": True, "rows": 75, "matching_rows": 75, "differences": []}
    assert sha256(csv_bytes(rows)).hexdigest() == REFERENCE_LABEL_SHA256
    assert [row["task"] for row in diagnostics if row["coverage_rescue_applied"]] == ["fold4_c0_b4"]


def test_nested_training_peers_exclude_target_and_outer_test():
    assert len(fixed_tasks()) == 75
    for task in fixed_tasks():
        assert task.bearing_id not in task.training_bearings
        assert task.outer_fold not in task.training_bearings
        assert len(task.training_bearings) == (4 if task.role == "held_out" else 3)
    with pytest.raises(FPTLabelContractError):
        FPTTask(True, 0, 1)


def test_changed_detector_results_cannot_be_published(tmp_path, replay):
    reference, rows, diagnostics = replay
    changed = [dict(row) for row in rows]
    changed[0]["fpt_idx"] += 1
    with pytest.raises(FPTLabelContractError, match="publication refused"):
        write_artifact(tmp_path / "bad", changed, reference, generation={}, diagnostics=diagnostics)
    assert not (tmp_path / "bad").exists()


def test_single_consumer_accepts_only_frozen_source(tmp_path, replay):
    manifest_path = publish(tmp_path / "valid", replay)
    store = FPTLabelStore(manifest_path)
    assert store.csv_sha256 == REFERENCE_LABEL_SHA256
    assert len(store.labels) == 75
    fold = store.labels_for(4, 0)
    assert fold.loc[fold.bearing_id == 4, "fpt_idx"].item() == 116
    original = store.labels_path.read_bytes()
    store.labels_path.write_bytes(original.replace(b"held_out", b"changed_", 1))
    with pytest.raises(FPTLabelContractError, match="immutable reference hash"):
        FPTLabelStore(manifest_path)


@pytest.mark.parametrize("field,value", [("source", "changed"), ("schema_version", 1)])
def test_alternative_manifests_rejected(tmp_path, replay, field, value):
    path = publish(tmp_path / "artifact", replay)
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(FPTLabelContractError):
        FPTLabelStore(path)


@pytest.mark.parametrize("raw_data", [None, {}, {"file_count": 1, "sha256": "a" * 64},
                                        {"file_count": 9216, "sha256": "a" * 64}])
def test_changed_raw_input_binding_rejected(tmp_path, replay, raw_data):
    path = publish(tmp_path / "artifact", replay)
    manifest = json.loads(path.read_text())
    manifest["raw_data"] = raw_data
    manifest["generation"]["raw_data"] = raw_data
    path.write_text(json.dumps(manifest))
    with pytest.raises(FPTLabelContractError, match="raw input"):
        FPTLabelStore(path)


@pytest.mark.parametrize("field,value", [("mode", "changed"), ("seed", 7),
                                          ("device", "cuda"),
                                          ("groupvae_training_invoked", True),
                                          ("comparison", {"passed": False})])
def test_changed_generation_provenance_rejected(tmp_path, replay, field, value):
    path = publish(tmp_path / "artifact", replay)
    manifest = json.loads(path.read_text())
    manifest["generation"][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(FPTLabelContractError, match="generation"):
        FPTLabelStore(path)


def test_model_settings_do_not_invalidate_labels_but_fpt_seed_does(tmp_path, replay, monkeypatch):
    path = publish(tmp_path / "artifact", replay)
    monkeypatch.setitem(cfg.PARAMS_R14F, "lr", 0.123)
    FPTLabelStore(path)
    monkeypatch.setattr(cfg, "FPT_RANDOM_SEED", 43)
    with pytest.raises(FPTLabelContractError, match="seed"):
        FPTLabelStore(path)


@pytest.mark.parametrize("condition", [1, 2])
def test_raw_loader_uses_canonical_condition_directory(tmp_path, monkeypatch, condition):
    from otsurv.raw_signal_loader import RawSignalLoader
    import otsurv.raw_signal_loader as raw_module

    canonical = tmp_path / cfg.CONDITION_NAMES[condition] / f"Bearing{condition + 1}_1"
    canonical.mkdir(parents=True)
    (canonical.parent / "Bearing1_1").mkdir()
    assert resolve_bearing_directory(tmp_path, condition, 1) == canonical
    monkeypatch.setattr(raw_module.cfg, "DATA_DIR", tmp_path)
    loader = RawSignalLoader(condition, normalization="none")
    assert loader._signal_path(1, 0) == canonical / "1.csv"


def test_score_bundle_tampering_rejected(tmp_path):
    shutil.copytree(ROOT / "fpt_detection" / "reference", tmp_path / "reference")
    ref = FPTReference(tmp_path / "reference")
    task = fixed_tasks()[0]
    (ref.directory / "scores" / f"{task.key}.npz").write_bytes(b"invalid")
    with pytest.raises(FPTLabelContractError, match="score file digest"):
        ref.scores(task)


def test_reference_manifest_tampering_rejected(tmp_path):
    shutil.copytree(ROOT / "fpt_detection" / "reference", tmp_path / "reference")
    path = tmp_path / "reference" / "reference.json"
    manifest = json.loads(path.read_text())
    manifest["historical_environment"]["device"] = "cuda"
    path.write_text(json.dumps(manifest))
    with pytest.raises(FPTLabelContractError, match="reference manifest"):
        FPTReference(path.parent)


def test_changed_raw_data_cannot_create_partial_official_artifact(tmp_path, replay):
    reference, rows, diagnostics = replay
    with pytest.raises(FPTLabelContractError, match="raw input"):
        write_artifact(tmp_path / "bad", rows, reference,
                       generation={"raw_data": {"file_count": 9216, "sha256": "a" * 64}},
                       diagnostics=diagnostics)
    assert not (tmp_path / "bad").exists()


def test_other_gpu_indices_rejected_before_query(monkeypatch):
    monkeypatch.setattr("device_policy.socket.gethostname", lambda: "z133-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("unauthorized GPU query")

    monkeypatch.setattr(subprocess, "run", forbidden)
    for devices in ([0], [3, 4], [4, 4], []):
        with pytest.raises(ValueError, match="only physical GPUs"):
            authorized_gpus(devices)


def test_gpu_uuid_and_busy_card_refusal(monkeypatch):
    calls = []

    def query(command, **kwargs):
        calls.append(command)
        return Namespace(stdout="4, GPU-four, NVIDIA GeForce RTX 4090, 1, 0\n"
                                  "5, GPU-five, NVIDIA GeForce RTX 4090, 1, 0\n")

    monkeypatch.setattr(subprocess, "run", query)
    devices = authorized_gpus([4, 5])
    assert "--id=4,5" in calls[0]
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: Namespace(
        stdout="4, GPU-four, NVIDIA GeForce RTX 4090, 2048, 50\n"))
    with pytest.raises(RuntimeError, match="busy"):
        authorized_gpus([4])
