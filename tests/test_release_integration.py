"""Core release checks for the public main-table workflow."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from paper.protocol import MODELS, cpu_threads, task_gpu_ids, tasks_for
from paper.runtime import PACKAGES, source_hashes


def configuration(**overrides):
    value = {
        "models": list(MODELS),
        "conditions": [0, 1, 2],
        "outer_folds": [1, 2, 3, 4, 5],
        "censoring_levels": [0.25, 0.5, 0.75],
        "device": "cuda",
        "gpu_ids": [4, 5],
        "seed": 0,
    }
    value.update(overrides)
    return value


def test_dependency_files_match_surv():
    from packaging.requirements import Requirement

    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if not line or line.startswith(("#", "--")):
            continue
        requirement = Requirement(line)
        version = importlib.metadata.version(requirement.name)
        assert str(requirement.specifier) == f"=={version}"
        pins[requirement.name] = version
    assert all(pins[key] == value for key, value in PACKAGES.items())
    environment = (ROOT / "environment.yml").read_text()
    assert "- -r requirements.txt" in environment
    for pin in ("python=3.10.20", "r-base=4.5.3", "cudatoolkit=11.2.2", "cudnn=8.1.0.77"):
        assert pin in environment


def test_full_matrix_and_dry_run_without_data_or_devices(monkeypatch, capsys):
    import run_paper_cross_validation as runner

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not access data, environment or GPUs")

    monkeypatch.setattr(runner, "environment_report", forbidden)
    monkeypatch.setattr(runner, "authorized_idle_gpus", forbidden)
    monkeypatch.setattr("fpt_detection.generation.raw_inventory", forbidden)
    assert len(tasks_for(configuration())) == 360
    assert runner.main([
        "--conditions", "0", "1", "2",
        "--censoring-levels", "0.25", "0.5", "0.75",
        "--gpu-ids", "4", "5", "--dry-run",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["task_count"] == 360
    args = runner.build_parser().parse_args([])
    assert args.conditions == [1] and args.censoring_levels == [0.5]
    assert not hasattr(args, "arms")


def test_main_device_contract():
    config = configuration()
    assert task_gpu_ids("otsurv", config) == [4, 5]
    assert task_gpu_ids("rawcnn", config) == [4]
    assert task_gpu_ids("coxph", config) == []
    assert cpu_threads("otsurv") == 37
    assert cpu_threads("coxph") == 18

    config["gpu_ids"] = [4]
    with pytest.raises(ValueError, match="exactly two"):
        tasks_for(config)
    config["device"] = "cpu"
    assert len(tasks_for(config)) == 360


def test_source_hashes_cover_existing_core_files():
    hashes = source_hashes(ROOT)
    assert "src/paper/reference.json" in hashes
    assert all((ROOT / name).is_file() and len(digest) == 64 for name, digest in hashes.items())


def test_fpt_cli_has_only_path_options(monkeypatch, tmp_path):
    import generate_fpt_labels as runner

    parser = runner.build_parser()
    assert {action.dest for action in parser._actions} == {"help", "data_dir", "output_dir"}
    calls = []
    monkeypatch.setattr("fpt_detection.generation.replay_labels",
                        lambda output, data: calls.append((output, data)))
    assert runner.main(["--data-dir", str(tmp_path / "data"),
                        "--output-dir", str(tmp_path / "labels")]) == 0
    assert calls == [(tmp_path / "labels", tmp_path / "data")]
