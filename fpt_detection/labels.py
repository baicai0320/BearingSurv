"""The single, hash-bound XJTU FPTv2-A coverage label contract.

Historical scores, not reference labels, drive detector replay. Reference
labels are used only after detection to reject changes to the paper endpoint.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg
from fpt_detection.coverage import (
    FPTV2CoverageRescuePolicy,
    detect_fpt_grouped_consensus_with_coverage,
)
from fpt_detection.detect_fpt_v2 import FPTV2DetectorConfig


REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
REFERENCE_LABEL_SHA256 = "e4ee7495a59daa674afcd883fae8f083a250ab469ffc1a2f77bda58610c9dd4e"
REFERENCE_MANIFEST_SHA256 = "992050fc4ace8481c5aa91ef907dac343ced71fe6bdf0e228ea0ead3f83ea24f"
REFERENCE_RAW_SHA256 = "8e2033d140ad0662821efdc9b9e858f1c2c0fd2ebbdee3fb491ae7fb2f835c64"
DETECTOR_VERSION = "fpt_v2_a_coverage"
LABEL_COLUMNS = (
    "outer_fold", "outer_held_out_bearing", "condition", "condition_id",
    "bearing_id", "role", "phase", "scheme", "seed", "training_bearings",
    "n_samples", "fpt_idx", "event_observed", "label_time_idx", "censoring_reason",
    "decision_is_stable", "eventual_is_stable", "detection_delay", "fusion_status",
    "selected_profile_sha256", "score_cache_key", "score_frame_sha256",
)
DECISION_COLUMNS = (
    "outer_fold", "outer_held_out_bearing", "condition", "condition_id",
    "bearing_id", "role", "phase", "scheme", "seed", "training_bearings",
    "n_samples", "fpt_idx", "event_observed", "label_time_idx", "censoring_reason",
    "decision_is_stable", "eventual_is_stable", "detection_delay", "fusion_status",
)


class FPTLabelContractError(ValueError):
    """The artifact is not the frozen downstream FPT source."""


def json_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_bytes(rows, columns=LABEL_COLUMNS) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LABEL_COLUMNS:
            raise FPTLabelContractError("FPT CSV columns must match the canonical ordered contract")
        return list(reader)


@dataclass(frozen=True)
class FPTTask:
    outer_fold: int
    condition_id: int
    bearing_id: int

    def __post_init__(self):
        for value in (self.outer_fold, self.condition_id, self.bearing_id):
            if isinstance(value, bool) or not isinstance(value, int):
                raise FPTLabelContractError("FPT task indices must be integers")
        if (self.outer_fold not in cfg.BEARING_IDS or self.bearing_id not in cfg.BEARING_IDS
                or self.condition_id not in cfg.CONDITIONS):
            raise FPTLabelContractError("FPT task is outside the fixed XJTU protocol")

    @property
    def key(self) -> str:
        return f"fold{self.outer_fold}_c{self.condition_id}_b{self.bearing_id}"

    @property
    def training_bearings(self) -> tuple[int, ...]:
        return tuple(b for b in cfg.BEARING_IDS if b not in {self.outer_fold, self.bearing_id})

    @property
    def role(self) -> str:
        return "held_out" if self.bearing_id == self.outer_fold else "development"


def fixed_tasks() -> tuple[FPTTask, ...]:
    return tuple(FPTTask(fold, condition, bearing) for fold in cfg.BEARING_IDS
                 for condition in cfg.CONDITIONS for bearing in cfg.BEARING_IDS)


def legacy_profile(outer_fold: int) -> dict:
    profile = cfg.FPT_V2_A_COVERAGE_PROFILES[outer_fold]
    return {
        "schema_version": 1, "shared": profile["shared"],
        "global_detector": profile["detector"],
        "conditions": {name: {} for name in cfg.CONDITION_NAMES},
    }


class FPTReference:
    """Read the portable, non-pickle score bundle exported from main."""

    def __init__(self, directory: Path = REFERENCE_DIR):
        self.directory = Path(directory).resolve()
        self.manifest_path = self.directory / "reference.json"
        self.sha256 = file_digest(self.manifest_path)
        if self.sha256 != REFERENCE_MANIFEST_SHA256:
            raise FPTLabelContractError("reference manifest differs from the frozen main score bundle")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("artifact_type") != "fpt_v2_a_coverage_score_reference":
            raise FPTLabelContractError("unexpected reference bundle type")
        if self.manifest.get("schema_version") != 1:
            raise FPTLabelContractError("unsupported reference bundle version")
        if self.manifest["labels"]["sha256"] != REFERENCE_LABEL_SHA256:
            raise FPTLabelContractError("reference is not the main paper label artifact")
        reference_path = self.directory / "labels.csv"
        if file_digest(reference_path) != REFERENCE_LABEL_SHA256:
            raise FPTLabelContractError("reference label CSV was changed")
        self.rows = read_csv_rows(reference_path)
        self.entries = self.manifest["entries"]
        if set(self.entries) != {task.key for task in fixed_tasks()}:
            raise FPTLabelContractError("reference must contain all 75 score trajectories")
        if cfg.FPT_RANDOM_SEED != 42 or self.manifest.get("seed") != 42:
            raise FPTLabelContractError("official FPT seed is fixed at 42")
        if self.manifest["fpt_configuration_sha256"] != json_digest(cfg.fpt_configuration_payload()):
            raise FPTLabelContractError("FPT configuration differs from the frozen reference")
        for task in fixed_tasks():
            entry = self.entries[task.key]
            if entry["training_bearings"] != list(task.training_bearings):
                raise FPTLabelContractError(f"invalid training provenance: {task.key}")
            if entry["selected_profile_sha256"] != json_digest(legacy_profile(task.outer_fold)):
                raise FPTLabelContractError(f"invalid fold profile: {task.key}")

    def scores(self, task: FPTTask) -> pd.DataFrame:
        entry = self.entries[task.key]
        path = self.directory / "scores" / f"{task.key}.npz"
        if file_digest(path) != entry["npz_sha256"]:
            raise FPTLabelContractError(f"score file digest mismatch: {task.key}")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(entry["columns"]):
                raise FPTLabelContractError(f"score schema mismatch: {task.key}")
            frame = pd.DataFrame({name: arrays[name] for name in entry["columns"]})
        validate_scores(frame, entry["n_samples"])
        return frame


def validate_scores(frame: pd.DataFrame, n_samples: int) -> None:
    required = {f"{kind}_{group}" for group in cfg.FPT_FEATURE_GROUPS
                for kind in ("reconstruction", "epistemic")} | {"timeline"}
    if set(frame.columns) != required or len(frame) != n_samples:
        raise FPTLabelContractError("score trajectory shape/schema mismatch")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise FPTLabelContractError("score trajectory contains non-finite values")
    if np.any(np.diff(frame["timeline"].to_numpy()) <= 0):
        raise FPTLabelContractError("score timeline must be strictly increasing")


def detect_task(task: FPTTask, scores: pd.DataFrame) -> dict:
    profile = cfg.FPT_V2_A_COVERAGE_PROFILES[task.outer_fold]
    return detect_fpt_grouped_consensus_with_coverage(
        scores, groups=cfg.FPT_FEATURE_GROUPS,
        primary_config=FPTV2DetectorConfig(**profile["detector"]),
        rescue_policy=FPTV2CoverageRescuePolicy(**cfg.FPT_V2_A_COVERAGE_RESCUE),
    )


def label_row(task: FPTTask, scores: pd.DataFrame, result: dict, provenance: dict) -> dict:
    """Convert a detector decision, never a reference label, into a survival label."""
    fpt = result.get("fpt_idx")
    index = -1 if fpt is None else int(fpt)
    if index < -1 or index >= len(scores):
        raise FPTLabelContractError("detector FPT is outside its trajectory")
    delay = result.get("detection_delay")
    return {
        "outer_fold": task.outer_fold, "outer_held_out_bearing": task.outer_fold,
        "condition": cfg.CONDITION_NAMES[task.condition_id], "condition_id": task.condition_id,
        "bearing_id": task.bearing_id, "role": task.role,
        "phase": "outer" if task.role == "held_out" else "inner", "scheme": "A",
        "seed": cfg.FPT_RANDOM_SEED,
        "training_bearings": "+".join(map(str, task.training_bearings)),
        "n_samples": len(scores), "fpt_idx": index, "event_observed": index >= 0,
        "label_time_idx": index if index >= 0 else len(scores) - 1,
        "censoring_reason": "none" if index >= 0 else "fpt_not_detected",
        "decision_is_stable": bool(result.get("decision_is_stable", False)),
        "eventual_is_stable": bool(result.get("eventual_is_stable", False)),
        "detection_delay": -1 if delay is None else int(delay),
        "fusion_status": str(result.get("fusion_status", "unknown")),
        "selected_profile_sha256": json_digest(legacy_profile(task.outer_fold)),
        "score_cache_key": provenance["score_cache_key"],
        "score_frame_sha256": provenance["score_frame_sha256"],
    }


def compare_rows(actual: list[dict], expected: list[dict], columns=DECISION_COLUMNS) -> dict:
    def index(rows):
        indexed = {}
        for row in rows:
            key = tuple(int(row[name]) for name in ("outer_fold", "condition_id", "bearing_id"))
            if key in indexed:
                raise FPTLabelContractError(f"duplicate comparison row: {key}")
            indexed[key] = row
        return indexed
    left, right = index(actual), index(expected)
    differences = []
    for key in sorted(set(left) | set(right)):
        if key not in left or key not in right:
            differences.append({"key": key, "field": "row", "actual": key in left, "expected": key in right})
            continue
        for field in columns:
            if str(left[key][field]) != str(right[key][field]):
                differences.append({"key": key, "field": field,
                                    "actual": left[key][field], "expected": right[key][field]})
    failed_keys = {tuple(item["key"]) for item in differences}
    return {"passed": not differences, "rows": len(left), "matching_rows": len(set(left) & set(right) - failed_keys),
            "differences": differences}


def validate_raw_identity(identity) -> None:
    # Frozen scores cannot legitimately be paired with a different raw corpus.
    expected = {"file_count": 9216, "sha256": REFERENCE_RAW_SHA256}
    if identity != expected:
        raise FPTLabelContractError("raw input identity differs from the registered XJTU reference corpus")


def validate_generation(generation) -> None:
    if not isinstance(generation, dict):
        raise FPTLabelContractError("FPT generation provenance is missing")
    validate_raw_identity(generation.get("raw_data"))
    expected = {"mode": "frozen_score_replay", "seed": 42, "device": "cpu",
                "groupvae_training_invoked": False,
                "comparison": {"passed": True, "rows": 75, "matching_rows": 75, "differences": []}}
    if any(generation.get(key) != value for key, value in expected.items()):
        raise FPTLabelContractError("FPT generation provenance does not certify an exact detector replay")


def write_artifact(directory: Path, rows: list[dict], reference: FPTReference,
                   *, generation: dict, diagnostics: list[dict]) -> Path:
    """Publish only a complete, byte-exact replay of the main label source."""
    payload = csv_bytes(rows)
    if sha256(payload).hexdigest() != REFERENCE_LABEL_SHA256:
        raise FPTLabelContractError("generated labels differ from the frozen main CSV; publication refused")
    validate_generation(generation)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    label_path = directory / cfg.FPT_LABELS_PATH.name
    label_path.write_bytes(payload)
    manifest = {
        "artifact_type": "fpt_v2_a_coverage_labels", "schema_version": 2,
        "dataset": cfg.DATASET_NAME, "detector_version": DETECTOR_VERSION,
        "label_rows": 75, "label_columns": list(LABEL_COLUMNS),
        "labels": {"filename": label_path.name, "sha256": REFERENCE_LABEL_SHA256},
        "reference_sha256": reference.sha256,
        "fpt_configuration": cfg.fpt_configuration_payload(),
        "fpt_configuration_sha256": json_digest(cfg.fpt_configuration_payload()),
        "generation": generation, "decisions": diagnostics,
        "raw_data": generation.get("raw_data"),
        "source": "main_frozen_scores_seed42",
    }
    manifest_path = directory / cfg.FPT_MANIFEST_PATH.name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    FPTLabelStore(manifest_path)
    return manifest_path


class FPTLabelStore:
    """The only downstream consumer; arbitrary regenerated labels fail closed."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest = self.manifest
        reference = FPTReference()
        required = {
            "artifact_type": "fpt_v2_a_coverage_labels", "schema_version": 2,
            "dataset": "xjtu", "detector_version": DETECTOR_VERSION,
            "label_rows": 75, "source": "main_frozen_scores_seed42",
            "reference_sha256": reference.sha256,
            "fpt_configuration_sha256": json_digest(cfg.fpt_configuration_payload()),
        }
        if any(manifest.get(key) != value for key, value in required.items()):
            raise FPTLabelContractError("FPT manifest does not identify the frozen main source")
        if manifest.get("label_columns") != list(LABEL_COLUMNS):
            raise FPTLabelContractError("FPT manifest columns differ from the canonical contract")
        if json_digest(manifest.get("fpt_configuration")) != required["fpt_configuration_sha256"]:
            raise FPTLabelContractError("FPT manifest configuration payload was changed")
        validate_raw_identity(manifest.get("raw_data"))
        validate_generation(manifest.get("generation"))
        expected = {"filename": cfg.FPT_LABELS_PATH.name, "sha256": REFERENCE_LABEL_SHA256}
        if manifest.get("labels") != expected:
            raise FPTLabelContractError("only the frozen main FPT CSV can be used for survival analysis")
        self.labels_path = self.manifest_path.parent / expected["filename"]
        self.csv_sha256 = file_digest(self.labels_path)
        if self.csv_sha256 != REFERENCE_LABEL_SHA256:
            raise FPTLabelContractError("FPT CSV differs from its immutable reference hash")
        self.manifest_sha256 = file_digest(self.manifest_path)
        self.labels = pd.read_csv(self.labels_path)
        # The byte hash fixes every row, seed, role, profile and training peer.
        if not compare_rows(read_csv_rows(self.labels_path), reference.rows, LABEL_COLUMNS)["passed"]:
            raise FPTLabelContractError("FPT CSV differs from the reference rows")

    def labels_for(self, outer_fold: int, condition: int) -> pd.DataFrame:
        FPTTask(outer_fold, condition, 1)
        rows = self.labels.loc[(self.labels["outer_fold"] == outer_fold)
                               & (self.labels["condition_id"] == condition)].sort_values("bearing_id")
        if tuple(rows["bearing_id"]) != cfg.BEARING_IDS:
            raise FPTLabelContractError("fold does not contain exactly the five protocol bearings")
        return rows.copy()
