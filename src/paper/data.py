"""The paper's feature and censoring protocol, separate from FPT detection."""

from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy

import config as public_config
from fpt_detection.labels import FPTLabelStore, file_digest
from fpt_detection.data_loader import _read_signal, resolve_bearing_directory
from paper.config import MOVING_AVERAGE_LAG


FEATURES = (
    "mean", "std", "skew", "kurtosis", "entropy", "rms", "max", "p2p",
    "crest", "clearence", "shape", "impulse",
)
OUTCOMES = ("bearing_id", "time_step", "Event", "Survival_time", "TrueEvent", "TrueTime")
RECIPE = "main_b0bb808_dual_axis_features_two_csv_roundtrips_v2"


def extract_features(values: np.ndarray, condition: int) -> dict:
    """Preserve main's twelve time-domain numerical definitions.

    In particular, 'clearence' retains its historical definition. Frequency
    features belonged to a separate table and were not baseline covariates.
    """
    frame = pd.DataFrame(np.asarray(values, dtype=np.float64))
    if frame.shape != (32768, 2) or not np.isfinite(frame).all().all():
        raise ValueError("paper features require finite [32768, 2] raw minutes")
    mean = frame.abs().mean().to_numpy()
    rms = np.asarray([np.sqrt((frame[c] ** 2).sum() / len(frame[c])) for c in frame])
    maximum = frame.abs().max().to_numpy()
    statistics = {
        "mean": mean, "std": frame.std().to_numpy(), "skew": frame.skew().to_numpy(),
        "kurtosis": frame.kurtosis().to_numpy(),
        "entropy": np.asarray([entropy(pd.cut(frame[c], 500).value_counts()) for c in frame]),
        "rms": rms, "max": maximum,
        "p2p": (frame.max().abs() + frame.min().abs()).to_numpy(),
        "crest": maximum / rms,
        "clearence": np.asarray([((np.sqrt(frame[c].abs())).sum() / len(frame[c])) ** 2 for c in frame]),
        "shape": rms / mean, "impulse": maximum / mean,
    }
    return {f"{prefix}_{name}": float(statistics[name][channel])
            for prefix, channel in (("h", 0), ("v", 1)) for name in FEATURES}


def prepare_features(condition: int, store: FPTLabelStore) -> Path:
    """Create a hash-checked raw-derived cache without replacing older tables."""
    source_hash = file_digest(Path(__file__))
    root = public_config.OUTPUT_DIR / "paper_features" / source_hash[:12] / f"C{condition + 1}"
    path, manifest_path = root / "features.csv", root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["recipe"] != RECIPE or manifest["source_sha256"] != source_hash
                or manifest["raw_data"] != store.manifest["raw_data"]
                or file_digest(path) != manifest["sha256"]):
            raise ValueError("paper feature cache differs from its input or source manifest")
        return path
    if root.exists():
        raise FileExistsError(f"incomplete feature cache must be inspected before retry: {root}")
    root.mkdir(parents=True)
    rows = []
    labels = store.labels_for(1, condition).set_index("bearing_id")
    for bearing in public_config.BEARING_IDS:
        directory = resolve_bearing_directory(public_config.DATA_DIR, condition, bearing)
        count = int(labels.loc[bearing, "n_samples"])
        for step in range(count):
            raw_path = directory / f"{step + 1}.csv"
            row = {"condition_id": condition, "bearing_id": bearing, "time_step": step}
            row.update(extract_features(_read_signal(raw_path), condition))
            rows.append(row)
        print(f"[paper features] C{condition + 1} B{bearing}: {count} raw minutes", flush=True)
    # main wrote per-bearing CSVs, read them during aggregation, then wrote
    # the covariate CSV. DSM's double-precision fit is sensitive to both passes.
    intermediate = pd.read_csv(io.StringIO(pd.DataFrame(rows).to_csv(index=False)))
    intermediate.to_csv(path, index=False)
    manifest = {"recipe": RECIPE, "source_sha256": source_hash, "sha256": file_digest(path),
                "condition": condition, "raw_data": store.manifest["raw_data"], "rows": len(rows)}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def outcome_hash(frame: pd.DataFrame) -> str:
    canonical = frame.loc[:, OUTCOMES].copy()
    canonical["Event"] = canonical["Event"].astype(np.int8)
    canonical["TrueEvent"] = canonical["TrueEvent"].astype(np.int8)
    canonical = canonical.sort_values(["bearing_id", "time_step"], kind="mergesort")
    return sha256(canonical.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()).hexdigest()


def build_fold(condition: int, fold: int, level: float, store: FPTLabelStore,
               feature_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Consume global NumPy RNG in exactly main's per-bearing order."""
    labels = store.labels_for(fold, condition).set_index("bearing_id")
    features = pd.read_csv(feature_path)
    frames = []
    for bearing in public_config.BEARING_IDS:
        raw = features.loc[features.bearing_id.eq(bearing)].sort_values("time_step").reset_index(drop=True)
        label = labels.loc[bearing]
        if len(raw) != int(label["n_samples"]):
            raise ValueError("paper feature length does not match the sole FPT source")
        boundary = int(label["label_time_idx"])
        raw = raw.iloc[:boundary + 1]
        frame = raw[["h_" + name for name in FEATURES]].copy()
        frame.columns = list(FEATURES)
        frame = frame.rolling(public_config.MOVING_AVERAGE_WINDOW_BY_CONDITION[condition]).mean().shift(MOVING_AVERAGE_LAG[condition])
        valid = frame.notna().all(axis=1)
        frame = frame.loc[valid].copy()
        frame["time_step"] = raw.loc[valid, "time_step"].to_numpy(dtype=int)
        frame["Survival_time"] = boundary - frame["time_step"] + 1
        frame["Event"] = bool(label["event_observed"])
        frame["bearing_id"] = bearing
        frame = frame.reset_index(drop=True)
        frame["TrueTime"] = frame["Survival_time"]
        frame["TrueEvent"] = frame["Event"].astype(bool)
        censored = frame.sample(frac=level, random_state=0).copy()
        rest = frame.loc[~frame.index.isin(censored.index)]
        censored["Survival_time"] = censored.apply(lambda row: np.random.randint(0, row["Survival_time"]), axis=1)
        censored.loc[censored.Survival_time.eq(0), "Survival_time"] = 1
        censored["Event"] = False
        frames.append(pd.concat([censored, rest]).reset_index(drop=True).sample(frac=1, random_state=0))
    data = pd.concat(frames, axis=0)
    development = data.loc[data.bearing_id.ne(fold)].reset_index(drop=True)
    test = data.loc[data.bearing_id.eq(fold)].reset_index(drop=True)
    if set(development.bearing_id) != set(public_config.BEARING_IDS) - {fold} or set(test.bearing_id) != {fold}:
        raise ValueError("outer bearing isolation failed")
    return development, test
