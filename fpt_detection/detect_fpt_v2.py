"""Causal and robust first-predicting-time detection for fixed FPT labels."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_GROUPS = ("FTF", "Fr", "BSF", "BPFO", "BPFI")


@dataclass(frozen=True)
class FPTV2DetectorConfig:
    """Configuration for score calibration, CUSUM, and band fusion."""

    healthy_window: int = 10
    smoothing_window: int = 5
    drift_sigma: float = 3.0
    threshold_sigma: float = 20.0
    persistence: int = 2
    score_mode: str = "composite"
    reconstruction_weight: float = 1.0
    epistemic_weight: float = 1.0
    log_transform: bool = True
    z_clip: float = 25.0
    fusion: str = "median"
    min_triggered_groups: int = 3
    consensus_mode: str = "onset_cluster"
    consensus_absolute_spread: float = 10.0
    consensus_relative_spread: float = 0.15
    max_relative_mad: float = 0.10

    def validate(self) -> None:
        if self.healthy_window < 5:
            raise ValueError("healthy_window must be at least 5")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be positive")
        if self.drift_sigma < 0 or self.threshold_sigma <= 0:
            raise ValueError("CUSUM drift/threshold must be non-negative/positive")
        if self.persistence < 1:
            raise ValueError("persistence must be positive")
        if self.score_mode not in {"composite", "reconstruction", "epistemic"}:
            raise ValueError(f"unsupported score_mode: {self.score_mode}")
        if self.reconstruction_weight < 0 or self.epistemic_weight < 0:
            raise ValueError("score weights cannot be negative")
        if self.score_mode == "composite" and (
            self.reconstruction_weight + self.epistemic_weight <= 0
        ):
            raise ValueError("at least one composite score weight must be positive")
        if self.z_clip <= 0:
            raise ValueError("z_clip must be positive")
        if self.fusion not in {"median", "trimmed_mean"}:
            raise ValueError(f"unsupported fusion: {self.fusion}")
        if self.min_triggered_groups < 1:
            raise ValueError("min_triggered_groups must be positive")
        if self.consensus_mode not in {"onset_cluster", "alarm_quorum"}:
            raise ValueError(f"unsupported consensus_mode: {self.consensus_mode}")
        if self.consensus_absolute_spread < 0:
            raise ValueError("consensus_absolute_spread cannot be negative")
        if self.consensus_relative_spread < 0:
            raise ValueError("consensus_relative_spread cannot be negative")
        if (
            self.consensus_mode == "onset_cluster"
            and self.consensus_absolute_spread == 0
            and self.consensus_relative_spread == 0
        ):
            raise ValueError("onset-cluster consensus requires a positive spread limit")
        if self.max_relative_mad < 0:
            raise ValueError("max_relative_mad cannot be negative")


def causal_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Return a trailing moving average that never reads future values."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if window < 1:
        raise ValueError("window must be positive")
    if not np.all(np.isfinite(x)):
        raise ValueError("values contain NaN or infinity")
    if window == 1 or x.size == 0:
        return x.copy()

    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    result = np.empty_like(x)
    for idx in range(x.size):
        start = max(0, idx - window + 1)
        result[idx] = (cumulative[idx + 1] - cumulative[start]) / (idx - start + 1)
    return result


def robust_healthy_standardize(values: np.ndarray, healthy_window: int,
                               z_clip: float) -> tuple[np.ndarray, dict]:
    """Standardize from the healthy prefix using median/MAD and IQR only."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if x.size <= healthy_window:
        raise ValueError(
            f"sequence length {x.size} must exceed healthy_window {healthy_window}"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("values contain NaN or infinity")

    baseline = x[:healthy_window]
    center = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - center)))
    q25, q75 = np.percentile(baseline, [25, 75])
    mad_scale = 1.4826 * mad
    iqr_scale = float(q75 - q25) / 1.349
    magnitude_floor = max(abs(center), float(np.max(np.abs(baseline))), 1.0) * 1e-8
    scale = max(mad_scale, iqr_scale, magnitude_floor)

    z = np.clip((x - center) / scale, -z_clip, z_clip)
    return z, {
        "center": center,
        "mad": mad,
        "iqr": float(q75 - q25),
        "scale": float(scale),
        "z_clip": float(z_clip),
    }


def _prepare_component(values: np.ndarray,
                       config: FPTV2DetectorConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    raw = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(raw)):
        raise ValueError("score component contains NaN or infinity")
    if np.any(raw < 0):
        raise ValueError("reconstruction and uncertainty scores must be non-negative")

    transformed = np.log1p(raw) if config.log_transform else raw.copy()
    smoothed = causal_moving_average(transformed, config.smoothing_window)
    standardized, baseline = robust_healthy_standardize(
        smoothed, config.healthy_window, config.z_clip
    )
    return smoothed, standardized, baseline


def _combine_group_score(features_df: pd.DataFrame, group: str,
                         config: FPTV2DetectorConfig) -> tuple[np.ndarray, dict]:
    recon_col = f"reconstruction_{group}"
    epistemic_col = f"epistemic_{group}"
    required = []
    if config.score_mode in {"composite", "reconstruction"}:
        required.append(recon_col)
    if config.score_mode in {"composite", "epistemic"}:
        required.append(epistemic_col)
    missing = [column for column in required if column not in features_df.columns]
    if missing:
        raise ValueError(f"missing FPT v2 score columns: {missing}")

    recon_smooth = recon_z = None
    epistemic_smooth = epistemic_z = None
    recon_baseline = epistemic_baseline = None
    if recon_col in required:
        recon_smooth, recon_z, recon_baseline = _prepare_component(
            features_df[recon_col].to_numpy(), config
        )
    if epistemic_col in required:
        epistemic_smooth, epistemic_z, epistemic_baseline = _prepare_component(
            features_df[epistemic_col].to_numpy(), config
        )

    if config.score_mode == "reconstruction":
        score_z = recon_z
    elif config.score_mode == "epistemic":
        score_z = epistemic_z
    else:
        total_weight = config.reconstruction_weight + config.epistemic_weight
        score_z = (
            config.reconstruction_weight * recon_z
            + config.epistemic_weight * epistemic_z
        ) / total_weight

    return score_z, {
        "reconstruction_raw": (
            features_df[recon_col].to_numpy(dtype=float)
            if recon_col in features_df.columns else None
        ),
        "reconstruction_smoothed": recon_smooth,
        "reconstruction_z": recon_z,
        "reconstruction_baseline": recon_baseline,
        "epistemic_raw": (
            features_df[epistemic_col].to_numpy(dtype=float)
            if epistemic_col in features_df.columns else None
        ),
        "epistemic_smoothed": epistemic_smooth,
        "epistemic_z": epistemic_z,
        "epistemic_baseline": epistemic_baseline,
        "score_z": score_z,
    }


def cusum_backtracked_fpt(score_z: np.ndarray, healthy_window: int,
                          drift_sigma: float, threshold_sigma: float,
                          persistence: int = 1) -> dict:
    """Run one-sided CUSUM and backtrack from a persistent alarm."""
    score = np.asarray(score_z, dtype=np.float64)
    if score.ndim != 1 or not np.all(np.isfinite(score)):
        raise ValueError("score_z must be a finite one-dimensional array")
    if score.size <= healthy_window:
        raise ValueError("score sequence is shorter than the healthy prefix")

    cusum = np.zeros(score.size, dtype=np.float64)
    above_count = 0
    first_above = None
    alarm_idx: Optional[int] = None

    for idx in range(healthy_window, score.size):
        cusum[idx] = max(0.0, cusum[idx - 1] + score[idx] - drift_sigma)
        if cusum[idx] > threshold_sigma:
            if above_count == 0:
                first_above = idx
            above_count += 1
            if above_count >= persistence:
                alarm_idx = idx
                break
        else:
            above_count = 0
            first_above = None

    fpt_idx: Optional[int] = None
    alarm_strength = 0.0
    if alarm_idx is not None:
        search_end = first_above if first_above is not None else alarm_idx
        resets = np.flatnonzero(cusum[:search_end] <= 1e-12)
        fpt_idx = int(resets[-1] + 1) if resets.size else int(healthy_window)
        alarm_strength = float(cusum[alarm_idx] / threshold_sigma)

    return {
        "fpt_idx": fpt_idx,
        "alarm_idx": alarm_idx,
        "alarm_strength": alarm_strength,
        "cusum_curve": cusum,
    }


def _trimmed_mean(values: np.ndarray) -> float:
    ordered = np.sort(values)
    if ordered.size >= 5:
        ordered = ordered[1:-1]
    return float(np.mean(ordered))


def _onset_spread(values: np.ndarray,
                  config: FPTV2DetectorConfig) -> tuple[float, float, float]:
    median = float(np.median(values))
    spread = float(np.max(values) - np.min(values))
    allowed = max(
        float(config.consensus_absolute_spread),
        float(config.consensus_relative_spread) * max(abs(median), 1.0),
    )
    return median, spread, allowed


def _band_dispersion(results: Dict[str, dict]) -> dict:
    """Summarize onset dispersion for a named set of triggered bands."""
    if not results:
        return {
            "median": float("nan"),
            "mad": float("nan"),
            "iqr": float("nan"),
            "range": float("nan"),
            "relative_mad": float("nan"),
            "relative_iqr": float("nan"),
            "relative_range": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
            "max_min_ratio": float("nan"),
        }

    values = np.asarray(
        [result["fpt_idx"] for result in results.values()], dtype=np.float64
    )
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    q25, q75 = np.percentile(values, [25, 75])
    iqr = float(q75 - q25)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    value_range = maximum - minimum
    denominator = max(abs(median), 1.0)
    return {
        "median": median,
        "mad": mad,
        "iqr": iqr,
        "range": value_range,
        "relative_mad": mad / denominator,
        "relative_iqr": iqr / denominator,
        "relative_range": value_range / denominator,
        "minimum": minimum,
        "maximum": maximum,
        "max_min_ratio": (
            maximum / minimum if minimum > 0.0 else float("nan")
        ),
    }


def _first_onset_cluster(
    all_triggered: Dict[str, dict],
    config: FPTV2DetectorConfig,
) -> tuple[Optional[int], Dict[str, dict], float]:
    """Return the first alarm-time subset with mutually consistent onsets."""
    if len(all_triggered) < config.min_triggered_groups:
        return None, all_triggered, float("nan")

    active: Dict[str, dict] = {}
    alarm_times = sorted({
        int(result["alarm_idx"]) for result in all_triggered.values()
    })
    for alarm_idx in alarm_times:
        for group, result in all_triggered.items():
            if int(result["alarm_idx"]) == alarm_idx:
                active[group] = result

        candidates = []
        for names in combinations(sorted(active), config.min_triggered_groups):
            values = np.asarray(
                [active[name]["fpt_idx"] for name in names], dtype=np.float64
            )
            median, spread, allowed = _onset_spread(values, config)
            if spread <= allowed + np.finfo(float).eps:
                relative_spread = spread / max(abs(median), 1.0)
                candidates.append((relative_spread, spread, median, names, allowed))

        if candidates:
            _, _, _, names, allowed = min(candidates)
            selected = {name: active[name] for name in names}
            return alarm_idx, selected, float(allowed)

    return None, all_triggered, float("nan")


def fuse_group_fpts(group_results: Dict[str, dict], n_samples: int,
                    config: FPTV2DetectorConfig) -> dict:
    """Freeze fusion when the first causal multi-band consensus is reached."""
    del n_samples
    all_triggered = {
        group: result for group, result in group_results.items()
        if result["fpt_idx"] is not None
    }
    for group, result in all_triggered.items():
        if result.get("alarm_idx") is None:
            raise ValueError(f"triggered group {group} has no alarm_idx")

    agreement_limit = float("nan")
    if config.consensus_mode == "onset_cluster":
        consensus_alarm_idx, triggered, agreement_limit = _first_onset_cluster(
            all_triggered, config
        )
    elif len(all_triggered) >= config.min_triggered_groups:
        ordered_alarms = sorted(
            int(result["alarm_idx"]) for result in all_triggered.values()
        )
        consensus_alarm_idx = ordered_alarms[config.min_triggered_groups - 1]
        triggered = {
            group: result for group, result in all_triggered.items()
            if int(result["alarm_idx"]) <= consensus_alarm_idx
        }
    else:
        consensus_alarm_idx = None
        triggered = all_triggered

    values = np.asarray(
        [result["fpt_idx"] for result in triggered.values()], dtype=np.float64
    )
    coverage = len(triggered) / max(len(group_results), 1)
    eventual_coverage = len(all_triggered) / max(len(group_results), 1)
    decision_dispersion = _band_dispersion(triggered)
    eventual_dispersion = _band_dispersion(all_triggered)
    median = decision_dispersion["median"]

    eventual_outlier_groups = []
    if all_triggered:
        eventual_median = eventual_dispersion["median"]
        eventual_allowed = max(
            float(config.consensus_absolute_spread),
            float(config.consensus_relative_spread)
            * max(abs(eventual_median), 1.0),
        )
        eventual_outlier_groups = [
            group for group, result in all_triggered.items()
            if abs(float(result["fpt_idx"]) - eventual_median)
            > eventual_allowed + np.finfo(float).eps
        ]
    else:
        eventual_allowed = float("nan")
    untriggered_groups = [
        group for group in group_results if group not in all_triggered
    ]

    global_fpt: Optional[int]
    if consensus_alarm_idx is None:
        global_fpt = None
        status = (
            "insufficient_groups"
            if len(all_triggered) < config.min_triggered_groups
            else "insufficient_consensus"
        )
    else:
        if config.fusion == "median":
            fused = median
        elif config.fusion == "trimmed_mean":
            fused = _trimmed_mean(values)
        else:  # ``FPTV2DetectorConfig.validate`` rejects other fusion modes.
            raise RuntimeError(f"unsupported fixed FPT fusion: {config.fusion}")
        global_fpt = int(round(fused))
        # Decision stability belongs to the causal quorum that fixed the FPT.
        # Later band events are retained in the eventual_* audit fields, but
        # they cannot retrospectively change an already frozen online decision.
        status = (
            "stable"
            if decision_dispersion["relative_mad"] <= config.max_relative_mad
            else "dispersed"
        )

    decision_is_stable = status == "stable"
    eventual_mad_is_stable = bool(
        global_fpt is not None
        and np.isfinite(eventual_dispersion["relative_mad"])
        and eventual_dispersion["relative_mad"] <= config.max_relative_mad
    )
    all_configured_bands_triggered = bool(
        group_results and len(all_triggered) == len(group_results)
    )
    all_triggered_bands_consistent = bool(
        all_triggered and not eventual_outlier_groups
    )
    eventual_is_stable = bool(
        eventual_mad_is_stable
        and all_configured_bands_triggered
        and all_triggered_bands_consistent
    )
    eventual_fusion_status = (
        "not_detected" if global_fpt is None
        else "incomplete_bands" if not all_configured_bands_triggered
        else "stable" if eventual_is_stable else "dispersed"
    )
    detection_delay = (
        None if global_fpt is None or consensus_alarm_idx is None
        else int(consensus_alarm_idx - global_fpt)
    )

    return {
        "fpt_idx": global_fpt,
        "fusion_status": status,
        # is_stable is retained as the backwards-compatible decision-time flag.
        "is_stable": decision_is_stable,
        "decision_is_stable": decision_is_stable,
        "eventual_mad_is_stable": eventual_mad_is_stable,
        "eventual_is_stable": eventual_is_stable,
        "eventual_fusion_status": eventual_fusion_status,
        "all_configured_bands_triggered": all_configured_bands_triggered,
        "band_coverage": float(coverage),
        "eventual_band_coverage": float(eventual_coverage),
        "consensus_alarm_idx": consensus_alarm_idx,
        "detection_delay": detection_delay,
        "consensus_mode": config.consensus_mode,
        "consensus_allowed_spread": agreement_limit,
        "consensus_reached": consensus_alarm_idx is not None,
        "triggered_groups": list(triggered),
        "all_triggered_groups": list(all_triggered),
        "untriggered_groups": untriggered_groups,
        "eventual_outlier_groups": eventual_outlier_groups,
        "eventual_outlier_allowed_spread": float(eventual_allowed),
        "all_triggered_bands_consistent": all_triggered_bands_consistent,
        "earliest_group": (
            min(triggered, key=lambda g: triggered[g]["fpt_idx"])
            if triggered else None
        ),
        "band_fpt_median": decision_dispersion["median"],
        "band_fpt_mad": decision_dispersion["mad"],
        "band_fpt_iqr": decision_dispersion["iqr"],
        "band_fpt_range": decision_dispersion["range"],
        "band_fpt_relative_mad": decision_dispersion["relative_mad"],
        "band_fpt_relative_iqr": decision_dispersion["relative_iqr"],
        "band_fpt_relative_range": decision_dispersion["relative_range"],
        "eventual_band_fpt_median": eventual_dispersion["median"],
        "eventual_band_fpt_mad": eventual_dispersion["mad"],
        "eventual_band_fpt_iqr": eventual_dispersion["iqr"],
        "eventual_band_fpt_range": eventual_dispersion["range"],
        "eventual_band_fpt_relative_mad": eventual_dispersion["relative_mad"],
        "eventual_band_fpt_relative_iqr": eventual_dispersion["relative_iqr"],
        "eventual_band_fpt_relative_range": eventual_dispersion["relative_range"],
        "eventual_band_fpt_min": eventual_dispersion["minimum"],
        "eventual_band_fpt_max": eventual_dispersion["maximum"],
        "eventual_band_fpt_max_min_ratio": eventual_dispersion["max_min_ratio"],
    }


def detect_fpt_grouped_consensus(
    features_df: pd.DataFrame,
    groups: Iterable[str] = DEFAULT_GROUPS,
    config: Optional[FPTV2DetectorConfig] = None,
) -> dict:
    """Detect band-level FPTs and fuse them using a robust consensus rule."""
    detector_config = config or FPTV2DetectorConfig()
    detector_config.validate()
    group_names = list(groups)
    if not group_names:
        raise ValueError("at least one fault-frequency group is required")
    if len(features_df) <= detector_config.healthy_window:
        raise ValueError("feature sequence is too short for FPT v2")

    group_results = {}
    for group in group_names:
        score_z, diagnostics = _combine_group_score(
            features_df, group, detector_config
        )
        result = cusum_backtracked_fpt(
            score_z=score_z,
            healthy_window=detector_config.healthy_window,
            drift_sigma=detector_config.drift_sigma,
            threshold_sigma=detector_config.threshold_sigma,
            persistence=detector_config.persistence,
        )
        result["diag"] = diagnostics
        group_results[group] = result

    fused = fuse_group_fpts(group_results, len(features_df), detector_config)
    fused.update({
        "detector_version": "fpt_v2",
        "n_samples": int(len(features_df)),
        "group_results": group_results,
        "config": asdict(detector_config),
    })
    return fused
