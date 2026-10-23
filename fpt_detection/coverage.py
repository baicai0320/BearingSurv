"""Coverage rescue for the fixed primary FPT detector.

The frozen primary detector remains the decision maker. This module may only
replace an ``insufficient_groups`` all-band miss with a new, stable three-band
consensus. Existing primary detections are returned unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Optional

import pandas as pd

from fpt_detection.detect_fpt_v2 import (
    DEFAULT_GROUPS,
    FPTV2DetectorConfig,
    detect_fpt_grouped_consensus,
)


PRIMARY_ALL_BAND_MISS_STATUS = "insufficient_groups"


@dataclass(frozen=True)
class FPTV2CoverageRescuePolicy:
    """A stricter secondary rule for a primary all-band miss.

    The policy keeps the primary healthy-prefix calibration, causal score
    transform, and consensus mode.  It only makes the secondary detector more
    responsive to reconstruction shocks, and requires three supporting bands
    before a rescue is accepted.
    """

    reconstruction_weight: float = 0.75
    epistemic_weight: float = 0.25
    smoothing_window: int = 7
    drift_sigma: float = 4.0
    threshold_sigma: float = 10.0
    persistence: int = 1
    fusion: str = "trimmed_mean"
    min_triggered_groups: int = 3
    consensus_absolute_spread: float = 5.0
    consensus_relative_spread: float = 0.15
    primary_miss_status: str = PRIMARY_ALL_BAND_MISS_STATUS

    def validate(self) -> None:
        """Reject policies that could weaken the primary safety boundary."""
        if self.primary_miss_status != PRIMARY_ALL_BAND_MISS_STATUS:
            raise ValueError(
                "coverage rescue may only run after an all-band "
                "insufficient_groups miss"
            )
        if self.min_triggered_groups < 3:
            raise ValueError("coverage rescue requires at least three bands")

        probe = FPTV2DetectorConfig(
            healthy_window=5,
            smoothing_window=int(self.smoothing_window),
            drift_sigma=float(self.drift_sigma),
            threshold_sigma=float(self.threshold_sigma),
            persistence=int(self.persistence),
            score_mode="composite",
            reconstruction_weight=float(self.reconstruction_weight),
            epistemic_weight=float(self.epistemic_weight),
            log_transform=True,
            z_clip=25.0,
            fusion=str(self.fusion),
            min_triggered_groups=int(self.min_triggered_groups),
            consensus_mode="onset_cluster",
            consensus_absolute_spread=float(self.consensus_absolute_spread),
            consensus_relative_spread=float(self.consensus_relative_spread),
            max_relative_mad=0.10,
        )
        probe.validate()

    def build_detector_config(
        self, primary_config: FPTV2DetectorConfig
    ) -> FPTV2DetectorConfig:
        """Build a rescue configuration while retaining primary calibration."""
        self.validate()
        primary_config.validate()
        rescue = replace(
            primary_config,
            smoothing_window=int(self.smoothing_window),
            drift_sigma=float(self.drift_sigma),
            threshold_sigma=float(self.threshold_sigma),
            persistence=int(self.persistence),
            score_mode="composite",
            reconstruction_weight=float(self.reconstruction_weight),
            epistemic_weight=float(self.epistemic_weight),
            fusion=str(self.fusion),
            min_triggered_groups=int(self.min_triggered_groups),
            consensus_absolute_spread=float(self.consensus_absolute_spread),
            consensus_relative_spread=float(self.consensus_relative_spread),
        )
        rescue.validate()
        return rescue


# This is the frozen coverage policy consumed by the public label generator.
FPTV2_A_COVERAGE_POLICY = FPTV2CoverageRescuePolicy()


_COVERAGE_DIAGNOSTIC_FIELDS = (
    "fpt_idx",
    "fusion_status",
    "decision_is_stable",
    "eventual_is_stable",
    "eventual_fusion_status",
    "band_coverage",
    "eventual_band_coverage",
    "consensus_alarm_idx",
    "detection_delay",
    "triggered_groups",
    "all_triggered_groups",
    "untriggered_groups",
    "band_fpt_median",
    "band_fpt_mad",
    "band_fpt_iqr",
    "band_fpt_range",
    "band_fpt_relative_mad",
    "band_fpt_relative_iqr",
    "band_fpt_relative_range",
    "eventual_band_fpt_median",
    "eventual_band_fpt_mad",
    "eventual_band_fpt_iqr",
    "eventual_band_fpt_range",
    "eventual_band_fpt_relative_mad",
    "eventual_band_fpt_relative_iqr",
    "eventual_band_fpt_relative_range",
)


def _diagnostic_snapshot(result: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Keep scalar coverage provenance without retaining score-array payloads."""
    if result is None:
        return {field: None for field in _COVERAGE_DIAGNOSTIC_FIELDS}
    snapshot = {field: result.get(field) for field in _COVERAGE_DIAGNOSTIC_FIELDS}
    snapshot["config"] = result.get("config")
    return snapshot


def _annotate_result(
    result: dict[str, Any],
    primary_result: dict[str, Any],
    policy: FPTV2CoverageRescuePolicy,
    rescue_config: FPTV2DetectorConfig,
    *,
    decision_source: str,
    attempted: bool,
    applied: bool,
    rescue_result: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Attach provenance fields without mutating a primary detector result."""
    output = dict(result)
    output.update({
        "decision_source": decision_source,
        "coverage_rescue_attempted": bool(attempted),
        "coverage_rescue_applied": bool(applied),
        "coverage_primary_fpt_idx": primary_result.get("fpt_idx"),
        "coverage_primary_fusion_status": str(
            primary_result.get("fusion_status", "unknown")
        ),
        "coverage_rescue_fpt_idx": (
            None if rescue_result is None else rescue_result.get("fpt_idx")
        ),
        "coverage_rescue_fusion_status": (
            None
            if rescue_result is None
            else str(rescue_result.get("fusion_status", "unknown"))
        ),
        "coverage_rescue_policy": asdict(policy),
        "coverage_rescue_config": asdict(rescue_config),
    })
    for prefix, snapshot in (
        ("coverage_primary", _diagnostic_snapshot(primary_result)),
        ("coverage_rescue", _diagnostic_snapshot(rescue_result)),
    ):
        for field, value in snapshot.items():
            output[f"{prefix}_{field}"] = value
    return output


def detect_fpt_grouped_consensus_with_coverage(
    features_df: pd.DataFrame,
    groups: Iterable[str] = DEFAULT_GROUPS,
    primary_config: Optional[FPTV2DetectorConfig] = None,
    rescue_policy: FPTV2CoverageRescuePolicy = FPTV2_A_COVERAGE_POLICY,
) -> dict[str, Any]:
    """Run FPTv2-A first and rescue only an all-band miss.

    A primary detection always wins, including a stable decision with only
    two supporting bands.  A disagreement between triggered bands is also not
    overridden.  The rescue path is available solely when the primary rule
    produced no single-band alarms at all and itself produces a stable,
    three-band consensus.
    """
    detector_config = primary_config or FPTV2DetectorConfig()
    rescue_config = rescue_policy.build_detector_config(detector_config)
    group_names = tuple(groups)
    primary = detect_fpt_grouped_consensus(
        features_df, groups=group_names, config=detector_config
    )

    if primary.get("fpt_idx") is not None:
        return _annotate_result(
            dict(primary), primary, rescue_policy, rescue_config,
            decision_source="primary",
            attempted=False,
            applied=False,
            rescue_result=None,
        )

    if primary.get("fusion_status") != rescue_policy.primary_miss_status:
        return _annotate_result(
            dict(primary), primary, rescue_policy, rescue_config,
            decision_source="primary_uncovered",
            attempted=False,
            applied=False,
            rescue_result=None,
        )

    rescue = detect_fpt_grouped_consensus(
        features_df, groups=group_names, config=rescue_config
    )
    rescue_groups = tuple(rescue.get("triggered_groups", ()))
    accepted = bool(
        rescue.get("fpt_idx") is not None
        and rescue.get("decision_is_stable", False)
        and len(rescue_groups) >= rescue_policy.min_triggered_groups
    )
    if not accepted:
        return _annotate_result(
            dict(primary), primary, rescue_policy, rescue_config,
            decision_source="primary_uncovered",
            attempted=True,
            applied=False,
            rescue_result=dict(rescue),
        )

    rescued = _annotate_result(
        dict(rescue), primary, rescue_policy, rescue_config,
        decision_source="coverage_rescue",
        attempted=True,
        applied=True,
        rescue_result=dict(rescue),
    )
    rescued["detector_version"] = "fpt_v2_a_coverage"
    return rescued


__all__ = [
    "FPTV2CoverageRescuePolicy",
    "FPTV2_A_COVERAGE_POLICY",
    "PRIMARY_ALL_BAND_MISS_STATUS",
    "detect_fpt_grouped_consensus_with_coverage",
]
