"""Formal survival-probability metrics and durable OOF artifacts.

The legacy evaluator remains the source of the project MAE variants.  This
module deliberately keeps formal IPCW metrics separate because the historical
Brier helper silently treats unsupported IPCW contributions as zero.  Every
unsupported metric is instead recorded as ``NaN`` with a machine-readable
reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_ipcw
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

from tools.Evaluations.D_Calibration import d_calibration
from tools.Evaluations.OneCalibration import one_calibration


DEFAULT_IBS_GRID_POINTS = 50
IPCW_TIME_CAP = 10.0
CALIBRATION_HORIZON = IPCW_TIME_CAP
CALIBRATION_BINS = 10
CALIBRATION_METHOD = "one_calibration_dagostino_nam"
CALIBRATION_HORIZON_RULE = "fixed_remaining_life_steps_v1"
IBS_DEFINITION = "graf_ipcw_integrated_brier_fixed_horizon_v1"
IBS_FOLLOWUP_RULE = "fixed_horizon_post_horizon_controls_v1"
CSV_NULL_REP = "null"


@dataclass(frozen=True)
class DevelopmentTimeGrid:
    """A fold-level grid defined entirely from the development cohort."""

    times: np.ndarray
    tau: float
    status: str
    reason: str | None = None
    time_cap: float = IPCW_TIME_CAP

    @property
    def is_valid(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class CanonicalSurvivalCurves:
    """Validated sample-by-time survival curves in ascending time order."""

    times: np.ndarray
    survival: np.ndarray
    status: str
    reason: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class ProbabilityMetricResult:
    """Formal metrics plus explicit per-metric status values."""

    c_index_ipcw: float
    ibs: float
    d_calib_pvalue: float
    c_index_status: str
    ibs_status: str
    d_calib_status: str
    time_grid: DevelopmentTimeGrid

    def result_fields(self) -> dict[str, object]:
        metric_status = ";".join(
            (
                f"CIndexIPCW={self.c_index_status}",
                f"IBS={self.ibs_status}",
                f"DCalibPValue={self.d_calib_status}",
            )
        )
        grid = self.time_grid.times
        return {
            "CIndexIPCW": self.c_index_ipcw,
            "CIndexIPCWDefinition": "uno_ipcw_sksurv",
            "IBS": self.ibs,
            "IBSDefinition": IBS_DEFINITION,
            "IBSFollowupRule": IBS_FOLLOWUP_RULE,
            "IBSEvaluationHorizon": self.time_grid.tau,
            "DCalibPValue": self.d_calib_pvalue,
            "CIndexIPCWStatus": self.c_index_status,
            "IBSStatus": self.ibs_status,
            "DCalibStatus": self.d_calib_status,
            "MetricStatus": metric_status,
            "IPCWReferenceCohort": "outer_development_only",
            "ProbabilityTimeGridMethod": "development_censoring_safe_ipcw_time_cap_v1",
            "ProbabilityTimeGridMin": float(grid[0]) if grid.size else np.nan,
            "ProbabilityTimeGridMax": float(grid[-1]) if grid.size else np.nan,
            "ProbabilityTimeGridCount": int(grid.size),
            "CIndexIPCWTau": self.time_grid.tau,
            "IPCWTimeCap": self.time_grid.time_cap,
        }


@dataclass(frozen=True)
class ProbabilityArtifactRecord:
    """Paths returned after durable per-fold OOF persistence."""

    oof_path: Path
    curve_path: Path
    curve_sha256: str
    metric_protocol_path: Path
    metric_protocol_sha256: str
    calibration_artifact_id: str

    def result_fields(self) -> dict[str, object]:
        return {
            "OOFPredictionsPath": str(self.oof_path),
            "SurvivalCurveArtifactPath": str(self.curve_path),
            "SurvivalCurveArtifactSHA256": self.curve_sha256,
            "ProbabilityMetricProtocolPath": str(self.metric_protocol_path),
            "ProbabilityMetricProtocolSHA256": self.metric_protocol_sha256,
            "CalibrationArtifactID": self.calibration_artifact_id,
            "CalibrationArtifactPath": "pending_finalization",
            "CalibrationArtifactSHA256": "pending_finalization",
            "CalibrationArtifactStatus": "pending_finalization",
            "CalibrationMethod": CALIBRATION_METHOD,
            "CalibrationHorizon": CALIBRATION_HORIZON,
            "CalibrationBins": CALIBRATION_BINS,
            "CalibrationHorizonRule": CALIBRATION_HORIZON_RULE,
        }


def _status_ok() -> str:
    return "ok"


def _status_na(reason: str) -> str:
    return f"na:{reason}"


def _exception_reason(prefix: str, exc: Exception) -> str:
    message = " ".join(str(exc).split())
    message = message.replace(";", ",")
    if not message:
        return f"{prefix}:{type(exc).__name__}"
    return f"{prefix}:{type(exc).__name__}:{message}"


def _as_1d_finite(values: Iterable[object], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name}_empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}_non_finite")
    return array


def _as_binary(values: Iterable[object], name: str) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}_not_binary") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name}_not_binary")
    return numeric.astype(bool)


def _survival_array(times: Iterable[object], events: Iterable[object], prefix: str):
    event_times = _as_1d_finite(times, f"{prefix}_times")
    event_indicators = _as_binary(events, f"{prefix}_events")
    if event_times.shape[0] != event_indicators.shape[0]:
        raise ValueError(f"{prefix}_time_event_length_mismatch")
    if np.any(event_times < 0.0):
        raise ValueError(f"{prefix}_negative_time")
    return Surv.from_arrays(event=event_indicators, time=event_times)


def _positive_censoring_probability(
    censoring: CensoringDistributionEstimator,
    times: Iterable[object],
) -> tuple[bool, str | None]:
    try:
        probabilities = np.asarray(censoring.predict_proba(np.asarray(times, dtype=float)))
    except Exception as exc:
        return False, f"development_censoring_support_query_failed:{type(exc).__name__}"
    if not np.isfinite(probabilities).all() or np.any(probabilities <= 0.0):
        return False, "development_censoring_support_zero"
    return True, None


def build_development_time_grid(
    development_event_times: Iterable[object],
    development_event_indicators: Iterable[object],
    *,
    grid_points: int = DEFAULT_IBS_GRID_POINTS,
    time_cap: float = IPCW_TIME_CAP,
) -> DevelopmentTimeGrid:
    """Build the single pre-model time grid for one outer development cohort.

    The upper endpoint is capped at the pre-registered IPCW horizon and must
    have strictly positive development-only censoring survival.  The cap is
    never selected from held-out outcomes; a held-out bearing can only make a
    pre-defined metric unavailable through the later support check.
    """

    if isinstance(grid_points, bool) or int(grid_points) != grid_points:
        return DevelopmentTimeGrid(
            times=np.empty(0, dtype=float),
            tau=np.nan,
            status=_status_na("invalid_grid_point_count"),
            reason="invalid_grid_point_count",
        )
    try:
        time_cap = float(time_cap)
    except (TypeError, ValueError):
        time_cap = float("nan")
    if not np.isfinite(time_cap) or time_cap <= 0.0:
        return DevelopmentTimeGrid(
            times=np.empty(0, dtype=float),
            tau=np.nan,
            status=_status_na("invalid_ipcw_time_cap"),
            reason="invalid_ipcw_time_cap",
            time_cap=time_cap,
        )
    grid_points = int(grid_points)
    if grid_points < 2:
        return DevelopmentTimeGrid(
            times=np.empty(0, dtype=float),
            tau=np.nan,
            status=_status_na("fewer_than_two_grid_points"),
            reason="fewer_than_two_grid_points",
        )
    try:
        y_development = _survival_array(
            development_event_times,
            development_event_indicators,
            "development",
        )
        development_times = np.asarray(y_development["time"], dtype=float)
        censoring = CensoringDistributionEstimator().fit(y_development)
    except Exception as exc:
        reason = f"invalid_development_cohort:{type(exc).__name__}"
        return DevelopmentTimeGrid(
            times=np.empty(0, dtype=float),
            tau=np.nan,
            status=_status_na(reason),
            reason=reason,
        )

    unique_times = np.unique(development_times)
    if unique_times.size < 2:
        reason = "development_followup_has_fewer_than_two_times"
        return DevelopmentTimeGrid(
            times=np.empty(0, dtype=float),
            tau=np.nan,
            status=_status_na(reason),
            reason=reason,
        )

    positive_support = []
    for value in unique_times:
        supported, _ = _positive_censoring_probability(censoring, [value])
        if supported:
            positive_support.append(float(value))
    if not positive_support:
        reason = "development_censoring_support_zero"
        return DevelopmentTimeGrid(
            times=np.empty(0, dtype=float),
            tau=np.nan,
            status=_status_na(reason),
            reason=reason,
        )

    lower = float(np.nextafter(unique_times[0], np.inf))
    upper = float(
        min(
            time_cap,
            np.nextafter(max(positive_support), -np.inf),
        )
    )
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        reason = "development_censoring_safe_interval_empty"
        return DevelopmentTimeGrid(
            times=np.empty(0, dtype=float),
            tau=np.nan,
            status=_status_na(reason),
            reason=reason,
        )

    grid = np.linspace(lower, upper, grid_points, dtype=float)
    supported, support_reason = _positive_censoring_probability(censoring, grid)
    if not supported:
        reason = support_reason or "development_censoring_support_zero"
        return DevelopmentTimeGrid(
            times=np.empty(0, dtype=float),
            tau=np.nan,
            status=_status_na(reason),
            reason=reason,
        )
    return DevelopmentTimeGrid(
        times=grid,
        tau=float(grid[-1]),
        status=_status_ok(),
        time_cap=time_cap,
    )


def _fixed_horizon_ipcw_integrated_brier_score(
    *,
    censoring: CensoringDistributionEstimator,
    test_times: np.ndarray,
    test_events: np.ndarray,
    survival_at_grid: np.ndarray,
    evaluation_times: np.ndarray,
) -> float:
    """Compute Graf IPCW IBS without querying irrelevant post-horizon follow-up.

    At a fixed evaluation time ``t``, the Graf Brier score only requires
    censoring survival at ``t`` and at observed event times up to ``t``.
    Querying ``G(T_i)`` for a test event after the fixed final horizon adds no
    contribution to any point on the grid, but scikit-survival rejects such a
    query when that far follow-up is outside the development support.  This
    implementation is numerically equivalent to scikit-survival's formula
    whenever the full test follow-up is supported.
    """

    times = _as_1d_finite(evaluation_times, "ibs_evaluation_times")
    observed_times = _as_1d_finite(test_times, "ibs_test_times")
    observed_events = _as_binary(test_events, "ibs_test_events")
    estimates = np.asarray(survival_at_grid, dtype=float)
    if observed_times.size != observed_events.size:
        raise ValueError("ibs_test_time_event_length_mismatch")
    if estimates.shape != (observed_times.size, times.size):
        raise ValueError("ibs_survival_grid_shape_mismatch")
    if not np.isfinite(estimates).all():
        raise ValueError("ibs_survival_grid_non_finite")
    if times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("ibs_evaluation_times_not_strictly_increasing")

    horizon = float(times[-1])
    event_within_horizon = observed_events & (observed_times <= horizon)
    grid_censoring = np.asarray(censoring.predict_proba(times), dtype=float)
    if not np.isfinite(grid_censoring).all() or np.any(grid_censoring <= 0.0):
        raise ValueError("ibs_grid_censoring_support_zero")

    case_censoring = np.ones(observed_times.size, dtype=float)
    if event_within_horizon.any():
        case_censoring[event_within_horizon] = censoring.predict_proba(
            observed_times[event_within_horizon]
        )
        if (
            not np.isfinite(case_censoring[event_within_horizon]).all()
            or np.any(case_censoring[event_within_horizon] <= 0.0)
        ):
            raise ValueError("ibs_case_censoring_support_zero")

    brier_scores = np.empty(times.size, dtype=float)
    for position, time_point in enumerate(times):
        is_case = observed_events & (observed_times <= time_point)
        is_control = observed_times > time_point
        contributions = np.zeros(observed_times.size, dtype=float)
        contributions[is_case] = (
            np.square(estimates[is_case, position]) / case_censoring[is_case]
        )
        contributions[is_control] = (
            np.square(1.0 - estimates[is_control, position])
            / grid_censoring[position]
        )
        brier_scores[position] = float(np.mean(contributions))

    return float(np.trapz(brier_scores, times) / (times[-1] - times[0]))


def canonicalize_survival_curves(survival_predictions: pd.DataFrame) -> CanonicalSurvivalCurves:
    """Validate and order a sample-by-time survival matrix without repair.

    Formal probability metrics may not use the legacy fill/clip/extrapolation
    fallbacks.  We therefore preserve the numeric values and report invalid
    curves rather than making them appear valid through a repair operation.
    """

    if not isinstance(survival_predictions, pd.DataFrame):
        return CanonicalSurvivalCurves(
            times=np.empty(0, dtype=float),
            survival=np.empty((0, 0), dtype=float),
            status=_status_na("survival_predictions_not_dataframe"),
            reason="survival_predictions_not_dataframe",
        )
    if survival_predictions.empty:
        return CanonicalSurvivalCurves(
            times=np.empty(0, dtype=float),
            survival=np.empty((0, 0), dtype=float),
            status=_status_na("survival_predictions_empty"),
            reason="survival_predictions_empty",
        )
    try:
        raw_times = np.asarray(survival_predictions.columns, dtype=float)
        raw_survival = survival_predictions.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        return CanonicalSurvivalCurves(
            times=np.empty(0, dtype=float),
            survival=np.empty((0, 0), dtype=float),
            status=_status_na("survival_curve_non_numeric"),
            reason=f"survival_curve_non_numeric:{type(exc).__name__}",
        )
    if raw_survival.ndim != 2 or raw_survival.shape[1] != raw_times.size:
        return CanonicalSurvivalCurves(
            times=np.empty(0, dtype=float),
            survival=np.empty((0, 0), dtype=float),
            status=_status_na("survival_curve_shape_invalid"),
            reason="survival_curve_shape_invalid",
        )
    if not np.isfinite(raw_times).all():
        return CanonicalSurvivalCurves(
            times=raw_times,
            survival=raw_survival,
            status=_status_na("survival_curve_non_finite_time"),
            reason="survival_curve_non_finite_time",
        )
    if np.any(raw_times < 0.0):
        return CanonicalSurvivalCurves(
            times=np.empty(0, dtype=float),
            survival=np.empty((0, 0), dtype=float),
            status=_status_na("survival_curve_negative_time"),
            reason="survival_curve_negative_time",
        )
    order = np.argsort(raw_times, kind="mergesort")
    times = raw_times[order]
    survival = raw_survival[:, order]
    if not np.isfinite(survival).all():
        return CanonicalSurvivalCurves(
            times=times,
            survival=survival,
            status=_status_na("survival_curve_non_finite"),
            reason="survival_curve_non_finite",
        )
    if np.unique(times).size != times.size:
        return CanonicalSurvivalCurves(
            times=times,
            survival=survival,
            status=_status_na("survival_curve_duplicate_time"),
            reason="survival_curve_duplicate_time",
        )
    tolerance = 1e-10
    if np.any(survival < -tolerance) or np.any(survival > 1.0 + tolerance):
        return CanonicalSurvivalCurves(
            times=times,
            survival=survival,
            status=_status_na("survival_curve_probability_out_of_range"),
            reason="survival_curve_probability_out_of_range",
        )
    if np.any(np.diff(survival, axis=1) > tolerance):
        return CanonicalSurvivalCurves(
            times=times,
            survival=survival,
            status=_status_na("survival_curve_not_monotone"),
            reason="survival_curve_not_monotone",
        )
    return CanonicalSurvivalCurves(times=times, survival=survival, status=_status_ok())


def _sample_survival_at_times(
    curves: CanonicalSurvivalCurves,
    target_times: Iterable[object],
) -> np.ndarray:
    if not curves.is_valid:
        raise ValueError(curves.reason or "survival_curve_invalid")
    targets = _as_1d_finite(target_times, "target_times")
    if targets.min() < curves.times[0] or targets.max() > curves.times[-1]:
        raise ValueError("target_times_outside_survival_curve_support")
    indices = np.searchsorted(curves.times, targets, side="right") - 1
    if np.any(indices < 0) or np.any(indices >= curves.times.size):
        raise ValueError("target_times_outside_survival_curve_support")
    return curves.survival[:, indices]


def event_probability_risk_at_horizon(
    survival_predictions: pd.DataFrame,
    horizon: float,
) -> tuple[np.ndarray, str]:
    """Return high-risk scores ``1 - S(horizon)`` from canonical curves.

    This adapter is for survival models that have no native Cox log-risk.
    The caller must supply a horizon fixed before model fitting from the outer
    development cohort.  Invalid or unsupported curves yield one explicit
    ``NaN`` score per prediction row so Uno/IPCW can be marked unavailable
    without disabling IBS or calibration.
    """

    count = len(survival_predictions) if isinstance(survival_predictions, pd.DataFrame) else 0
    scores = np.full(count, np.nan, dtype=float)
    try:
        numeric_horizon = float(horizon)
        if not np.isfinite(numeric_horizon) or numeric_horizon < 0.0:
            raise ValueError('invalid_risk_horizon')
        curves = canonicalize_survival_curves(survival_predictions)
        if not curves.is_valid:
            raise ValueError(curves.reason or 'survival_curve_invalid')
        if numeric_horizon < curves.times[0] or numeric_horizon > curves.times[-1]:
            raise ValueError('risk_horizon_outside_survival_curve_support')
        scores = 1.0 - _sample_survival_at_times(curves, [numeric_horizon])[:, 0]
        if not np.isfinite(scores).all():
            raise ValueError('risk_score_non_finite')
        if np.any(scores < -1e-10) or np.any(scores > 1.0 + 1e-10):
            raise ValueError('risk_score_out_of_range')
        return np.clip(scores, 0.0, 1.0), _status_ok()
    except Exception as exc:
        reason = str(exc) or type(exc).__name__
        return scores, _status_na(f'event_probability_risk_unavailable:{reason}')


def _new_censoring_estimator(y_development):
    try:
        return CensoringDistributionEstimator().fit(y_development), None
    except Exception as exc:
        return None, f"development_censoring_fit_failed:{type(exc).__name__}"


def evaluate_formal_probability_metrics(
    *,
    development_event_times: Iterable[object],
    development_event_indicators: Iterable[object],
    test_event_times: Iterable[object],
    test_event_indicators: Iterable[object],
    log_risk: Iterable[object],
    survival_predictions: pd.DataFrame,
    time_grid: DevelopmentTimeGrid,
) -> ProbabilityMetricResult:
    """Evaluate Uno/IPCW C-index, IPCW IBS, and D-calibration explicitly.

    ``time_grid`` must have been created once from the outer development
    cohort before entering the model loop.  No test-driven truncation or
    per-model grid replacement occurs in this function.
    """

    nan = float("nan")
    try:
        y_development = _survival_array(
            development_event_times,
            development_event_indicators,
            "development",
        )
        y_test = _survival_array(test_event_times, test_event_indicators, "test")
        test_times = np.asarray(y_test["time"], dtype=float)
        test_events = np.asarray(y_test["event"], dtype=bool)
    except Exception as exc:
        reason = _exception_reason("invalid_metric_inputs", exc)
        return ProbabilityMetricResult(
            c_index_ipcw=nan,
            ibs=nan,
            d_calib_pvalue=nan,
            c_index_status=_status_na(reason),
            ibs_status=_status_na(reason),
            d_calib_status=_status_na(reason),
            time_grid=time_grid,
        )

    risks = None
    risk_input_reason = None
    try:
        risks = _as_1d_finite(log_risk, "risk_score")
        if risks.size != test_times.size:
            raise ValueError("risk_score_test_length_mismatch")
    except Exception as exc:
        risk_input_reason = _exception_reason("invalid_risk_score", exc)

    curves = canonicalize_survival_curves(survival_predictions)
    curves_align_with_test = (
        curves.is_valid and curves.survival.shape[0] == test_times.size
    )
    censoring, censoring_reason = _new_censoring_estimator(y_development)

    c_index = nan
    if not time_grid.is_valid:
        c_index_status = _status_na(time_grid.reason or "development_time_grid_invalid")
    elif risk_input_reason is not None:
        c_index_status = _status_na(risk_input_reason)
    elif not np.any(y_development["event"]):
        c_index_status = _status_na("development_has_no_observed_event")
    elif censoring is None:
        c_index_status = _status_na(censoring_reason or "development_censoring_fit_failed")
    elif not np.any(test_times < time_grid.tau):
        c_index_status = _status_na("no_test_followup_before_ipcw_tau")
    elif not np.any(test_events & (test_times < time_grid.tau)):
        c_index_status = _status_na("no_observed_test_event_before_ipcw_tau")
    else:
        supported, support_reason = _positive_censoring_probability(
            censoring, [time_grid.tau]
        )
        if not supported:
            c_index_status = _status_na(support_reason or "development_censoring_support_zero")
        else:
            try:
                c_index = float(
                    concordance_index_ipcw(
                        y_development,
                        y_test,
                        risks,
                        tau=float(time_grid.tau),
                    )[0]
                )
                c_index_status = (
                    _status_ok() if np.isfinite(c_index) else _status_na("non_finite_ipcw_cindex")
                )
                if not np.isfinite(c_index):
                    c_index = nan
            except Exception as exc:
                c_index = nan
                c_index_status = _status_na(_exception_reason("ipcw_cindex_failed", exc))

    ibs = nan
    if not time_grid.is_valid:
        ibs_status = _status_na(time_grid.reason or "development_time_grid_invalid")
    elif censoring is None:
        ibs_status = _status_na(censoring_reason or "development_censoring_fit_failed")
    elif not curves.is_valid:
        ibs_status = _status_na(curves.reason or "survival_curve_invalid")
    elif not curves_align_with_test:
        ibs_status = _status_na("survival_curve_test_row_count_mismatch")
    elif time_grid.times.size < 2:
        ibs_status = _status_na("fewer_than_two_time_grid_points")
    elif (
        time_grid.times[0] < test_times.min()
        or time_grid.times[-1] >= test_times.max()
    ):
        ibs_status = _status_na("predefined_time_grid_outside_test_followup")
    else:
        # The fixed horizon makes test follow-up after tau a control at every
        # grid point. It must not create an unrelated development-support
        # query, while observed events through tau still require G(T_i).
        ibs_case_times = test_times[test_events & (test_times <= time_grid.tau)]
        ibs_support_times = np.concatenate((time_grid.times, ibs_case_times))
        supported, support_reason = _positive_censoring_probability(
            censoring, ibs_support_times
        )
        if not supported:
            ibs_status = _status_na(support_reason or "development_censoring_support_zero")
        else:
            try:
                survival_at_grid = _sample_survival_at_times(curves, time_grid.times)
                if survival_at_grid.shape != (test_times.size, time_grid.times.size):
                    raise ValueError("survival_grid_shape_mismatch")
                ibs = _fixed_horizon_ipcw_integrated_brier_score(
                    censoring=censoring,
                    test_times=test_times,
                    test_events=test_events,
                    survival_at_grid=survival_at_grid,
                    evaluation_times=time_grid.times,
                )
                ibs_status = _status_ok() if np.isfinite(ibs) else _status_na("non_finite_ibs")
                if not np.isfinite(ibs):
                    ibs = nan
            except Exception as exc:
                ibs = nan
                ibs_status = _status_na(_exception_reason("ipcw_ibs_failed", exc))

    d_calib = nan
    if not curves.is_valid:
        d_calib_status = _status_na(curves.reason or "survival_curve_invalid")
    elif not curves_align_with_test:
        d_calib_status = _status_na("survival_curve_test_row_count_mismatch")
    elif test_times.min() < curves.times[0] or test_times.max() > curves.times[-1]:
        d_calib_status = _status_na("test_followup_outside_survival_curve_support")
    else:
        try:
            survival_at_observed_times = _sample_survival_at_times(curves, test_times)
            diagonal_probabilities = np.diag(survival_at_observed_times)
            d_calib = float(
                d_calibration(
                    diagonal_probabilities,
                    test_events.astype(int),
                    num_bins=CALIBRATION_BINS,
                )[0]
            )
            d_calib_status = (
                _status_ok() if np.isfinite(d_calib) else _status_na("non_finite_d_calibration")
            )
            if not np.isfinite(d_calib):
                d_calib = nan
        except Exception as exc:
            d_calib = nan
            d_calib_status = _status_na(_exception_reason("d_calibration_failed", exc))

    return ProbabilityMetricResult(
        c_index_ipcw=c_index,
        ibs=ibs,
        d_calib_pvalue=d_calib,
        c_index_status=c_index_status,
        ibs_status=ibs_status,
        d_calib_status=d_calib_status,
        time_grid=time_grid,
    )


def fpt_reference_mae(
    predicted_medians: Iterable[object],
    reference_times: Iterable[object],
    reference_events: Iterable[object],
) -> tuple[float, str]:
    """Return detector-reference MAE, excluding an entire FPT-miss fold."""

    try:
        observed = _as_binary(reference_events, "reference_events")
    except Exception as exc:
        return float("nan"), _status_na(f"invalid_fpt_reference:{type(exc).__name__}")

    predicted = np.asarray(predicted_medians).reshape(-1)
    reference = np.asarray(reference_times).reshape(-1)
    if not (predicted.size == reference.size == observed.size):
        return float("nan"), _status_na("invalid_fpt_reference:ValueError")
    if not observed.any():
        return float("nan"), _status_na("fpt_reference_missing_for_fold")
    try:
        predicted_observed = _as_1d_finite(
            predicted[observed], "predicted_medians"
        )
        reference_observed = _as_1d_finite(
            reference[observed], "reference_times"
        )
    except Exception as exc:
        return float("nan"), _status_na(f"invalid_fpt_reference:{type(exc).__name__}")
    return (
        float(np.mean(np.abs(predicted_observed - reference_observed))),
        _status_ok(),
    )


def probability_artifact_root(results_path: str | Path) -> Path:
    results_path = Path(results_path)
    return results_path.with_name(f"{results_path.stem}_probability_artifacts")


def _safe_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "value"


def _censoring_token(value: float) -> str:
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p")


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_exclusive_json(path: Path, payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != serialized:
            raise FileExistsError(f"probability metric protocol collision: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
    return _file_sha256(path)


def _metric_protocol(
    *,
    development_time_grid: DevelopmentTimeGrid,
    condition: str,
    censoring_level: float,
    outer_fold: int,
    development_outcome_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "condition": str(condition),
        "censoring_level": float(censoring_level),
        "outer_fold": int(outer_fold),
        "development_outcome_sha256": str(development_outcome_sha256),
        "ipcw_reference_cohort": "outer_development_only",
        "c_index_ipcw_definition": "uno_ipcw_sksurv",
        "ibs_definition": IBS_DEFINITION,
        "ibs_followup_rule": IBS_FOLLOWUP_RULE,
        "ibs_evaluation_horizon": float(development_time_grid.tau),
        "time_grid_method": "development_censoring_safe_ipcw_time_cap_v1",
        "time_grid_status": development_time_grid.status,
        "time_grid_reason": development_time_grid.reason,
        "time_grid": [float(value) for value in development_time_grid.times],
        "ipcw_tau": float(development_time_grid.tau),
        "ipcw_time_cap": float(development_time_grid.time_cap),
        "calibration_method": CALIBRATION_METHOD,
        "calibration_horizon": CALIBRATION_HORIZON,
        "calibration_bins": CALIBRATION_BINS,
        "calibration_horizon_rule": CALIBRATION_HORIZON_RULE,
    }


def _curve_path(
    root: Path,
    *,
    condition: str,
    censoring_level: float,
    outer_fold: int,
    model_name: str,
    model_params_sha256: str,
) -> Path:
    token = "_".join(
        (
            _safe_component(condition),
            f"censoring{_censoring_token(censoring_level)}",
            f"fold{int(outer_fold)}",
            _safe_component(model_name),
            _safe_component(model_params_sha256)[:12],
        )
    )
    return root / "curves" / f"{token}.npz"


def _calibration_artifact_id(
    *,
    condition: str,
    censoring_level: float,
    model_name: str,
    input_representation: str,
) -> str:
    return "|".join(
        (
            str(condition),
            _censoring_token(censoring_level),
            str(model_name),
            str(input_representation),
        )
    )


def persist_probability_oof_artifacts(
    *,
    results_path: str | Path,
    condition: str,
    censoring_level: float,
    outer_fold: int,
    model_name: str,
    input_representation: str,
    model_params_sha256: str,
    label_artifact_sha256: str,
    label_manifest_sha256: str,
    development_outcome_sha256: str,
    test_data: pd.DataFrame,
    log_risk: Iterable[object],
    risk_score_definition: str = "cox_log_risk",
    risk_score_status: str = "ok",
    risk_score_is_log_risk: bool = True,
    predicted_medians: Iterable[object],
    predicted_median_status: str = "ok",
    predicted_median_unreached: Iterable[object] | None = None,
    predicted_median_rule: str = "not_recorded",
    survival_predictions: pd.DataFrame,
    time_grid: DevelopmentTimeGrid,
) -> ProbabilityArtifactRecord:
    """Persist raw sample-level OOF predictions and the complete curve matrix.

    Curves are written once per ``(condition, censoring, fold, model)`` NPZ.
    The OOF CSV only contains scalar fields and a row pointer, allowing later
    calibration and bootstrap jobs to reload data without process memory.
    """

    required = {
        "bearing_id",
        "time_step",
        "Survival_time",
        "Event",
        "TrueTime",
        "TrueEvent",
    }
    if not isinstance(test_data, pd.DataFrame):
        raise TypeError("test_data must be a pandas DataFrame")
    missing = sorted(required - set(test_data.columns))
    if missing:
        raise ValueError(f"test_data missing OOF columns: {missing}")
    count = len(test_data)
    risks = np.asarray(log_risk, dtype=float).reshape(-1)
    medians = np.asarray(predicted_medians, dtype=float).reshape(-1)
    if risks.size != count or medians.size != count:
        raise ValueError("OOF scalar predictions must align with test_data")
    if predicted_median_unreached is None:
        median_unreached = None
    else:
        median_unreached = _as_binary(
            predicted_median_unreached, "predicted_median_unreached"
        )
        if median_unreached.size != count:
            raise ValueError("predicted median support must align with test_data")
    if predicted_median_status == "ok":
        if median_unreached is None:
            median_unreached = np.zeros(count, dtype=bool)
        if median_unreached.any() and not np.isnan(medians[median_unreached]).all():
            raise ValueError("unreached predicted medians must be recorded as NaN")
        if not np.isfinite(medians[~median_unreached]).all():
            raise ValueError("supported predicted medians must be finite")
        median_unreached_output = median_unreached.astype(int)
    elif median_unreached is None:
        # A non-finite raw survival matrix has no meaningful per-row median
        # support classification. Persist an explicit unavailable value rather
        # than coercing it to either reached or unreached.
        if not np.isnan(medians).all():
            raise ValueError(
                "unavailable predicted medians must be recorded as NaN"
            )
        median_unreached_output = np.full(count, np.nan, dtype=float)
    else:
        if median_unreached.any() and not np.isnan(medians[median_unreached]).all():
            raise ValueError("unreached predicted medians must be recorded as NaN")
        if not np.isfinite(medians[~median_unreached]).all():
            raise ValueError("supported predicted medians must be finite")
        median_unreached_output = median_unreached.astype(int)
    event_indicators = _as_binary(test_data["Event"], "test_event_indicators")
    reference_event_indicators = _as_binary(
        test_data["TrueEvent"], "reference_fpt_event_indicators"
    )

    curves = canonicalize_survival_curves(survival_predictions)
    if curves.survival.shape[0] != count:
        raise ValueError("survival predictions must align with test_data")

    results_path = Path(results_path)
    root = probability_artifact_root(results_path)
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "metric_protocols" / (
        f"{_safe_component(condition)}_censoring{_censoring_token(censoring_level)}_"
        f"fold{int(outer_fold)}.json"
    )
    protocol_hash = _write_exclusive_json(
        protocol_path,
        _metric_protocol(
            development_time_grid=time_grid,
            condition=condition,
            censoring_level=censoring_level,
            outer_fold=outer_fold,
            development_outcome_sha256=development_outcome_sha256,
        ),
    )

    curve_path = _curve_path(
        root,
        condition=condition,
        censoring_level=censoring_level,
        outer_fold=outer_fold,
        model_name=model_name,
        model_params_sha256=model_params_sha256,
    )
    if curve_path.exists():
        raise FileExistsError(f"survival curve artifact already exists: {curve_path}")
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "condition": str(condition),
        "censoring_level": float(censoring_level),
        "outer_fold": int(outer_fold),
        "model_name": str(model_name),
        "input_representation": str(input_representation),
        "model_params_sha256": str(model_params_sha256),
        "risk_score_definition": str(risk_score_definition),
        "risk_score_status": str(risk_score_status),
        "risk_score_is_log_risk": bool(risk_score_is_log_risk),
        "curve_status": curves.status,
        "curve_reason": curves.reason,
    }
    np.savez_compressed(
        curve_path,
        time_grid=curves.times,
        survival=curves.survival,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    curve_hash = _file_sha256(curve_path)

    oof_path = root / "oof_predictions.csv"
    artifact_id = _calibration_artifact_id(
        condition=condition,
        censoring_level=censoring_level,
        model_name=model_name,
        input_representation=input_representation,
    )
    rows = pd.DataFrame(
        {
            "Condition": str(condition),
            "CensoringLevel": float(censoring_level),
            "OuterFold": int(outer_fold),
            "ModelName": str(model_name),
            "InputRepresentation": str(input_representation),
            "Bearing": test_data["bearing_id"].to_numpy(copy=True),
            "TimeStep": test_data["time_step"].to_numpy(copy=True),
            "SurvivalTime": test_data["Survival_time"].to_numpy(copy=True),
            "EventIndicator": event_indicators.astype(int),
            "ReferenceFPTTime": test_data["TrueTime"].to_numpy(copy=True),
            "ReferenceFPTEvent": reference_event_indicators.astype(int),
            "RiskScore": risks,
            "RiskScoreDefinition": str(risk_score_definition),
            "RiskScoreStatus": str(risk_score_status),
            "LogRisk": (
                risks if risk_score_is_log_risk else np.full(count, np.nan, dtype=float)
            ),
            "PredictedMedian": medians,
            "PredictedMedianStatus": str(predicted_median_status),
            "PredictedMedianUnreached": median_unreached_output,
            "PredictedMedianRule": str(predicted_median_rule),
            "SurvivalCurveStatus": str(curves.status),
            "LabelArtifactSHA256": str(label_artifact_sha256),
            "LabelManifestSHA256": str(label_manifest_sha256),
            "ModelParamsSHA256": str(model_params_sha256),
            "DevelopmentOutcomeSHA256": str(development_outcome_sha256),
            "ProbabilityMetricProtocolPath": str(protocol_path),
            "ProbabilityMetricProtocolSHA256": protocol_hash,
            "SurvivalCurveArtifactPath": str(curve_path),
            "SurvivalCurveArtifactSHA256": curve_hash,
            "CurveRowIndex": np.arange(count, dtype=int),
            "CalibrationArtifactID": artifact_id,
        }
    )
    header = not oof_path.exists()
    rows.to_csv(
        oof_path,
        mode="a",
        header=header,
        index=False,
        na_rep=CSV_NULL_REP,
    )
    return ProbabilityArtifactRecord(
        oof_path=oof_path,
        curve_path=curve_path,
        curve_sha256=curve_hash,
        metric_protocol_path=protocol_path,
        metric_protocol_sha256=protocol_hash,
        calibration_artifact_id=artifact_id,
    )


def _load_oof_survival_at_horizon(
    oof_rows: pd.DataFrame,
    horizon: float,
) -> tuple[np.ndarray, str | None]:
    cache: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    values = np.empty(len(oof_rows), dtype=float)
    for position, (_, row) in enumerate(oof_rows.iterrows()):
        path = Path(str(row["SurvivalCurveArtifactPath"]))
        if not path.exists():
            return values, f"curve_artifact_missing:{path.name}"
        key = str(path)
        if key not in cache:
            try:
                artifact_hash = _file_sha256(path)
                expected_hash = str(row["SurvivalCurveArtifactSHA256"])
                if artifact_hash != expected_hash:
                    return values, "curve_artifact_sha256_mismatch"
                with np.load(path, allow_pickle=False) as artifact:
                    times = np.asarray(artifact["time_grid"], dtype=float)
                    survival = np.asarray(artifact["survival"], dtype=float)
            except Exception as exc:
                return values, f"curve_artifact_unreadable:{type(exc).__name__}"
            if (
                times.ndim != 1
                or survival.ndim != 2
                or survival.shape[1] != times.size
                or times.size == 0
                or not np.isfinite(times).all()
                or not np.isfinite(survival).all()
                or np.any(np.diff(times) <= 0.0)
                or np.any(survival < 0.0)
                or np.any(survival > 1.0)
                or np.any(np.diff(survival, axis=1) > 1e-10)
            ):
                return values, "curve_artifact_invalid"
            cache[key] = (times, survival, artifact_hash)
        times, survival, artifact_hash = cache[key]
        if artifact_hash != str(row["SurvivalCurveArtifactSHA256"]):
            return values, "curve_artifact_sha256_mismatch"
        curve_row = int(row["CurveRowIndex"])
        if curve_row < 0 or curve_row >= survival.shape[0]:
            return values, "curve_row_index_out_of_range"
        if horizon < times[0] or horizon > times[-1]:
            return values, "fixed_calibration_horizon_outside_curve_support"
        column = int(np.searchsorted(times, horizon, side="right") - 1)
        values[position] = survival[curve_row, column]
    return values, None


def _calibration_output_path(root: Path, calibration_artifact_id: str) -> Path:
    return root / "calibration" / f"{_safe_component(calibration_artifact_id)}.csv"


def finalize_oof_calibration_artifacts(
    results_path: str | Path,
) -> tuple[dict[str, dict[str, object]], Path | None, str | None]:
    """Build fixed-horizon one-calibration curves by rereading OOF sidecars.

    The fixed horizon is a predeclared protocol constant.  The function does
    not receive in-memory predictions, which keeps later bootstrap/calibration
    work reproducible from saved artifacts alone.
    """

    root = probability_artifact_root(results_path)
    oof_path = root / "oof_predictions.csv"
    if not oof_path.exists():
        return {}, None, None
    oof = pd.read_csv(oof_path)
    required = {
        "Condition",
        "CensoringLevel",
        "ModelName",
        "InputRepresentation",
        "SurvivalTime",
        "EventIndicator",
        "SurvivalCurveArtifactPath",
        "SurvivalCurveArtifactSHA256",
        "CurveRowIndex",
        "CalibrationArtifactID",
    }
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"OOF prediction artifact missing columns: {missing}")

    artifacts: dict[str, dict[str, object]] = {}
    for artifact_id, group in oof.groupby("CalibrationArtifactID", sort=True):
        path = _calibration_output_path(root, str(artifact_id))
        if path.exists():
            raise FileExistsError(f"calibration artifact already exists: {path}")
        status = _status_ok()
        pvalue = float("nan")
        expected: list[float] = [float("nan")] * CALIBRATION_BINS
        observed: list[float] = [float("nan")] * CALIBRATION_BINS
        bin_sizes: list[int] = [0] * CALIBRATION_BINS
        try:
            survival_at_horizon, reason = _load_oof_survival_at_horizon(
                group, CALIBRATION_HORIZON
            )
            if reason is not None:
                raise ValueError(reason)
            event_times = _as_1d_finite(group["SurvivalTime"], "oof_survival_times")
            event_indicators = _as_binary(group["EventIndicator"], "oof_event_indicators")
            if event_times.size != survival_at_horizon.size:
                raise ValueError("oof_curve_length_mismatch")
            pvalue, observed, expected = one_calibration(
                survival_at_horizon,
                event_times,
                event_indicators.astype(int),
                CALIBRATION_HORIZON,
                num_bins=CALIBRATION_BINS,
                method="DN",
            )
            pvalue = float(pvalue)
            if not np.isfinite(pvalue):
                raise ValueError("non_finite_one_calibration_pvalue")
            event_probabilities = 1.0 - survival_at_horizon
            sorted_indices = np.argsort(-event_probabilities)
            bin_sizes = [
                int(values.size)
                for values in np.array_split(sorted_indices, CALIBRATION_BINS)
            ]
        except Exception as exc:
            status = _status_na(_exception_reason("one_calibration_failed", exc))
            pvalue = float("nan")

        first = group.iloc[0]
        frame = pd.DataFrame(
            {
                "CalibrationArtifactID": str(artifact_id),
                "Condition": str(first["Condition"]),
                "CensoringLevel": float(first["CensoringLevel"]),
                "ModelName": str(first["ModelName"]),
                "InputRepresentation": str(first["InputRepresentation"]),
                "Method": CALIBRATION_METHOD,
                "Horizon": CALIBRATION_HORIZON,
                "HorizonRule": CALIBRATION_HORIZON_RULE,
                "Bins": CALIBRATION_BINS,
                "Bin": np.arange(1, CALIBRATION_BINS + 1, dtype=int),
                "BinSize": bin_sizes,
                "ExpectedEventProbability": expected,
                "ObservedEventProbability": observed,
                "OneCalibrationPValue": pvalue,
                "OOFRowCount": int(len(group)),
                "OOFSourcePath": str(oof_path),
                "MetricStatus": status,
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, mode="x", index=False, na_rep=CSV_NULL_REP)
        artifacts[str(artifact_id)] = {
            "CalibrationArtifactPath": str(path),
            "CalibrationArtifactSHA256": _file_sha256(path),
            "CalibrationArtifactStatus": status,
        }
    return artifacts, oof_path, _file_sha256(oof_path)


def attach_finalized_calibration_artifacts(
    result_frame: pd.DataFrame,
    artifacts: dict[str, dict[str, object]],
    *,
    oof_path: Path | None,
    oof_sha256: str | None,
) -> pd.DataFrame:
    """Attach finalized aggregate calibration paths to persisted result rows."""

    updated = result_frame.copy()
    if "CalibrationArtifactID" not in updated.columns:
        return updated
    for index, artifact_id in updated["CalibrationArtifactID"].items():
        details = artifacts.get(str(artifact_id))
        if details is not None:
            for field, value in details.items():
                updated.loc[index, field] = value
        if oof_path is not None:
            updated.loc[index, "OOFPredictionsPath"] = str(oof_path)
        if oof_sha256 is not None:
            updated.loc[index, "OOFPredictionsSHA256"] = oof_sha256
    return updated
