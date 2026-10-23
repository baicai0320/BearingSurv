"""Numerical evaluator bodies retained from the frozen main revision."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import math
import warnings
import numpy as np
import pandas as pd
from tools.evaluator import LifelinesEvaluator
from tools.Evaluations.util import predict_median_survival_time
from tools.probability_metrics import canonicalize_survival_curves, evaluate_formal_probability_metrics, fpt_reference_mae
from paper.training import stable_breslow_survival_probabilities
from utility.survival import Survival
MEDIAN_UNREACHED_STATUS = "na:median_unreached_on_evaluation_grid"

def uncensored_mae_from_observed_truth(
    predicted_times,
    true_times,
    true_events,
) -> float:
    """Compute MAE only where detector truth contains an observed event."""

    predicted = np.asarray(predicted_times, dtype=float)
    truth = np.asarray(true_times, dtype=float)
    observed = np.asarray(true_events, dtype=bool)
    if predicted.ndim != 1 or truth.ndim != 1 or observed.ndim != 1:
        raise ValueError("MAETrue inputs must be one-dimensional")
    if not (len(predicted) == len(truth) == len(observed)):
        raise ValueError("MAETrue inputs must be aligned")
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        raise ValueError("MAETrue inputs must be finite")
    if not observed.any():
        return float("nan")
    return float(np.mean(np.abs(truth[observed] - predicted[observed])))

def predict_coxph_survival_with_log_domain_fallback(
    model,
    train_features,
    train_event_times,
    train_event_indicators,
    test_features,
    evaluation_times,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    """Return native CoxPH curves, falling back only for baseline underflow.

    ``CoxPHSurvivalAnalysis`` fits the regression coefficients in the log-risk
    domain but its native Breslow estimator exponentiates risk-set scores. At
    high censoring that can turn a finite fitted Cox model into an all-NaN
    baseline. The fallback preserves the same Breslow definition and uses only
    the outer-development cohort, evaluated with log-sum-exp denominators.
    """

    test_log_risks = np.asarray(model.predict(test_features), dtype=float).reshape(-1)
    # scikit-survival emits this warning before exposing an all-NaN Breslow
    # baseline. The validated fallback below records that condition in the
    # result row, so suppress only this expected upstream warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
            module=r"sksurv\.linear_model\.coxph",
        )
        native_curves = Survival.predict_survival_function(
            model, test_features, evaluation_times
        )
    if not isinstance(native_curves, pd.DataFrame):
        raise TypeError("CoxPH did not return DataFrame survival curves")
    native_values = native_curves.to_numpy(dtype=float)
    if np.isfinite(native_values).all():
        return native_curves, test_log_risks, "native_sksurv_breslow"

    train_log_risks = np.asarray(model.predict(train_features), dtype=float).reshape(-1)
    fallback_values = stable_breslow_survival_probabilities(
        train_log_risks,
        train_event_times,
        train_event_indicators,
        test_log_risks,
        evaluation_times,
    )
    expected_shape = (len(test_log_risks), len(evaluation_times))
    if fallback_values.shape != expected_shape:
        raise RuntimeError(
            "log-domain CoxPH Breslow fallback returned an invalid curve shape"
        )
    if not np.isfinite(fallback_values).all():
        raise RuntimeError("log-domain CoxPH Breslow fallback returned non-finite curves")
    return (
        pd.DataFrame(fallback_values, columns=np.asarray(evaluation_times)),
        test_log_risks,
        "logsumexp_breslow_fallback",
    )

@dataclass(frozen=True)
class SurvivalEvaluation:
    """Survival metrics and auditable median-support information for one fold."""

    mae_hinge: float
    mae_margin: float
    mae_pseudo: float
    d_calib: float
    median_survs: np.ndarray
    median_unreached_mask: np.ndarray | None = None
    median_prediction_status: str = "ok"
    median_evaluation_grid_max: float = np.nan
    d_calib_status: str = "ok"

def median_unreached_mask_from_survival_predictions(
    surv_preds: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Identify curves whose median is not supported by their prediction grid.

    The historical evaluator extrapolates a median beyond the final supplied
    time point.  That extrapolation becomes infinite when a numerically valid
    curve is exactly one across the entire grid, as can happen when a Cox risk
    underflows on a held-out bearing.  Formal results therefore use a common
    support rule: median-derived metrics are unavailable unless every curve
    reaches 0.5 on the supplied development-derived grid.  Probability curves
    themselves remain unchanged for IPCW and calibration metrics.
    """

    if not isinstance(surv_preds, pd.DataFrame):
        raise TypeError("survival predictions must be a DataFrame")
    if surv_preds.empty:
        raise ValueError("survival predictions must be non-empty")
    try:
        times = np.asarray(surv_preds.columns, dtype=float)
        curves = surv_preds.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("survival predictions must be numeric") from exc
    if curves.ndim != 2 or curves.shape[1] != times.size:
        raise ValueError("survival prediction shape is invalid")
    if not np.isfinite(times).all() or not np.isfinite(curves).all():
        raise ValueError("survival predictions must be finite")
    if times.size == 0:
        raise ValueError("survival prediction grid is empty")
    return np.all(curves > 0.5, axis=1), float(np.max(times))

def evaluate_survival_predictions(
    surv_preds: pd.DataFrame,
    test_event_times,
    test_event_indicators,
    reference_event_times,
    reference_event_indicators,
    *,
    context: str,
) -> SurvivalEvaluation:
    """Evaluate one prediction matrix without hiding undefined median metrics.

    A median is only a point prediction when each curve reaches 0.5 on the
    supplied grid.  If one or more curves do not, the three legacy MAE
    variants are recorded as unavailable rather than extrapolating an
    arbitrary finite value or terminating unrelated probability metrics.
    """

    try:
        median_unreached_mask, median_evaluation_grid_max = (
            median_unreached_mask_from_survival_predictions(surv_preds)
        )
        lifelines_eval = LifelinesEvaluator(
            surv_preds.T,
            test_event_times,
            test_event_indicators,
            reference_event_times,
            reference_event_indicators,
        )
        try:
            d_calib = float(lifelines_eval.d_calibration()[0])
            if not np.isfinite(d_calib):
                raise ValueError("non_finite_d_calibration")
            d_calib_status = "ok"
        except Exception as exc:
            # D-calibration is an independent diagnostic. A numerical corner
            # case in that statistic must not discard an already trained
            # model, its MAE metrics, or its probability sidecars.
            d_calib = np.nan
            d_calib_status = f"na:legacy_d_calibration_failed:{type(exc).__name__}"
        if np.any(median_unreached_mask):
            # Do not invoke the legacy whole-fold MAE implementation here: it
            # calls the extrapolating median helper and converts a valid
            # unsupported curve into inf/nan.  Supported rows still retain
            # their direct grid-crossing median in the OOF sidecar.
            mae_hinge = np.nan
            mae_margin = np.nan
            mae_pseudo = np.nan
            median_survs = np.full(len(surv_preds), np.nan, dtype=float)
            supported_indices = np.flatnonzero(~median_unreached_mask)
            if supported_indices.size:
                supported_eval = LifelinesEvaluator(
                    surv_preds.iloc[supported_indices].T,
                    np.asarray(test_event_times)[supported_indices],
                    np.asarray(test_event_indicators)[supported_indices],
                    reference_event_times,
                    reference_event_indicators,
                )
                median_survs[supported_indices] = (
                    supported_eval.predict_time_from_curve(
                        predict_median_survival_time
                    )
                )
            median_prediction_status = MEDIAN_UNREACHED_STATUS
        else:
            mae_hinge = lifelines_eval.mae(method="Hinge")
            mae_margin = lifelines_eval.mae(method="Margin")
            mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
            median_survs = lifelines_eval.predict_time_from_curve(
                predict_median_survival_time
            )
            median_prediction_status = "ok"
    except Exception as exc:
        raise RuntimeError(
            f"{context}: survival evaluation failed "
            f"({type(exc).__name__}): {exc}"
        ) from exc

    metrics = {}
    if d_calib_status == "ok":
        metrics["DCalib"] = d_calib
    if median_prediction_status == "ok":
        metrics.update(
            {
                "MAEHinge": mae_hinge,
                "MAEMargin": mae_margin,
                "MAEPseudo": mae_pseudo,
            }
        )
    non_finite = [
        name for name, value in metrics.items()
        if not np.isfinite(float(value))
    ]
    median_survs = np.asarray(median_survs, dtype=float)
    if median_survs.ndim != 1 or median_survs.size != len(surv_preds):
        non_finite.append("median_survival_time")
    elif median_prediction_status == "ok":
        if not np.isfinite(median_survs).all():
            non_finite.append("median_survival_time")
    else:
        unsupported_values = median_survs[median_unreached_mask]
        supported_values = median_survs[~median_unreached_mask]
        if (
            not np.isnan(unsupported_values).all()
            or not np.isfinite(supported_values).all()
        ):
            non_finite.append("median_survival_time")
    if non_finite:
        raise RuntimeError(
            f"{context}: survival evaluation returned non-finite values: "
            f"{', '.join(non_finite)}"
        )

    return SurvivalEvaluation(
        mae_hinge=float(mae_hinge),
        mae_margin=float(mae_margin),
        mae_pseudo=float(mae_pseudo),
        d_calib=float(d_calib),
        median_survs=median_survs,
        median_unreached_mask=median_unreached_mask,
        median_prediction_status=median_prediction_status,
        median_evaluation_grid_max=median_evaluation_grid_max,
        d_calib_status=d_calib_status,
    )

def _evaluate_curves(
    curves: pd.DataFrame,
    *,
    development_data: pd.DataFrame,
    test_data: pd.DataFrame,
    risks: np.ndarray,
    probability_time_grid,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None, str]:
    canonical = canonicalize_survival_curves(curves)
    if not canonical.is_valid:
        raise RuntimeError(f"R14-F returned invalid survival curves: {canonical.status}")
    prepared = pd.DataFrame(canonical.survival, columns=canonical.times)
    if prepared.shape[0] != len(test_data):
        raise RuntimeError("R14-F survival curves do not align with the held-out rows")
    median_unreached = np.all(prepared.to_numpy(dtype=float) > 0.5, axis=1)
    median_status = "ok"
    if median_unreached.any():
        median_status = "na:median_unreached_on_evaluation_grid"
        medians = np.full(len(test_data), np.nan, dtype=float)
        mae_hinge = np.nan
        mae_margin = np.nan
        mae_pseudo = np.nan
        legacy_dcalib = np.nan
        legacy_dcalib_status = median_status
    else:
        evaluator = LifelinesEvaluator(
            prepared.T,
            test_data["Survival_time"].to_numpy(copy=True),
            test_data["Event"].to_numpy(copy=True),
            development_data["Survival_time"].to_numpy(copy=True),
            development_data["Event"].to_numpy(copy=True),
        )
        medians = np.asarray(
            evaluator.predict_time_from_curve(predict_median_survival_time),
            dtype=float,
        )
        mae_hinge = float(evaluator.mae(method="Hinge"))
        mae_margin = float(evaluator.mae(method="Margin"))
        mae_pseudo = float(evaluator.mae(method="Pseudo_obs"))
        try:
            legacy_dcalib = float(evaluator.d_calibration()[0])
            if not math.isfinite(legacy_dcalib):
                raise ValueError("non-finite legacy D-calibration")
            legacy_dcalib_status = "ok"
        except Exception as exc:
            legacy_dcalib = np.nan
            legacy_dcalib_status = f"na:legacy_d_calibration_failed:{type(exc).__name__}"
    probability = evaluate_formal_probability_metrics(
        development_event_times=development_data["Survival_time"],
        development_event_indicators=development_data["Event"],
        test_event_times=test_data["Survival_time"],
        test_event_indicators=test_data["Event"],
        log_risk=risks,
        survival_predictions=prepared,
        time_grid=probability_time_grid,
    )
    if median_status == "ok":
        true_mae = uncensored_mae_from_observed_truth(
            medians,
            test_data["TrueTime"].to_numpy(copy=True),
            test_data["TrueEvent"].to_numpy(copy=True),
        )
        true_mae_status = "ok" if math.isfinite(true_mae) else "na:true_reference_missing_for_fold"
        mae_fpt, mae_fpt_status = fpt_reference_mae(
            medians,
            test_data["TrueTime"].to_numpy(copy=True),
            test_data["TrueEvent"].to_numpy(copy=True),
        )
    else:
        true_mae = np.nan
        true_mae_status = median_status
        mae_fpt = np.nan
        mae_fpt_status = median_status
    metrics = {
        "MAEHinge": mae_hinge,
        "MAEMargin": mae_margin,
        "MAEPseudo": mae_pseudo,
        "MAETrue": true_mae,
        "MAETrueStatus": true_mae_status,
        "MAEFPTRef": mae_fpt,
        "MAEFPTRefStatus": mae_fpt_status,
        "LegacyDCalib": legacy_dcalib,
        "LegacyDCalibStatus": legacy_dcalib_status,
        "CIndexIPCW": probability.c_index_ipcw,
        "CIndexIPCWStatus": probability.c_index_status,
        "IBS": probability.ibs,
        "IBSStatus": probability.ibs_status,
        "DCalibPValue": probability.d_calib_pvalue,
        "DCalibPValueStatus": probability.d_calib_status,
        "ProbabilityGridStatus": probability.time_grid.status,
        "ProbabilityGridTau": probability.time_grid.tau,
        "MedianPredictionStatus": median_status,
        "MedianUnreachedCount": int(median_unreached.sum()),
    }
    return metrics, medians, median_unreached, median_status
