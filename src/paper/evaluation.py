"""The shared nine-metric paper output contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper._evaluation_reference import _evaluate_curves, evaluate_survival_predictions, uncensored_mae_from_observed_truth, SurvivalEvaluation
from tools.probability_metrics import canonicalize_survival_curves, evaluate_formal_probability_metrics, fpt_reference_mae
from paper.runtime import METRICS


def reference_record(model, condition, fold, censoring):
    payload = json.loads(Path(__file__).with_name("reference.json").read_text())
    records = [r for r in payload["records"] if (r["model"], r["condition"], r["fold"], r["censoring"]) == (model, condition, fold, censoring)]
    if len(records) != 1:
        raise ValueError("paper reference does not contain exactly one matching task")
    return records[0]


def evaluate(model, curves, risks, development, test, grid):
    if model == "otsurv":
        metrics, medians, _, _ = _evaluate_curves(curves, development_data=development, test_data=test,
                                                risks=risks, probability_time_grid=grid)
        metrics["DCalib"] = metrics["LegacyDCalib"]
        return metrics, medians
    canonical = canonicalize_survival_curves(curves)
    if not canonical.is_valid:
        legacy = SurvivalEvaluation(np.nan, np.nan, np.nan, np.nan, np.full(len(test), np.nan),
                                    median_prediction_status=canonical.status, d_calib_status=canonical.status)
    else:
        prepared = curves.copy().clip(0, 1)
        prepared.loc[:, 0.0] = 1.0
        legacy = evaluate_survival_predictions(prepared, test.Survival_time, test.Event,
                                              development.Survival_time, development.Event, context=model)
    probability = evaluate_formal_probability_metrics(
        development_event_times=development.Survival_time, development_event_indicators=development.Event,
        test_event_times=test.Survival_time, test_event_indicators=test.Event,
        log_risk=risks, survival_predictions=curves, time_grid=grid,
    )
    if legacy.median_prediction_status == "ok":
        mae_fpt, mae_status = fpt_reference_mae(legacy.median_survs, test.TrueTime, test.TrueEvent)
        true_mae = uncensored_mae_from_observed_truth(legacy.median_survs, test.TrueTime, test.TrueEvent)
        true_status = "ok" if np.isfinite(true_mae) else "na:true_reference_missing_for_fold"
    else:
        mae_fpt = true_mae = np.nan
        mae_status = true_status = legacy.median_prediction_status
    return {"MAEFPTRef": mae_fpt, "MAEFPTRefStatus": mae_status, "MAETrue": true_mae,
            "MAETrueStatus": true_status,
            "MAEHinge": legacy.mae_hinge, "MAEMargin": legacy.mae_margin, "MAEPseudo": legacy.mae_pseudo,
            "CIndexIPCW": probability.c_index_ipcw, "CIndexIPCWStatus": probability.c_index_status,
            "IBS": probability.ibs, "IBSStatus": probability.ibs_status,
            "DCalibPValue": probability.d_calib_pvalue, "DCalibPValueStatus": probability.d_calib_status,
            "DCalib": legacy.d_calib, "LegacyDCalibStatus": legacy.d_calib_status,
            "SurvivalCurveStatus": canonical.status,
            "ProbabilityGridStatus": probability.time_grid.status,
            "MedianPredictionStatus": legacy.median_prediction_status}, legacy.median_survs


def compare_metrics(metrics, reference, *, rtol=1e-5, atol=1e-6):
    comparison = {}
    for name in METRICS:
        actual, expected = metrics.get(name), reference["metrics"][name]
        actual = float(actual) if actual is not None and np.isfinite(actual) else None
        passed = actual is None if expected is None else actual is not None and bool(np.isclose(actual, expected, rtol=rtol, atol=atol))
        comparison[name] = {"actual": actual, "reference": expected, "passed": passed,
                            "absolute_difference": abs(actual - expected) if actual is not None and expected is not None else None}
    statuses = {}
    for name in ("MAEFPTRefStatus", "MAETrueStatus", "LegacyDCalibStatus", "CIndexIPCWStatus",
                 "IBSStatus", "DCalibPValueStatus", "MedianPredictionStatus"):
        reference_name = "DCalibStatus" if name == "DCalibPValueStatus" and reference["model"] != "otsurv" else name
        expected = reference["statuses"].get(reference_name)
        if expected is not None:
            statuses[name] = {"actual": metrics.get(name), "reference": expected,
                              "passed": metrics.get(name) == expected}
    return {"passed": all(row["passed"] for row in (*comparison.values(), *statuses.values())),
            "rtol": rtol, "atol": atol, "metrics": comparison, "statuses": statuses}
