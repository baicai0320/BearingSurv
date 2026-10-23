import numpy as np
import pandas as pd
from scipy.stats import chisquare
import warnings

from tools.Evaluations.custom_types import NumericArrayLike
from tools.Evaluations.util import check_and_convert, predict_prob_from_curve


# Keep this aligned with the formal curve audit: harmless interpolation round-
# off at a probability boundary is normalized, while material violations still
# make the metric unavailable.
PROBABILITY_BOUNDARY_TOLERANCE = 1e-10


def d_calibration_pycox(
        predicted_survival_curves: pd.DataFrame,
        event_time: NumericArrayLike,
        event_indicator: NumericArrayLike,
        num_bins: int = 10
) -> (float, np.ndarray):
    """
    Calculate the D-Calibration score for PyCox models.
    Parameters
    ----------
    predicted_survival_curves: pd.DataFrame
        The predicted survival curves from the PyCox model.
    event_time: NumericArrayLike
        The true event times.
    event_indicator: NumericArrayLike
        The true event indicators.
    num_bins: int
        The number of bins to use for the D-Calibration score.

    Returns
    -------
    float
        The D-Calibration score.
    np.ndarray
        The binning of the D-Calibration score.
    """
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'PyCoxEvaluator' from Evaluator.py.", DeprecationWarning)
    # Checking the format of the data
    true_event_times, uncensor_status = check_and_convert(event_time, event_indicator)
    # Extracting the time buckets
    time_coordinates = predicted_survival_curves.index.values
    # computing the Survival function, and set the small negative value to zero
    survival_curves = predicted_survival_curves.values.T
    survival_curves[survival_curves < 0] = 0

    predict_probs = []
    for i in range(survival_curves.shape[0]):
        predict_prob = predict_prob_from_curve(survival_curves[i, :], time_coordinates, true_event_times[i])
        predict_probs.append(predict_prob)
    predict_probs = np.array(predict_probs)
    return d_calibration(predict_probs, uncensor_status, num_bins)


def d_calibration(
        predict_probs,
        event_indicators,
        num_bins: int = 10
) -> (float, np.ndarray):
    """
    Calculate the D-Calibration score.
    Parameters
    ----------
    predict_probs: np.ndarray
        The predicted survival probabilities at individual's event/censor time.
    event_indicators: np.ndarray
        The event indicators.
    num_bins: int
        The number of bins to use for the D-Calibration score.

    Returns
    -------
    pvalue: float
        The p-value of the D-Calibration test.
    combine_binning: np.ndarray
        The binning histogram of the D-Calibration test.
    """
    num_bins = _validate_num_bins(num_bins)
    probabilities, event_mask = _validate_d_calibration_inputs(
        predict_probs, event_indicators
    )
    quantile = np.linspace(1.0, 0.0, num_bins + 1)

    event_probabilities = probabilities[event_mask]
    # ``np.digitize`` returns ``num_bins + 1`` for the lower endpoint of a
    # descending bin vector. S(t)=0 is a valid survival probability and must
    # belong to the final D-calibration bin rather than indexing past it.
    event_positions = np.digitize(event_probabilities, quantile)
    event_positions = np.clip(event_positions, 1, num_bins)
    event_binning = np.bincount(
        event_positions - 1, minlength=num_bins
    ).astype(float, copy=False)

    censored_probabilities = probabilities[~event_mask]
    censor_binning = np.zeros(num_bins, dtype=float)
    for probability in censored_probabilities:
        censor_binning += create_censor_binning(probability, num_bins)

    combine_binning = event_binning + censor_binning
    if not np.isfinite(combine_binning).all() or combine_binning.sum() <= 0.0:
        raise ValueError("d_calibration_binning_is_invalid")
    _, pvalue = chisquare(combine_binning)
    if not np.isfinite(pvalue):
        raise ValueError("d_calibration_pvalue_is_non_finite")
    return pvalue, combine_binning


def d_calibration_sksurv(
        predicted_survival_curves: pd.DataFrame,
        event_time: NumericArrayLike,
        event_indicator: NumericArrayLike,
        num_bins: int = 10
) -> (float, np.ndarray):
    """
    Calculate the D-Calibration score for scikit-survival models.
    :param predicted_survival_curves: pd.DataFrame
        The predicted survival curves from the scikit-survival model. The index is the time coordinates.
    :param event_time: NumericArrayLike
        The true event times.
    :param event_indicator: NumericArrayLike
        The true event indicators.
    :param num_bins: int
        The number of bins to use for the D-Calibration score.
    :return:
    pvalue: float
        The p-value of the D-Calibration test.
    combine_binning: np.ndarray
        The binning histogram of the D-Calibration test.
    """
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'ScikitSurvivalEvaluator' from Evaluator.py.", DeprecationWarning)
    # Checking the format of the data
    true_event_times, uncensor_status = check_and_convert(event_time, event_indicator)

    predict_probs = []
    for i in range(predicted_survival_curves.shape[0]):
        predict_prob = predict_prob_from_curve(predicted_survival_curves[i].y, predicted_survival_curves[i].x,
                                               true_event_times[i])
        predict_probs.append(predict_prob)
    predict_probs = np.array(predict_probs)

    return d_calibration(predict_probs, uncensor_status, num_bins)


def d_calibration_pysurvival(
        model,
        X,
        T,
        E,
        num_bins=10
) -> (np.ndarray, float):
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'PyCoxEvaluator' from Evaluator.py.", DeprecationWarning)
    # Checking the format of the data
    true_event_times, uncensor_status = check_and_convert(T, E)
    # computing the Survival function
    survival_curves = model.predict_survival(X, None)

    # Extracting the time buckets
    time_coordinates = model.times

    # if isinstance(model, RandomSurvivalForestModel):
    #     time_coordinates.pop()
    #     survival_curves = np.delete(survival_curves, -1, 1)
    #     print('use rsf, delete last time point')

    predict_probs = []
    for i in range(survival_curves.shape[0]):
        predict_prob = predict_prob_from_curve(survival_curves[i, :], time_coordinates, true_event_times[i])
        predict_probs.append(predict_prob)
    predict_probs = np.array(predict_probs)
    return d_calibration(predict_probs, uncensor_status, num_bins)


def create_censor_binning(
        probability: float,
        num_bins: int
) -> np.ndarray:
    """
    For censoring instance,
    b1 will be the infimum probability of the bin that contains S(c),
    for the bin of [b1, b2) which contains S(c), probability = (S(c) - b1) / S(c)
    for the rest of the bins, [b2, b3), [b3, b4), etc., probability = 1 / (B * S(c)), where B is the number of bins.
    :param probability: float
        The predicted probability at the censored time of a censoring instance.
    :param num_bins: int
        The number of bins to use for the D-Calibration score.
    :return:
    final_binning: np.ndarray
        The "split" histogram of this censored subject.
    """
    num_bins = _validate_num_bins(num_bins)
    probability = _normalize_survival_probabilities(
        np.asarray([probability], dtype=float)
    )[0]

    if probability == 1.0:
        return np.full(num_bins, 1.0 / num_bins, dtype=float)
    if probability == 0.0:
        # For a censored subject with S(c)=0, the conditional distribution is
        # degenerate at zero, so all of its mass belongs to the final bin.
        final_binning = np.zeros(num_bins, dtype=float)
        final_binning[-1] = 1.0
        return final_binning

    quantile = np.linspace(1.0, 0.0, num_bins + 1)
    position = int(np.digitize(probability, quantile))
    position = int(np.clip(position, 1, num_bins))
    bin_index = position - 1
    lower_edge = quantile[bin_index + 1]

    final_binning = np.zeros(num_bins, dtype=float)
    final_binning[bin_index] = (probability - lower_edge) / probability
    if bin_index + 1 < num_bins:
        final_binning[bin_index + 1:] = 1.0 / (num_bins * probability)
    return final_binning


def _validate_num_bins(num_bins: int) -> int:
    """Return a positive integral bin count without silently truncating input."""

    if isinstance(num_bins, (bool, np.bool_)):
        raise ValueError("num_bins must be a positive integer")
    try:
        normalized = int(num_bins)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("num_bins must be a positive integer") from exc
    if normalized < 1 or normalized != num_bins:
        raise ValueError("num_bins must be a positive integer")
    return normalized


def _validate_d_calibration_inputs(
        predict_probs,
        event_indicators,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the probability/event pairing before constructing bins."""

    try:
        probabilities = np.asarray(predict_probs, dtype=float).reshape(-1)
        events = np.asarray(event_indicators, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("d_calibration inputs must be numeric one-dimensional arrays") from exc
    if probabilities.size == 0:
        raise ValueError("d_calibration requires at least one sample")
    if probabilities.size != events.size:
        raise ValueError("predict_probs and event_indicators must have matching lengths")
    probabilities = _normalize_survival_probabilities(probabilities)
    if not np.isfinite(events).all() or not np.isin(events, (0.0, 1.0)).all():
        raise ValueError("event_indicators must contain only 0 or 1")
    return probabilities, events.astype(bool)


def _normalize_survival_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Normalize insignificant interpolation overshoot at [0, 1] boundaries."""

    if not np.isfinite(probabilities).all() or np.any(
            probabilities < -PROBABILITY_BOUNDARY_TOLERANCE
    ) or np.any(probabilities > 1.0 + PROBABILITY_BOUNDARY_TOLERANCE):
        raise ValueError("survival probabilities must be finite and in [0, 1]")
    return np.clip(probabilities, 0.0, 1.0)
