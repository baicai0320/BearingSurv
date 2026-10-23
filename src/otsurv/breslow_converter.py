"""Breslow estimator and NLL survival curve conversion.

Converts OTSurv model outputs (log-risk scores or logits) into
standard survival curve DataFrames compatible with RULSurv evaluation.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared floor-detection constants
# ---------------------------------------------------------------------------
# These constants are the single source of truth for the "Breslow floor
# plateau" detection threshold used by the diagnostic plotting scripts
# (``predict_mean_curves.py``, ``predict_stratified_curves.py``,
# ``plot_otsurv_isd_rul.py``) and by the bug-condition / preservation
# tests. Keeping them here (module-level) avoids the historical bug where
# three scripts hard-coded the same numbers and drifted out of sync.
#
# Meaning:
#   SURVIVAL_FLOOR        — the legacy 0.001 lower bound that used to be
#                            enforced by ``np.clip`` in BreslowConverter;
#                            retained as a detection threshold so
#                            diagnostic plots can still flag any residual
#                            floor plateau in cached/old outputs.
#   SURVIVAL_FLOOR_TOL    — numerical tolerance around SURVIVAL_FLOOR
#                            when identifying a plateau.
#   MIN_FLOOR_TAIL_POINTS — minimum consecutive floor-valued points along
#                            the time axis required to call a tail a
#                            "plateau".
SURVIVAL_FLOOR = 0.001
SURVIVAL_FLOOR_TOL = 0.005
MIN_FLOOR_TAIL_POINTS = 3


# Upper bound on the cumulative-hazard argument of ``np.exp(-·)``.
# ``np.exp(-_EXP_ARG_MAX)`` is a finite near-zero float64 value and remains
# far below the removed historical 0.001 floor. Applying the bound in log
# space prevents either the relative risk or cumulative hazard from
# overflowing before their product is formed.
_EXP_ARG_MAX = 700.0
_LOG_EXP_ARG_MAX = np.log(_EXP_ARG_MAX)


def detect_otsurv_floor_tail(surv, continuous_times, y_test_t):
    """Detect whether an OTSurv survival matrix exhibits a floor plateau
    that starts inside the test lifetime.

    This is the metric-side counterpart of :func:`trim_breslow_floor_tail`:
    instead of trimming the curve for plotting, it inspects the per-row
    survival probabilities and reports the earliest ``t*`` at which any
    test sample enters a persistent floor tail of length ≥
    ``MIN_FLOOR_TAIL_POINTS`` whose onset time is ``≤ max(y_test_t)``.

    Used by :mod:`src.run_cross_validation` to emit the metric–
    visualization consistency flag (``OTSurvFloorTail`` CSV column +
    stdout warning) described by Property 4 of the spec design.

    Args:
        surv: 2-D array ``[N_test, n_times]`` of survival probabilities,
            **before** any downstream 0.001 clip. Typically the output of
            ``BreslowConverter.predict_survival(...).to_numpy()``.
        continuous_times: 1-D array aligned with ``surv``'s time axis.
        y_test_t: 1-D array of test-sample survival times (the lifetime
            inside which any plateau is considered "inside the test
            lifetime" for the bug-condition).

    Returns:
        The earliest ``continuous_times[idx]`` (as ``float``) where any
        row of ``surv`` enters a qualifying floor tail. Returns ``-1.0``
        when no qualifying plateau exists or when inputs are empty/
        malformed (same sentinel used by the ``OTSurvFloorTail`` CSV
        column for non-OTSurv rows and OTSurv rows that do not hit the
        plateau).
    """
    surv = np.asarray(surv, dtype=float)
    continuous_times = np.asarray(continuous_times, dtype=float)
    y_test_t = np.asarray(y_test_t, dtype=float)

    if surv.ndim != 2 or surv.size == 0:
        return -1.0
    if surv.shape[1] != continuous_times.shape[0]:
        return -1.0
    if y_test_t.size == 0:
        return -1.0

    max_test_t = float(np.max(y_test_t))
    near_floor_all = surv <= (SURVIVAL_FLOOR + SURVIVAL_FLOOR_TOL)

    earliest_t = None
    for row in range(surv.shape[0]):
        near_floor = near_floor_all[row]
        for idx in np.flatnonzero(near_floor):
            tail_len = len(continuous_times) - int(idx)
            if (
                tail_len >= MIN_FLOOR_TAIL_POINTS
                and np.all(near_floor[idx:])
                and continuous_times[int(idx)] <= max_test_t
            ):
                # Only flag as a floor plateau if the tail values are
                # near-constant AND near the SURVIVAL_FLOOR level (the bug).
                # Natural exponential decay or underflow to near-zero
                # is correct behavior (requirement 3.5).
                tail_values = surv[row, int(idx):]
                tail_max = float(np.max(tail_values))
                if tail_max >= SURVIVAL_FLOOR / 10.0:
                    tail_min = float(np.min(tail_values))
                    ratio = tail_min / tail_max
                    if ratio > 0.5:
                        candidate = float(continuous_times[int(idx)])
                        if earliest_t is None or candidate < earliest_t:
                            earliest_t = candidate
                # Earliest qualifying idx for this row found; stop
                # scanning further indices for this row.
                break

    return earliest_t if earliest_t is not None else -1.0


def trim_breslow_floor_tail(times, survival_curve):
    """Trim a trailing "floor plateau" from an OTSurv Breslow survival curve.

    Diagnostic helper reused by the plotting scripts. It detects the
    historical ``np.clip(·, 0.001, 1.0)`` plateau produced by the
    unfixed BreslowConverter (or any equivalent downstream clip) and
    returns the curve truncated just before the plateau starts.

    After the task 3 fix removes the hard 0.001 lower bound, this helper
    will normally return the curve untouched (``truncated_at=None``) and
    is kept as a defensive safety net for reloaded/cached outputs.

    A "plateau" is defined as a tail where all values are ≤ the threshold
    AND the values are near-constant (range < 1e-6). This distinguishes
    the artificial 0.001 floor (the bug) from natural exponential decay
    to 0 (correct behavior per requirement 3.5).

    Args:
        times: 1-D array of time points aligned with ``survival_curve``.
        survival_curve: 1-D array of survival probabilities S(t).

    Returns:
        (plot_times, plot_curve, truncated_at): if a qualifying plateau
        is found, arrays truncated to exclude the plateau and the time
        at which the plateau starts; otherwise the original arrays and
        ``truncated_at=None``.
    """
    times = np.asarray(times)
    survival_curve = np.asarray(survival_curve)
    near_floor = survival_curve <= (SURVIVAL_FLOOR + SURVIVAL_FLOOR_TOL)

    for idx in np.flatnonzero(near_floor):
        tail_len = len(survival_curve) - idx
        if tail_len >= MIN_FLOOR_TAIL_POINTS and np.all(near_floor[idx:]):
            # Only trim if the tail is a constant plateau near the
            # SURVIVAL_FLOOR level (the bug), not natural decay or
            # underflow to near-zero (correct behavior).
            tail_values = survival_curve[idx:]
            tail_max = float(np.max(tail_values))
            if tail_max >= SURVIVAL_FLOOR / 10.0:
                tail_min = float(np.min(tail_values))
                ratio = tail_min / tail_max
                if ratio > 0.5:
                    end = max(idx, 1)
                    return times[:end], survival_curve[:end], times[idx]

    return times, survival_curve, None


class BreslowConverter:
    """Breslow estimator: generates survival curves from Cox log-risk scores.

    Estimates the baseline cumulative hazard H_0(t) from training data,
    then computes individual survival S(t|x) = S_0(t)^exp(risk) for
    test samples.
    """

    def __init__(self):
        self.baseline_hazard_times_ = None
        self.baseline_cumulative_hazard_ = None
        # Cox risks are identifiable only up to a common additive constant.
        # Keep the fitted coordinate origin and cumulative baseline in log
        # space so large, finite logits are never exponentiated directly.
        self.log_risk_center_ = None
        self.baseline_log_cumulative_hazard_ = None
        self._centered_log_cumulative_hazard_ = None
        # Upper bound of the baseline hazard's training support. Time
        # points past this value trigger constant right-extrapolation of
        # H_0 (the only extrapolation mode currently supported).
        self.t_support_max_ = None
        # Boolean mask (per time point) set by the most recent call to
        # ``predict_survival`` that marks time points beyond
        # ``t_support_max_``. Downstream consumers (e.g. OTSurv floor
        # warning) can read this to flag extrapolated segments.
        self.last_extrapolation_mask_ = None

    def fit(self, log_risks: np.ndarray, times: np.ndarray,
            events: np.ndarray) -> "BreslowConverter":
        """Estimate baseline cumulative hazard from training data.

        Args:
            log_risks: Training set log-risk scores [N_train,].
            times: Training set survival times [N_train,].
            events: Training set event indicators [N_train,] (1=event).

        Returns:
            self
        """
        log_risks = np.asarray(log_risks, dtype=np.float64).ravel()
        times = np.asarray(times, dtype=np.float64).ravel()
        events = np.asarray(events, dtype=np.float64).ravel()

        if not (len(log_risks) == len(times) == len(events)):
            raise ValueError("log_risks, times, and events must have equal length")
        if not np.isfinite(log_risks).all():
            raise ValueError("log_risks must contain only finite values")
        if not np.isfinite(times).all():
            raise ValueError("times must contain only finite values")
        if not np.isfinite(events).all() or not np.all((events == 0) | (events == 1)):
            raise ValueError("events must contain only 0 or 1")

        self.log_risk_center_ = (
            float(np.max(log_risks)) if log_risks.size else 0.0
        )

        # Sort by time to build risk sets efficiently
        order = np.argsort(times)
        times_sorted = times[order]
        events_sorted = events[order]
        log_risks_sorted = log_risks[order]
        centered_log_risks = log_risks_sorted - self.log_risk_center_

        # Find unique event times (only where events occurred)
        event_mask = events_sorted == 1
        unique_event_times = np.unique(times_sorted[event_mask])

        # Build baseline cumulative hazard H_0(t)
        baseline_hazard_times = []
        centered_log_cumulative_hazard_values = []
        centered_log_cum_h = -np.inf

        for t_i in unique_event_times:
            # d_i: number of events at time t_i
            d_i = np.sum((times_sorted == t_i) & event_mask)

            # R(t_i): risk set — all subjects with survival time >= t_i
            risk_set_mask = times_sorted >= t_i
            risk_set_log_risks = centered_log_risks[risk_set_mask]

            # Re-center each risk-set denominator. The maximum-risk subject
            # may have left a later risk set, so one global shift alone does
            # not prevent exp underflow for every event time.
            risk_set_max = float(np.max(risk_set_log_risks))
            log_risk_sum = risk_set_max + np.log(
                np.exp(risk_set_log_risks - risk_set_max).sum()
            )
            log_increment = np.log(float(d_i)) - log_risk_sum
            centered_log_cum_h = np.logaddexp(
                centered_log_cum_h, log_increment
            )

            baseline_hazard_times.append(t_i)
            centered_log_cumulative_hazard_values.append(centered_log_cum_h)

        self.baseline_hazard_times_ = np.array(baseline_hazard_times)
        self._centered_log_cumulative_hazard_ = np.asarray(
            centered_log_cumulative_hazard_values, dtype=np.float64
        )
        self.baseline_log_cumulative_hazard_ = (
            self._centered_log_cumulative_hazard_ - self.log_risk_center_
        )
        # Preserve the public baseline attribute in conventional, uncentered
        # Cox coordinates. Prediction uses the exact log representation.
        with np.errstate(over="ignore", under="ignore"):
            baseline_cumulative_hazard = np.exp(
                self.baseline_log_cumulative_hazard_
            )
        self.baseline_cumulative_hazard_ = np.nan_to_num(
            baseline_cumulative_hazard,
            nan=0.0,
            posinf=np.finfo(np.float64).max,
            neginf=0.0,
        )
        # Record baseline-hazard support upper bound so predict_survival
        # can mark time points that fall into the constant right-
        # extrapolation segment.
        if self.baseline_hazard_times_.size > 0:
            self.t_support_max_ = float(self.baseline_hazard_times_.max())
        else:
            self.t_support_max_ = None
        return self

    def predict_survival(self, log_risks: np.ndarray,
                         time_points: np.ndarray,
                         extrapolation: str = "constant") -> pd.DataFrame:
        """Generate survival curves for test samples.

        Args:
            log_risks: Test set log-risk scores [N_test,].
            time_points: Evaluation time points array (continuous_times).
            extrapolation: How to handle ``time_points`` beyond
                ``self.t_support_max_``. Currently only ``"constant"``
                (right-constant extrapolation of H_0, the historical
                default) is supported. The kwarg is kept for forward
                compatibility so future callers can opt into
                ``"nan"``/``"zero_hazard"`` semantics without breaking
                the current contract.

        Returns:
            pd.DataFrame: Shape [N_test, len(time_points)], columns are
                time_points values, values are survival probabilities
                clipped to [0.0, 1.0]. No artificial 0.001 lower bound is
                applied.
        """
        if (
            self.baseline_hazard_times_ is None
            or self._centered_log_cumulative_hazard_ is None
            or self.log_risk_center_ is None
        ):
            raise RuntimeError("Must call fit() before predict_survival().")

        if extrapolation != "constant":
            raise ValueError(
                f"Unsupported extrapolation mode {extrapolation!r}; "
                f"only 'constant' is currently implemented."
            )

        log_risks = np.asarray(log_risks, dtype=np.float64).ravel()
        time_points = np.asarray(time_points, dtype=np.float64).ravel()
        if not np.isfinite(log_risks).all():
            raise ValueError("log_risks must contain only finite values")
        if not np.isfinite(time_points).all():
            raise ValueError("time_points must contain only finite values")

        # Interpolate H_0 at requested time_points using step function
        # H_0(t) = cumulative hazard at the largest event time <= t
        centered_log_h0_at_times = np.full(
            len(time_points), -np.inf, dtype=np.float64
        )
        for i, t in enumerate(time_points):
            # Find all baseline hazard times <= t
            mask = self.baseline_hazard_times_ <= t
            if np.any(mask):
                centered_log_h0_at_times[i] = (
                    self._centered_log_cumulative_hazard_[mask][-1]
                )
            # else: H_0(t) = 0 (before first event time)

        # Record which time points fall in the right-extrapolation region.
        if self.t_support_max_ is not None:
            self.last_extrapolation_mask_ = time_points > self.t_support_max_
        else:
            self.last_extrapolation_mask_ = np.zeros(
                len(time_points), dtype=bool
            )

        # S(t|x) = exp(-H_0(t) * exp(risk)). Form the cumulative-hazard
        # argument in log space so neither factor is exponentiated alone.
        shifted_log_risks = log_risks - self.log_risk_center_
        survival = np.ones(
            (len(log_risks), len(time_points)), dtype=np.float64
        )
        positive_hazard = np.isfinite(centered_log_h0_at_times)
        if np.any(positive_hazard):
            log_exponent_arg = (
                shifted_log_risks[:, np.newaxis]
                + centered_log_h0_at_times[positive_hazard][np.newaxis, :]
            )
            # Preserve the historical finite exp(-700) lower limit without
            # ever evaluating exp(large_log_risk).
            log_exponent_arg = np.minimum(
                log_exponent_arg, _LOG_EXP_ARG_MAX
            )
            exponent_arg = np.exp(log_exponent_arg)
            survival[:, positive_hazard] = np.exp(-exponent_arg)

        # Guard against any arithmetic edge case after the finite-input checks.
        survival = np.nan_to_num(survival, nan=0.0, posinf=1.0, neginf=0.0)

        # Clip to [0.0, 1.0] — no artificial lower bound. Values that
        # underflow to exactly 0 remain 0.
        survival = np.clip(survival, 0.0, 1.0)

        # Ensure S=1.0 at time 0
        for i, t in enumerate(time_points):
            if t == 0:
                survival[:, i] = 1.0

        return pd.DataFrame(survival, columns=time_points)


def nll_predict_survival(logits: np.ndarray) -> np.ndarray:
    """Convert NLL discrete loss logits to survival probabilities.

    Computes discrete survival curve from model logits:
        hazards = sigmoid(logits)
        S(t) = cumprod(1 - hazards)

    Args:
        logits: Model output logits [N, K] where K is the number of
            discrete time bins.

    Returns:
        np.ndarray: Survival probabilities [N, K], monotonically
            non-increasing along the time axis.
    """
    logits = np.asarray(logits, dtype=np.float64)

    # hazards = sigmoid(logits)
    hazards = 1.0 / (1.0 + np.exp(-logits))

    # S(t) = cumprod(1 - hazards) along time axis
    survival = np.cumprod(1.0 - hazards, axis=1)

    return survival
