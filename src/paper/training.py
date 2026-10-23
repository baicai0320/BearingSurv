from abc import ABC, abstractmethod
import copy
import logging
import math
import time
from hashlib import sha256
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from paper import config as cfg
try:
    from bnnsurv import models
except ImportError:
    # BNNSurv is optional for raw-signal-only OTSurv/R13 runs. Keeping the
    # import lazy lets those independent models run in a minimal PyTorch env.
    models = None
from utility.mtlr import mtlr

# 添加 dsm 模块路径
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from dsm import DeepSurvivalMachines, DeepRecurrentSurvivalMachines


BNNSURV_NUMERICAL_STATUS_OK = "ok"
BNNSURV_BRESLOW_STRATEGY = "logsumexp_breslow_v1"


def stable_breslow_survival_probabilities(
    train_log_risks,
    train_event_times,
    train_event_indicators,
    test_log_risks,
    evaluation_times,
) -> np.ndarray:
    """Evaluate Breslow survival curves in the log domain.

    ``sksurv``'s :class:`BreslowEstimator` evaluates ``exp(log_risk)``
    directly.  Neural log-risks can be finite while still exceeding the
    floating-point exponent range, which turns the baseline hazard and all
    downstream survival probabilities into ``NaN``.  This implementation
    preserves the Breslow definition, including tied events, but evaluates
    each risk-set denominator with log-sum-exp and only exponentiates the
    final cumulative hazard when that operation is representable.
    """

    train_risks = np.asarray(train_log_risks, dtype=np.float64).reshape(-1)
    train_times = np.asarray(train_event_times, dtype=np.float64).reshape(-1)
    train_events_raw = np.asarray(train_event_indicators).reshape(-1)
    test_risks = np.asarray(test_log_risks, dtype=np.float64).reshape(-1)
    targets = np.asarray(evaluation_times, dtype=np.float64).reshape(-1)

    if train_risks.size == 0:
        raise ValueError("BNNSurv Breslow fit requires at least one development row")
    if not (
        train_risks.size == train_times.size == train_events_raw.size
    ):
        raise ValueError("BNNSurv Breslow development arrays must align")
    if not np.isfinite(train_risks).all() or not np.isfinite(test_risks).all():
        raise ValueError("BNNSurv Breslow log-risks must be finite")
    if not np.isfinite(train_times).all() or not np.isfinite(targets).all():
        raise ValueError("BNNSurv Breslow times must be finite")
    if np.any(train_times < 0.0) or np.any(targets < 0.0):
        raise ValueError("BNNSurv Breslow times must be non-negative")
    try:
        train_events_numeric = train_events_raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("BNNSurv Breslow event indicators must be binary") from exc
    if (
        not np.isfinite(train_events_numeric).all()
        or not np.isin(train_events_numeric, (0.0, 1.0)).all()
    ):
        raise ValueError("BNNSurv Breslow event indicators must be binary")
    train_events = train_events_numeric.astype(bool)

    unique_times = np.unique(train_times)
    log_cumulative_hazard = np.full(unique_times.size, -np.inf, dtype=np.float64)
    cumulative_log_value = -np.inf
    for position, event_time in enumerate(unique_times):
        risk_set_scores = train_risks[train_times >= event_time]
        if risk_set_scores.size == 0:
            raise RuntimeError("BNNSurv Breslow encountered an empty risk set")
        max_score = float(np.max(risk_set_scores))
        with np.errstate(over="ignore", under="ignore"):
            log_denominator = max_score + float(
                np.log(np.exp(risk_set_scores - max_score).sum())
            )
        event_count = int(
            np.count_nonzero(train_events & (train_times == event_time))
        )
        if event_count:
            log_increment = float(np.log(event_count) - log_denominator)
            cumulative_log_value = float(
                np.logaddexp(cumulative_log_value, log_increment)
            )
        log_cumulative_hazard[position] = cumulative_log_value

    target_positions = np.searchsorted(unique_times, targets, side="right") - 1
    target_log_hazard = np.full(targets.size, -np.inf, dtype=np.float64)
    supported = target_positions >= 0
    target_log_hazard[supported] = log_cumulative_hazard[target_positions[supported]]
    with np.errstate(over="ignore", under="ignore"):
        log_subject_hazard = (
            test_risks[:, np.newaxis] + target_log_hazard[np.newaxis, :]
        )

    survival = np.ones(log_subject_hazard.shape, dtype=np.float64)
    # exp(-exp(x)) underflows to zero after this point.  Avoid evaluating the
    # nested exponential beyond its representable range.
    zero_survival_threshold = float(
        np.log(-np.log(np.nextafter(0.0, 1.0)))
    )
    zero_mask = log_subject_hazard >= zero_survival_threshold
    survival[zero_mask] = 0.0
    finite_mask = np.isfinite(log_subject_hazard) & ~zero_mask
    survival[finite_mask] = np.exp(-np.exp(log_subject_hazard[finite_mask]))
    if not np.isfinite(survival).all():
        raise RuntimeError("BNNSurv log-domain Breslow returned non-finite survival")
    return survival


if models is not None:
    class FullDevelopmentMCD(models.MCD):
        """BNNSurv MCD with deterministic, inclusive fixed-epoch batches.

        The upstream implementation unconditionally uses ``drop_last=True``.
        That silently excludes development rows whenever the cohort size is
        not divisible by the configured batch size, which is incompatible with
        the outer-CV full-development fitting contract.
        """

        is_bnnsurv_mcd_ = True

        def __init__(self, *args, random_seed=0, **kwargs):
            super().__init__(*args, **kwargs)
            self.random_seed = int(random_seed)

        @staticmethod
        def _tensor_is_finite(tf_module, value) -> bool:
            return bool(
                tf_module.reduce_all(tf_module.math.is_finite(value)).numpy()
            )

        def _mark_training_numerical_failure(
            self,
            reason: str,
            *,
            epoch: int | None = None,
            batch: int | None = None,
        ) -> None:
            self.fit_numerical_status_ = f"na:bnnsurv_training_{reason}"
            self.fit_numerical_failure_ = {
                "reason": str(reason),
                "epoch": None if epoch is None else int(epoch),
                "batch": None if batch is None else int(batch),
            }
            self.fixed_fit_epochs_ = int(getattr(self, "n_epochs", 0))
            self.fixed_fit_row_count_ = int(len(getattr(self, "training_data", ())))
            self.fixed_fit_batch_count_ = int(
                getattr(self, "fixed_fit_batch_count_", 0)
            )
            self.fixed_fit_dropped_rows_ = 0
            self.fixed_fit_validation_used_ = False
            self.fixed_fit_final_loss_ = np.nan
            self.fixed_fit_completed_ = False

        def _invalid_prediction(
            self,
            sample_count: int,
            row_count: int,
            time_count: int,
            reason: str,
        ):
            self.last_prediction_status_ = f"na:bnnsurv_prediction_{reason}"
            self.last_prediction_log_risk_min_ = np.nan
            self.last_prediction_log_risk_max_ = np.nan
            return np.full(
                (int(sample_count), int(row_count), int(time_count)),
                np.nan,
                dtype=np.float64,
            )

        def fit(self, X, y_time, y_event):
            tf_module = getattr(models, 'tf', None)
            if tf_module is None:
                raise ImportError('BNNSurv TensorFlow backend is unavailable')

            features = np.asarray(X, dtype=np.float32)
            event_times = np.asarray(y_time, dtype=np.float32).reshape(-1)
            events = np.asarray(y_event, dtype=np.int32).reshape(-1)
            if features.ndim != 2 or features.shape[0] == 0:
                raise ValueError('BNNSurv fixed fit requires a non-empty 2D feature matrix')
            if event_times.size != features.shape[0] or events.size != features.shape[0]:
                raise ValueError('BNNSurv feature/outcome lengths must align')
            if not np.isfinite(features).all() or not np.isfinite(event_times).all():
                raise ValueError('BNNSurv fixed fit received non-finite values')
            if not np.isin(events, (0, 1)).all():
                raise ValueError('BNNSurv event indicators must be binary')

            self.training_data = features
            self.t_train = event_times
            self.e_train = events
            self.fit_numerical_status_ = BNNSURV_NUMERICAL_STATUS_OK
            self.fit_numerical_failure_ = None
            self.last_prediction_status_ = "not_run"
            self.last_prediction_log_risk_min_ = np.nan
            self.last_prediction_log_risk_max_ = np.nan
            self.breslow_prediction_strategy_ = BNNSURV_BRESLOW_STRATEGY
            self.fixed_fit_completed_ = False
            self.model = models.make_mcd_model(
                input_shape=features.shape[1:],
                output_dim=2,
                layers=self.layers,
                activation_fn=self.activation_fn,
                dropout_rate=self.dropout,
                regularization_pen=self.l2_reg,
            )
            batch_size = int(self.batch_size)
            if batch_size <= 0:
                raise ValueError('BNNSurv fixed fit requires batch_size > 0')
            batch_count = 0
            last_loss = np.nan
            for epoch in range(int(self.n_epochs)):
                index = np.arange(features.shape[0])
                np.random.RandomState(self.random_seed + epoch).shuffle(index)
                for batch, start in enumerate(range(0, features.shape[0], batch_size)):
                    batch_index = index[start:min(start + batch_size, features.shape[0])]
                    batch_times = event_times[batch_index]
                    batch_events = events[batch_index]
                    batch_features = tf_module.convert_to_tensor(
                        features[batch_index], dtype=tf_module.float32
                    )
                    event_tensor = tf_module.expand_dims(
                        tf_module.convert_to_tensor(batch_events, dtype=tf_module.int32),
                        axis=1,
                    )
                    # ``bnnsurv.models`` does not re-export its private
                    # risk-set helper. Its documented definition is y_j >=
                    # y_i for each row i, which is reproduced here directly.
                    riskset = batch_times[None, :] >= batch_times[:, None]
                    riskset_tensor = tf_module.convert_to_tensor(
                        riskset, dtype=tf_module.bool
                    )
                    with tf_module.GradientTape() as tape:
                        logits = self.model(batch_features, training=True)
                        loss = self.loss_fn(
                            y_true=[event_tensor, riskset_tensor], y_pred=logits
                        )
                    gradients = tape.gradient(loss, self.model.trainable_weights)
                    if not self._tensor_is_finite(tf_module, loss):
                        self._mark_training_numerical_failure(
                            "loss_non_finite", epoch=epoch, batch=batch
                        )
                        return self
                    if any(gradient is None for gradient in gradients):
                        self._mark_training_numerical_failure(
                            "gradient_missing", epoch=epoch, batch=batch
                        )
                        return self
                    if not all(
                        self._tensor_is_finite(tf_module, gradient)
                        for gradient in gradients
                    ):
                        self._mark_training_numerical_failure(
                            "gradient_non_finite", epoch=epoch, batch=batch
                        )
                        return self
                    try:
                        self.optimizer.apply_gradients(
                            zip(gradients, self.model.trainable_weights)
                        )
                    except Exception:
                        self._mark_training_numerical_failure(
                            "optimizer_apply_failed", epoch=epoch, batch=batch
                        )
                        return self
                    if not all(
                        self._tensor_is_finite(tf_module, weight)
                        for weight in self.model.trainable_weights
                    ):
                        self._mark_training_numerical_failure(
                            "weight_non_finite", epoch=epoch, batch=batch
                        )
                        return self
                    last_loss = float(tf_module.reduce_mean(loss).numpy())
                    batch_count += 1
                    self.fixed_fit_batch_count_ = int(batch_count)

            self.fixed_fit_epochs_ = int(self.n_epochs)
            self.fixed_fit_row_count_ = int(features.shape[0])
            self.fixed_fit_batch_count_ = int(batch_count)
            self.fixed_fit_dropped_rows_ = 0
            self.fixed_fit_validation_used_ = False
            self.fixed_fit_final_loss_ = last_loss
            self.fixed_fit_completed_ = True
            return self

        def predict_survival(self, X, event_times, n_post_samples=100):
            """Return posterior curves without direct exponentiation overflow.

            A numerical failure is represented by an all-``NaN`` curve plus a
            machine-readable status.  The runner persists that raw output and
            records dependent metrics as ``null`` instead of terminating the
            other models in the same cross-validation shard.
            """

            features = np.asarray(X, dtype=np.float32)
            times = np.asarray(event_times, dtype=np.float64).reshape(-1)
            samples = int(n_post_samples)
            row_count = int(features.shape[0]) if features.ndim >= 1 else 0
            if (
                features.ndim != 2
                or row_count == 0
                or times.size == 0
                or samples < 1
                or not np.isfinite(features).all()
                or not np.isfinite(times).all()
                or np.any(times < 0.0)
            ):
                return self._invalid_prediction(
                    samples, row_count, times.size, "input_invalid"
                )
            if getattr(self, "fit_numerical_status_", BNNSURV_NUMERICAL_STATUS_OK) != (
                BNNSURV_NUMERICAL_STATUS_OK
            ):
                return self._invalid_prediction(
                    samples, row_count, times.size, "training_non_finite"
                )
            try:
                train_predictions = np.asarray(
                    self.model.predict(self.training_data, verbose=False),
                    dtype=np.float64,
                ).reshape(-1)
                if train_predictions.size != self.training_data.shape[0]:
                    raise ValueError("development_log_risk_shape_invalid")
                posterior_predictions = np.empty(
                    (samples, row_count), dtype=np.float64
                )
                for sample_index in range(samples):
                    risk_prediction = np.asarray(
                        self.model.predict(features, verbose=False),
                        dtype=np.float64,
                    ).reshape(-1)
                    if risk_prediction.size != row_count:
                        raise ValueError("test_log_risk_shape_invalid")
                    posterior_predictions[sample_index] = risk_prediction
                all_predictions = np.concatenate(
                    (train_predictions, posterior_predictions.reshape(-1))
                )
                if not np.isfinite(all_predictions).all():
                    raise ValueError("log_risk_non_finite")

                curves = np.empty((samples, row_count, times.size), dtype=np.float64)
                for sample_index, risk_prediction in enumerate(posterior_predictions):
                    curves[sample_index] = stable_breslow_survival_probabilities(
                        train_predictions,
                        self.t_train,
                        self.e_train,
                        risk_prediction,
                        times,
                    )
                if not np.isfinite(curves).all():
                    raise RuntimeError("survival_curve_non_finite")
            except Exception as exc:
                return self._invalid_prediction(
                    samples, row_count, times.size, type(exc).__name__.lower()
                )

            self.last_prediction_status_ = BNNSURV_NUMERICAL_STATUS_OK
            self.last_prediction_log_risk_min_ = float(np.min(all_predictions))
            self.last_prediction_log_risk_max_ = float(np.max(all_predictions))
            return curves
else:
    FullDevelopmentMCD = None


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class BaseRegressor (ABC):
    """
    Base class for regressors.
    """
    def __init__ (self):
        """Initilizes inputs and targets variables."""

    @abstractmethod
    def make_model (self, params=None):
        """
        This method is an abstract method to be implemented
        by a concrete classifier. Must return a sklearn-compatible
        estimator object implementing 'fit'.
        """

    @abstractmethod
    def get_hyperparams (self):
        """Method"""

    @abstractmethod
    def get_hyperparams (self, condition):
        """Method"""

    def get_tuneable_params (self):
        return self.get_hyperparams()

    def get_best_params (self):
        return self.get_hyperparams()

    def get_estimator (self, params=None):
        return self.make_model(params)

class CoxPH(BaseRegressor):
    def make_model (self, params=None):
        model_params = cfg.PARAMS_CPH.copy()
        if params:
            model_params.update(params)
        return CoxPHSurvivalAnalysis(**model_params)
    """
    def get_hyperparams (self):
        return {'n_iter': [100000],
                'tol': [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]}
    """
    def get_hyperparams (self, condition):
        if condition == 0:
            return {'tol': 1e-09, 'n_iter': 100}
        elif condition == 1:
            return {'tol': 1e-09, 'n_iter': 100}
        elif condition == 2:
            return {'tol': 1e-09, 'n_iter': 100}
        else:
            raise ValueError("Invalid condition, choose {0, 1, 2}")

class CoxBoost (BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_COXBOOST.copy()
        if params:
            model_params.update(params)
        return GradientBoostingSurvivalAnalysis(**model_params)
    """
    def get_hyperparams(self):
        return {'learning_rate': [0.1, 0.5, 1.0],
                'n_estimators': [100, 200, 400],
                'max_depth': [3, 5, 7],
                'min_samples_split': [float(x) for x in np.linspace(0.1, 0.9, 5, endpoint=True)],
                'min_samples_leaf': [float(x) for x in np.linspace(0.1, 0.5, 5, endpoint=True)],
                'max_features': [None, "log2", "sqrt"],
                'dropout_rate': [float(x) for x in np.linspace(0.0, 0.9, 10, endpoint=True)],
                'subsample': [float(x) for x in np.linspace(0.1, 1.0, 10, endpoint=True)]}
    """  
    def get_hyperparams(self, condition):
        if condition == 0:
            return {'learning_rate': 0.1, 'n_estimators': 100, 'loss': 'coxph',
                    'dropout_rate': 0.4, 'max_features': 'log2', 'max_depth': 3,
                    'min_samples_split': 5, 'min_samples_leaf': 3, 'subsample': 1.0}
        elif condition == 1:
            return {'learning_rate': 0.1, 'n_estimators': 200, 'loss': 'coxph',
                    'dropout_rate': 0.4, 'max_features': 'log2', 'max_depth': 5,
                    'min_samples_split': 10, 'min_samples_leaf': 5, 'subsample': 1.0}
        elif condition == 2:
            return {'learning_rate': 0.1, 'n_estimators': 400, 'loss': 'coxph',
                    'dropout_rate': 0.4, 'max_features': 'log2', 'max_depth': 7,
                    'min_samples_split': 20, 'min_samples_leaf': 10, 'subsample': 1.0}
        else:
            raise ValueError("Invalid condition, choose {0, 1, 2}")

class RSF(BaseRegressor):
    def make_model (self, params=None):
        model_params = cfg.PARAMS_RSF.copy()
        if params:
            model_params.update(params)
        return RandomSurvivalForest(**model_params)
    """
    def get_hyperparams (self):
        return {'max_depth': [3, 5, 7],
                'n_estimators': [100, 200, 400],
                'min_samples_split': [float(x) for x in np.linspace(0.1, 0.9, 5, endpoint=True)],
                'min_samples_leaf': [float(x) for x in np.linspace(0.1, 0.5, 5, endpoint=True)]}
    """
    def get_hyperparams (self, n_condition):
        if n_condition == 0:
            return  {'n_estimators': 100, 'min_samples_split': 5, 'min_samples_leaf': 3,  'max_depth': 3}
        elif n_condition == 1:
            return  {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 5,  'max_depth': 5}
        elif n_condition == 2:
            return  {'n_estimators': 400, 'min_samples_split': 20, 'min_samples_leaf': 10,  'max_depth': 7}
        else:
            raise ValueError("Invalid condition, choose {0, 1, 2}")

class MTLR(BaseRegressor):
    def make_model(self, num_features, num_time_bins, config):
        model = mtlr(in_features=num_features, num_time_bins=num_time_bins, config=config)
        return model
    """
    def get_hyperparams(self):
        return {'batch_size' : [32, 64, 128],
                'dropout' : [0.25, 0.5, 0.6],
                'lr' : [0.00008],
                'c1': [0.01],
                'num_epochs': [100, 500, 1000, 5000],
                'hidden_size': [32, 50, 60, 128]}
    """
    def get_hyperparams(self, condition):
        if condition == 0:
            return {'batch_size': 32, 'dropout': 0.25, 'lr': 0.00008, 'c1': 0.01, 'num_epochs': 5000, 'hidden_size': 16}
        elif condition == 1:
            return {'batch_size': 64, 'dropout': 0.25, 'lr': 0.00008, 'c1': 0.01, 'num_epochs': 5000, 'hidden_size': 32}
        elif condition == 2:
            return {'batch_size': 128, 'dropout': 0.25, 'lr': 0.00008, 'c1': 0.01, 'num_epochs': 5000, 'hidden_size': 64}
        else:
            raise ValueError("Invalid condition, choose {0, 1, 2}")

class DSM(BaseRegressor):
    def make_model(self, params=None):
        model_params = cfg.PARAMS_DSM.copy()
        if params:
            model_params.update(params)
        return DeepSurvivalMachines(
            k=model_params['k'],
            layers=model_params['layers'],
            distribution=model_params['distribution'],
            random_seed=int(model_params.get('random_seed', 0)),
        )
    
    def get_hyperparams(self, condition):
        # DSM 超参数配置 - 从 config.py 读取并根据工况调整
        # 注意: 这些是经过调优的参数，不同工况可能需要不同配置
        base_params = cfg.PARAMS_DSM.copy()
        
        # 根据工况调整参数（可以通过网格搜索优化这些值）
        if condition == 0:
            # 工况0: 使用默认配置
            return base_params
        elif condition == 1:
            # 工况1: 可能需要更多混合分布
            base_params['k'] = 6
            return base_params
        elif condition == 2:
            # 工况2: 可能需要更深的网络
            base_params['k'] = 6
            base_params['layers'] = [64]
            return base_params
        else:
            raise ValueError("Invalid condition, choose {0, 1, 2}")

class BNNSurv(BaseRegressor):
    def make_model(self, params=None):
        if models is None:
            raise ImportError(
                "BNNSurv requires the optional 'bnnsurv' package. "
                "Install it before selecting the BNNSurv model."
            )
        model_params = cfg.PARAMS_BNN.copy()
        if params:
            model_params.update(params)
        random_seed = int(model_params.pop('random_seed', 0))
        if FullDevelopmentMCD is None:
            raise ImportError('BNNSurv TensorFlow backend is unavailable')
        return FullDevelopmentMCD(random_seed=random_seed, **model_params)
    """
    def get_hyperparams(self):
        return {'batch_size' : [32, 64, 128],
                'learning_rate' : [1e-2, 5e-3, 1e-3],
                'num_epochs': [10, 15, 20],
                'layers': [[16], [32], [64], [128]]}
    """
    def get_hyperparams(self, condition):
        if condition == 0:
            return {'batch_size' : 32, 'learning_rate' : 0.001, 'num_epochs': 100, 'layers': [16]}
        elif condition == 1:
            return {'batch_size' : 64, 'learning_rate' : 0.001, 'num_epochs': 100, 'layers': [32]}
        elif condition == 2:
            return {'batch_size' : 128, 'learning_rate' : 0.001, 'num_epochs': 100, 'layers': [64]}
        else:
            raise ValueError("Invalid condition, choose {0, 1, 2}")

class OTSurvRegressor(BaseRegressor):
    """Shared numerical helpers for the final ranking endpoint only."""
    @staticmethod
    def _validate_amplitude_sample_meta(sample_meta, n_samples, *, context):
        if not isinstance(sample_meta, dict):
            raise ValueError(
                f"{context} requires bearing_id and time_step sample metadata"
            )
        bearing_ids = np.asarray(sample_meta.get('bearing_id'))
        time_steps = np.asarray(sample_meta.get('time_step'))
        if (
            bearing_ids.ndim != 1
            or time_steps.ndim != 1
            or len(bearing_ids) != n_samples
            or len(time_steps) != n_samples
        ):
            raise ValueError(
                f"{context} requires aligned one-dimensional sample metadata"
            )
        try:
            bearing_numbers = bearing_ids.astype(np.float64)
            time_numbers = time_steps.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context} sample metadata must be numeric") from exc
        if (
            not np.isfinite(bearing_numbers).all()
            or not np.isfinite(time_numbers).all()
            or np.any(bearing_numbers != np.floor(bearing_numbers))
            or np.any(time_numbers < 0.0)
            or np.any(time_numbers != np.floor(time_numbers))
        ):
            raise ValueError(
                f"{context} requires finite integer bearing_id/time_step values"
            )
        keys = np.column_stack(
            (bearing_numbers.astype(np.int64), time_numbers.astype(np.int64))
        )
        if len(np.unique(keys, axis=0)) != n_samples:
            raise ValueError(f"{context} requires unique bearing_id/time_step pairs")
        return keys[:, 0], keys[:, 1]

    @staticmethod
    def _initialize_amplitude_head_from_train_cox(
        model,
        train_amplitudes,
        train_times,
        train_events,
        alpha,
    ):
        """Warm-start the optional amplitude head with inner-train Cox risk."""
        if getattr(model, 'amplitude_head', None) is None:
            raise ValueError("Cox amplitude initialization requires an amplitude head")
        train_amplitudes = np.asarray(train_amplitudes, dtype=np.float64)
        train_times = np.asarray(train_times, dtype=np.float64)
        train_events = np.asarray(train_events, dtype=bool)
        if (
            train_amplitudes.ndim != 2
            or train_amplitudes.shape[1] != model.amplitude_head.in_features
            or len(train_amplitudes) != len(train_times)
            or len(train_times) != len(train_events)
            or not np.isfinite(train_amplitudes).all()
            or not np.isfinite(train_times).all()
            or np.any(train_times <= 0.0)
        ):
            raise ValueError("Cox amplitude initialization received invalid train data")
        if not train_events.any():
            raise ValueError("Cox amplitude initialization requires an observed train event")
        if not np.isfinite(float(alpha)) or float(alpha) < 0.0:
            raise ValueError("amplitude Cox alpha must be finite and non-negative")

        from sksurv.util import Surv

        cox_model = CoxPHSurvivalAnalysis(alpha=float(alpha))
        cox_model.fit(
            train_amplitudes,
            Surv.from_arrays(event=train_events, time=train_times),
        )
        coefficients = np.asarray(cox_model.coef_, dtype=np.float32).reshape(-1)
        feature_dim = int(model.amplitude_head.in_features)
        if coefficients.shape != (feature_dim,) or not np.isfinite(coefficients).all():
            raise RuntimeError("Cox amplitude initialization produced invalid coefficients")

        with torch.no_grad():
            model.amplitude_head.weight.copy_(
                torch.as_tensor(
                    coefficients.reshape(1, feature_dim),
                    device=model.amplitude_head.weight.device,
                    dtype=model.amplitude_head.weight.dtype,
                )
            )
            if model.amplitude_head.bias is not None:
                model.amplitude_head.bias.zero_()
        return coefficients

    @staticmethod
    def _emit(message: str, verbose: bool = False, log_fn=None):
        if verbose:
            print(message, flush=True)
        if log_fn is not None:
            log_fn(message)

    @staticmethod
    def _resolve_device(device) -> torch.device:
        """Validate *device* and fall back to CPU when GPU is unavailable.

        Args:
            device: Requested ``torch.device`` or device string.

        Returns:
            Resolved ``torch.device``.
        """
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            logger.info(
                "CUDA device requested but not available – falling back to CPU."
            )
            device = torch.device('cpu')
        return device

class R13OTSurvRegressor(OTSurvRegressor):
    """Adapter for the end-to-end R13 self-reference OT model.

    R13 does not use the legacy prototype-attention input contract.  It keeps
    each bearing's raw minutes as one causal trajectory, builds both branches
    in a shared Torch forward pass, and optimizes their summed log-risk with
    the exact complete-risk-set Cox objective.
    """

    MODE = 'r13'

    def make_model(self, params=None, mode=None):
        """Instantiate :class:`otsurv.model_r13_otsurv.R13OTSurv`."""
        from otsurv.model_r13_otsurv import R13OTSurv

        model_params = dict(cfg.PARAMS_R13_OTSURV)
        if params:
            model_params.update(params)
        requested_mode = mode if mode is not None else model_params.get('mode', self.MODE)
        if requested_mode != self.MODE:
            raise ValueError("R13OTSurvRegressor requires mode='r13'")
        self._validate_r13_params(model_params)
        return R13OTSurv(
            window_length=int(model_params['window_length']),
            reference_length=int(model_params['r13_reference_length']),
            transport_batch_size=int(model_params['r13_transport_batch_size']),
            ot_epsilon=float(model_params['ot_epsilon']),
            ot_num_iters=int(model_params['ot_num_iters']),
            ot_stoperr=float(model_params['ot_stoperr']),
            shape_projection_trainable=bool(
                model_params['r13_shape_projection_trainable']
            ),
            sinkhorn_checkpoint=bool(model_params['r13_sinkhorn_checkpoint']),
        )

    def get_hyperparams(self, condition):
        if condition not in {0, 1, 2}:
            raise ValueError("Invalid condition, choose {0, 1, 2}")
        params = dict(cfg.PARAMS_R13_OTSURV)
        params.update(cfg.PARAMS_R13_OTSURV_CONDITION_OVERRIDES.get(condition, {}))
        self._validate_r13_params(params)
        return params

    @staticmethod
    def _validate_r13_params(params):
        if not isinstance(params, dict):
            raise ValueError("R13 parameters must be a dictionary")
        expected_keys = set(cfg.PARAMS_R13_OTSURV)
        missing = sorted(expected_keys - set(params))
        unexpected = sorted(set(params) - expected_keys)
        if missing:
            raise ValueError(f"R13 parameters are missing required keys: {missing}")
        if unexpected:
            raise ValueError(
                "R13 parameters include unsupported legacy keys: "
                f"{unexpected}"
            )
        if params['mode'] != R13OTSurvRegressor.MODE:
            raise ValueError("R13 parameters require mode='r13'")
        if params['loss_fn'] != 'cox':
            raise ValueError("R13 supports only exact Cox training")
        if params['raw_signal_normalization'] != 'none':
            raise ValueError("R13 requires physical-scale raw_signal_normalization='none'")
        for key in (
            'r13_shape_projection_trainable',
            'r13_sinkhorn_checkpoint',
            'r13_orthogonalize_trajectory_to_main',
        ):
            if not isinstance(params[key], bool):
                raise ValueError(f"{key} must be a boolean")
        if params['r13_amplitude_init'] not in {'random', 'cox_train'}:
            raise ValueError("r13_amplitude_init must be 'random' or 'cox_train'")
        integer_keys = (
            'window_length',
            'stride',
            'r13_reference_length',
            'r13_transport_batch_size',
            'ot_num_iters',
            'max_epochs',
            'early_stopping_patience',
            'min_epochs_before_stopping',
        )
        for key in integer_keys:
            value = params[key]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{key} must be an integer")
            if int(value) < 1:
                raise ValueError(f"{key} must be positive")
        for key in ('r13_amplitude_freeze_epochs', 'r13_shape_projection_freeze_epochs'):
            value = params[key]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{key} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{key} must be non-negative")
        numeric_keys = (
            'ot_epsilon',
            'ot_stoperr',
            'lr',
            'r13_weight_decay',
            'r13_amplitude_cox_alpha',
            'r13_amplitude_anchor_weight',
            'r13_amplitude_aux_cox_weight',
            'r13_ot_head_l2_weight',
            'r13_ot_risk_orthogonality_weight',
            'r13_risk_temperature',
        )
        for key in numeric_keys:
            value = params[key]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"{key} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{key} must be finite")
        window_length = int(params['window_length'])
        stride = int(params['stride'])
        if window_length != 512 or stride != window_length:
            raise ValueError(
                "R13 requires 512-sample non-overlapping waveform windows"
            )
        if float(params['ot_epsilon']) <= 0.0:
            raise ValueError("ot_epsilon must be positive")
        if float(params['ot_stoperr']) < 0.0:
            raise ValueError("ot_stoperr must be non-negative")
        if float(params['lr']) <= 0.0:
            raise ValueError("R13 lr must be positive")
        if float(params['r13_weight_decay']) < 0.0:
            raise ValueError("r13_weight_decay must be non-negative")
        if float(params['r13_risk_temperature']) <= 0.0:
            raise ValueError("r13_risk_temperature must be positive")
        for key in (
            'r13_amplitude_cox_alpha',
            'r13_amplitude_anchor_weight',
            'r13_amplitude_aux_cox_weight',
            'r13_ot_head_l2_weight',
            'r13_ot_risk_orthogonality_weight',
        ):
            if float(params[key]) < 0.0:
                raise ValueError(f"{key} must be non-negative")
        if (
            params['r13_amplitude_init'] == 'random'
            and float(params['r13_amplitude_anchor_weight']) > 0.0
        ):
            raise ValueError(
                "r13_amplitude_anchor_weight requires r13_amplitude_init='cox_train'"
            )
        if int(params['max_epochs']) < 1:
            raise ValueError("R13 max_epochs must be positive")
        if int(params['early_stopping_patience']) < 1:
            raise ValueError("R13 early_stopping_patience must be positive")
        if int(params['min_epochs_before_stopping']) < 1:
            raise ValueError("R13 min_epochs_before_stopping must be positive")
        if int(params['min_epochs_before_stopping']) > int(params['max_epochs']):
            raise ValueError("R13 min_epochs_before_stopping exceeds max_epochs")
        warmup_epochs = max(
            int(params['r13_amplitude_freeze_epochs']),
            int(params['r13_shape_projection_freeze_epochs']),
        )
        if int(params['max_epochs']) <= warmup_epochs:
            raise ValueError(
                "R13 max_epochs must exceed the longest warm-up period so "
                "checkpoint selection occurs after warm-up"
            )

    @staticmethod
    def _r13_apply_risk_temperature(risks, params):
        """Apply the rank-preserving post-Cox calibration scale."""

        values = np.asarray(risks, dtype=np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError("R13 log risks must be finite before calibration")
        temperature = float(params['r13_risk_temperature'])
        scaled = values * temperature
        if not np.isfinite(scaled).all():
            raise RuntimeError("R13 risk temperature produced non-finite risks")
        return scaled.astype(np.float32, copy=False)

    @staticmethod
    def _stack_r13_signals(X_raw, device):
        if not isinstance(X_raw, (list, tuple)) or not X_raw:
            raise ValueError("R13 requires a non-empty list of raw signals")
        signals = []
        expected_length = None
        for signal in X_raw:
            if (
                not torch.is_tensor(signal)
                or signal.ndim != 2
                or signal.shape[1] != 2
                or signal.shape[0] < 1
                or not torch.isfinite(signal).all()
            ):
                raise ValueError("R13 raw signals must be finite [samples, 2] tensors")
            if expected_length is None:
                expected_length = int(signal.shape[0])
            elif int(signal.shape[0]) != expected_length:
                raise ValueError("R13 requires equal raw-signal lengths in one forward")
            signals.append(signal.to(device=device, dtype=torch.float32))
        return torch.stack(signals, dim=0)

    @staticmethod
    def _r13_metadata(sample_meta, n_samples, device, *, context):
        bearing_ids, time_steps = OTSurvRegressor._validate_amplitude_sample_meta(
            sample_meta,
            n_samples,
            context=context,
        )
        return (
            torch.as_tensor(bearing_ids, device=device, dtype=torch.long),
            torch.as_tensor(time_steps, device=device, dtype=torch.long),
        )

    @staticmethod
    def _calibration_statistics(values, *, allow_constant):
        if not torch.is_tensor(values) or values.ndim != 2 or values.shape[0] < 1:
            raise ValueError("R13 calibration requires a non-empty [rows, features] tensor")
        if not torch.isfinite(values).all():
            raise ValueError("R13 calibration values must be finite")
        center = values.mean(dim=0)
        scale = values.std(dim=0, unbiased=False)
        epsilon = torch.finfo(values.dtype).eps
        keep = scale > epsilon
        if not allow_constant and not bool(keep.all()):
            raise ValueError("R13 calibration requires nonzero training variance")
        if not bool(keep.any()):
            raise ValueError("R13 calibration has no nonconstant features")
        safe_center = torch.where(keep, center, torch.zeros_like(center))
        safe_scale = torch.where(keep, scale, torch.ones_like(scale))
        return safe_center.detach(), safe_scale.detach(), keep.detach()

    def _fit_r13_calibrations(
        self,
        model,
        train_signals,
        train_bearings,
        train_steps,
    ):
        """Fit every non-learned transform from inner-train bearings only."""
        was_training = model.training
        model.eval()
        with torch.no_grad():
            amplitude_raw = model.amplitude_features(
                train_signals, train_bearings, train_steps
            )
            amplitude_center, amplitude_scale, _ = self._calibration_statistics(
                amplitude_raw, allow_constant=False
            )
            model.set_amplitude_calibration(amplitude_center, amplitude_scale)

            token_raw = model.raw_shape_tokens(train_signals)
            token_center, token_scale, _ = self._calibration_statistics(
                token_raw.reshape(-1, token_raw.shape[-1]), allow_constant=True
            )
            model.set_token_calibration(token_center, token_scale)

            calibration_output = model(
                train_signals,
                train_bearings,
                train_steps,
                return_components=True,
                apply_trajectory_calibration=False,
            )
            trajectory_raw = calibration_output['ot_trajectory_raw']
            trajectory_center, trajectory_scale, trajectory_keep = (
                self._calibration_statistics(trajectory_raw, allow_constant=True)
            )
            model.set_trajectory_calibration(
                trajectory_center,
                trajectory_scale,
                trajectory_keep,
            )
        if was_training:
            model.train()
        return {
            'amplitude_feature_dim': int(amplitude_raw.shape[1]),
            'token_feature_dim': int(token_raw.shape[-1]),
            'trajectory_feature_dim': int(trajectory_raw.shape[1]),
            'trajectory_retained_feature_count': int(trajectory_keep.sum().item()),
            'token_rows': int(token_raw.shape[0] * token_raw.shape[1]),
        }

    def _initialize_r13_amplitude_head(
        self,
        model,
        train_signals,
        train_bearings,
        train_steps,
        train_times,
        train_events,
        params,
    ):
        """Fit the legacy-stable amplitude start on inner-train rows only."""

        initialization = params['r13_amplitude_init']
        if initialization == 'random':
            return None
        if initialization != 'cox_train':
            raise ValueError(f"unsupported R13 amplitude initialization: {initialization!r}")

        was_training = model.training
        model.eval()
        with torch.no_grad():
            raw_amplitudes = model.amplitude_features(
                train_signals, train_bearings, train_steps
            )
            amplitudes = (
                raw_amplitudes - model.amplitude_center.view(1, -1)
            ) / model.amplitude_scale.view(1, -1)
        if was_training:
            model.train()
        coefficients = self._initialize_amplitude_head_from_train_cox(
            model,
            amplitudes.detach().cpu().numpy(),
            train_times.detach().cpu().numpy(),
            train_events.detach().cpu().numpy(),
            alpha=float(params['r13_amplitude_cox_alpha']),
        )
        return torch.as_tensor(
            coefficients,
            device=model.amplitude_head.weight.device,
            dtype=model.amplitude_head.weight.dtype,
        )

    def _fit_r13_trajectory_main_residualization(
        self,
        model,
        train_signals,
        train_bearings,
        train_steps,
        params,
    ):
        """Fit the train-only OT residual view used by the joint endpoint."""

        model.clear_trajectory_main_residualization()
        if not bool(params['r13_orthogonalize_trajectory_to_main']):
            return {
                'enabled': False,
                'projection_rank': 0,
                'projection_l2': 0.0,
                'retained_feature_count': int(model.trajectory_keep.sum().item()),
            }

        was_training = model.training
        model.eval()
        with torch.no_grad():
            output = model(
                train_signals,
                train_bearings,
                train_steps,
                return_components=True,
            )
            main_risk = output['amplitude_log_risk'].squeeze(1)
            trajectory = output['ot_trajectory']
        if was_training:
            model.train()

        main_values = main_risk.detach().cpu().numpy().astype(np.float64, copy=False)
        trajectory_values = trajectory.detach().cpu().numpy().astype(np.float64, copy=False)
        design = np.column_stack((np.ones(len(main_values), dtype=np.float64), main_values))
        projection, _, rank, _ = np.linalg.lstsq(design, trajectory_values, rcond=None)
        residual = trajectory_values - design @ projection
        center = residual.mean(axis=0)
        scale = residual.std(axis=0)
        threshold = np.finfo(np.float32).eps
        keep = scale > threshold
        if not np.isfinite(projection).all() or not np.isfinite(residual).all():
            raise RuntimeError("R13 trajectory-main residualization produced non-finite values")
        if not bool(keep.any()):
            raise RuntimeError("R13 trajectory-main residualization removed every feature")
        safe_center = np.where(keep, center, 0.0).astype(np.float32)
        safe_scale = np.where(keep, scale, 1.0).astype(np.float32)
        model.set_trajectory_main_residualization(
            projection.astype(np.float32),
            safe_center,
            safe_scale,
            keep,
        )
        cross_moment = design.T @ residual / float(len(design))
        return {
            'enabled': True,
            'projection_rank': int(rank),
            'projection_l2': float(np.linalg.norm(projection)),
            'retained_feature_count': int(keep.sum()),
            'max_train_cross_moment': float(np.max(np.abs(cross_moment))),
        }

    @staticmethod
    def _r13_risk_orthogonality_loss(amplitude_log_risk, ot_log_risk):
        """Squared centered correlation, with a stable zero-variance limit."""

        amplitude = amplitude_log_risk.reshape(-1)
        ot_risk = ot_log_risk.reshape(-1)
        amplitude_centered = amplitude - amplitude.mean()
        ot_centered = ot_risk - ot_risk.mean()
        numerator = (amplitude_centered * ot_centered).sum().square()
        denominator = amplitude_centered.square().sum() * ot_centered.square().sum()
        epsilon = torch.finfo(amplitude.dtype).eps
        return numerator / (denominator + epsilon)

    def _r13_train_objective(
        self,
        model,
        output,
        train_times,
        train_events,
        anchor_coefficients,
        params,
    ):
        """Return the complete-risk-set Cox objective and stabilizing terms."""

        from otsurv.losses import partial_ll_loss

        cox_loss = partial_ll_loss(
            output['logits'], train_times, train_events
        )['loss']
        amplitude_cox_loss = partial_ll_loss(
            output['amplitude_log_risk'], train_times, train_events
        )['loss']
        anchor_loss = cox_loss.new_zeros(())
        if anchor_coefficients is not None:
            anchor_loss = 0.5 * (
                model.amplitude_head.weight.reshape(-1) - anchor_coefficients
            ).square().sum()
        ot_head_l2 = 0.5 * model.ot_head.weight.square().sum()
        risk_orthogonality_loss = self._r13_risk_orthogonality_loss(
            output['amplitude_log_risk'], output['ot_log_risk']
        )
        total_loss = (
            cox_loss
            + float(params['r13_amplitude_aux_cox_weight']) * amplitude_cox_loss
            + float(params['r13_amplitude_anchor_weight']) * anchor_loss
            + float(params['r13_ot_head_l2_weight']) * ot_head_l2
            + float(params['r13_ot_risk_orthogonality_weight'])
            * risk_orthogonality_loss
        )
        return {
            'loss': total_loss,
            'cox_loss': cox_loss,
            'amplitude_cox_loss': amplitude_cox_loss,
            'anchor_loss': anchor_loss,
            'ot_head_l2': ot_head_l2,
            'risk_orthogonality_loss': risk_orthogonality_loss,
        }

    @staticmethod
    def _r13_apply_phase_freeze(model, params, epoch):
        """Suppress updates during the residual warm-up without breaking the graph."""

        amplitude_frozen = epoch < int(params['r13_amplitude_freeze_epochs'])
        shape_frozen = (
            bool(params['r13_shape_projection_trainable'])
            and epoch < int(params['r13_shape_projection_freeze_epochs'])
        )
        if amplitude_frozen:
            for parameter in model.amplitude_head.parameters():
                parameter.grad = None
        if shape_frozen:
            for parameter in model.shape_token_projection.parameters():
                parameter.grad = None
        if amplitude_frozen and shape_frozen:
            return 'residual_warmup'
        if amplitude_frozen:
            return 'amplitude_warmup'
        if shape_frozen:
            return 'shape_warmup'
        return 'joint'

    @staticmethod
    def _finite_gradient_l2(model):
        squared_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError("R13 produced a non-finite parameter gradient")
            squared_norm += float(parameter.grad.detach().square().sum().item())
        return math.sqrt(squared_norm)

    def predict(
        self,
        model,
        X_raw: List[torch.Tensor],
        params: dict,
        device,
        verbose: bool = True,
        sample_meta: dict = None,
    ) -> np.ndarray:
        """Predict one R13 log-risk for each supplied raw minute."""
        self._validate_r13_params(params)
        device = self._resolve_device(device)
        model = model.to(device)
        signals = self._stack_r13_signals(X_raw, device)
        bearings, steps = self._r13_metadata(
            sample_meta, len(X_raw), device, context='R13 prediction metadata'
        )
        model.eval()
        with torch.no_grad():
            output = model(signals, bearings, steps)
        raw_risks = output['logits'].squeeze(1).detach().cpu().numpy().astype(np.float32)
        risks = self._r13_apply_risk_temperature(raw_risks, params)
        if verbose:
            print(f"[R13OTSurv] predicted {len(risks)} samples", flush=True)
        return risks

    def predict_components(
        self,
        model,
        X_raw: List[torch.Tensor],
        params: dict,
        device,
        verbose: bool = True,
        sample_meta: dict = None,
    ) -> Dict[str, np.ndarray]:
        """Return the R13 amplitude, OT-distance, and trajectory branches."""
        self._validate_r13_params(params)
        device = self._resolve_device(device)
        model = model.to(device)
        signals = self._stack_r13_signals(X_raw, device)
        bearings, steps = self._r13_metadata(
            sample_meta, len(X_raw), device, context='R13 component metadata'
        )
        model.eval()
        with torch.no_grad():
            output = model(signals, bearings, steps, return_components=True)
        result = {
            'log_risk': self._r13_apply_risk_temperature(
                output['logits'].squeeze(1).cpu().numpy().astype(np.float32), params
            ),
            'amplitude_log_risk': self._r13_apply_risk_temperature(
                output['amplitude_log_risk'].squeeze(1).cpu().numpy().astype(np.float32),
                params,
            ),
            'ot_log_risk': self._r13_apply_risk_temperature(
                output['ot_log_risk'].squeeze(1).cpu().numpy().astype(np.float32),
                params,
            ),
            'ot_distances': output['ot_distances'].cpu().numpy().astype(np.float32),
            'ot_trajectory': output['ot_trajectory'].cpu().numpy().astype(np.float32),
            'ot_trajectory_raw': output['ot_trajectory_raw'].cpu().numpy().astype(np.float32),
        }
        for name in ('sinkhorn_hard_marginal_error', 'sinkhorn_converged'):
            value = output.get(name)
            if value is not None:
                result[name] = value.cpu().numpy().copy()
        if not np.allclose(
            result['log_risk'],
            result['amplitude_log_risk'] + result['ot_log_risk'],
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError("R13 component decomposition is invalid")
        if verbose:
            print(f"[R13OTSurv] predicted components for {len(X_raw)} samples", flush=True)
        return result

class R14BOTSurvRegressor(R13OTSurvRegressor):
    """C2-only R14-B hybrid raw-CNN/log-normal AFT adapter.

    This adapter intentionally does not inherit R13's Cox objective.  It only
    reuses the audited raw-signal stacking, metadata validation, causal R13
    feature construction, and grouped split helpers.  FPT targets are explicit
    arguments to both fit paths; hiding them in ``sample_meta`` is rejected.
    """

    MODE = 'r14b'

    SUPPORTED_CONDITION = 1

    AFT_HORIZON = 1024.0

    WARMUP_EPOCHS = 6

    TEACHER_EPOCHS = 30

    def make_model(self, params=None, mode=None):
        from otsurv.model_r14b_otsurv import R14BOTSurv

        model_params = dict(cfg.PARAMS_R14B_OTSURV)
        if params:
            model_params.update(params)
        requested_mode = mode if mode is not None else model_params.get('mode', self.MODE)
        if requested_mode != self.MODE:
            raise ValueError("R14-B requires mode='r14b'")
        self._validate_r14b_params(model_params)
        model = R14BOTSurv(
            signal_length=int(model_params['signal_length']),
            window_length=int(model_params['window_length']),
            reference_lengths=(
                int(model_params['r14b_reference_length_short']),
                int(model_params['r14b_reference_length_medium']),
                int(model_params['r14b_reference_length_long']),
            ),
            transport_batch_size=int(model_params['r14b_transport_batch_size']),
            ot_epsilon=float(model_params['ot_epsilon']),
            ot_num_iters=int(model_params['ot_num_iters']),
            ot_stoperr=float(model_params['ot_stoperr']),
            shape_projection_trainable=bool(model_params['r14b_shape_projection_trainable']),
            sinkhorn_checkpoint=bool(model_params['r14b_sinkhorn_checkpoint']),
            state_hidden_dim=int(model_params['r14b_state_hidden_dim']),
            sigma_init=float(model_params['r14b_sigma_init']),
        )
        model.r14b_architecture_ = {
            'input': '[32768, 2] physical-scale dual-channel vibration',
            'raw_cnn': '64 non-overlapping 512-sample windows + Conv1d/GroupNorm/ReLU',
            'source_features': 'R13 14D amplitude + calibrated 21D OT trajectory + 64D minute embedding',
            'state': 'causal short/medium/long references + signed rates + maturity + handwritten gated cell',
            'state_hidden_dim': int(model.state_hidden_dim),
            'endpoint': 'mu_bias + mu_rank + 0.75*tanh(delta_fpt_head)',
            'risk': 'negative_aft_location',
            'survival': 'lognormal_aft_survival_s0_one_monotone',
            'aft_horizon': float(self.AFT_HORIZON),
        }
        model.native_risk_score_definition_ = 'negative_aft_location'
        model.survival_curve_definition_ = 'lognormal_aft_s0_one_monotone'
        model.fpt_label_channel_ = 'explicit_fit_arguments'
        return model

    def get_hyperparams(self, condition):
        try:
            condition_value = float(condition)
        except (TypeError, ValueError, OverflowError):
            condition_value = float('nan')
        if (
            isinstance(condition, (bool, np.bool_))
            or not math.isfinite(condition_value)
            or condition_value != float(self.SUPPORTED_CONDITION)
        ):
            raise ValueError(
                'R14-B is currently a C2-only architecture candidate; condition must be 1'
            )
        params = dict(cfg.PARAMS_R14B_OTSURV)
        self._validate_r14b_params(params)
        return params

    @classmethod
    def _validate_r14b_params(cls, params):
        if not isinstance(params, dict):
            raise ValueError('R14-B parameters must be a dictionary')
        expected = set(cfg.PARAMS_R14B_OTSURV)
        missing = sorted(expected - set(params))
        unexpected = sorted(set(params) - expected)
        if missing:
            raise ValueError(f'R14-B parameters are missing required keys: {missing}')
        if unexpected:
            raise ValueError(
                'R14-B parameters include unsupported keys: ' f'{unexpected}'
            )
        if params['mode'] != cls.MODE:
            raise ValueError("R14-B parameters require mode='r14b'")
        if params['loss_fn'] != 'hybrid_cox_aft':
            raise ValueError("R14-B requires loss_fn='hybrid_cox_aft'")
        if params['raw_signal_normalization'] != 'none':
            raise ValueError('R14-B requires physical-scale raw_signal_normalization=none')
        bool_keys = ('r14b_shape_projection_trainable', 'r14b_sinkhorn_checkpoint')
        for key in bool_keys:
            if not isinstance(params[key], bool):
                raise ValueError(f'{key} must be a boolean')
        int_keys = (
            'signal_length', 'window_length', 'stride',
            'r14b_reference_length_short', 'r14b_reference_length_medium',
            'r14b_reference_length_long', 'r14b_transport_batch_size',
            'r14b_state_hidden_dim', 'r14b_warmup_epochs', 'r14b_teacher_epochs',
            'ot_num_iters', 'max_epochs', 'early_stopping_patience',
            'min_epochs_before_stopping',
        )
        for key in int_keys:
            value = params[key]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'{key} must be an integer')
            if int(value) < 1:
                raise ValueError(f'{key} must be positive')
        numeric_keys = (
            'r14b_sigma_init', 'r14b_delta_fpt_scale', 'ot_epsilon', 'ot_stoperr',
            'lr', 'r14b_weight_decay', 'r14b_aft_horizon',
        )
        for key in numeric_keys:
            value = params[key]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f'{key} must be a finite number')
            if not math.isfinite(float(value)):
                raise ValueError(f'{key} must be finite')
        if params['r14b_loss_profile'] not in {'balanced', 'fpt_aggressive'}:
            raise ValueError("r14b_loss_profile must be 'balanced' or 'fpt_aggressive'")
        if int(params['window_length']) != 512 or int(params['stride']) != 512:
            raise ValueError('R14-B requires 512-sample non-overlapping waveform windows')
        if int(params['signal_length']) < 512 or int(params['signal_length']) % 512:
            raise ValueError('R14-B signal_length must be a positive multiple of 512')
        refs = (
            int(params['r14b_reference_length_short']),
            int(params['r14b_reference_length_medium']),
            int(params['r14b_reference_length_long']),
        )
        if any(right <= left for left, right in zip(refs[:-1], refs[1:])):
            raise ValueError('R14-B reference lengths must be strictly increasing')
        if float(params['ot_epsilon']) <= 0.0 or float(params['ot_stoperr']) < 0.0:
            raise ValueError('R14-B OT numerical parameters are invalid')
        if float(params['r14b_sigma_init']) <= 0.15 or float(params['r14b_sigma_init']) >= 1.50:
            raise ValueError('R14-B sigma_init must lie inside (0.15, 1.50)')
        if not math.isclose(float(params['r14b_delta_fpt_scale']), 0.75, abs_tol=1e-12):
            raise ValueError('R14-B delta_fpt scale is fixed at 0.75')
        if int(params['r14b_warmup_epochs']) != cls.WARMUP_EPOCHS:
            raise ValueError('R14-B ranking warmup is fixed at six epochs')
        if int(params['r14b_teacher_epochs']) != cls.TEACHER_EPOCHS:
            raise ValueError('R14-B teacher training is fixed at 30 epochs')
        if float(params['r14b_aft_horizon']) != cls.AFT_HORIZON:
            raise ValueError('R14-B AFT horizon is fixed at 1024')
        if float(params['lr']) <= 0.0 or float(params['r14b_weight_decay']) < 0.0:
            raise ValueError('R14-B optimizer parameters are invalid')
        if int(params['min_epochs_before_stopping']) > int(params['max_epochs']):
            raise ValueError('R14-B min_epochs_before_stopping exceeds max_epochs')
        if int(params['max_epochs']) <= int(params['r14b_warmup_epochs']):
            raise ValueError('R14-B max_epochs must exceed the six-epoch warmup')

    @staticmethod
    def _require_fpt_targets(fpt_times, fpt_events, n_samples, *, context):
        if fpt_times is None or fpt_events is None:
            raise ValueError(
                f'R14-B {context} requires explicit fpt_times and fpt_events; '
                'FPT labels may not be hidden in sample_meta'
            )
        times = np.asarray(fpt_times, dtype=np.float32).reshape(-1)
        events = np.asarray(fpt_events, dtype=np.float32).reshape(-1)
        if len(times) != int(n_samples) or len(events) != int(n_samples):
            raise ValueError('R14-B FPT targets must align with raw signals')
        if not np.isfinite(times).all() or not np.isfinite(events).all():
            raise ValueError('R14-B FPT targets must be finite')
        if np.any(times < 0.0) or not np.isin(events, (0.0, 1.0)).all():
            raise ValueError('R14-B FPT times must be non-negative and events binary')
        return times, events

    @staticmethod
    def _r14b_sigma_numpy(model):
        sigma = float(model.sigma.detach().cpu().item())
        if not math.isfinite(sigma) or not 0.15 <= sigma <= 1.50:
            raise RuntimeError('R14-B learned sigma is outside [0.15, 1.50]')
        return sigma

    def _fit_r14b_calibrations(self, model, train_signals, train_bearings, train_steps):
        """Fit amplitude, token, OT, and minute-embedding transforms on train only."""
        def safe_statistics(values):
            values = torch.as_tensor(values)
            if values.ndim != 2 or values.shape[0] < 1 or not torch.isfinite(values).all():
                raise ValueError('R14-B calibration values must be finite [rows, features]')
            center = values.mean(dim=0)
            scale = values.std(dim=0, unbiased=False)
            keep = scale > torch.finfo(values.dtype).eps
            # Constant columns carry no trainable information, but they are a
            # valid feature in a tiny development shard.  Preserve their mean,
            # use unit scale, and let the normalized value become exactly zero.
            safe_scale = torch.where(keep, scale, torch.ones_like(scale))
            if not bool(keep.any()):
                keep = torch.ones_like(keep, dtype=torch.bool)
            return center.detach(), safe_scale.detach(), keep.detach()

        was_training = model.training
        model.eval()
        with torch.no_grad():
            amplitude_raw = model.amplitude_features(train_signals, train_bearings, train_steps)
            ac, ass, _ = safe_statistics(amplitude_raw)
            model.set_amplitude_calibration(ac, ass)
            token_raw = model.raw_shape_tokens(train_signals)
            tc, ts, _ = safe_statistics(token_raw.reshape(-1, token_raw.shape[-1]))
            model.set_token_calibration(tc, ts)
            probe = model(
                train_signals, train_bearings, train_steps,
                return_components=True, apply_trajectory_calibration=False,
            )
            trajectory_raw = probe['ot_trajectory_raw']
            trc, trs, trk = safe_statistics(trajectory_raw)
            model.set_trajectory_calibration(trc, trs, trk)
            minute_raw = probe['minute_embedding']
            mc, ms, _ = safe_statistics(minute_raw)
            model.set_minute_calibration(mc, ms)
        if was_training:
            model.train()
        return {
            'amplitude_feature_dim': int(amplitude_raw.shape[1]),
            'trajectory_feature_dim': int(trajectory_raw.shape[1]),
            'minute_embedding_dim': int(minute_raw.shape[1]),
            'trajectory_retained_feature_count': int(trk.sum().item()),
            'fit_scope': 'inner_train_only',
        }

    def _objective(self, model, output, survival_times, events, fpt_times, fpt_events, teacher_scores, params, *, include_fpt):
        components = model.loss_components(
            output,
            survival_times,
            events,
            fpt_times,
            fpt_events,
            teacher_scores,
            include_fpt=include_fpt,
        )
        # The two pre-registered profiles differ only in fixed loss emphasis;
        # balanced is the protocol default and retains the stated weights.
        if params['r14b_loss_profile'] == 'fpt_aggressive':
            total = (
                0.25 * components['cox']
                + 1.25 * (components['aft'] if include_fpt else components['aft'] * 0.0)
                + 1.00 * (components['smooth_fpt'] if include_fpt else components['smooth_fpt'] * 0.0)
                + 0.50 * components['teacher_rank']
            )
            components['loss'] = total
        return components

    def _predict_output(self, model, X_raw, params, device, sample_meta):
        self._validate_r14b_params(params)
        device = self._resolve_device(device)
        model = model.to(device)
        signals = self._stack_r13_signals(X_raw, device)
        bearings, steps = self._r13_metadata(
            sample_meta, len(X_raw), device, context='R14-B prediction metadata'
        )
        model.eval()
        with torch.no_grad():
            return model(signals, bearings, steps, return_components=True)

    def predict(self, model, X_raw, params, device, verbose=True, sample_meta=None):
        output = self._predict_output(model, X_raw, params, device, sample_meta)
        values = output['risk'].squeeze(1).cpu().numpy().astype(np.float32)
        if not np.isfinite(values).all():
            raise RuntimeError('R14-B predicted risk is non-finite')
        if verbose:
            print(f'[R14BOTSurv] predicted {len(values)} samples', flush=True)
        return values

    def predict_components(self, model, X_raw, params, device, verbose=True, sample_meta=None):
        output = self._predict_output(model, X_raw, params, device, sample_meta)
        result = {
            'risk': output['risk'].squeeze(1).cpu().numpy().astype(np.float32),
            'log_risk': output['risk'].squeeze(1).cpu().numpy().astype(np.float32),
            'mu': output['mu'].squeeze(1).cpu().numpy().astype(np.float32),
            'mu_rank': output['mu_rank'].squeeze(1).cpu().numpy().astype(np.float32),
            'delta_fpt': output['delta_fpt'].squeeze(1).cpu().numpy().astype(np.float32),
            'sigma': float(output['sigma'].detach().cpu().item()),
            'minute_embedding': output['minute_embedding'].cpu().numpy().astype(np.float32),
            'source_features': output['source_features'].cpu().numpy().astype(np.float32),
            'causal_state': output['causal_state'].cpu().numpy().astype(np.float32),
            'reference_maturity': output['reference_maturity'].cpu().numpy().astype(np.float32),
            'ot_trajectory': output['ot_trajectory'].cpu().numpy().astype(np.float32),
            'ot_trajectory_raw': output['ot_trajectory_raw'].cpu().numpy().astype(np.float32),
        }
        if not np.allclose(result['risk'], -result['mu'], rtol=1e-6, atol=1e-7):
            raise RuntimeError('R14-B native risk provenance is invalid')
        if verbose:
            print(f'[R14BOTSurv] predicted components for {len(result["risk"])} samples', flush=True)
        return result

    def predict_survival(
        self, model, X_raw, event_times, params, device, n_post_samples=1,
        verbose=True, sample_meta=None,
    ):
        import pandas as pd

        self._validate_r14b_params(params)
        if isinstance(n_post_samples, bool) or int(n_post_samples) != n_post_samples:
            raise ValueError('R14-B native AFT requires integer n_post_samples=1')
        if int(n_post_samples) != 1:
            raise ValueError('R14-B native AFT is deterministic and supports n_post_samples=1')
        times = np.asarray(event_times, dtype=np.float64).reshape(-1)
        if times.size < 1 or not np.isfinite(times).all() or np.any(times < 0.0):
            raise ValueError('R14-B AFT event_times must be finite and non-negative')
        if np.unique(times).size != times.size:
            raise ValueError('R14-B AFT event_times must be unique')
        if np.max(times) > float(params['r14b_aft_horizon']):
            raise ValueError('R14-B AFT event_times exceed the fixed C2 horizon 1024')
        output = self._predict_output(model, X_raw, params, device, sample_meta)
        with torch.no_grad():
            curves = model.aft_survival(output['mu'], times).cpu().numpy().astype(np.float64)
        frame = pd.DataFrame(curves, columns=times)
        if 0.0 in frame.columns:
            frame.loc[:, 0.0] = 1.0
        else:
            frame.insert(0, 0.0, 1.0)
        return frame

class R14FOTSurvRegressor(R13OTSurvRegressor):
    """Experimental fixed-30 joint R13/R14-B dual-endpoint adapter.

    R14-F is intentionally separate from R14-E.  Its R13 branch receives its
    own complete-risk-set Cox objective on every epoch.  R14-B receives the
    just-updated R13 scores as a detached pairwise-ranking teacher target, so
    its distillation loss cannot alter R13 parameters.  The class supports no
    inner bearing selection and is therefore not a formal replacement path.
    """

    MODE = 'r14f'

    SUPPORTED_CONDITIONS = frozenset({0, 1, 2})

    DEFAULT_CONDITION = 1

    FIXED_EPOCHS = 30

    AFT_HORIZON = 1024.0

    RANK_ANCHOR_CONTRACT = 'trainable_r13_condition_matched_detached_teacher'

    TIME_PROFILE = 'r14b_balanced_shared_teacher'

    TEACHER_TIMING = 'post_r13_update_same_epoch_detached'

    FIT_SCOPE = 'outer_development_joint_fixed30'

    TRAINING_PROTOCOL = 'experimental_fixed30_shared_teacher_not_formal'

    DIAGNOSTIC_SWEEP_FIT_SCOPE = 'outer_development_joint_checkpoint_sweep'

    DIAGNOSTIC_SWEEP_PROTOCOL = (
        'diagnostic_fixed40_shared_teacher_checkpoint_sweep_no_selection'
    )

    @classmethod
    def _rank_condition(cls, params):
        """Return the explicit XJTU condition of the live R13 teacher."""

        cls._validate_r14f_params(params, validate_children=False)
        value = params.get('r14f_condition')
        if isinstance(value, (bool, np.bool_)):
            raise ValueError('R14-F r14f_condition must be one of 0, 1, 2')
        try:
            condition = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('R14-F r14f_condition must be one of 0, 1, 2') from exc
        if condition != value or condition not in cls.SUPPORTED_CONDITIONS:
            raise ValueError('R14-F r14f_condition must be one of 0, 1, 2')
        return condition

    @classmethod
    def _rank_params(cls, params):
        """Return the registered R13 objective for the matching condition."""

        cls._validate_r14f_params(params, validate_children=False)
        rank_condition = cls._rank_condition(params)
        rank = R13OTSurvRegressor().get_hyperparams(rank_condition)
        R13OTSurvRegressor._validate_r13_params(rank)
        if rank.get('mode') != R13OTSurvRegressor.MODE:
            raise RuntimeError('R14-F rank branch did not resolve to R13 mode')
        if int(rank.get('max_epochs', -1)) != cls.FIXED_EPOCHS:
            raise RuntimeError('R14-F requires the registered 30-epoch R13 budget')
        return rank

    @classmethod
    def _time_params(cls, params):
        """Translate the R14-F envelope into the balanced R14-B contract."""

        cls._validate_r14f_params(params, validate_children=False)
        time_params = {key: params[key] for key in cfg.PARAMS_R14B_OTSURV}
        time_params['mode'] = R14BOTSurvRegressor.MODE
        time_params['loss_fn'] = 'hybrid_cox_aft'
        time_params['r14b_loss_profile'] = 'balanced'
        R14BOTSurvRegressor._validate_r14b_params(time_params)
        return time_params

    @classmethod
    def _validate_r14f_params(cls, params, *, validate_children=True):
        if not isinstance(params, dict):
            raise ValueError('R14-F parameters must be a dictionary')
        expected = set(cfg.PARAMS_R14F_OTSURV)
        missing = sorted(expected - set(params))
        unexpected = sorted(set(params) - expected)
        if missing:
            raise ValueError(f'R14-F parameters are missing required keys: {missing}')
        if unexpected:
            raise ValueError(f'R14-F parameters include unsupported keys: {unexpected}')
        if params.get('mode') != cls.MODE:
            raise ValueError("R14-F parameters require mode='r14f'")
        if params.get('loss_fn') != 'joint_trainable_r13_detached_teacher_r14b_time_expert':
            raise ValueError('R14-F requires its joint detached-teacher loss contract')
        if params.get('r14f_rank_anchor_contract') != cls.RANK_ANCHOR_CONTRACT:
            raise ValueError('R14-F requires the trainable detached R13 teacher contract')
        if params.get('r14f_time_profile') != cls.TIME_PROFILE:
            raise ValueError('R14-F requires the shared-teacher balanced R14-B profile')
        if params.get('r14f_teacher_timing') != cls.TEACHER_TIMING:
            raise ValueError('R14-F requires post-update same-epoch detached teacher scores')
        condition = params.get('r14f_condition')
        if (
            isinstance(condition, (bool, np.bool_))
            or not isinstance(condition, (int, np.integer))
            or int(condition) not in cls.SUPPORTED_CONDITIONS
        ):
            raise ValueError('R14-F r14f_condition must be one of 0, 1, 2')
        if int(params.get('r14f_fixed_epochs', -1)) != cls.FIXED_EPOCHS:
            raise ValueError('R14-F fixed epoch budget is exactly 30')
        if params.get('r14b_loss_profile') != 'balanced':
            raise ValueError('R14-F must retain R14-B balanced time-expert losses')
        if float(params.get('r14b_aft_horizon', np.nan)) != cls.AFT_HORIZON:
            raise ValueError('R14-F AFT horizon is fixed at 1024')
        # The only intended R14-B change is its teacher source.  Architecture,
        # optimizer, warm-up, and AFT objective remain exactly registered.
        for key, expected_value in cfg.PARAMS_R14B_OTSURV.items():
            if key in {'mode', 'loss_fn'}:
                continue
            if params.get(key) != expected_value:
                raise ValueError(
                    'R14-F must retain every R14-B parameter unchanged; '
                    f'{key}={params.get(key)!r} != {expected_value!r}'
                )
        if validate_children:
            cls._rank_params(params)
            cls._time_params(params)

    @staticmethod
    def _require_model(model):
        from otsurv.model_r14f_otsurv import R14FOTSurv

        if not isinstance(model, R14FOTSurv):
            raise TypeError('R14-F requires an R14FOTSurv model instance')
        return model

    @staticmethod
    def _reject_hidden_fpt_labels(sample_meta):
        if not isinstance(sample_meta, dict):
            return
        hidden = {'fpt_times', 'fpt_events', 'TrueTime', 'TrueEvent'}.intersection(
            sample_meta
        )
        if hidden:
            raise ValueError(
                'R14-F FPT labels must be passed through explicit '
                'fpt_times/fpt_events arguments, not sample_meta'
            )

    @classmethod
    def _parse_fixed_epochs(cls, fixed_epochs, params):
        if isinstance(fixed_epochs, bool):
            raise ValueError('R14-F fixed epochs must be the integer 30')
        try:
            epochs = int(fixed_epochs)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('R14-F fixed epochs must be the integer 30') from exc
        if epochs != fixed_epochs or epochs != int(params['r14f_fixed_epochs']):
            raise ValueError('R14-F fixed epochs must be the integer 30')
        return epochs

    @staticmethod
    def _resolve_branch_devices(device, *, rank_device=None, time_device=None):
        base = torch.device(device)
        rank = torch.device(rank_device) if rank_device is not None else base
        time = torch.device(time_device) if time_device is not None else base
        for name, candidate in (('rank_device', rank), ('time_device', time)):
            if candidate.type == 'cuda':
                if not torch.cuda.is_available():
                    raise RuntimeError(f'R14-F {name} requests CUDA but CUDA is unavailable')
                index = torch.cuda.current_device() if candidate.index is None else candidate.index
                if index < 0 or index >= torch.cuda.device_count():
                    raise ValueError(
                        f'R14-F {name}={candidate} is outside visible CUDA devices'
                    )
            elif candidate.type != 'cpu':
                raise ValueError(f'R14-F {name} must be a CPU or CUDA device')
        return rank, time

    @staticmethod
    def _assert_disjoint_branch_parameters(model):
        rank_ids = {id(parameter) for parameter in model.rank_anchor.parameters()}
        time_ids = {id(parameter) for parameter in model.time_expert.parameters()}
        if rank_ids & time_ids:
            raise RuntimeError('R14-F rank and time branches unexpectedly share parameters')

    @staticmethod
    def _teacher_score_sha256(scores):
        values = np.ascontiguousarray(
            np.asarray(scores, dtype=np.float32).reshape(-1)
        )
        if values.size < 1 or not np.isfinite(values).all():
            raise RuntimeError('R14-F teacher scores must be finite and non-empty')
        return sha256(values.tobytes()).hexdigest()

    def make_model(self, params=None, mode=None):
        from otsurv.model_r14f_otsurv import R14FOTSurv

        model_params = dict(cfg.PARAMS_R14F_OTSURV)
        if params:
            model_params.update(params)
        requested_mode = mode if mode is not None else model_params.get('mode', self.MODE)
        if requested_mode != self.MODE:
            raise ValueError("R14-F requires mode='r14f'")
        self._validate_r14f_params(model_params)
        rank_params = self._rank_params(model_params)
        time_params = self._time_params(model_params)

        # Both branches begin from the same reproducible CPU RNG position, as
        # independent model constructions rather than shared parameters.
        initial_rng = torch.get_rng_state()
        rank_anchor = R13OTSurvRegressor().make_model(rank_params)
        rank_training_rng = torch.get_rng_state().clone()
        torch.set_rng_state(initial_rng)
        time_expert = R14BOTSurvRegressor().make_model(time_params)
        time_training_rng = torch.get_rng_state().clone()

        model = R14FOTSurv(rank_anchor, time_expert)
        self._assert_disjoint_branch_parameters(model)
        model.r14f_rank_training_rng_state_ = rank_training_rng
        model.r14f_time_training_rng_state_ = time_training_rng
        model.r14f_architecture_ = {
            'rank_endpoint': (
                f'trainable_r13_c{self._rank_condition(model_params) + 1}_cox_log_risk'
            ),
            'time_endpoint': 'r14b_balanced_lognormal_aft',
            'teacher_source': 'same_container_r13_post_update_logits_detached',
            'gradient_path': 'r14b_distillation_to_r13_none',
            'risk': 'rank_anchor_only',
            'survival': 'time_expert_only',
            'aft_horizon': float(self.AFT_HORIZON),
            'rank_condition': int(self._rank_condition(model_params)),
            'training_protocol': self.TRAINING_PROTOCOL,
        }
        model.r14f_rank_params_ = dict(rank_params)
        model.r14f_time_params_ = dict(time_params)
        model.r14f_fpt_label_channel_ = 'explicit_fit_arguments_time_expert_only'
        model.r14f_fpt_label_scope_ = 'not_fitted'
        return model

    def get_hyperparams(self, condition):
        if isinstance(condition, (bool, np.bool_)) or condition not in self.SUPPORTED_CONDITIONS:
            raise ValueError('R14-F condition must be one of 0, 1, 2')
        params = dict(cfg.PARAMS_R14F_OTSURV)
        params['r14f_condition'] = int(condition)
        self._validate_r14f_params(params)
        return params

    def fit(
        self,
        model,
        X_raw,
        t,
        e,
        params,
        device,
        **kwargs,
    ):
        raise ValueError(
            'R14-F intentionally has no selection fit; use fixed_epochs=30 '
            'on the four-bearing outer development cohort'
        )

    def fit_checkpoint_sweep(
        self,
        model,
        X_raw,
        t,
        e,
        params,
        device,
        *,
        sweep_epochs,
        fpt_times=None,
        fpt_events=None,
        sample_meta=None,
        log_fn=None,
        verbose=True,
        collect_diagnostics=True,
        rank_device=None,
        time_device=None,
        epoch_state_observer=None,
    ):
        """Fit one diagnostic trajectory and expose post-epoch CPU snapshots.

        This path is deliberately separate from the fixed-30 R14-F contract.
        It exists only for a pre-registered outer-test checkpoint sweep: the
        observer receives model state after each joint optimizer update, and
        must not load or score a held-out bearing during fitting.
        """

        return self.fit_fixed_epochs(
            model,
            X_raw,
            t,
            e,
            params,
            device,
            fixed_epochs=sweep_epochs,
            fpt_times=fpt_times,
            fpt_events=fpt_events,
            sample_meta=sample_meta,
            log_fn=log_fn,
            verbose=verbose,
            collect_diagnostics=collect_diagnostics,
            rank_device=rank_device,
            time_device=time_device,
            epoch_state_observer=epoch_state_observer,
            diagnostic_checkpoint_sweep=True,
        )

    def fit_fixed_epochs(
        self,
        model,
        X_raw,
        t,
        e,
        params,
        device,
        *,
        fixed_epochs,
        fpt_times=None,
        fpt_events=None,
        sample_meta=None,
        log_fn=None,
        verbose=True,
        collect_diagnostics=True,
        rank_device=None,
        time_device=None,
        epoch_state_observer=None,
        diagnostic_checkpoint_sweep=False,
    ):
        """Jointly train both endpoints on all four development bearings.

        The R13 update deliberately precedes the R14-B forward pass.  A fresh
        no-grad R13 output is transferred only after ``detach`` and supplies
        R14-B's pairwise ranking term.  The rank optimizer is cleared before
        R14-B backpropagation, making a reverse-gradient violation explicit.
        """

        self._validate_r14f_params(params)
        self._reject_hidden_fpt_labels(sample_meta)
        model = self._require_model(model)
        self._assert_disjoint_branch_parameters(model)
        if not isinstance(diagnostic_checkpoint_sweep, bool):
            raise ValueError('R14-F diagnostic_checkpoint_sweep must be boolean')
        if diagnostic_checkpoint_sweep:
            if isinstance(fixed_epochs, bool):
                raise ValueError('R14-F checkpoint-sweep epochs must be a positive integer')
            try:
                epochs = int(fixed_epochs)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    'R14-F checkpoint-sweep epochs must be a positive integer'
                ) from exc
            if epochs != fixed_epochs or epochs < 1:
                raise ValueError('R14-F checkpoint-sweep epochs must be a positive integer')
            fit_scope = self.DIAGNOSTIC_SWEEP_FIT_SCOPE
            training_protocol = self.DIAGNOSTIC_SWEEP_PROTOCOL
        else:
            epochs = self._parse_fixed_epochs(fixed_epochs, params)
            fit_scope = self.FIT_SCOPE
            training_protocol = self.TRAINING_PROTOCOL
        if epoch_state_observer is not None and not callable(epoch_state_observer):
            raise TypeError('R14-F epoch_state_observer must be callable')
        rank_params = self._rank_params(params)
        if diagnostic_checkpoint_sweep:
            # R13's registered fixed-30 profile is unchanged; only this
            # isolated diagnostic trajectory extends its optimizer budget.
            rank_params = dict(rank_params)
            rank_params['max_epochs'] = int(epochs)
            R13OTSurvRegressor._validate_r13_params(rank_params)
        time_params = self._time_params(params)
        if epochs > int(rank_params['max_epochs']) or epochs > int(time_params['max_epochs']):
            raise ValueError('R14-F fixed epochs exceed a branch epoch budget')
        rank_device, time_device = self._resolve_branch_devices(
            device,
            rank_device=rank_device,
            time_device=time_device,
        )
        rank_model = model.rank_anchor.to(rank_device)
        time_model = model.time_expert.to(time_device)
        model.rank_anchor = rank_model
        model.time_expert = time_model

        n_total = len(X_raw)
        if n_total < 1:
            raise ValueError('R14-F requires at least one development raw signal')
        times = np.asarray(t, dtype=np.float32).reshape(-1)
        events = np.asarray(e, dtype=np.float32).reshape(-1)
        if len(times) != n_total or len(events) != n_total:
            raise ValueError('R14-F survival labels must align with raw signals')
        if (
            not np.isfinite(times).all()
            or not np.isfinite(events).all()
            or not np.isin(events, (0.0, 1.0)).all()
        ):
            raise ValueError('R14-F survival labels must be finite with binary events')
        fpt_t, fpt_e = R14BOTSurvRegressor._require_fpt_targets(
            fpt_times,
            fpt_events,
            n_total,
            context='R14-F fixed-epoch fit',
        )

        rank_signals_all = self._stack_r13_signals(X_raw, rank_device)
        rank_bearings_all, rank_steps_all = self._r13_metadata(
            sample_meta,
            n_total,
            rank_device,
            context='R14-F rank training metadata',
        )
        ordered = np.lexsort(
            (
                rank_steps_all.detach().cpu().numpy(),
                rank_bearings_all.detach().cpu().numpy(),
            )
        ).astype(int)
        rank_order = torch.as_tensor(ordered, device=rank_device, dtype=torch.long)
        rank_signals = rank_signals_all.index_select(0, rank_order)
        rank_bearings = rank_bearings_all.index_select(0, rank_order)
        rank_steps = rank_steps_all.index_select(0, rank_order)
        rank_times = torch.as_tensor(times[ordered], device=rank_device)
        rank_events = torch.as_tensor(events[ordered], device=rank_device)
        if int(rank_events.sum().item()) < 1:
            raise ValueError('R14-F development cohort requires an observed survival event')

        time_signals_all = self._stack_r13_signals(X_raw, time_device)
        time_bearings_all, time_steps_all = self._r13_metadata(
            sample_meta,
            n_total,
            time_device,
            context='R14-F time training metadata',
        )
        time_order = torch.as_tensor(ordered, device=time_device, dtype=torch.long)
        time_signals = time_signals_all.index_select(0, time_order)
        time_bearings = time_bearings_all.index_select(0, time_order)
        time_steps = time_steps_all.index_select(0, time_order)
        time_times = torch.as_tensor(times[ordered], device=time_device)
        time_events = torch.as_tensor(events[ordered], device=time_device)
        time_fpt_t = torch.as_tensor(fpt_t[ordered], device=time_device)
        time_fpt_e = torch.as_tensor(fpt_e[ordered], device=time_device)
        if not torch.equal(
            rank_bearings.detach().cpu(), time_bearings.detach().cpu()
        ) or not torch.equal(rank_steps.detach().cpu(), time_steps.detach().cpu()):
            raise RuntimeError('R14-F branch sample ordering diverged')

        train_bearings = tuple(
            sorted(int(value) for value in rank_bearings.detach().cpu().unique().tolist())
        )
        rank_calibration = self._fit_r13_calibrations(
            rank_model, rank_signals, rank_bearings, rank_steps
        )
        rank_anchor_coefficients = self._initialize_r13_amplitude_head(
            rank_model,
            rank_signals,
            rank_bearings,
            rank_steps,
            rank_times,
            rank_events,
            rank_params,
        )
        rank_residualization = self._fit_r13_trajectory_main_residualization(
            rank_model,
            rank_signals,
            rank_bearings,
            rank_steps,
            rank_params,
        )
        time_calibration = R14BOTSurvRegressor()._fit_r14b_calibrations(
            time_model, time_signals, time_bearings, time_steps
        )
        rank_model.r13_calibration_ = {
            **rank_calibration,
            'trajectory_main_residualization': rank_residualization,
            'fit_scope': fit_scope,
            'train_bearings': train_bearings,
            'validation_bearing': None,
        }
        rank_model.r13_optimization_ = {
            'amplitude_initialization': str(rank_params['r13_amplitude_init']),
            'amplitude_cox_alpha': float(rank_params['r13_amplitude_cox_alpha']),
            'amplitude_anchor_l2': (
                float(rank_anchor_coefficients.norm().item())
                if rank_anchor_coefficients is not None else None
            ),
            'amplitude_freeze_epochs': int(rank_params['r13_amplitude_freeze_epochs']),
            'shape_projection_freeze_epochs': int(
                rank_params['r13_shape_projection_freeze_epochs']
            ),
            'trajectory_main_residualization': rank_residualization,
        }
        rank_model.train_bearing_ids_ = train_bearings
        rank_model.val_bearing_ids_ = ()
        rank_model.validation_bearing_id_ = None
        time_model.r14b_calibration_ = {
            **time_calibration,
            'fit_scope': fit_scope,
            'train_bearings': train_bearings,
            'validation_bearing': None,
        }
        time_model.train_bearing_ids_ = train_bearings
        time_model.val_bearing_ids_ = ()
        time_model.validation_bearing_id_ = None
        time_model.r14b_fpt_label_scope_ = fit_scope
        time_model.r14b_fpt_label_channel_ = 'explicit_fit_arguments'

        rank_optimizer = torch.optim.AdamW(
            (parameter for parameter in rank_model.parameters() if parameter.requires_grad),
            lr=float(rank_params['lr']),
            weight_decay=float(rank_params['r13_weight_decay']),
        )
        time_optimizer = torch.optim.AdamW(
            (parameter for parameter in time_model.parameters() if parameter.requires_grad),
            lr=float(time_params['lr']),
            weight_decay=float(time_params['r14b_weight_decay']),
        )
        rank_records = []
        time_records = []
        time_regressor = R14BOTSurvRegressor()
        last_teacher_scores = None
        self._emit(
            f'[R14FOTSurv] samples={n_total} bearings={train_bearings} '
            f'epochs={epochs} rank_device={rank_device} time_device={time_device} '
            f'protocol={training_protocol}',
            verbose=verbose,
            log_fn=log_fn,
        )

        for epoch in range(epochs):
            rank_started = time.perf_counter()
            rank_model.train()
            rank_optimizer.zero_grad(set_to_none=True)
            rank_output = rank_model(
                rank_signals,
                rank_bearings,
                rank_steps,
                return_components=True,
            )
            rank_objective = self._r13_train_objective(
                rank_model,
                rank_output,
                rank_times,
                rank_events,
                rank_anchor_coefficients,
                rank_params,
            )
            if not torch.isfinite(rank_objective['loss']):
                raise RuntimeError('R14-F R13 branch produced a non-finite loss')
            rank_objective['loss'].backward()
            rank_phase = self._r13_apply_phase_freeze(rank_model, rank_params, epoch)
            rank_gradient_l2 = self._finite_gradient_l2(rank_model)
            rank_optimizer.step()
            # Clearing the branch before R14-B backprop makes any accidental
            # reverse path detectable rather than merely hidden in old grads.
            rank_optimizer.zero_grad(set_to_none=True)

            rank_model.eval()
            with torch.no_grad():
                live_teacher = rank_model(
                    rank_signals,
                    rank_bearings,
                    rank_steps,
                )['logits'].reshape(-1).detach()
            if live_teacher.requires_grad or live_teacher.grad_fn is not None:
                raise RuntimeError('R14-F teacher target was not detached from R13')
            if not torch.isfinite(live_teacher).all():
                raise RuntimeError('R14-F live R13 teacher produced non-finite scores')
            teacher_scores = live_teacher.to(
                device=time_device,
                dtype=torch.float32,
                non_blocking=rank_device != time_device,
            ).detach()
            if teacher_scores.requires_grad or teacher_scores.grad_fn is not None:
                raise RuntimeError('R14-F teacher transfer retained an autograd path')
            last_teacher_scores = teacher_scores.detach().cpu().numpy().astype(
                np.float32, copy=True
            )
            rank_seconds = float(time.perf_counter() - rank_started)

            time_started = time.perf_counter()
            include_fpt = epoch >= int(time_params['r14b_warmup_epochs'])
            time_model.train()
            time_optimizer.zero_grad(set_to_none=True)
            time_output = time_model(
                time_signals,
                time_bearings,
                time_steps,
                return_components=True,
            )
            time_objective = time_regressor._objective(
                time_model,
                time_output,
                time_times,
                time_events,
                time_fpt_t,
                time_fpt_e,
                teacher_scores,
                time_params,
                include_fpt=include_fpt,
            )
            if not torch.isfinite(time_objective['loss']):
                raise RuntimeError('R14-F R14-B branch produced a non-finite loss')
            time_objective['loss'].backward()
            if any(parameter.grad is not None for parameter in rank_model.parameters()):
                raise RuntimeError('R14-F R14-B distillation loss reached the R13 branch')
            time_gradient_l2 = self._finite_gradient_l2(time_model)
            time_optimizer.step()
            time_seconds = float(time.perf_counter() - time_started)

            rank_record = {
                'epoch': int(epoch + 1),
                'phase': rank_phase,
                'training_stage': fit_scope,
                'fixed_epoch_budget': int(epochs),
                'train_loss': float(rank_objective['loss'].detach().item()),
                'train_cox_loss': float(rank_objective['cox_loss'].detach().item()),
                'train_amplitude_cox_loss': float(
                    rank_objective['amplitude_cox_loss'].detach().item()
                ),
                'train_amplitude_anchor_loss': float(
                    rank_objective['anchor_loss'].detach().item()
                ),
                'train_ot_head_l2': float(rank_objective['ot_head_l2'].detach().item()),
                'train_risk_orthogonality_loss': float(
                    rank_objective['risk_orthogonality_loss'].detach().item()
                ),
                'gradient_l2': float(rank_gradient_l2),
                'epoch_seconds': rank_seconds,
                'r14b_distillation_gradient_to_rank': False,
            }
            time_record = {
                'epoch': int(epoch + 1),
                'phase': 'fpt_joint' if include_fpt else 'ranking_warmup',
                'training_stage': fit_scope,
                'fixed_epoch_budget': int(epochs),
                'train_loss': float(time_objective['loss'].detach().item()),
                'train_cox_loss': float(time_objective['cox'].detach().item()),
                'train_aft_loss': float(time_objective['aft'].detach().item()),
                'train_smooth_fpt_loss': float(
                    time_objective['smooth_fpt'].detach().item()
                ),
                'train_teacher_rank_loss': float(
                    time_objective['teacher_rank'].detach().item()
                ),
                'gradient_l2': float(time_gradient_l2),
                'epoch_seconds': time_seconds,
                'teacher_source': 'live_r13_post_update_detached',
                'teacher_requires_grad': False,
                'teacher_gradient_to_rank': False,
            }
            rank_records.append(rank_record)
            time_records.append(time_record)
            if epoch_state_observer is not None:
                epoch_state_observer(
                    model,
                    int(epoch + 1),
                    {'rank': dict(rank_record), 'time': dict(time_record)},
                )
            self._emit(
                f'[R14FOTSurv] epoch {epoch + 1:3d}/{epochs} '
                f'rank={rank_record["train_loss"]:.6f} '
                f'time={time_record["train_loss"]:.6f} '
                f'teacher={time_record["train_teacher_rank_loss"]:.6f}',
                verbose=verbose,
                log_fn=log_fn,
            )

        if last_teacher_scores is None:
            raise RuntimeError('R14-F did not produce a live teacher score vector')
        rank_model.best_epoch = int(epochs)
        rank_model.eval_iterations_per_epoch = 1
        rank_model.eval_iterations = int(epochs)
        rank_model.r13_best_validation_ = None
        rank_model.r13_refit_ = {
            'epochs': int(epochs),
            'fit_scope': fit_scope,
            'train_bearings': train_bearings,
        }
        rank_model.r13_probability_calibration_ = {
            'method': 'positive_risk_temperature',
            'temperature': float(rank_params['r13_risk_temperature']),
            'applied_to': 'breslow_development_and_prediction_risks',
            'training_objective_temperature': 1.0,
        }
        rank_model.eval()
        with torch.no_grad():
            final_rank_output = rank_model(rank_signals, rank_bearings, rank_steps)
        final_rank_risks = self._r13_apply_risk_temperature(
            final_rank_output['logits'].squeeze(1).detach().cpu().numpy(),
            rank_params,
        )
        rank_model.train_log_risks_ = final_rank_risks.copy()
        rank_model.train_t_ = times[ordered].astype(np.float32, copy=True)
        rank_model.train_e_ = events[ordered].astype(np.float32, copy=True)
        rank_model.otsurv_epoch_diagnostics_ = rank_records
        rank_model.otsurv_diagnostics_status_ = {
            'requested': bool(collect_diagnostics),
            'config_enabled': True,
            'effective': bool(collect_diagnostics),
            'collected': bool(collect_diagnostics),
        }

        time_model.best_epoch = int(epochs)
        time_model.r14b_best_validation_ = None
        time_model.r14b_refit_ = {
            'epochs': int(epochs),
            'fit_scope': fit_scope,
        }
        time_model.eval()
        with torch.no_grad():
            final_time_output = time_model(time_signals, time_bearings, time_steps)
        final_time_risks = (
            final_time_output['risk'].squeeze(1).detach().cpu().numpy().astype(np.float32)
        )
        time_model.train_log_risks_ = final_time_risks.copy()
        time_model.train_t_ = times[ordered].astype(np.float32, copy=True)
        time_model.train_e_ = events[ordered].astype(np.float32, copy=True)
        time_model.train_fpt_times_ = fpt_t[ordered].astype(np.float32, copy=True)
        time_model.train_fpt_events_ = fpt_e[ordered].astype(np.float32, copy=True)
        time_model.otsurv_epoch_diagnostics_ = time_records
        time_model.otsurv_diagnostics_status_ = {
            'requested': bool(collect_diagnostics),
            'config_enabled': True,
            'effective': bool(collect_diagnostics),
            'collected': bool(collect_diagnostics),
        }

        rank_condition = self._rank_condition(params)
        teacher_provenance = {
            'architecture': 'live_shared_r13_rank_anchor',
            'fit_scope': fit_scope,
            'fixed_epochs': int(epochs),
            'frozen': False,
            'rank_condition': int(rank_condition),
            'rank_condition_name': f'C{rank_condition + 1}',
            'rank_risk_temperature': float(rank_params['r13_risk_temperature']),
            'teacher_timing': self.TEACHER_TIMING,
            'teacher_target': 'r13_logits_after_own_optimizer_step',
            'gradient_path': 'r14b_distillation_to_r13_none_detached',
            'r13_own_loss': 'exact_cox_plus_registered_r13_stabilizers',
            'train_bearings': train_bearings,
            'teacher_scores_sha256': self._teacher_score_sha256(last_teacher_scores),
        }
        time_model.r14b_teacher_ = dict(teacher_provenance)
        time_model.teacher_scores_ = last_teacher_scores.copy()
        model.best_epoch = int(epochs)
        model.train_log_risks_ = final_rank_risks.copy()
        model.train_t_ = times[ordered].astype(np.float32, copy=True)
        model.train_e_ = events[ordered].astype(np.float32, copy=True)
        model.r14f_rank_epoch_diagnostics_ = rank_records
        model.r14f_time_epoch_diagnostics_ = time_records
        model.otsurv_epoch_diagnostics_ = time_records
        model.otsurv_diagnostics_status_ = {
            'requested': bool(collect_diagnostics),
            'config_enabled': True,
            'effective': bool(collect_diagnostics),
            'collected': bool(collect_diagnostics),
        }
        model.train_bearing_ids_ = train_bearings
        model.val_bearing_ids_ = ()
        model.validation_bearing_id_ = None
        model.r14f_refit_ = {
            'rank_anchor_epochs': int(epochs),
            'time_expert_epochs': int(epochs),
            'rank_condition': int(rank_condition),
            'fit_scope': fit_scope,
            'training_protocol': training_protocol,
        }
        model.r14f_rank_anchor_ = {
            'architecture': f'trainable_r13_c{rank_condition + 1}_shared_teacher',
            'fit_scope': fit_scope,
            'frozen': False,
            'rank_condition': int(rank_condition),
            'risk_definition': 'trainable_r13_cox_log_risk',
            'optimized_by': 'r13_own_exact_cox_objective_only',
            'r14b_distillation_gradient': 'detached_none',
            'params': dict(rank_params),
            'train_bearings': train_bearings,
            'validation_bearing': None,
        }
        model.r14f_time_expert_ = {
            'architecture': 'r14b_balanced_lognormal_aft_shared_r13_teacher',
            'fit_scope': fit_scope,
            'params': dict(time_params),
            'train_bearings': train_bearings,
            'validation_bearing': None,
            'teacher_provenance': dict(teacher_provenance),
        }
        model.r14f_teacher_ = dict(teacher_provenance)
        model.r14f_architecture_ = {
            **dict(model.r14f_architecture_),
            'training_protocol': training_protocol,
        }
        model.r14f_rank_params_ = dict(rank_params)
        model.r14f_time_params_ = dict(time_params)
        model.r14f_fpt_label_scope_ = fit_scope
        return model, final_rank_risks, model.train_t_, model.train_e_, (
            time_records if collect_diagnostics else None
        )

    def predict(
        self,
        model,
        X_raw,
        params,
        device,
        verbose=True,
        sample_meta=None,
        *,
        rank_device=None,
    ):
        self._validate_r14f_params(params)
        model = self._require_model(model)
        branch_device, _ = self._resolve_branch_devices(
            device, rank_device=rank_device
        )
        values = R13OTSurvRegressor().predict(
            model.rank_anchor,
            X_raw,
            self._rank_params(params),
            branch_device,
            verbose=False,
            sample_meta=sample_meta,
        )
        if not np.isfinite(values).all():
            raise RuntimeError('R14-F predicted R13 rank risks are non-finite')
        if verbose:
            print(f'[R14FOTSurv] predicted {len(values)} R13 rank risks', flush=True)
        return values

    def predict_survival(
        self,
        model,
        X_raw,
        event_times,
        params,
        device,
        n_post_samples=1,
        verbose=True,
        sample_meta=None,
        *,
        time_device=None,
    ):
        self._validate_r14f_params(params)
        model = self._require_model(model)
        _, branch_device = self._resolve_branch_devices(
            device, time_device=time_device
        )
        curves = R14BOTSurvRegressor().predict_survival(
            model.time_expert,
            X_raw,
            event_times,
            self._time_params(params),
            branch_device,
            n_post_samples=n_post_samples,
            verbose=False,
            sample_meta=sample_meta,
        )
        if verbose:
            print(f'[R14FOTSurv] predicted {len(curves)} R14-B time curves', flush=True)
        return curves

    def predict_components(
        self,
        model,
        X_raw,
        params,
        device,
        verbose=True,
        sample_meta=None,
        *,
        rank_device=None,
        time_device=None,
    ):
        self._validate_r14f_params(params)
        model = self._require_model(model)
        rank_branch_device, time_branch_device = self._resolve_branch_devices(
            device,
            rank_device=rank_device,
            time_device=time_device,
        )
        rank = R13OTSurvRegressor().predict_components(
            model.rank_anchor,
            X_raw,
            self._rank_params(params),
            rank_branch_device,
            verbose=False,
            sample_meta=sample_meta,
        )
        time = R14BOTSurvRegressor().predict_components(
            model.time_expert,
            X_raw,
            self._time_params(params),
            time_branch_device,
            verbose=False,
            sample_meta=sample_meta,
        )
        result = {
            'risk': rank['log_risk'],
            'log_risk': rank['log_risk'],
            'r13_anchor_risk': rank['log_risk'],
            'time_mu': time['mu'],
            'time_sigma': time['sigma'],
            'time_expert_risk': time['risk'],
            'rank_anchor': rank,
            'time_expert': time,
        }
        if not np.allclose(result['risk'], result['r13_anchor_risk'], rtol=1e-6, atol=1e-7):
            raise RuntimeError('R14-F rank endpoint decomposition is invalid')
        if not np.allclose(result['time_expert_risk'], -result['time_mu'], rtol=1e-6, atol=1e-7):
            raise RuntimeError('R14-F time endpoint decomposition is invalid')
        if verbose:
            print(f'[R14FOTSurv] predicted components for {len(result["risk"])} samples', flush=True)
        return result

class RawCNNCoxRegressor(BaseRegressor):
    """Input-matched pure raw CNN baseline with an exact Cox endpoint.

    This is intentionally not a renamed legacy mean-pooling OTSurv model.  It
    owns a separate CNN, optimizer, and parameter contract, and contains no
    amplitude branch, transport, prototypes, trajectory features, or warm
    start.  The only common survival component is the generic Breslow
    conversion used by every Cox model in this repository.
    """

    MODE = 'raw-cnn'

    INPUT_REPRESENTATION = 'dual_channel_raw_vibration'

    COX_OBJECTIVE = 'exact_complete_risk_set_breslow_ties'

    def make_model(self, params=None, mode=None):
        """Create an independently initialized pure waveform CNN."""
        from otsurv.model_raw_cnn_cox import RawCNNCox

        model_params = dict(cfg.PARAMS_RAW_CNN_COX)
        if params:
            model_params.update(params)
        requested_mode = mode if mode is not None else model_params.get('mode', self.MODE)
        if requested_mode != self.MODE:
            raise ValueError("RawCNNCoxRegressor requires mode='raw-cnn'")
        self._validate_raw_cnn_params(model_params)
        model = RawCNNCox(
            signal_length=int(model_params['signal_length']),
            window_length=int(model_params['window_length']),
            hidden_dim=int(model_params['hidden_dim']),
            base_channels=int(model_params['base_channels']),
        )
        model.raw_cnn_architecture_ = {
            'input': (
                f'[{int(model_params["signal_length"])}, 2] '
                'dual-channel raw vibration'
            ),
            'window_length': int(model_params['window_length']),
            'stride': int(model_params['stride']),
            'window_count': int(model.window_count),
            'encoder': 'Conv1d+GroupNorm+ReLU x3 -> AdaptiveAvgPool1d -> Linear',
            'pooling': 'mean_over_waveform_windows',
            'risk_head': 'linear_log_risk_without_bias',
            'loss': self.COX_OBJECTIVE,
        }
        return model

    def get_hyperparams(self, condition):
        if condition not in {0, 1, 2}:
            raise ValueError("Invalid condition, choose {0, 1, 2}")
        return dict(cfg.PARAMS_RAW_CNN_COX)

    @classmethod
    def _validate_raw_cnn_params(cls, params):
        """Fail closed so this baseline cannot acquire hidden OT/R13 options."""
        if not isinstance(params, dict):
            raise ValueError("Raw-CNN parameters must be a dictionary")
        expected_keys = set(cfg.PARAMS_RAW_CNN_COX)
        missing = sorted(expected_keys - set(params))
        unexpected = sorted(set(params) - expected_keys)
        if missing:
            raise ValueError(f"Raw-CNN parameters are missing required keys: {missing}")
        if unexpected:
            raise ValueError(
                "Raw-CNN parameters include unsupported non-baseline keys: "
                f"{unexpected}"
            )
        if params['mode'] != cls.MODE:
            raise ValueError("Raw-CNN parameters require mode='raw-cnn'")
        if params['loss_fn'] != 'cox':
            raise ValueError("Raw-CNN supports only exact Cox training")
        if params['raw_signal_normalization'] != 'none':
            raise ValueError("Raw-CNN requires physical-scale raw_signal_normalization='none'")

        integer_keys = (
            'signal_length',
            'window_length',
            'stride',
            'hidden_dim',
            'base_channels',
            'batch_size',
            'max_epochs',
            'early_stopping_patience',
            'min_epochs_before_stopping',
        )
        for key in integer_keys:
            value = params[key]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"Raw-CNN {key} must be an integer")
            if int(value) < 1:
                raise ValueError(f"Raw-CNN {key} must be positive")
        for key in ('lr', 'weight_decay'):
            value = params[key]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"Raw-CNN {key} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"Raw-CNN {key} must be finite")

        supported_lengths = {
            int(cfg.PARAMS_RAW_CNN_COX['signal_length']),
            int(cfg.PARAMS_PHM_RAW_CNN_COX['signal_length']),
        }
        if int(params['signal_length']) not in supported_lengths:
            raise ValueError(
                'Raw-CNN supports only configured XJTU or PHM raw signal lengths'
            )
        if int(params['window_length']) != 512 or int(params['stride']) != 512:
            raise ValueError("Raw-CNN requires 512-sample non-overlapping windows")
        if int(params['signal_length']) % int(params['window_length']):
            raise ValueError("Raw-CNN signal_length must tile exactly into waveform windows")
        if float(params['lr']) <= 0.0:
            raise ValueError("Raw-CNN lr must be positive")
        if float(params['weight_decay']) < 0.0:
            raise ValueError("Raw-CNN weight_decay must be non-negative")
        if int(params['min_epochs_before_stopping']) > int(params['max_epochs']):
            raise ValueError(
                "Raw-CNN min_epochs_before_stopping must not exceed max_epochs"
            )

    @staticmethod
    def _resolve_device(device) -> torch.device:
        if isinstance(device, str):
            device = torch.device(device)
        if not isinstance(device, torch.device):
            raise TypeError("Raw-CNN device must be a torch.device or device string")
        if device.type == 'cuda' and not torch.cuda.is_available():
            logger.info("CUDA device requested but unavailable; using CPU for Raw-CNN.")
            return torch.device('cpu')
        return device

    @staticmethod
    def _emit(message, *, verbose=False, log_fn=None):
        if verbose:
            print(message, flush=True)
        if log_fn is not None:
            log_fn(message)

    @staticmethod
    def resolve_diagnostics_collection_status(collect_diagnostics: bool):
        """Match the repository's requested-versus-effective diagnostics rule."""
        diag_cfg = getattr(cfg, 'OTSURV_DIAGNOSTICS', {})
        if diag_cfg is None:
            diag_cfg = {}
        if not isinstance(diag_cfg, dict):
            raise ValueError("OTSURV_DIAGNOSTICS must be a dictionary")
        requested = bool(collect_diagnostics)
        config_enabled = bool(diag_cfg.get('enabled', True))
        return {
            'requested': requested,
            'config_enabled': config_enabled,
            'effective': requested and config_enabled,
        }

    @classmethod
    def _validate_raw_signals(cls, X_raw, params):
        if not isinstance(X_raw, (list, tuple)) or not X_raw:
            raise ValueError("Raw-CNN requires a non-empty list of raw signals")
        expected_length = int(params['signal_length'])
        signals = []
        for signal in X_raw:
            if (
                not torch.is_tensor(signal)
                or signal.ndim != 2
                or signal.shape[0] != expected_length
                or signal.shape[1] != 2
                or not torch.isfinite(signal).all()
            ):
                raise ValueError(
                    f"Raw-CNN raw signals must be finite [{expected_length}, 2] tensors"
                )
            signals.append(signal.detach().to(dtype=torch.float32).contiguous())
        return signals

    @staticmethod
    def _validate_labels(t, e, n_samples, *, context):
        times = np.asarray(t, dtype=np.float32).reshape(-1)
        events = np.asarray(e, dtype=np.float32).reshape(-1)
        if len(times) != n_samples or len(events) != n_samples:
            raise ValueError(f"Raw-CNN {context} labels must align with raw signals")
        if not np.isfinite(times).all() or not np.isfinite(events).all():
            raise ValueError(f"Raw-CNN {context} labels must be finite")
        if np.any(times < 0.0):
            raise ValueError(f"Raw-CNN {context} survival times must be non-negative")
        if not np.isin(events, (0.0, 1.0)).all():
            raise ValueError(f"Raw-CNN {context} event indicators must be binary")
        return times, events

    @staticmethod
    def _validated_sample_meta(sample_meta, n_samples, *, context):
        if not isinstance(sample_meta, dict):
            raise ValueError(f"Raw-CNN {context} requires sample metadata")
        bearing_ids = np.asarray(sample_meta.get('bearing_id'))
        time_steps = np.asarray(sample_meta.get('time_step'))
        for name, values in (('bearing_id', bearing_ids), ('time_step', time_steps)):
            if (
                values.ndim != 1
                or len(values) != n_samples
                or not np.issubdtype(values.dtype, np.number)
                or not np.isfinite(values).all()
                or np.any(values != np.floor(values))
            ):
                raise ValueError(
                    f"Raw-CNN {context} requires finite integer {name} metadata"
                )
        if np.any(time_steps < 0):
            raise ValueError("Raw-CNN time_step values must be non-negative")
        bearing_ids = bearing_ids.astype(np.int64, copy=False)
        time_steps = time_steps.astype(np.int64, copy=False)
        identities = np.column_stack((bearing_ids, time_steps))
        if np.unique(identities, axis=0).shape[0] != n_samples:
            raise ValueError("Raw-CNN requires unique bearing_id/time_step pairs")
        return bearing_ids, time_steps

    @classmethod
    def _make_bearing_validation_split(cls, times, sample_meta, validation_bearing_id):
        bearings, steps = cls._validated_sample_meta(
            sample_meta, len(times), context='formal validation'
        )
        validation_mask = bearings == int(validation_bearing_id)
        if not validation_mask.any():
            raise ValueError(
                f"validation bearing {validation_bearing_id!r} is absent from Raw-CNN data"
            )
        if validation_mask.all():
            raise ValueError("Raw-CNN validation bearing would consume all training data")
        ordered_indices = np.lexsort((steps, bearings)).astype(int)
        ordered_validation_mask = validation_mask[ordered_indices]
        train_indices = ordered_indices[~ordered_validation_mask]
        val_indices = ordered_indices[ordered_validation_mask]
        train_bearings = tuple(sorted(int(value) for value in np.unique(bearings[train_indices])))
        val_bearings = tuple(sorted(int(value) for value in np.unique(bearings[val_indices])))
        if val_bearings != (int(validation_bearing_id),):
            raise RuntimeError("Raw-CNN grouped validation selected an unexpected bearing")
        if set(train_bearings) & set(val_bearings):
            raise RuntimeError("Raw-CNN grouped validation leaked a bearing")
        return ordered_indices, train_indices, val_indices, train_bearings, val_bearings

    @staticmethod
    def _signal_batch(signals, indices, device):
        batch = torch.stack([signals[int(index)] for index in indices], dim=0)
        return batch.to(device=device, dtype=torch.float32, non_blocking=False)

    @classmethod
    def _predict_logits(cls, model, signals, indices, batch_size, device):
        indices = np.asarray(indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError("Raw-CNN prediction indices must be one-dimensional")
        if batch_size < 1:
            raise ValueError("Raw-CNN batch_size must be positive")
        logits_parts = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                logits = model(cls._signal_batch(signals, batch_indices, device))
                if logits.shape != (len(batch_indices), 1) or not torch.isfinite(logits).all():
                    raise RuntimeError("Raw-CNN prediction produced invalid [batch, 1] logits")
                logits_parts.append(logits.detach())
        if not logits_parts:
            return np.empty(0, dtype=np.float32)
        return torch.cat(logits_parts, dim=0).squeeze(1).cpu().numpy().astype(np.float32)

    @classmethod
    def _evaluate_exact_cox_loss(cls, model, signals, times, events, indices, batch_size, device):
        logits = cls._predict_logits(model, signals, indices, batch_size, device)
        if len(logits) == 0:
            return float('nan'), float('nan')
        from otsurv.losses import partial_ll_loss

        index_array = np.asarray(indices, dtype=int)
        risk_tensor = torch.as_tensor(logits, device=device, dtype=torch.float32)
        time_tensor = torch.as_tensor(times[index_array], device=device, dtype=torch.float32)
        event_tensor = torch.as_tensor(events[index_array], device=device, dtype=torch.float32)
        loss = partial_ll_loss(risk_tensor, time_tensor, event_tensor)['loss']
        if not torch.isfinite(loss):
            raise RuntimeError("Raw-CNN validation produced a non-finite Cox loss")
        return float(loss.item()), float(np.mean(np.abs(logits)))

    @classmethod
    def _train_exact_cox_epoch(
        cls,
        model,
        signals,
        times,
        events,
        indices,
        batch_size,
        optimizer,
        device,
        *,
        permutation=None,
    ):
        """Replay micro-batches through the exact complete-risk-set gradient.

        The first pass materializes no graph and obtains all logits for the
        selected cohort.  The second pass applies the analytical full-Cox
        logit gradient to each micro-batch, which is algebraically equivalent
        to backpropagating the objective over the entire risk set at once.
        """
        from otsurv.losses import cox_breslow_loss_and_logit_gradients

        indices = np.asarray(indices, dtype=int)
        if len(indices) < 1:
            raise ValueError("Raw-CNN exact Cox training requires at least one sample")
        if batch_size < 1:
            raise ValueError("Raw-CNN batch_size must be positive")
        if permutation is None:
            permutation = np.random.permutation(len(indices))
        else:
            permutation = np.asarray(permutation, dtype=int)
        if (
            permutation.ndim != 1
            or len(permutation) != len(indices)
            or not np.array_equal(np.sort(permutation), np.arange(len(indices)))
        ):
            raise ValueError("Raw-CNN permutation must contain every cohort row once")
        ordered_indices = indices[permutation]
        batches = [
            ordered_indices[start:min(start + batch_size, len(ordered_indices))]
            for start in range(0, len(ordered_indices), batch_size)
        ]

        model.train()
        initial_logits = []
        with torch.no_grad():
            for batch_indices in batches:
                logits = model(cls._signal_batch(signals, batch_indices, device))
                if logits.shape != (len(batch_indices), 1) or not torch.isfinite(logits).all():
                    raise RuntimeError("Raw-CNN exact Cox first pass produced invalid logits")
                initial_logits.append(logits.detach().reshape(-1))
        full_logits = torch.cat(initial_logits, dim=0)
        full_times = torch.as_tensor(
            times[ordered_indices], device=device, dtype=full_logits.dtype
        )
        full_events = torch.as_tensor(
            events[ordered_indices], device=device, dtype=full_logits.dtype
        )
        cox_loss, logit_gradients = cox_breslow_loss_and_logit_gradients(
            full_logits, full_times, full_events
        )
        if not torch.isfinite(cox_loss) or not torch.isfinite(logit_gradients).all():
            raise RuntimeError("Raw-CNN exact Cox loss or gradient is non-finite")

        optimizer.zero_grad(set_to_none=True)
        offset = 0
        for batch_indices in batches:
            batch_size_actual = len(batch_indices)
            live_logits = model(cls._signal_batch(signals, batch_indices, device)).reshape(-1)
            if live_logits.shape != (batch_size_actual,) or not torch.isfinite(live_logits).all():
                raise RuntimeError("Raw-CNN exact Cox replay produced invalid logits")
            surrogate = torch.sum(
                live_logits * logit_gradients[offset:offset + batch_size_actual]
            )
            surrogate.backward()
            offset += batch_size_actual
        if offset != len(ordered_indices):
            raise RuntimeError("Raw-CNN exact Cox replay did not cover the full cohort")

        squared_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError("Raw-CNN produced a non-finite parameter gradient")
            squared_norm += float(parameter.grad.detach().square().sum().item())
        optimizer.step()
        return {
            'loss': float(cox_loss.item()),
            'mean_abs_risk': float(full_logits.detach().abs().mean().item()),
            'gradient_l2': math.sqrt(squared_norm),
            'event_count': int(full_events.sum().item()),
        }

    @staticmethod
    def _diagnostic_split(log_risks, times, events, bearing_ids, time_steps):
        return {
            'log_risk': np.asarray(log_risks, dtype=np.float32).copy(),
            'times': np.asarray(times, dtype=np.float32).copy(),
            'events': np.asarray(events, dtype=np.float32).copy(),
            'sample_meta': {
                'bearing_id': np.asarray(bearing_ids, dtype=np.int64).copy(),
                'time_step': np.asarray(time_steps, dtype=np.int64).copy(),
            },
        }

    def fit(
        self,
        model,
        X_raw: List[torch.Tensor],
        t: np.ndarray,
        e: np.ndarray,
        params: dict,
        device,
        verbose: bool = True,
        sample_meta: dict = None,
        log_fn=None,
        validation_bearing_id=None,
        collect_diagnostics: bool = True,
    ) -> Tuple:
        """Select an epoch on a disjoint validation bearing using exact Cox."""
        self._validate_raw_cnn_params(params)
        if validation_bearing_id is None:
            raise ValueError("Raw-CNN formal training requires validation_bearing_id")
        device = self._resolve_device(device)
        model = model.to(device)
        signals = self._validate_raw_signals(X_raw, params)
        times, events = self._validate_labels(t, e, len(signals), context='training')
        bearing_ids, time_steps = self._validated_sample_meta(
            sample_meta, len(signals), context='training'
        )
        (
            ordered_indices,
            train_indices,
            val_indices,
            train_bearings,
            val_bearings,
        ) = self._make_bearing_validation_split(times, sample_meta, validation_bearing_id)
        if int(events[train_indices].sum()) < 1:
            raise ValueError("Raw-CNN inner-train data requires at least one observed event")

        diagnostics_status = self.resolve_diagnostics_collection_status(
            collect_diagnostics
        )
        collect_diagnostics = diagnostics_status['effective']
        diagnostics_status['collected'] = False
        model.otsurv_diagnostics_status_ = dict(diagnostics_status)
        model.raw_cnn_diagnostics_status_ = dict(diagnostics_status)
        model.otsurv_epoch_diagnostics_ = []
        model.raw_cnn_epoch_diagnostics_ = model.otsurv_epoch_diagnostics_
        model.train_bearing_ids_ = train_bearings
        model.val_bearing_ids_ = val_bearings
        model.validation_bearing_id_ = int(validation_bearing_id)

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(params['lr']),
            weight_decay=float(params['weight_decay']),
        )
        max_epochs = int(params['max_epochs'])
        patience = int(params['early_stopping_patience'])
        min_epochs = int(params['min_epochs_before_stopping'])
        batch_size = int(params['batch_size'])
        best_state = None
        best_epoch = 0
        best_val_loss = float('inf')
        epochs_without_improve = 0
        self._emit(
            f"[RawCNNCox] samples={len(signals)} train={len(train_indices)} "
            f"val={len(val_indices)} windows=64 device={device} lr={params['lr']} "
            f"cox={self.COX_OBJECTIVE}",
            verbose=verbose,
            log_fn=log_fn,
        )

        for epoch in range(max_epochs):
            epoch_started = time.perf_counter()
            train_metrics = self._train_exact_cox_epoch(
                model,
                signals,
                times,
                events,
                train_indices,
                batch_size,
                optimizer,
                device,
            )
            val_loss, val_mean_abs_risk = self._evaluate_exact_cox_loss(
                model,
                signals,
                times,
                events,
                val_indices,
                batch_size,
                device,
            )
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
            record = {
                'epoch': int(epoch + 1),
                'phase': 'inner_bearing_epoch_selection',
                'training_stage': 'inner_train_selection',
                'train_loss': train_metrics['loss'],
                'train_cox_loss': train_metrics['loss'],
                'train_mean_abs_risk': train_metrics['mean_abs_risk'],
                'train_event_count': train_metrics['event_count'],
                'val_loss': float(val_loss),
                'val_mean_abs_risk': float(val_mean_abs_risk),
                'best_val_loss': float(best_val_loss),
                'selection_eligible': True,
                'gradient_l2': train_metrics['gradient_l2'],
                'epoch_seconds': float(time.perf_counter() - epoch_started),
            }
            model.otsurv_epoch_diagnostics_.append(record)
            self._emit(
                f"  [RawCNNCox] epoch {epoch + 1:3d}/{max_epochs} "
                f"train_loss={train_metrics['loss']:.6f} val_loss={val_loss:.6f} "
                f"best={best_val_loss:.6f} grad_l2={train_metrics['gradient_l2']:.6f}",
                verbose=verbose,
                log_fn=log_fn,
            )
            if epoch + 1 >= min_epochs and epochs_without_improve >= patience:
                self._emit(
                    f"[RawCNNCox] early stopping at epoch {epoch + 1}; best={best_epoch}",
                    verbose=verbose,
                    log_fn=log_fn,
                )
                break

        if best_state is None:
            raise RuntimeError("Raw-CNN training did not produce a valid checkpoint")
        model.load_state_dict(best_state)
        model.best_epoch = int(best_epoch)
        model.eval_iterations_per_epoch = 1
        model.eval_iterations = int(best_epoch)
        model.raw_cnn_selection_ = {
            'fit_scope': 'inner_train_only',
            'train_bearings': train_bearings,
            'validation_bearing': int(validation_bearing_id),
            'best_epoch': int(best_epoch),
            'best_validation_loss': float(best_val_loss),
            'cox_objective': self.COX_OBJECTIVE,
        }
        model.raw_cnn_refit_ = None
        model.raw_cnn_breslow_cohort_ = {
            'fit_scope': 'outer_development_selection_model',
            'bearings': tuple(
                sorted(int(value) for value in np.unique(bearing_ids[ordered_indices]))
            ),
        }
        train_risks = self._predict_logits(
            model, signals, ordered_indices, batch_size, device
        )
        ordered_times = times[ordered_indices].copy()
        ordered_events = events[ordered_indices].copy()
        model.train_log_risks_ = train_risks
        model.train_t_ = ordered_times
        model.train_e_ = ordered_events

        diagnostics = None
        if collect_diagnostics:
            train_risks_diag = self._predict_logits(
                model, signals, train_indices, batch_size, device
            )
            val_risks_diag = self._predict_logits(
                model, signals, val_indices, batch_size, device
            )
            diagnostics = {
                'train': self._diagnostic_split(
                    train_risks_diag,
                    times[train_indices],
                    events[train_indices],
                    bearing_ids[train_indices],
                    time_steps[train_indices],
                ),
                'val': self._diagnostic_split(
                    val_risks_diag,
                    times[val_indices],
                    events[val_indices],
                    bearing_ids[val_indices],
                    time_steps[val_indices],
                ),
            }
        diagnostics_status['collected'] = diagnostics is not None
        model.otsurv_diagnostics_status_ = dict(diagnostics_status)
        model.raw_cnn_diagnostics_status_ = dict(diagnostics_status)
        return model, train_risks, ordered_times, ordered_events, diagnostics

    def fit_fixed_epochs(
        self,
        model,
        X_raw: List[torch.Tensor],
        t: np.ndarray,
        e: np.ndarray,
        params: dict,
        device,
        *,
        fixed_epochs: int,
        sample_meta: dict = None,
        log_fn=None,
        verbose: bool = True,
        collect_diagnostics: bool = True,
    ) -> Tuple:
        """Freshly fit all development bearings for the selected epoch count.

        There is deliberately no validation input, early stopping, checkpoint
        selection, warm start, or copied model state in this final phase.
        """
        self._validate_raw_cnn_params(params)
        try:
            refit_epochs = int(fixed_epochs)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Raw-CNN fixed refit epochs must be a positive integer"
            ) from exc
        if isinstance(fixed_epochs, bool) or refit_epochs != fixed_epochs:
            raise ValueError("Raw-CNN fixed refit epochs must be a positive integer")
        if refit_epochs < 1 or refit_epochs > int(params['max_epochs']):
            raise ValueError(
                "Raw-CNN fixed refit epochs must be within the configured max_epochs"
            )
        device = self._resolve_device(device)
        model = model.to(device)
        signals = self._validate_raw_signals(X_raw, params)
        times, events = self._validate_labels(t, e, len(signals), context='fixed refit')
        bearing_ids, time_steps = self._validated_sample_meta(
            sample_meta, len(signals), context='fixed refit'
        )
        ordered_indices = np.lexsort((time_steps, bearing_ids)).astype(int)
        train_bearings = tuple(
            sorted(int(value) for value in np.unique(bearing_ids[ordered_indices]))
        )
        if int(events[ordered_indices].sum()) < 1:
            raise ValueError("Raw-CNN full-development refit requires an observed event")

        diagnostics_status = self.resolve_diagnostics_collection_status(
            collect_diagnostics
        )
        collect_diagnostics = diagnostics_status['effective']
        diagnostics_status['collected'] = False
        model.otsurv_diagnostics_status_ = dict(diagnostics_status)
        model.raw_cnn_diagnostics_status_ = dict(diagnostics_status)
        model.otsurv_epoch_diagnostics_ = []
        model.raw_cnn_epoch_diagnostics_ = model.otsurv_epoch_diagnostics_
        model.train_bearing_ids_ = train_bearings
        model.val_bearing_ids_ = ()
        model.validation_bearing_id_ = None
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(params['lr']),
            weight_decay=float(params['weight_decay']),
        )
        batch_size = int(params['batch_size'])
        self._emit(
            f"[RawCNNCoxRefit] samples={len(signals)} train={len(ordered_indices)} "
            f"epochs={refit_epochs} windows=64 device={device} lr={params['lr']} "
            f"cox={self.COX_OBJECTIVE}",
            verbose=verbose,
            log_fn=log_fn,
        )
        for epoch in range(refit_epochs):
            epoch_started = time.perf_counter()
            train_metrics = self._train_exact_cox_epoch(
                model,
                signals,
                times,
                events,
                ordered_indices,
                batch_size,
                optimizer,
                device,
            )
            record = {
                'epoch': int(epoch + 1),
                'phase': 'full_development_refit',
                'training_stage': 'full_development_refit',
                'fixed_epoch_budget': int(refit_epochs),
                'train_loss': train_metrics['loss'],
                'train_cox_loss': train_metrics['loss'],
                'train_mean_abs_risk': train_metrics['mean_abs_risk'],
                'train_event_count': train_metrics['event_count'],
                'selection_eligible': False,
                'gradient_l2': train_metrics['gradient_l2'],
                'epoch_seconds': float(time.perf_counter() - epoch_started),
            }
            model.otsurv_epoch_diagnostics_.append(record)
            self._emit(
                f"  [RawCNNCoxRefit] epoch {epoch + 1:3d}/{refit_epochs} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"grad_l2={train_metrics['gradient_l2']:.6f}",
                verbose=verbose,
                log_fn=log_fn,
            )

        model.best_epoch = int(refit_epochs)
        model.eval_iterations_per_epoch = 1
        model.eval_iterations = int(refit_epochs)
        model.raw_cnn_selection_ = None
        model.raw_cnn_refit_ = {
            'fit_scope': 'outer_development_refit_all_bearings',
            'train_bearings': train_bearings,
            'epochs': int(refit_epochs),
            'early_stopping': False,
            'cox_objective': self.COX_OBJECTIVE,
        }
        model.raw_cnn_breslow_cohort_ = {
            'fit_scope': 'outer_development_refit_all_bearings',
            'bearings': train_bearings,
        }
        train_risks = self._predict_logits(
            model, signals, ordered_indices, batch_size, device
        )
        ordered_times = times[ordered_indices].copy()
        ordered_events = events[ordered_indices].copy()
        model.train_log_risks_ = train_risks
        model.train_t_ = ordered_times
        model.train_e_ = ordered_events
        diagnostics = None
        if collect_diagnostics:
            diagnostics = {
                'train': self._diagnostic_split(
                    train_risks,
                    ordered_times,
                    ordered_events,
                    bearing_ids[ordered_indices],
                    time_steps[ordered_indices],
                )
            }
        diagnostics_status['collected'] = diagnostics is not None
        model.otsurv_diagnostics_status_ = dict(diagnostics_status)
        model.raw_cnn_diagnostics_status_ = dict(diagnostics_status)
        return model, train_risks, ordered_times, ordered_events, diagnostics

    def predict(
        self,
        model,
        X_raw: List[torch.Tensor],
        params: dict,
        device,
        verbose: bool = True,
        sample_meta: dict = None,
    ) -> np.ndarray:
        """Predict one raw-CNN Cox log-risk per raw vibration minute."""
        del sample_meta  # The pure CNN does not consume trajectory metadata at inference.
        self._validate_raw_cnn_params(params)
        device = self._resolve_device(device)
        model = model.to(device)
        signals = self._validate_raw_signals(X_raw, params)
        risks = self._predict_logits(
            model,
            signals,
            np.arange(len(signals), dtype=int),
            int(params['batch_size']),
            device,
        )
        if verbose:
            print(f"[RawCNNCox] predicted {len(risks)} samples", flush=True)
        return risks

    def predict_survival(
        self,
        model,
        X_raw: List[torch.Tensor],
        event_times: np.ndarray,
        params: dict,
        device,
        n_post_samples: int = 1,
        verbose: bool = True,
        sample_meta: dict = None,
    ):
        """Convert raw-CNN risks to evaluator-ready Breslow survival curves.

        The shared evaluator consumes a ``[samples, times]`` DataFrame, so
        unlike the historical posterior-sampling wrapper this deterministic
        Cox baseline returns the Breslow DataFrame directly.
        """
        if n_post_samples != 1:
            raise ValueError("Raw-CNN+Cox is deterministic and supports n_post_samples=1")
        if not all(hasattr(model, name) for name in ('train_log_risks_', 'train_t_', 'train_e_')):
            raise RuntimeError(
                "Raw-CNN survival prediction requires the fitted development Breslow cohort"
            )
        from otsurv.breslow_converter import BreslowConverter

        test_risks = self.predict(
            model,
            X_raw,
            params,
            device,
            verbose=verbose,
            sample_meta=sample_meta,
        )
        converter = BreslowConverter().fit(
            model.train_log_risks_, model.train_t_, model.train_e_
        )
        return converter.predict_survival(test_risks, event_times)
