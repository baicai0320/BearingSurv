"""Public adapters over the final paper's preserved numerical implementations."""

from __future__ import annotations

import copy
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
import torch

from paper import config as cfg
from paper import training
from paper.data import FEATURES
from otsurv.raw_signal_loader import RawSignalLoader
from utility.mtlr import DeepMTLR, train_mtlr_fixed_epochs, make_mtlr_prediction
from utility.survival import Survival, make_time_bins


MODEL_NAMES = ("otsurv", "rawcnn", "coxph", "coxboost", "rsf", "mtlr", "dsm", "bnnsurv")
BASELINES = {"rawcnn": training.RawCNNCoxRegressor, "coxph": training.CoxPH,
             "coxboost": training.CoxBoost, "rsf": training.RSF, "mtlr": training.MTLR,
             "dsm": training.DSM, "bnnsurv": training.BNNSurv}


def seed_everything(seed=0):
    if seed != cfg.PAPER_SEED:
        raise ValueError("the paper profile requires seed 0")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def baseline_parameters(name, condition, cpu_workers=18):
    params = BASELINES[name]().get_hyperparams(condition)
    if name == "rsf":
        params.update(random_state=0, n_jobs=cpu_workers)
    elif name == "coxboost":
        params.update(random_state=0)
    elif name in {"dsm", "bnnsurv"}:
        params.update(random_seed=0)
    return params


class OTSurv:
    """Dual-endpoint OTSurv with the fixed paper checkpoint schedule.

    FPT targets are explicit training-only inputs to the time endpoint.
    Prediction takes only raw signals and their causal sample identities.
    """

    def __init__(self, condition=1, *, device="cpu", time_device=None):
        if condition not in cfg.REPORT_EPOCHS:
            raise ValueError("condition must be 0, 1, or 2")
        self.condition = condition
        self.device = torch.device(device)
        self.time_device = torch.device(time_device or device)
        self.regressor = training.R14FOTSurvRegressor()
        self.params = self.regressor.get_hyperparams(condition)
        self.model = None
        self.checkpoints = {}

    def effective_parameters(self):
        rank = self.regressor._rank_params(self.params)
        rank["max_epochs"] = cfg.TRAIN_EPOCHS
        return {"parameters": dict(self.params), "rank_parameters": rank,
                "time_parameters": self.regressor._time_params(self.params),
                "train_epochs": cfg.TRAIN_EPOCHS,
                "time_cnn_window_batch_size": cfg.TIME_CNN_WINDOW_BATCH_SIZES[self.condition],
                "checkpoint_epoch": cfg.REPORT_EPOCHS[self.condition], "seed": 0}

    def fit(self, signals, times, events, *, bearing_ids, time_steps, fpt_times, fpt_events,
            log_fn=print):
        torch.manual_seed(cfg.PAPER_SEED)
        self.model = self.regressor.make_model(self.params)
        self.model.time_expert.RAW_CNN_WINDOW_BATCH_SIZE = cfg.TIME_CNN_WINDOW_BATCH_SIZES[self.condition]
        log_fn(f"[otsurv] train_epochs={cfg.TRAIN_EPOCHS}; report_epoch={cfg.REPORT_EPOCHS[self.condition]}; "
               f"time_cnn_window_batch_size={cfg.TIME_CNN_WINDOW_BATCH_SIZES[self.condition]}")

        def capture(model, epoch, record):
            if epoch in cfg.CHECKPOINT_EPOCHS:
                self.checkpoints[int(epoch)] = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            log_fn(f"[otsurv] epoch {epoch}/{cfg.TRAIN_EPOCHS}")

        self.regressor.fit_checkpoint_sweep(
            self.model, signals, np.asarray(times, dtype=np.float32), np.asarray(events, dtype=np.float32),
            self.params, self.device, sweep_epochs=cfg.TRAIN_EPOCHS,
            fpt_times=np.asarray(fpt_times, dtype=np.float32), fpt_events=np.asarray(fpt_events, dtype=np.float32),
            sample_meta={"bearing_id": np.asarray(bearing_ids), "time_step": np.asarray(time_steps)},
            rank_device=self.device, time_device=self.time_device, verbose=False,
            collect_diagnostics=True, epoch_state_observer=capture,
        )
        if tuple(self.checkpoints) != cfg.CHECKPOINT_EPOCHS:
            raise RuntimeError("the 40-epoch trajectory did not produce every registered checkpoint")
        self.model.load_state_dict(self.checkpoints[cfg.REPORT_EPOCHS[self.condition]])
        self.model.eval()
        return self

    def predict_risk(self, signals, *, bearing_ids, time_steps):
        if self.model is None:
            raise RuntimeError("fit or load OTSurv before prediction")
        return self.regressor.predict(
            self.model, signals, self.params, self.device, verbose=False,
            sample_meta={"bearing_id": np.asarray(bearing_ids), "time_step": np.asarray(time_steps)},
            rank_device=self.device,
        )

    def predict_survival(self, signals, evaluation_times, *, bearing_ids, time_steps):
        if self.model is None:
            raise RuntimeError("fit or load OTSurv before prediction")
        return self.regressor.predict_survival(
            self.model, signals, evaluation_times, self.params, self.device, verbose=False,
            sample_meta={"bearing_id": np.asarray(bearing_ids), "time_step": np.asarray(time_steps)},
            time_device=self.time_device,
        )

    def save(self, path):
        if self.model is None:
            raise RuntimeError("cannot save an unfitted model")
        target = Path(path)
        with target.open("xb") as handle:
            torch.save({"format": "bearingsurv_otsurv_v1", "condition": self.condition,
                        "parameters": self.effective_parameters(),
                        "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()}}, handle)

    @classmethod
    def load(cls, path, *, device="cpu", time_device=None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format"] != "bearingsurv_otsurv_v1":
            raise ValueError("unsupported OTSurv checkpoint")
        result = cls(payload["condition"], device=device, time_device=time_device)
        if payload["parameters"] != result.effective_parameters():
            raise ValueError("checkpoint parameters differ from the paper profile")
        result.model = result.regressor.make_model(result.params)
        result.model.time_expert.RAW_CNN_WINDOW_BATCH_SIZE = cfg.TIME_CNN_WINDOW_BATCH_SIZES[result.condition]
        result.model.load_state_dict(payload["state_dict"])
        result.model.rank_anchor.to(result.device)
        result.model.time_expert.to(result.time_device)
        result.model.eval()
        return result


class PaperBaseline:
    """One of the seven fixed reference protocols, without a search interface."""

    def __init__(self, name, condition, *, device="cpu", cpu_workers=18):
        self.name, self.condition, self.device = name, condition, torch.device(device)
        self.builder = BASELINES[name]()
        self.params = baseline_parameters(name, condition, cpu_workers)
        self.fit_metadata = {}

    def fit(self, development, *, outer_fold):
        self.development = development.copy()
        self.scaler = StandardScaler().fit(development[list(FEATURES)])
        self.x = pd.DataFrame(self.scaler.transform(development[list(FEATURES)]), columns=FEATURES)
        t, e = development.Survival_time.to_numpy(), development.Event.to_numpy()
        if self.name == "rawcnn":
            loader = RawSignalLoader(self.condition, normalization="none")
            signals = loader.load_bearing_signals_from_dataframe(development)
            meta = {"bearing_id": development.bearing_id.to_numpy(), "time_step": development.time_step.to_numpy()}
            # main selected RawCNN after the seed-0 R13 refit. R13 consumes CPU
            # Torch RNG only during construction, not during its deterministic fit.
            seed_everything()
            anchor = training.R13OTSurvRegressor()
            del_model = anchor.make_model(anchor.get_hyperparams(self.condition))
            del del_model
            model = self.builder.make_model(self.params)
            model, *_ = self.builder.fit(model, signals, t, e, self.params, self.device,
                                         validation_bearing_id=outer_fold % 5 + 1,
                                         sample_meta=meta, verbose=False, collect_diagnostics=False)
            epochs = int(model.best_epoch)
            self.selection_diagnostics = copy.deepcopy(model.otsurv_epoch_diagnostics_)
            self.fit_metadata = {"selected_epoch": epochs, "validation_bearing": outer_fold % 5 + 1,
                                 "selection": copy.deepcopy(model.raw_cnn_selection_),
                                 "initialization_protocol": "main_r13_refit_cpu_rng_successor"}
            del model
            seed_everything()
            model = self.builder.make_model(self.params)
            self.model, *_ = self.builder.fit_fixed_epochs(
                model, signals, t, e, self.params, self.device, fixed_epochs=epochs,
                sample_meta=meta, verbose=False, collect_diagnostics=False,
            )
        elif self.name == "mtlr":
            self.bins = make_time_bins(t.copy(), event=e.copy())
            self.mtlr_config = training.dotdict({**cfg.PARAMS_MTLR, **self.params, "early_stop": False})
            self.model = DeepMTLR(in_features=len(FEATURES), num_time_bins=len(self.bins), config=self.mtlr_config)
            data = self.x.copy()
            data["Survival_time"], data["Event"] = t, e.astype(int)
            self.model = train_mtlr_fixed_epochs(self.model, data, self.bins, self.mtlr_config,
                                                random_state=0, reset_model=True, device=self.device)
        elif self.name == "dsm":
            self.model = self.builder.make_model(self.params)
            self.model.fit(self.x, t, e, full_fit=True, iters=self.params["iters"],
                           learning_rate=self.params["learning_rate"], batch_size=self.params["batch_size"],
                           pretrain_iters=self.params["pretrain_iters"])
        elif self.name == "bnnsurv":
            self.model = self.builder.make_model(self.params)
            self.model.fit(self.x.to_numpy(), t, e)
            self.fit_metadata["fit_numerical_status"] = self.model.fit_numerical_status_
        else:
            self.model = self.builder.make_model(self.params)
            self.model.fit(self.x, Surv.from_arrays(e.astype(bool), t))
        return self

    def predict(self, test, evaluation_times):
        from paper._evaluation_reference import predict_coxph_survival_with_log_domain_fallback
        from tools.probability_metrics import event_probability_risk_at_horizon

        x = pd.DataFrame(self.scaler.transform(test[list(FEATURES)]), columns=FEATURES)
        risk = None
        if self.name == "rawcnn":
            signals = RawSignalLoader(self.condition, normalization="none").load_bearing_signals_from_dataframe(test)
            meta = {"bearing_id": test.bearing_id.to_numpy(), "time_step": test.time_step.to_numpy()}
            risk = self.builder.predict(self.model, signals, self.params, self.device, verbose=False, sample_meta=meta)
            curves = self.builder.predict_survival(self.model, signals, evaluation_times, self.params,
                                                   self.device, verbose=False, sample_meta=meta)
        elif self.name == "mtlr":
            features = torch.tensor(x.to_numpy(), dtype=torch.float, device=self.device)
            values, times, _ = make_mtlr_prediction(self.model, features, self.bins, self.mtlr_config)
            curves = pd.DataFrame(values.detach().cpu().numpy(), columns=times.detach().cpu().numpy())
        elif self.name == "coxph":
            curves, risk, strategy = predict_coxph_survival_with_log_domain_fallback(
                self.model, self.x, self.development.Survival_time, self.development.Event, x, evaluation_times)
            self.fit_metadata["breslow_strategy"] = strategy
        else:
            curves = Survival.predict_survival_function(
                self.model, x.to_numpy() if self.name in {"dsm", "bnnsurv"} else x,
                evaluation_times, n_post_samples=100)
            if self.name in {"coxboost", "rsf"}:
                risk = np.asarray(self.model.predict(x), dtype=float)
        if 0.0 not in curves.columns:
            curves.insert(0, 0.0, 1.0)
        if risk is None:
            risk, _ = event_probability_risk_at_horizon(curves, 10.0)
        if self.name == "bnnsurv":
            self.fit_metadata["prediction_numerical_status"] = self.model.last_prediction_status_
        return curves, np.asarray(risk)
