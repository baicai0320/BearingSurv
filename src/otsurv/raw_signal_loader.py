"""Validated XJTU-SY raw vibration loading for the two public raw models."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

import config as cfg
from fpt_detection.data_loader import _read_signal, resolve_bearing_directory


SIGNAL_LENGTH = 32768
N_CHANNELS = 2


class RawSignalDataError(RuntimeError):
    """Raised when a source vibration file cannot be used as model input."""


class RawSignalLoader:
    """Load finite XJTU vibration minutes with an explicit normalization mode."""

    NORMALIZATION_MODES = frozenset({"per_signal_zscore", "none"})

    def __init__(self, condition: int, *, normalization: str = "per_signal_zscore") -> None:
        if int(condition) not in cfg.CONDITIONS:
            raise ValueError(f"unsupported XJTU condition: {condition}")
        if normalization not in self.NORMALIZATION_MODES:
            allowed = ", ".join(sorted(self.NORMALIZATION_MODES))
            raise ValueError(f"normalization must be one of: {allowed}")
        self.condition = int(condition)
        self.raw_path = Path(cfg.RAW_DATA_PATH_XJTU[self.condition])
        self.normalization = str(normalization)

    @staticmethod
    def _read_csv(
        path: Path, *, normalization: str = "per_signal_zscore"
    ) -> np.ndarray:
        try:
            values = _read_signal(path, dtype=np.float32)
        except (OSError, ValueError) as exc:
            raise RawSignalDataError(f"cannot read raw signal {path}: {exc}") from exc
        if normalization == "none":
            return values
        std = values.std(axis=0)
        if np.any(std <= 0.0):
            raise RawSignalDataError(f"raw signal has a zero-variance channel: {path}")
        return (values - values.mean(axis=0)) / std

    def _signal_path(self, bearing_id: int, time_step: int) -> Path:
        if int(bearing_id) not in cfg.BEARING_IDS or int(time_step) < 0:
            raise ValueError("bearing_id and time_step are outside the XJTU protocol")
        bearing_path = resolve_bearing_directory(cfg.DATA_DIR, self.condition, int(bearing_id))
        return bearing_path / f"{int(time_step) + 1}.csv"

    def load_bearing_signals(
        self, bearing_id: int, sample_indices: Sequence[int] | np.ndarray
    ) -> list[torch.Tensor]:
        indices = np.asarray(sample_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("sample_indices must be a one-dimensional integer array")
        signals = []
        for time_step in indices.tolist():
            path = self._signal_path(int(bearing_id), int(time_step))
            if not path.is_file():
                raise RawSignalDataError(f"raw signal is missing: {path}")
            signals.append(
                torch.from_numpy(self._read_csv(path, normalization=self.normalization))
            )
        return signals

    def load_bearing_signals_from_dataframe(self, data: pd.DataFrame) -> list[torch.Tensor]:
        required = {"bearing_id", "time_step"}
        if missing := required - set(data.columns):
            raise ValueError(f"raw signal loading needs columns: {sorted(missing)}")
        values = data.loc[:, ["bearing_id", "time_step"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.array_equal(values, np.floor(values)):
            raise ValueError("bearing_id and time_step must be finite integer values")
        if data.duplicated(["bearing_id", "time_step"]).any():
            raise ValueError("raw signal rows must have unique bearing_id/time_step pairs")
        signals: list[torch.Tensor] = []
        for bearing_id, time_step in values.astype(int):
            signals.extend(self.load_bearing_signals(bearing_id, np.array([time_step])))
        return signals
