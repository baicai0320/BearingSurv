"""Pure raw-waveform CNN Cox model used as the R13 input-matched baseline.

This module intentionally contains no R13 or legacy OTSurv components.  A
minute of dual-channel vibration is tiled into non-overlapping waveform
windows, encoded independently, mean pooled, and mapped directly to one Cox
log-risk score.
"""

from __future__ import annotations

import torch
from torch import nn


def _group_count(channels: int) -> int:
    """Choose a small GroupNorm group count that divides *channels*."""
    groups = min(8, int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


class RawCNNCox(nn.Module):
    """Raw dual-channel CNN followed by mean pooling and a Cox risk head.

    The production contract fixes a minute to ``[32768, 2]`` and therefore
    creates 64 non-overlapping windows of 512 samples.  ``signal_length`` is
    an explicit constructor argument only so focused unit tests can exercise
    the tensor contract with a smaller, exactly tiled signal; the production
    regressor rejects any value other than 32768.
    """

    CHANNELS = 2
    PRODUCTION_SIGNAL_LENGTH = 32768
    PRODUCTION_WINDOW_LENGTH = 512
    PRODUCTION_WINDOW_COUNT = 64

    def __init__(
        self,
        *,
        signal_length: int = PRODUCTION_SIGNAL_LENGTH,
        window_length: int = PRODUCTION_WINDOW_LENGTH,
        hidden_dim: int = 64,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        signal_length = int(signal_length)
        window_length = int(window_length)
        hidden_dim = int(hidden_dim)
        base_channels = int(base_channels)
        if signal_length < 1 or window_length < 1:
            raise ValueError("signal_length and window_length must be positive")
        if signal_length % window_length:
            raise ValueError("signal_length must tile exactly into CNN windows")
        if hidden_dim < 1 or base_channels < 1:
            raise ValueError("hidden_dim and base_channels must be positive")

        self.signal_length = signal_length
        self.window_length = window_length
        self.window_count = signal_length // window_length
        self.hidden_dim = hidden_dim
        self.base_channels = base_channels

        # GroupNorm has no running state, so micro-batch replay gives the same
        # per-sample logits used by the complete-risk-set Cox objective.
        self.window_encoder = nn.Sequential(
            nn.Conv1d(self.CHANNELS, base_channels, kernel_size=9, padding=4, bias=False),
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(_group_count(base_channels * 2), base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels * 2, hidden_dim, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.window_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Cox's intercept is not identifiable; Breslow estimates the baseline.
        self.risk_head = nn.Linear(hidden_dim, 1, bias=False)

    def _validate_signals(self, signals: torch.Tensor) -> None:
        if not torch.is_tensor(signals):
            raise TypeError("RawCNNCox signals must be a torch.Tensor")
        if signals.ndim != 3 or signals.shape[2] != self.CHANNELS:
            raise ValueError(
                "RawCNNCox expects raw signals with shape [batch, samples, 2]"
            )
        if signals.shape[1] != self.signal_length:
            raise ValueError(
                f"RawCNNCox expects {self.signal_length} samples per minute, "
                f"got {signals.shape[1]}"
            )
        if signals.shape[0] < 1 or not torch.isfinite(signals).all():
            raise ValueError("RawCNNCox signals must be non-empty and finite")

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        """Return one finite log-risk score per raw vibration minute."""
        self._validate_signals(signals)
        batch_size = int(signals.shape[0])
        windows = signals.reshape(
            batch_size,
            self.window_count,
            self.window_length,
            self.CHANNELS,
        )
        # Conv1d consumes [windows, channels, samples].
        window_batch = windows.reshape(
            batch_size * self.window_count,
            self.window_length,
            self.CHANNELS,
        ).permute(0, 2, 1)
        encoded = self.window_encoder(window_batch).flatten(start_dim=1)
        embeddings = self.window_projection(encoded).reshape(
            batch_size, self.window_count, self.hidden_dim
        )
        pooled = embeddings.mean(dim=1)
        logits = self.risk_head(pooled)
        if logits.shape != (batch_size, 1) or not torch.isfinite(logits).all():
            raise RuntimeError("RawCNNCox produced invalid Cox log-risk scores")
        return logits
