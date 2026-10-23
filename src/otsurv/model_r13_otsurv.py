"""End-to-end R13 self-reference OT survival model.

R13 keeps the useful causal feature construction from the development runner,
but evaluates both survival branches in one PyTorch graph:

* a 14-dimensional causal amplitude-compression branch;
* a per-channel waveform-shape branch whose self-reference Sinkhorn distances
  are compressed into the 21-dimensional R13 trajectory; and
* one additive log-risk trained with a complete-risk-set Cox objective.

The train-only calibration statistics are registered buffers.  They are fitted
by :class:`tools.regressors.R13OTSurvRegressor` and therefore remain fixed for
validation and outer-test bearings.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .sinkhornknopp import SemiCurrSinkhornKnopp_stable


_CHECKPOINT_ACCEPTS_USE_REENTRANT = (
    "use_reentrant" in inspect.signature(checkpoint).parameters
)


class R13OTSurv(nn.Module):
    """Causal self-reference Sinkhorn survival model used by R13.

    The model consumes complete, time-stamped bearing trajectories.  Rows may
    arrive in any order; all causal quantities are built after sorting each
    bearing by ``time_step`` and scattered back to the caller's row order.
    """

    CHANNELS = 2
    AMPLITUDE_DIM = 14
    TOKEN_DIM_PER_CHANNEL = 9
    TOKEN_DIM = CHANNELS * TOKEN_DIM_PER_CHANNEL
    TRAJECTORY_DIM = 21
    HISTORY_LAGS = (1, 4, 12)
    DEFAULT_SPECTRAL_BAND_EDGES = (1, 8, 32, 96, 257)

    def __init__(
        self,
        *,
        window_length: int = 512,
        reference_length: int = 8,
        transport_batch_size: int = 4,
        ot_epsilon: float = 0.5,
        ot_num_iters: int = 500,
        ot_stoperr: float = 1e-10,
        shape_projection_trainable: bool = True,
        sinkhorn_checkpoint: bool = True,
        spectral_band_edges: tuple[int, ...] = DEFAULT_SPECTRAL_BAND_EDGES,
    ) -> None:
        super().__init__()
        if isinstance(window_length, bool) or int(window_length) != window_length:
            raise ValueError("R13 window_length must be an integer")
        if int(window_length) < 4 or int(window_length) % 2:
            raise ValueError("R13 window_length must be an even integer of at least 4")
        if isinstance(reference_length, bool) or int(reference_length) != reference_length:
            raise ValueError("R13 reference_length must be an integer")
        if int(reference_length) < 1:
            raise ValueError("R13 reference_length must be positive")
        if isinstance(transport_batch_size, bool) or int(transport_batch_size) != transport_batch_size:
            raise ValueError("R13 transport_batch_size must be an integer")
        if int(transport_batch_size) < 1:
            raise ValueError("R13 transport_batch_size must be positive")
        if not math.isfinite(float(ot_epsilon)) or float(ot_epsilon) <= 0.0:
            raise ValueError("R13 ot_epsilon must be finite and positive")
        if isinstance(ot_num_iters, bool) or int(ot_num_iters) != ot_num_iters:
            raise ValueError("R13 ot_num_iters must be an integer")
        if int(ot_num_iters) < 1:
            raise ValueError("R13 ot_num_iters must be positive")
        if not math.isfinite(float(ot_stoperr)) or float(ot_stoperr) < 0.0:
            raise ValueError("R13 ot_stoperr must be finite and non-negative")

        edges = tuple(int(value) for value in spectral_band_edges)
        if len(edges) < 2 or edges[0] != 1 or edges[-1] != int(window_length) // 2 + 1:
            raise ValueError(
                "R13 spectral bands must start at 1 and cover the complete rFFT support"
            )
        if any(right <= left for left, right in zip(edges[:-1], edges[1:])):
            raise ValueError("R13 spectral band edges must be strictly increasing")
        token_width = 3 + len(edges) - 1 + 2
        if token_width != self.TOKEN_DIM_PER_CHANNEL:
            raise ValueError("R13 token layout must retain nine features per channel")

        self.mode = "r13"
        self.window_length = int(window_length)
        self.reference_length = int(reference_length)
        self.transport_batch_size = int(transport_batch_size)
        self.sinkhorn_checkpoint = bool(sinkhorn_checkpoint)
        self.spectral_band_edges = edges

        # Cox log-risk has no identifiable intercept; Breslow estimates the
        # baseline hazard after fitting, so both branch heads are bias-free.
        self.amplitude_head = nn.Linear(self.AMPLITUDE_DIM, 1, bias=False)
        self.ot_head = nn.Linear(self.TRAJECTORY_DIM, 1, bias=False)
        # The legacy conditional model starts from its fitted amplitude risk
        # with a zero OT residual.  Preserve that stable starting point here;
        # the OT head and the shape projection are still jointly trainable.
        nn.init.zeros_(self.ot_head.weight)
        # The identity initialization preserves the R13 token geometry at the
        # first step while letting the Sinkhorn path adapt jointly with Cox.
        self.shape_token_projection = nn.Linear(
            self.TOKEN_DIM_PER_CHANNEL,
            self.TOKEN_DIM_PER_CHANNEL,
            bias=False,
        )
        with torch.no_grad():
            self.shape_token_projection.weight.copy_(
                torch.eye(self.TOKEN_DIM_PER_CHANNEL)
            )
        self.shape_token_projection.requires_grad_(bool(shape_projection_trainable))

        self.sinkhorn = SemiCurrSinkhornKnopp_stable(
            num_iters=int(ot_num_iters),
            numItermax=int(ot_num_iters),
            epsilon=float(ot_epsilon),
            gamma=0.0,
            stoperr=float(ot_stoperr),
            rho=1.0,
            semi_use=False,
            use_dummy=False,
        )

        self.register_buffer("amplitude_center", torch.zeros(self.AMPLITUDE_DIM))
        self.register_buffer("amplitude_scale", torch.ones(self.AMPLITUDE_DIM))
        self.register_buffer("token_center", torch.zeros(self.TOKEN_DIM))
        self.register_buffer("token_scale", torch.ones(self.TOKEN_DIM))
        self.register_buffer("trajectory_center", torch.zeros(self.TRAJECTORY_DIM))
        self.register_buffer("trajectory_scale", torch.ones(self.TRAJECTORY_DIM))
        self.register_buffer(
            "trajectory_keep", torch.ones(self.TRAJECTORY_DIM, dtype=torch.bool)
        )
        # Optional train-only residualization of the standardized trajectory
        # against the initialized amplitude risk.  It is the differentiable
        # deployment counterpart of R13's old orthogonal-to-main OT view.
        self.register_buffer(
            "trajectory_main_projection", torch.zeros(2, self.TRAJECTORY_DIM)
        )
        self.register_buffer(
            "trajectory_residual_center", torch.zeros(self.TRAJECTORY_DIM)
        )
        self.register_buffer(
            "trajectory_residual_scale", torch.ones(self.TRAJECTORY_DIM)
        )
        self.register_buffer(
            "trajectory_residual_keep",
            torch.ones(self.TRAJECTORY_DIM, dtype=torch.bool),
        )
        self.register_buffer("amplitude_calibration_ready", torch.tensor(False))
        self.register_buffer("token_calibration_ready", torch.tensor(False))
        self.register_buffer("trajectory_calibration_ready", torch.tensor(False))
        self.register_buffer(
            "trajectory_main_residualization_ready", torch.tensor(False)
        )

    @staticmethod
    def _as_signal_batch(raw_signals) -> torch.Tensor:
        if torch.is_tensor(raw_signals):
            signals = raw_signals
        elif isinstance(raw_signals, Sequence) and raw_signals:
            if not all(torch.is_tensor(signal) for signal in raw_signals):
                raise ValueError("R13 raw signals must be tensors")
            signals = torch.stack(tuple(raw_signals), dim=0)
        else:
            raise ValueError("R13 requires a non-empty raw-signal batch")
        if (
            signals.ndim != 3
            or signals.shape[0] < 1
            or signals.shape[2] != R13OTSurv.CHANNELS
            or not torch.is_floating_point(signals)
            or not torch.isfinite(signals).all()
        ):
            raise ValueError("R13 raw signals must be finite [samples, length, 2] tensors")
        return signals

    @staticmethod
    def _as_metadata(value, *, name: str, n_samples: int, device: torch.device) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device)
        if tensor.ndim != 1 or tensor.numel() != n_samples:
            raise ValueError(f"R13 {name} must align with every raw signal")
        if not torch.isfinite(tensor.to(dtype=torch.float64)).all():
            raise ValueError(f"R13 {name} must be finite")
        numeric = tensor.to(dtype=torch.float64)
        if (numeric != torch.floor(numeric)).any():
            raise ValueError(f"R13 {name} must contain integer values")
        return numeric.to(dtype=torch.long)

    def _validated_metadata(self, bearing_ids, time_steps, n_samples, device):
        bearings = self._as_metadata(
            bearing_ids,
            name="bearing_id",
            n_samples=n_samples,
            device=device,
        )
        steps = self._as_metadata(
            time_steps,
            name="time_step",
            n_samples=n_samples,
            device=device,
        )
        if (steps < 0).any():
            raise ValueError("R13 time_step must be non-negative")
        keys = torch.stack((bearings, steps), dim=1)
        if torch.unique(keys, dim=0).shape[0] != n_samples:
            raise ValueError("R13 requires unique bearing_id/time_step pairs")
        return bearings, steps

    @staticmethod
    def _ordered_positions(bearing_ids: torch.Tensor, time_steps: torch.Tensor):
        groups = []
        for bearing_id in torch.unique(bearing_ids, sorted=True):
            positions = torch.nonzero(bearing_ids == bearing_id, as_tuple=False).flatten()
            order = torch.argsort(time_steps.index_select(0, positions))
            ordered = positions.index_select(0, order)
            ordered_steps = time_steps.index_select(0, ordered)
            if ordered_steps.numel() > 1 and (ordered_steps[1:] <= ordered_steps[:-1]).any():
                raise ValueError("R13 requires increasing time_step values within each bearing")
            groups.append((ordered, ordered_steps))
        return groups

    @classmethod
    def _causal_history(
        cls,
        values: torch.Tensor,
        bearing_ids: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError("R13 causal history requires [samples, features]")
        blocks_by_row = values.new_zeros((values.shape[0], values.shape[1] * 6))
        for positions, ordered_steps in cls._ordered_positions(bearing_ids, time_steps):
            ordered = values.index_select(0, positions)
            displacement = (ordered - ordered[0]).abs()
            running_max = torch.cummax(displacement, dim=0).values
            path_length = torch.zeros_like(ordered)
            if ordered.shape[0] > 1:
                increments = (ordered[1:] - ordered[:-1]).abs()
                cumulative = torch.cumsum(increments, dim=0)
                elapsed = (ordered_steps[1:] - ordered_steps[0]).to(dtype=ordered.dtype)
                path_length[1:] = cumulative / elapsed.unsqueeze(1)
            slope_blocks = []
            local_indices = torch.arange(
                ordered.shape[0], device=ordered.device, dtype=torch.long
            )
            for lag in cls.HISTORY_LAGS:
                earlier = (local_indices - int(lag)).clamp_min(0)
                delta_t = (ordered_steps - ordered_steps.index_select(0, earlier)).to(
                    dtype=ordered.dtype
                )
                safe_delta_t = delta_t.clamp_min(1.0)
                slopes = (ordered - ordered.index_select(0, earlier)).abs() / safe_delta_t.unsqueeze(1)
                slopes = torch.where(
                    (local_indices == 0).unsqueeze(1),
                    torch.zeros_like(slopes),
                    slopes,
                )
                slope_blocks.append(slopes)
            group_history = torch.cat(
                (displacement, running_max, path_length, *slope_blocks), dim=1
            )
            blocks_by_row = blocks_by_row.index_copy(0, positions, group_history)
        return blocks_by_row

    def amplitude_features(
        self, signals: torch.Tensor, bearing_ids: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        signal_power = signals.square().mean(dim=1)
        if (signal_power <= 0.0).any() or not torch.isfinite(signal_power).all():
            raise ValueError("R13 amplitude features require positive finite signal power")
        log_rms = 0.5 * signal_power.log()
        history = self._causal_history(log_rms, bearing_ids, time_steps)
        features = torch.cat((log_rms, history), dim=1)
        if features.shape[1] != self.AMPLITUDE_DIM or not torch.isfinite(features).all():
            raise RuntimeError("R13 amplitude feature construction failed")
        return features

    def raw_shape_tokens(self, signals: torch.Tensor) -> torch.Tensor:
        if signals.shape[1] % self.window_length:
            raise ValueError("R13 signals must tile exactly into non-overlapping windows")
        n_samples, signal_length, _ = signals.shape
        n_windows = signal_length // self.window_length
        if n_windows < 1:
            raise ValueError("R13 requires at least one waveform window per minute")
        windows = signals.reshape(
            n_samples, n_windows, self.window_length, self.CHANNELS
        )
        minute_rms = signals.square().mean(dim=1).sqrt()
        if (minute_rms <= 0.0).any() or not torch.isfinite(minute_rms).all():
            raise ValueError("R13 shape tokens require positive finite minute RMS")
        unit_windows = windows / minute_rms[:, None, None, :]
        centered = unit_windows - unit_windows.mean(dim=2, keepdim=True)
        local_rms = centered.square().mean(dim=2).sqrt().clamp_min(
            torch.finfo(centered.dtype).tiny
        )
        relative_log_rms = local_rms.log()

        spectrum = torch.fft.rfft(centered, dim=2)
        power = spectrum.abs().square()
        positive_power = power[:, :, 1:, :]
        power_sum = positive_power.sum(dim=2, keepdim=True)
        if (power_sum <= 0.0).any() or not torch.isfinite(power_sum).all():
            raise ValueError("R13 shape-token PSD has non-positive energy")
        normalized_psd = positive_power / power_sum
        frequencies = torch.arange(
            1,
            self.window_length // 2 + 1,
            dtype=centered.dtype,
            device=centered.device,
        ).view(1, 1, -1, 1)
        centroid = (normalized_psd * frequencies).sum(dim=2) / float(
            self.window_length // 2
        )
        entropy = -(
            normalized_psd
            * normalized_psd.clamp_min(torch.finfo(centered.dtype).tiny).log()
        ).sum(dim=2) / math.log(float(self.window_length // 2))
        bands = [
            normalized_psd[:, :, left - 1 : right - 1, :].sum(dim=2)
            for left, right in zip(self.spectral_band_edges[:-1], self.spectral_band_edges[1:])
        ]
        crest = centered.abs().amax(dim=2) / local_rms
        excess_kurtosis = centered.pow(4).mean(dim=2) / local_rms.pow(4) - 3.0
        per_channel = torch.stack(
            (
                relative_log_rms,
                centroid,
                entropy,
                *bands,
                crest,
                excess_kurtosis,
            ),
            dim=-1,
        )
        tokens = per_channel.reshape(n_samples, n_windows, self.TOKEN_DIM)
        if not torch.isfinite(tokens).all():
            raise RuntimeError("R13 shape-token construction produced non-finite values")
        return tokens

    def _project_shape_tokens(self, raw_tokens: torch.Tensor) -> torch.Tensor:
        if not bool(self.token_calibration_ready.item()):
            raise RuntimeError("R13 token calibration must be fitted before forward")
        calibrated = (
            raw_tokens - self.token_center.view(1, 1, -1)
        ) / self.token_scale.view(1, 1, -1)
        calibrated = calibrated.clamp(min=-6.0, max=6.0)
        per_channel = calibrated.reshape(
            calibrated.shape[0],
            calibrated.shape[1],
            self.CHANNELS,
            self.TOKEN_DIM_PER_CHANNEL,
        )
        projected = self.shape_token_projection(per_channel)
        result = projected.reshape(
            calibrated.shape[0], calibrated.shape[1], self.TOKEN_DIM
        )
        if not torch.isfinite(result).all():
            raise RuntimeError("R13 token projection produced non-finite values")
        return result

    def _causal_self_reference(
        self,
        tokens: torch.Tensor,
        bearing_ids: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> torch.Tensor:
        references = torch.zeros_like(tokens)
        for positions, _ in self._ordered_positions(bearing_ids, time_steps):
            ordered = tokens.index_select(0, positions)
            prefix = torch.cumsum(ordered[: self.reference_length], dim=0)
            local_positions = torch.arange(
                ordered.shape[0], device=ordered.device, dtype=torch.long
            )
            prefix_index = local_positions.clamp(max=self.reference_length - 1)
            counts = (prefix_index + 1).to(dtype=ordered.dtype).view(-1, 1, 1)
            group_references = prefix.index_select(0, prefix_index) / counts
            references = references.index_copy(0, positions, group_references)
        if not torch.isfinite(references).all():
            raise RuntimeError("R13 causal self-reference construction failed")
        return references

    def _regularized_ot_objective(
        self, transport: torch.Tensor, cost: torch.Tensor
    ) -> torch.Tensor:
        if transport.shape != cost.shape or transport.ndim != 3:
            raise ValueError("R13 OT objective requires matching [batch, windows, windows]")
        width = transport.shape[1]
        if width < 1 or transport.shape[2] != width:
            raise ValueError("R13 self-reference OT requires square transport matrices")
        uniform_pair_mass = 1.0 / float(width * width)
        entropy = (
            transport
            * (
                transport.clamp_min(torch.finfo(transport.dtype).tiny).log()
                - math.log(uniform_pair_mass)
            )
        ).sum(dim=(1, 2))
        return (transport * cost).sum(dim=(1, 2)) + self.sinkhorn.epsilon * entropy

    def _sinkhorn_values_only(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        values, _, _ = self._sinkhorn_values(source, target)
        return values

    def _sinkhorn_values(
        self, source: torch.Tensor, target: torch.Tensor):
        feature_width = source.shape[-1]
        channel_width = self.TOKEN_DIM_PER_CHANNEL
        scale = float(feature_width**0.5)
        channel_scale = float(channel_width**0.5)
        cross_cost = torch.cdist(source, target) / scale
        source_cost = torch.cdist(source, source) / scale
        target_cost = torch.cdist(target, target) / scale
        cross_plan, cross_diag = self.sinkhorn.cost_forward_batched(
            cross_cost, final=False, return_diagnostics=True
        )
        source_plan, source_diag = self.sinkhorn.cost_forward_batched(
            source_cost, final=False, return_diagnostics=True
        )
        target_plan, target_diag = self.sinkhorn.cost_forward_batched(
            target_cost, final=False, return_diagnostics=True
        )
        overall = self._regularized_ot_objective(cross_plan, cross_cost) - 0.5 * (
            self._regularized_ot_objective(source_plan, source_cost)
            + self._regularized_ot_objective(target_plan, target_cost)
        )
        delta = source.unsqueeze(2) - target.unsqueeze(1)
        source_delta = source.unsqueeze(2) - source.unsqueeze(1)
        target_delta = target.unsqueeze(2) - target.unsqueeze(1)
        channel_values = []
        for channel in range(self.CHANNELS):
            left = channel * channel_width
            right = left + channel_width
            cross_channel_cost = torch.linalg.vector_norm(
                delta[..., left:right], dim=-1
            ) / channel_scale
            source_channel_cost = torch.linalg.vector_norm(
                source_delta[..., left:right], dim=-1
            ) / channel_scale
            target_channel_cost = torch.linalg.vector_norm(
                target_delta[..., left:right], dim=-1
            ) / channel_scale
            channel_values.append(
                (cross_plan * cross_channel_cost).sum(dim=(1, 2))
                - 0.5
                * (
                    (source_plan * source_channel_cost).sum(dim=(1, 2))
                    + (target_plan * target_channel_cost).sum(dim=(1, 2))
                )
            )
        values = torch.stack((overall, *channel_values), dim=1)
        hard_error = torch.stack(
            (
                cross_diag["hard_marginal_error"],
                source_diag["hard_marginal_error"],
                target_diag["hard_marginal_error"],
            ),
            dim=1,
        ).amax(dim=1)
        converged = (
            cross_diag["converged"]
            & source_diag["converged"]
            & target_diag["converged"]
        )
        if not torch.isfinite(values).all():
            raise RuntimeError("R13 Sinkhorn distances are non-finite")
        return values, hard_error, converged

    def _self_reference_distances(self, tokens: torch.Tensor, references: torch.Tensor):
        value_parts = []
        hard_error_parts = []
        convergence_parts = []
        for start in range(0, tokens.shape[0], self.transport_batch_size):
            stop = min(start + self.transport_batch_size, tokens.shape[0])
            source = tokens[start:stop]
            target = references[start:stop]
            use_checkpoint = (
                self.sinkhorn_checkpoint
                and self.training
                and torch.is_grad_enabled()
                and (source.requires_grad or target.requires_grad)
            )
            if use_checkpoint:
                checkpoint_kwargs = (
                    {"use_reentrant": False}
                    if _CHECKPOINT_ACCEPTS_USE_REENTRANT
                    else {}
                )
                values = checkpoint(
                    self._sinkhorn_values_only,
                    source,
                    target,
                    **checkpoint_kwargs,
                )
                hard_error = None
                converged = None
            else:
                values, hard_error, converged = self._sinkhorn_values(source, target)
            value_parts.append(values)
            if hard_error is not None:
                hard_error_parts.append(hard_error)
            if converged is not None:
                convergence_parts.append(converged)
        distances = torch.cat(value_parts, dim=0)
        diagnostics = {
            "hard_marginal_error": (
                torch.cat(hard_error_parts, dim=0)
                if hard_error_parts
                else None
            ),
            "converged": (
                torch.cat(convergence_parts, dim=0)
                if convergence_parts
                else None
            ),
        }
        return distances, diagnostics

    def _trajectory_features(
        self,
        distances: torch.Tensor,
        bearing_ids: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> torch.Tensor:
        history = self._causal_history(distances, bearing_ids, time_steps)
        features = torch.cat((distances, history), dim=1)
        if features.shape[1] != self.TRAJECTORY_DIM or not torch.isfinite(features).all():
            raise RuntimeError("R13 trajectory feature construction failed")
        return features

    def set_amplitude_calibration(self, center, scale) -> None:
        center = torch.as_tensor(center, dtype=self.amplitude_center.dtype, device=self.amplitude_center.device)
        scale = torch.as_tensor(scale, dtype=self.amplitude_scale.dtype, device=self.amplitude_scale.device)
        if center.shape != (self.AMPLITUDE_DIM,) or scale.shape != (self.AMPLITUDE_DIM,):
            raise ValueError("R13 amplitude calibration must have shape [14]")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale <= 0.0).any():
            raise ValueError("R13 amplitude calibration must be finite and positive")
        self.amplitude_center.copy_(center)
        self.amplitude_scale.copy_(scale)
        self.amplitude_calibration_ready.fill_(True)

    def set_token_calibration(self, center, scale) -> None:
        center = torch.as_tensor(center, dtype=self.token_center.dtype, device=self.token_center.device)
        scale = torch.as_tensor(scale, dtype=self.token_scale.dtype, device=self.token_scale.device)
        if center.shape != (self.TOKEN_DIM,) or scale.shape != (self.TOKEN_DIM,):
            raise ValueError("R13 token calibration must have shape [18]")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale <= 0.0).any():
            raise ValueError("R13 token calibration must be finite and positive")
        self.token_center.copy_(center)
        self.token_scale.copy_(scale)
        self.token_calibration_ready.fill_(True)

    def set_trajectory_calibration(self, center, scale, keep=None) -> None:
        center = torch.as_tensor(center, dtype=self.trajectory_center.dtype, device=self.trajectory_center.device)
        scale = torch.as_tensor(scale, dtype=self.trajectory_scale.dtype, device=self.trajectory_scale.device)
        if center.shape != (self.TRAJECTORY_DIM,) or scale.shape != (self.TRAJECTORY_DIM,):
            raise ValueError("R13 trajectory calibration must have shape [21]")
        if keep is None:
            keep = torch.ones(self.TRAJECTORY_DIM, dtype=torch.bool, device=self.trajectory_keep.device)
        else:
            keep = torch.as_tensor(keep, dtype=torch.bool, device=self.trajectory_keep.device)
        if keep.shape != (self.TRAJECTORY_DIM,) or not bool(keep.any()):
            raise ValueError("R13 trajectory calibration requires one retained feature")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale <= 0.0).any():
            raise ValueError("R13 trajectory calibration must be finite and positive")
        self.trajectory_center.copy_(center)
        self.trajectory_scale.copy_(scale)
        self.trajectory_keep.copy_(keep)
        self.trajectory_calibration_ready.fill_(True)

    def set_trajectory_main_residualization(
        self, projection, center, scale, keep
    ) -> None:
        """Set the train-only trajectory residual transform.

        ``projection`` maps ``[1, amplitude_log_risk]`` onto the initially
        standardized trajectory.  The second calibration restores stable
        feature scale after the projection is removed.
        """

        projection = torch.as_tensor(
            projection,
            dtype=self.trajectory_main_projection.dtype,
            device=self.trajectory_main_projection.device,
        )
        center = torch.as_tensor(
            center,
            dtype=self.trajectory_residual_center.dtype,
            device=self.trajectory_residual_center.device,
        )
        scale = torch.as_tensor(
            scale,
            dtype=self.trajectory_residual_scale.dtype,
            device=self.trajectory_residual_scale.device,
        )
        keep = torch.as_tensor(
            keep,
            dtype=torch.bool,
            device=self.trajectory_residual_keep.device,
        )
        if projection.shape != (2, self.TRAJECTORY_DIM):
            raise ValueError("R13 trajectory-main projection must have shape [2, 21]")
        if (
            center.shape != (self.TRAJECTORY_DIM,)
            or scale.shape != (self.TRAJECTORY_DIM,)
            or keep.shape != (self.TRAJECTORY_DIM,)
        ):
            raise ValueError("R13 trajectory residual calibration must have shape [21]")
        if (
            not torch.isfinite(projection).all()
            or not torch.isfinite(center).all()
            or not torch.isfinite(scale).all()
            or (scale <= 0.0).any()
            or not bool(keep.any())
        ):
            raise ValueError("R13 trajectory residual calibration must be finite")
        self.trajectory_main_projection.copy_(projection)
        self.trajectory_residual_center.copy_(center)
        self.trajectory_residual_scale.copy_(scale)
        self.trajectory_residual_keep.copy_(keep)
        self.trajectory_main_residualization_ready.fill_(True)

    def clear_trajectory_main_residualization(self) -> None:
        """Disable a previously fitted trajectory-main residual transform."""

        self.trajectory_main_projection.zero_()
        self.trajectory_residual_center.zero_()
        self.trajectory_residual_scale.fill_(1.0)
        self.trajectory_residual_keep.fill_(True)
        self.trajectory_main_residualization_ready.fill_(False)

    def calibration_ready(self) -> bool:
        return bool(
            self.amplitude_calibration_ready.item()
            and self.token_calibration_ready.item()
            and self.trajectory_calibration_ready.item()
        )

    def forward(
        self,
        raw_signals,
        bearing_ids,
        time_steps,
        *,
        return_components: bool = False,
        apply_trajectory_calibration: bool = True,
    ):
        signals = self._as_signal_batch(raw_signals).to(
            device=self.amplitude_head.weight.device,
            dtype=self.amplitude_head.weight.dtype
        )
        bearing_ids, time_steps = self._validated_metadata(
            bearing_ids,
            time_steps,
            signals.shape[0],
            signals.device,
        )
        if apply_trajectory_calibration and not self.calibration_ready():
            raise RuntimeError("R13 train-only calibrations must be fitted before forward")
        if not bool(self.amplitude_calibration_ready.item()) or not bool(self.token_calibration_ready.item()):
            raise RuntimeError("R13 amplitude and token calibrations must be fitted before forward")

        amplitude_raw = self.amplitude_features(signals, bearing_ids, time_steps)
        amplitude = (
            amplitude_raw - self.amplitude_center.view(1, -1)
        ) / self.amplitude_scale.view(1, -1)
        amplitude_log_risk = self.amplitude_head(amplitude)

        raw_tokens = self.raw_shape_tokens(signals)
        tokens = self._project_shape_tokens(raw_tokens)
        references = self._causal_self_reference(tokens, bearing_ids, time_steps)
        distances, sinkhorn_diagnostics = self._self_reference_distances(tokens, references)
        trajectory_raw = self._trajectory_features(distances, bearing_ids, time_steps)
        if apply_trajectory_calibration:
            trajectory = (
                trajectory_raw - self.trajectory_center.view(1, -1)
            ) / self.trajectory_scale.view(1, -1)
            trajectory = torch.where(
                self.trajectory_keep.view(1, -1),
                trajectory,
                torch.zeros_like(trajectory),
            )
            if bool(self.trajectory_main_residualization_ready.item()):
                # Keep the projection reference fixed for the OT path.  The
                # main branch still receives the complete Cox gradient via
                # ``logits`` below, while OT cannot rewrite it through this
                # train-only residualization transform.
                main_reference = amplitude_log_risk.detach()
                design = torch.cat((torch.ones_like(main_reference), main_reference), dim=1)
                trajectory = trajectory - design @ self.trajectory_main_projection
                trajectory = (
                    trajectory - self.trajectory_residual_center.view(1, -1)
                ) / self.trajectory_residual_scale.view(1, -1)
                trajectory = torch.where(
                    self.trajectory_residual_keep.view(1, -1),
                    trajectory,
                    torch.zeros_like(trajectory),
                )
        else:
            trajectory = trajectory_raw
        ot_log_risk = self.ot_head(trajectory)
        logits = amplitude_log_risk + ot_log_risk
        if not torch.isfinite(logits).all():
            raise RuntimeError("R13 produced non-finite log risks")

        output = {"logits": logits}
        if return_components:
            output.update(
                {
                    "amplitude_log_risk": amplitude_log_risk,
                    "ot_log_risk": ot_log_risk,
                    "amplitude_features": amplitude,
                    "raw_shape_tokens": raw_tokens,
                    "ot_distances": distances,
                    "ot_trajectory": trajectory,
                    "ot_trajectory_raw": trajectory_raw,
                    "sinkhorn_hard_marginal_error": sinkhorn_diagnostics[
                        "hard_marginal_error"
                    ],
                    "sinkhorn_converged": sinkhorn_diagnostics["converged"],
                }
            )
        return output
