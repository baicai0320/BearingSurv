"""R14-B C2 hybrid raw-CNN / calibrated-trajectory AFT model.

The model deliberately has one location parameter for both discrimination and
survival curves.  Raw physical-scale waveform minutes, the calibrated R13
amplitude features, and the calibrated R13 OT trajectory are fused before a
small hand-written causal state recurrence.  There is no second Cox branch:
``risk`` is always ``-mu`` and the native curve is the log-normal AFT curve
formed from the same ``mu`` and global ``sigma``.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .losses import (
    fpt_smooth_l1_loss,
    lognormal_aft_loss,
    r13_teacher_pairwise_rank_loss,
)
from .model_r13_otsurv import R13OTSurv


def _group_count(channels: int) -> int:
    groups = min(8, int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


class _CausalStateCell(nn.Module):
    """GRU-equivalent gated cell built from ordinary linear operations.

    Keeping the recurrence explicit avoids initializing the environment's
    cuDNN RNN backend and makes the causal update straightforward to audit.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_gate = nn.Linear(input_dim, 2 * hidden_dim, bias=True)
        self.hidden_gate = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.input_candidate = nn.Linear(input_dim, hidden_dim, bias=True)
        self.hidden_candidate = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        reset, update = torch.sigmoid(
            self.input_gate(inputs) + self.hidden_gate(hidden)
        ).chunk(2, dim=-1)
        candidate = torch.tanh(
            self.input_candidate(inputs) + reset * self.hidden_candidate(hidden)
        )
        return update * hidden + (1.0 - update) * candidate


class R14BOTSurv(R13OTSurv):
    """C2-only R14-B model with a native log-normal AFT endpoint."""

    MODE = "r14b"
    RAW_CNN_HIDDEN_DIM = 64
    REFERENCE_COUNT = 3
    SIGNED_RATE_LAGS = (1, 4, 12)
    FPT_DELTA_SCALE = 0.75
    SIGMA_MIN = 0.15
    SIGMA_MAX = 1.50
    # A full development cohort contains many 64-window minutes.  Chunking
    # this encoder-only operation keeps the exact downstream risk set intact,
    # while checkpointing prevents Conv1d activations from dominating CUDA RAM.
    # C3's longest four-bearing development cohort needs a smaller replay
    # block to leave headroom for the checkpointed backward recomputation.
    # Chunks are independent under this encoder (GroupNorm, not BatchNorm),
    # so this changes peak memory only, not the full-risk-set objective.
    RAW_CNN_WINDOW_BATCH_SIZE = 8192

    # The source vector is exactly the three requested branches.
    SOURCE_DIM = 14 + 21 + RAW_CNN_HIDDEN_DIM

    def __init__(
        self,
        *,
        signal_length: int = 32768,
        window_length: int = 512,
        reference_lengths: tuple[int, int, int] = (4, 8, 16),
        transport_batch_size: int = 32,
        ot_epsilon: float = 0.5,
        ot_num_iters: int = 500,
        ot_stoperr: float = 1e-10,
        shape_projection_trainable: bool = True,
        sinkhorn_checkpoint: bool = True,
        state_hidden_dim: int = 96,
        sigma_init: float = 0.60,
    ) -> None:
        lengths = tuple(reference_lengths)
        if len(lengths) != self.REFERENCE_COUNT:
            raise ValueError("R14-B requires exactly three reference lengths")
        if any(
            isinstance(value, bool) or int(value) != value or int(value) < 1
            for value in lengths
        ):
            raise ValueError("R14-B reference lengths must be positive integers")
        lengths = tuple(int(value) for value in lengths)
        if any(right <= left for left, right in zip(lengths[:-1], lengths[1:])):
            raise ValueError("R14-B reference lengths must be strictly increasing")
        for name, value in (
            ("signal_length", signal_length),
            ("window_length", window_length),
            ("transport_batch_size", transport_batch_size),
            ("ot_num_iters", ot_num_iters),
        ):
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"R14-B {name} must be an integer")
        if int(window_length) < 4 or int(window_length) % 2:
            raise ValueError("R14-B window_length must be an even integer of at least 4")
        if int(signal_length) < int(window_length):
            raise ValueError("R14-B signal_length must contain at least one window")
        if int(signal_length) % int(window_length):
            raise ValueError("R14-B signal_length must tile exactly into 512-sample windows")
        if (
            isinstance(state_hidden_dim, bool)
            or not isinstance(state_hidden_dim, numbers.Integral)
            or int(state_hidden_dim) != state_hidden_dim
            or int(state_hidden_dim) < 1
        ):
            raise ValueError("R14-B state_hidden_dim must be positive")
        if not math.isfinite(float(sigma_init)) or not (
            self.SIGMA_MIN < float(sigma_init) < self.SIGMA_MAX
        ):
            raise ValueError("R14-B sigma_init must lie strictly inside [0.15, 1.50]")

        # This supplies the audited R13 amplitude/OT feature construction and
        # all train-only calibration buffers.  The inherited Cox heads are
        # removed: R14-B has one location head and one bounded FPT correction.
        super().__init__(
            window_length=int(window_length),
            reference_length=lengths[1],
            transport_batch_size=int(transport_batch_size),
            ot_epsilon=float(ot_epsilon),
            ot_num_iters=int(ot_num_iters),
            ot_stoperr=float(ot_stoperr),
            shape_projection_trainable=bool(shape_projection_trainable),
            sinkhorn_checkpoint=bool(sinkhorn_checkpoint),
        )
        self.mode = self.MODE
        self.signal_length = int(signal_length)
        self.window_length = int(window_length)
        self.window_count = self.signal_length // self.window_length
        self.reference_lengths = lengths
        self.state_hidden_dim = int(state_hidden_dim)
        self.source_dim = int(self.AMPLITUDE_DIM + self.TRAJECTORY_DIM + self.RAW_CNN_HIDDEN_DIM)
        self.state_input_dim = int(self.source_dim * 7 + self.REFERENCE_COUNT)

        # Do not leave an independently usable R13 Cox endpoint in the module.
        self._modules.pop("amplitude_head", None)
        self._modules.pop("ot_head", None)

        self.raw_cnn_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, padding=4, bias=False),
            nn.GroupNorm(_group_count(32), 32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(_group_count(64), 64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, self.RAW_CNN_HIDDEN_DIM, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(_group_count(self.RAW_CNN_HIDDEN_DIM), self.RAW_CNN_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.minute_projection = nn.Linear(
            self.RAW_CNN_HIDDEN_DIM, self.RAW_CNN_HIDDEN_DIM, bias=False
        )
        self.state_encoder = _CausalStateCell(self.state_input_dim, self.state_hidden_dim)
        self.mu_rank_head = nn.Linear(self.state_hidden_dim, 1, bias=False)
        self.delta_fpt_head = nn.Linear(self.state_hidden_dim, 1, bias=False)
        nn.init.zeros_(self.delta_fpt_head.weight)
        self.fpt_head = self.delta_fpt_head
        self.mu_bias = nn.Parameter(torch.zeros(()))

        # Parameterize sigma with a bounded sigmoid.  The inverse transform
        # gives the requested 0.60 starting value without clipping gradients.
        fraction = (float(sigma_init) - self.SIGMA_MIN) / (
            self.SIGMA_MAX - self.SIGMA_MIN
        )
        raw_sigma = math.log(fraction / (1.0 - fraction))
        self.raw_sigma = nn.Parameter(torch.tensor(raw_sigma, dtype=torch.float32))

        self.register_buffer(
            "minute_center", torch.zeros(self.RAW_CNN_HIDDEN_DIM)
        )
        self.register_buffer(
            "minute_scale", torch.ones(self.RAW_CNN_HIDDEN_DIM)
        )
        self.register_buffer("minute_calibration_ready", torch.tensor(False))

    @property
    def minute_encoder(self) -> nn.Module:
        """Compatibility alias exposing the physical-scale CNN encoder."""

        return self.raw_cnn_encoder

    @property
    def sigma(self) -> torch.Tensor:
        return self.SIGMA_MIN + (self.SIGMA_MAX - self.SIGMA_MIN) * torch.sigmoid(
            self.raw_sigma
        )

    def calibration_ready(self) -> bool:
        """Require every train-only source calibration before deployment."""

        return bool(
            super().calibration_ready() and self.minute_calibration_ready.item()
        )

    def set_minute_calibration(self, center, scale) -> None:
        center = torch.as_tensor(center, device=self.minute_center.device, dtype=self.minute_center.dtype)
        scale = torch.as_tensor(scale, device=self.minute_scale.device, dtype=self.minute_scale.dtype)
        expected = (self.RAW_CNN_HIDDEN_DIM,)
        if center.shape != expected or scale.shape != expected:
            raise ValueError("R14-B minute calibration must have shape [64]")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale <= 0.0).any():
            raise ValueError("R14-B minute calibration must be finite and positive")
        self.minute_center.copy_(center)
        self.minute_scale.copy_(scale)
        self.minute_calibration_ready.fill_(True)

    def raw_minute_embeddings(self, signals: torch.Tensor) -> torch.Tensor:
        if signals.ndim != 3 or signals.shape[2] != 2:
            raise ValueError("R14-B raw signals must have shape [N, samples, 2]")
        if signals.shape[1] != self.signal_length:
            raise ValueError(
                f"R14-B expects {self.signal_length} samples per minute, got {signals.shape[1]}"
            )
        windows = signals.reshape(
            signals.shape[0], self.window_count, self.window_length, 2
        )
        window_batch = windows.reshape(
            signals.shape[0] * self.window_count, self.window_length, 2
        ).permute(0, 2, 1)
        encoded_chunks = []
        for start in range(0, window_batch.shape[0], self.RAW_CNN_WINDOW_BATCH_SIZE):
            stop = min(start + self.RAW_CNN_WINDOW_BATCH_SIZE, window_batch.shape[0])
            window_chunk = window_batch[start:stop]
            if self.training and torch.is_grad_enabled():
                # Raw signals are not optimization variables.  Marking the
                # detached chunk as a checkpoint input preserves parameter
                # gradients while avoiding a retained encoder activation graph.
                checkpoint_input = window_chunk.detach().requires_grad_(True)
                encoded_chunk = checkpoint(
                    self.raw_cnn_encoder,
                    checkpoint_input,
                    use_reentrant=True,
                )
            else:
                encoded_chunk = self.raw_cnn_encoder(window_chunk)
            encoded_chunks.append(encoded_chunk.flatten(start_dim=1))
        encoded = torch.cat(encoded_chunks, dim=0)
        projected = self.minute_projection(encoded).reshape(
            signals.shape[0], self.window_count, self.RAW_CNN_HIDDEN_DIM
        )
        embeddings = projected.mean(dim=1)
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("R14-B raw CNN minute embeddings are non-finite")
        return embeddings

    def _minute_features(self, signals: torch.Tensor) -> torch.Tensor:
        raw = self.raw_minute_embeddings(signals)
        if bool(self.minute_calibration_ready.item()):
            raw = (raw - self.minute_center.view(1, -1)) / self.minute_scale.view(1, -1)
            raw = raw.clamp(min=-6.0, max=6.0)
        return raw

    def _causal_references(
        self, source: torch.Tensor, bearing_ids: torch.Tensor, time_steps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        references = source.new_zeros((source.shape[0], self.REFERENCE_COUNT, source.shape[1]))
        maturity = source.new_zeros((source.shape[0], self.REFERENCE_COUNT))
        lengths = torch.as_tensor(self.reference_lengths, device=source.device, dtype=source.dtype)
        for positions, _ in self._ordered_positions(bearing_ids, time_steps):
            ordered = source.index_select(0, positions)
            prefix = torch.cumsum(ordered, dim=0)
            local = torch.arange(ordered.shape[0], device=source.device, dtype=torch.long)
            group_references = source.new_zeros(
                (positions.numel(), self.REFERENCE_COUNT, source.shape[1])
            )
            group_maturity = source.new_zeros(
                (positions.numel(), self.REFERENCE_COUNT)
            )
            for ref_index, length in enumerate(self.reference_lengths):
                prefix_index = local.clamp(max=length - 1)
                count = (prefix_index + 1).to(dtype=source.dtype).unsqueeze(1)
                ref = prefix.index_select(0, prefix_index) / count
                group_references[:, ref_index, :] = ref
                group_maturity[:, ref_index] = torch.minimum(
                    local.to(dtype=source.dtype) + 1.0, lengths[ref_index]
                ) / lengths[ref_index]
            references = references.index_copy(0, positions, group_references)
            maturity = maturity.index_copy(0, positions, group_maturity)
        if not torch.isfinite(references).all() or not torch.isfinite(maturity).all():
            raise RuntimeError("R14-B causal reference construction failed")
        return references, maturity

    def _signed_rates(
        self, source: torch.Tensor, bearing_ids: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        rates = source.new_zeros((source.shape[0], len(self.SIGNED_RATE_LAGS) * source.shape[1]))
        for positions, ordered_steps in self._ordered_positions(bearing_ids, time_steps):
            ordered = source.index_select(0, positions)
            local = torch.arange(ordered.shape[0], device=source.device, dtype=torch.long)
            blocks = []
            for lag in self.SIGNED_RATE_LAGS:
                earlier = (local - int(lag)).clamp_min(0)
                delta_t = (ordered_steps - ordered_steps.index_select(0, earlier)).to(source.dtype)
                block = (ordered - ordered.index_select(0, earlier)) / delta_t.clamp_min(1.0).unsqueeze(1)
                block = torch.where((local == 0).unsqueeze(1), torch.zeros_like(block), block)
                blocks.append(block)
            rates[positions] = torch.cat(blocks, dim=1)
        if not torch.isfinite(rates).all():
            raise RuntimeError("R14-B signed rate construction failed")
        return rates

    def _state_inputs(
        self, source: torch.Tensor, bearing_ids: torch.Tensor, time_steps: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        references, maturity = self._causal_references(source, bearing_ids, time_steps)
        rates = self._signed_rates(source, bearing_ids, time_steps)
        state_inputs = torch.cat((source, references.flatten(1), rates, maturity), dim=1)
        if state_inputs.shape[1] != self.state_input_dim or not torch.isfinite(state_inputs).all():
            raise RuntimeError("R14-B state input construction failed")
        return state_inputs, {
            "causal_references": references,
            "signed_rates": rates,
            "reference_maturity": maturity,
        }

    def _encode_state(
        self, state_inputs: torch.Tensor, bearing_ids: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        state = state_inputs.new_zeros((state_inputs.shape[0], self.state_hidden_dim))
        for positions, _ in self._ordered_positions(bearing_ids, time_steps):
            ordered_inputs = state_inputs.index_select(0, positions)
            hidden = state_inputs.new_zeros((1, self.state_hidden_dim))
            outputs = []
            for row in ordered_inputs:
                hidden = self.state_encoder(row.unsqueeze(0), hidden)
                outputs.append(hidden)
            state = state.index_copy(0, positions, torch.cat(outputs, dim=0))
        if not torch.isfinite(state).all():
            raise RuntimeError("R14-B causal state encoder produced non-finite values")
        return state

    def survival_from_mu(self, mu: torch.Tensor, times) -> torch.Tensor:
        """Evaluate monotone ``S(t)=Phi_bar((log1p(t)-mu)/sigma)``."""

        return self.aft_survival(mu, times)

    def aft_survival(self, mu: torch.Tensor, times) -> torch.Tensor:
        location = mu.reshape(-1, 1)
        grid = torch.as_tensor(times, device=location.device, dtype=location.dtype).reshape(-1)
        if grid.numel() < 1 or not torch.isfinite(grid).all() or (grid < 0.0).any():
            raise ValueError("AFT time grid must be finite and non-negative")
        if torch.unique(grid).numel() != grid.numel():
            raise ValueError("AFT time grid must contain unique time points")
        order = torch.argsort(grid)
        sorted_grid = grid.index_select(0, order)
        z = (torch.log1p(sorted_grid).view(1, -1) - location) / self.sigma
        curves = 0.5 * torch.erfc(z / math.sqrt(2.0))
        curves = torch.cummin(curves, dim=1).values.clamp(0.0, 1.0)
        zero_mask = sorted_grid == 0.0
        if bool(zero_mask.any()):
            curves = torch.where(
                zero_mask.view(1, -1),
                torch.ones_like(curves),
                curves,
            )
        inverse = torch.argsort(order)
        curves = curves.index_select(1, inverse)
        if not torch.isfinite(curves).all():
            raise RuntimeError("R14-B AFT survival curve is non-finite")
        return curves

    def forward(
        self,
        raw_signals,
        bearing_ids,
        time_steps,
        *,
        return_components: bool = False,
        apply_trajectory_calibration: bool = True,
        evaluation_times=None,
    ):
        signals = self._as_signal_batch(raw_signals).to(
            device=self.raw_cnn_encoder[0].weight.device,
            dtype=self.raw_cnn_encoder[0].weight.dtype,
        )
        bearings, steps = self._validated_metadata(
            bearing_ids, time_steps, signals.shape[0], signals.device
        )
        if apply_trajectory_calibration and not self.calibration_ready():
            raise RuntimeError("R14-B train-only calibrations must be fitted before forward")
        if not bool(self.amplitude_calibration_ready.item()) or not bool(self.token_calibration_ready.item()):
            raise RuntimeError("R14-B amplitude and token calibrations must be fitted before forward")

        amplitude_raw = self.amplitude_features(signals, bearings, steps)
        amplitude = (amplitude_raw - self.amplitude_center.view(1, -1)) / self.amplitude_scale.view(1, -1)
        raw_tokens = self.raw_shape_tokens(signals)
        tokens = self._project_shape_tokens(raw_tokens)
        references = self._causal_self_reference(tokens, bearings, steps)
        distances, sinkhorn_diagnostics = self._self_reference_distances(tokens, references)
        trajectory_raw = self._trajectory_features(distances, bearings, steps)
        if apply_trajectory_calibration:
            trajectory = (trajectory_raw - self.trajectory_center.view(1, -1)) / self.trajectory_scale.view(1, -1)
            trajectory = torch.where(self.trajectory_keep.view(1, -1), trajectory, torch.zeros_like(trajectory))
        else:
            trajectory = trajectory_raw
        minute_embedding = self._minute_features(signals)
        source = torch.cat((amplitude, trajectory, minute_embedding), dim=1)
        state_inputs, state_diagnostics = self._state_inputs(source, bearings, steps)
        state = self._encode_state(state_inputs, bearings, steps)
        mu_rank = self.mu_rank_head(state)
        delta_fpt = self.FPT_DELTA_SCALE * torch.tanh(self.delta_fpt_head(state))
        mu = self.mu_bias + mu_rank + delta_fpt
        risk = -mu
        if not torch.isfinite(mu).all() or not torch.isfinite(risk).all():
            raise RuntimeError("R14-B produced non-finite AFT locations")
        output = {
            "logits": risk,
            "risk": risk,
            "mu": mu,
            "mu_rank": mu_rank,
            "delta_fpt": delta_fpt,
            "sigma": self.sigma,
        }
        if evaluation_times is not None:
            output["survival"] = self.aft_survival(mu, evaluation_times)
        if return_components:
            output.update(
                {
                    "amplitude_features": amplitude,
                    "ot_trajectory": trajectory,
                    "ot_trajectory_raw": trajectory_raw,
                    "ot_distances": distances,
                    "minute_embedding": minute_embedding,
                    "source_features": source,
                    "state_inputs": state_inputs,
                    "causal_state": state,
                    **state_diagnostics,
                    "sinkhorn_hard_marginal_error": sinkhorn_diagnostics["hard_marginal_error"],
                    "sinkhorn_converged": sinkhorn_diagnostics["converged"],
                }
            )
        return output

    def loss_components(
        self,
        output: dict[str, torch.Tensor],
        survival_times,
        events,
        fpt_times,
        fpt_events,
        teacher_risk,
        *,
        include_fpt: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Return the fixed R14-B loss decomposition."""

        from .losses import partial_ll_loss

        cox = partial_ll_loss(output["risk"], survival_times, events)["loss"]
        teacher = r13_teacher_pairwise_rank_loss(output["risk"], teacher_risk)
        aft = lognormal_aft_loss(output["mu"], output["sigma"], fpt_times, fpt_events)
        smooth = fpt_smooth_l1_loss(output["mu"], fpt_times, fpt_events)
        if not include_fpt:
            aft = aft * 0.0
            smooth = smooth * 0.0
        total = 0.35 * cox + 1.00 * aft + 0.75 * smooth + 0.50 * teacher
        if not torch.isfinite(total):
            raise RuntimeError("R14-B hybrid loss produced a non-finite value")
        return {
            "loss": total,
            "cox": cox,
            "aft": aft,
            "smooth_fpt": smooth,
            "teacher_rank": teacher,
        }
