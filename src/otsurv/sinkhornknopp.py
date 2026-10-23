"""Differentiable semi-relaxed Sinkhorn solvers used by OTSurv.

The original implementation moved the cost matrix to a detached CPU tensor and
always appended a dummy column.  That made the transport plan constant with
respect to both window and prototype embeddings, and coupled two independent
modelling choices.  This module keeps the solve on the input device, uses
log-domain updates, and exposes marginal relaxation and the absorbing node as
separate switches.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch


SolverReturn = Union[
    torch.Tensor,
    Tuple[torch.Tensor, int],
    Tuple[torch.Tensor, Dict[str, torch.Tensor]],
    Tuple[torch.Tensor, int, Dict[str, torch.Tensor]],
]

HARD_MARGINAL_TOLERANCE = 1e-10


class SK_Class(torch.nn.Module):
    """Small compatibility base class for the historical solver API."""

    def set_rho(self, rho: float) -> None:
        self._validate_rho(rho)
        self.rho = float(rho)

    def get_rho(self) -> Optional[float]:
        return getattr(self, "rho", None)

    def set_gamma(self, gamma: float) -> None:
        if not math.isfinite(float(gamma)) or float(gamma) < 0.0:
            raise ValueError("gamma must be finite and non-negative")
        self.gamma = float(gamma)

    def get_gamma(self) -> Optional[float]:
        return getattr(self, "gamma", None)

    @staticmethod
    def _validate_rho(rho: float) -> None:
        if not 0.0 < float(rho) <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {rho!r}")


class SemiCurrSinkhornKnopp(SK_Class):
    """Entropic OT with an optional KL-relaxed target marginal.

    The source marginal (one unit of mass spread uniformly over windows) is
    always enforced.  With ``semi_use=False`` this is balanced Sinkhorn and the
    target marginal is also enforced.  With ``semi_use=True`` only prototype
    columns use the standard unbalanced exponent
    ``gamma / (gamma + epsilon)``.  An enabled dummy column remains a hard
    marginal constraint, so its mass is always ``1 - rho``.

    ``use_dummy`` independently appends an absorbing target.  Its prior mass is
    ``1 - rho`` while the real prototypes share ``rho``.  Without the dummy,
    prototypes share all target mass and ``rho`` has no effect on the marginal.

    ``num_iters`` is the actual iteration budget.  ``numItermax`` is retained
    only as a backwards-compatible hard safety cap, so the effective budget is
    ``min(num_iters, numItermax)``.
    """

    def __init__(
        self,
        num_iters: int = 3,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        stoperr: float = 1e-6,
        numItermax: Optional[int] = None,
        rho: float = 1.0,
        semi_use: bool = True,
        prior: Optional[torch.Tensor] = None,
        use_dummy: bool = True,
    ) -> None:
        super().__init__()
        if (
            isinstance(num_iters, bool)
            or int(num_iters) != num_iters
            or int(num_iters) < 1
        ):
            raise ValueError("num_iters must be an integer of at least 1")
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(float(gamma)) or float(gamma) < 0.0:
            raise ValueError("gamma must be finite and non-negative")
        if not math.isfinite(float(stoperr)) or float(stoperr) < 0.0:
            raise ValueError("stoperr must be finite and non-negative")
        if numItermax is not None and (
            isinstance(numItermax, bool)
            or int(numItermax) != numItermax
            or int(numItermax) < 1
        ):
            raise ValueError("numItermax must be an integer of at least 1 when provided")
        self._validate_rho(rho)

        self.num_iters = int(num_iters)
        self.numItermax = int(numItermax) if numItermax is not None else self.num_iters
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.stoperr = float(stoperr)
        self.rho = float(rho)
        self.semi_use = bool(semi_use)
        self.use_dummy = bool(use_dummy)
        self.prior = prior
        self.last_diagnostics: Optional[Dict[str, torch.Tensor]] = None

    @property
    def iteration_budget(self) -> int:
        return min(self.num_iters, self.numItermax)

    def reset(self) -> None:
        """Clear diagnostics retained for inspection between calls."""
        self.last_diagnostics = None

    def _prototype_prior(
        self,
        num_prototypes: int,
        device: torch.device,
        dtype: torch.dtype,
        pred_order: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.prior is None:
            return torch.full(
                (num_prototypes,),
                1.0 / num_prototypes,
                device=device,
                dtype=dtype,
            )

        prior = torch.as_tensor(self.prior, device=device, dtype=dtype).reshape(-1)
        if prior.numel() != num_prototypes:
            raise ValueError(
                f"prior has {prior.numel()} entries for {num_prototypes} prototypes"
            )
        if not torch.isfinite(prior).all() or (prior < 0).any():
            raise ValueError("prior must contain finite, non-negative values")
        prior_sum = prior.sum()
        if prior_sum <= 0:
            raise ValueError("prior must have positive total mass")
        prior = prior / prior_sum

        if pred_order is not None:
            order = torch.as_tensor(pred_order, device=device, dtype=torch.long).reshape(-1)
            if order.numel() != num_prototypes:
                raise ValueError("pred_order must contain one index per prototype")
            reordered = torch.empty_like(prior)
            reordered = reordered.scatter(0, order, prior)
            prior = reordered
        return prior

    def _problem(
        self,
        cost: torch.Tensor,
        pred_order: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, k = cost.shape
        source = torch.full((n,), 1.0 / n, device=cost.device, dtype=cost.dtype)
        prototype_prior = self._prototype_prior(k, cost.device, cost.dtype, pred_order)

        if self.use_dummy:
            dummy_cost = torch.zeros((n, 1), device=cost.device, dtype=cost.dtype)
            expanded_cost = torch.cat((cost, dummy_cost), dim=1)
            target = torch.cat(
                (
                    self.rho * prototype_prior,
                    cost.new_tensor([1.0 - self.rho]),
                )
            )
        else:
            expanded_cost = cost
            target = prototype_prior
        return expanded_cost, source, target

    @staticmethod
    def _safe_log(probability: torch.Tensor) -> torch.Tensor:
        negative_inf = torch.full_like(probability, -torch.inf)
        return torch.where(probability > 0, probability.log(), negative_inf)

    def _project_hard_dummy(
        self,
        log_kernel: torch.Tensor,
        log_v: torch.Tensor,
        source: torch.Tensor,
        dummy_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Enforce source and dummy marginals with a differentiable scalar solve.

        For fixed real-prototype potentials, the dummy potential is the unique
        root of a monotone logistic mass equation.  A safeguarded Newton solve
        keeps every iterate on the computation graph. ``numItermax`` is the
        explicit sub-solve budget and failure to meet the hard marginal is an
        error, rather than a silently approximate transport plan.
        """
        real_logits = log_kernel[:, :-1] + log_v[:-1].unsqueeze(0)
        real_log_partition = torch.logsumexp(real_logits, dim=1)

        if self.rho == 1.0:
            real_plan = source.unsqueeze(1) * torch.softmax(real_logits, dim=1)
            projected_log_v = torch.cat(
                (log_v[:-1], log_v.new_full((1,), -torch.inf))
            )
            full_plan = torch.cat(
                (real_plan, source.new_zeros((source.numel(), 1))), dim=1
            )
            return full_plan, projected_log_v, 0, source.new_zeros(())

        target_fraction = dummy_target / source.sum()
        if (
            not torch.isfinite(target_fraction).detach().item()
            or not 0.0 < target_fraction.detach().item() < 1.0
        ):
            raise RuntimeError(
                "dummy target is not representable strictly between zero and total mass"
            )
        log_odds = torch.log(target_fraction) - torch.log1p(-target_fraction)
        total_source = source.sum()
        lower = real_log_partition.min() + log_odds
        upper = real_log_partition.max() + log_odds
        log_dummy_scale = (
            log_odds + (source * real_log_partition).sum() / total_source
        ).clamp(min=lower, max=upper)
        derivative_floor = torch.finfo(source.dtype).tiny
        # Hard absorbing mass is independent of the outer Sinkhorn stopping
        # tolerance. A loose fixed-point tolerance must never weaken it.
        projection_tolerance = HARD_MARGINAL_TOLERANCE
        projection_converged = False
        projection_residual = source.new_tensor(torch.inf)
        projection_iterations = 0

        for projection_iterations in range(1, self.numItermax + 1):
            dummy_fraction = torch.sigmoid(log_dummy_scale - real_log_partition)
            mass_error = (source * dummy_fraction).sum() - dummy_target
            projection_residual = mass_error.abs()
            if projection_residual.detach().item() <= projection_tolerance:
                projection_converged = True
                break

            move_lower = mass_error < 0
            lower = torch.where(move_lower, log_dummy_scale, lower)
            upper = torch.where(move_lower, upper, log_dummy_scale)
            derivative = (source * dummy_fraction * (1.0 - dummy_fraction)).sum()
            newton_candidate = log_dummy_scale - mass_error / derivative.clamp_min(
                derivative_floor
            )
            midpoint = 0.5 * (lower + upper)
            use_newton = (
                torch.isfinite(newton_candidate)
                & (newton_candidate > lower)
                & (newton_candidate < upper)
                & (derivative > derivative_floor)
            )
            log_dummy_scale = torch.where(use_newton, newton_candidate, midpoint)

        dummy_fraction = torch.sigmoid(log_dummy_scale - real_log_partition)
        projection_residual = (
            (source * dummy_fraction).sum() - dummy_target
        ).abs()
        projection_converged = (
            projection_converged
            or projection_residual.detach().item() <= projection_tolerance
        )
        if (
            not torch.isfinite(projection_residual).detach().item()
            or not projection_converged
        ):
            raise RuntimeError(
                "hard dummy projection did not converge within numItermax="
                f"{self.numItermax}: residual={projection_residual.detach().item():.6e}"
            )

        real_conditional = torch.softmax(real_logits, dim=1)
        real_plan = source.unsqueeze(1) * (1.0 - dummy_fraction).unsqueeze(1) * real_conditional
        dummy_plan = (source * dummy_fraction).unsqueeze(1)
        projected_log_v = torch.cat(
            (log_v[:-1], log_dummy_scale.reshape(1))
        )
        return (
            torch.cat((real_plan, dummy_plan), dim=1),
            projected_log_v,
            projection_iterations,
            projection_residual,
        )

    def _project_hard_dummy_batched(
        self,
        log_kernel: torch.Tensor,
        log_v: torch.Tensor,
        source: torch.Tensor,
        dummy_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized counterpart of :meth:`_project_hard_dummy`.

        Every batch member remains an independent OT problem.  Vectorizing the
        scalar hard-dummy projection removes the per-sample CUDA launch and
        synchronization overhead without pooling source or target marginals
        across bearings/minutes.
        """
        batch_size = log_kernel.shape[0]
        real_logits = log_kernel[:, :, :-1] + log_v[:, :-1].unsqueeze(1)
        real_log_partition = torch.logsumexp(real_logits, dim=2)

        if self.rho == 1.0:
            real_plan = source.unsqueeze(2) * torch.softmax(real_logits, dim=2)
            projected_log_v = torch.cat(
                (log_v[:, :-1], log_v.new_full((batch_size, 1), -torch.inf)),
                dim=1,
            )
            full_plan = torch.cat(
                (real_plan, source.new_zeros((batch_size, source.shape[1], 1))),
                dim=2,
            )
            return (
                full_plan,
                projected_log_v,
                torch.zeros(batch_size, device=source.device, dtype=torch.int64),
                source.new_zeros(batch_size),
                torch.ones(batch_size, device=source.device, dtype=torch.bool),
            )

        total_source = source.sum(dim=1)
        target_fraction = dummy_target / total_source
        if (
            not torch.isfinite(target_fraction).all().detach().item()
            or not ((target_fraction > 0.0) & (target_fraction < 1.0)).all().detach().item()
        ):
            raise RuntimeError(
                "dummy target is not representable strictly between zero and total mass"
            )

        log_odds = torch.log(target_fraction) - torch.log1p(-target_fraction)
        lower = real_log_partition.min(dim=1).values + log_odds
        upper = real_log_partition.max(dim=1).values + log_odds
        log_dummy_scale = (
            log_odds
            + (source * real_log_partition).sum(dim=1) / total_source
        ).clamp(min=lower, max=upper)
        derivative_floor = torch.finfo(source.dtype).tiny
        projection_tolerance = HARD_MARGINAL_TOLERANCE
        projection_converged = torch.zeros(
            batch_size, device=source.device, dtype=torch.bool
        )
        projection_residual = source.new_full((batch_size,), torch.inf)
        projection_iterations = torch.zeros(
            batch_size, device=source.device, dtype=torch.int64
        )

        for iteration in range(1, self.numItermax + 1):
            dummy_fraction = torch.sigmoid(
                log_dummy_scale.unsqueeze(1) - real_log_partition
            )
            mass_error = (source * dummy_fraction).sum(dim=1) - dummy_target
            projection_residual = mass_error.abs()
            active = ~projection_converged
            projection_iterations = torch.where(
                active,
                torch.full_like(projection_iterations, iteration),
                projection_iterations,
            )
            just_converged = active & (
                projection_residual.detach() <= projection_tolerance
            )
            projection_converged = projection_converged | just_converged
            needs_update = active & ~just_converged
            if not needs_update.any().detach().item():
                break

            move_lower = mass_error < 0
            next_lower = torch.where(move_lower, log_dummy_scale, lower)
            next_upper = torch.where(move_lower, upper, log_dummy_scale)
            derivative = (
                source * dummy_fraction * (1.0 - dummy_fraction)
            ).sum(dim=1)
            newton_candidate = log_dummy_scale - mass_error / derivative.clamp_min(
                derivative_floor
            )
            midpoint = 0.5 * (next_lower + next_upper)
            use_newton = (
                torch.isfinite(newton_candidate)
                & (newton_candidate > next_lower)
                & (newton_candidate < next_upper)
                & (derivative > derivative_floor)
            )
            next_log_dummy_scale = torch.where(
                use_newton, newton_candidate, midpoint
            )
            lower = torch.where(needs_update, next_lower, lower)
            upper = torch.where(needs_update, next_upper, upper)
            log_dummy_scale = torch.where(
                needs_update, next_log_dummy_scale, log_dummy_scale
            )

        dummy_fraction = torch.sigmoid(
            log_dummy_scale.unsqueeze(1) - real_log_partition
        )
        projection_residual = (
            (source * dummy_fraction).sum(dim=1) - dummy_target
        ).abs()
        projection_converged = projection_converged | (
            projection_residual.detach() <= projection_tolerance
        )
        if (
            not torch.isfinite(projection_residual).all().detach().item()
            or not projection_converged.all().detach().item()
        ):
            max_residual = projection_residual.detach().max().item()
            raise RuntimeError(
                "hard dummy projection did not converge within numItermax="
                f"{self.numItermax}: max_residual={max_residual:.6e}"
            )

        real_conditional = torch.softmax(real_logits, dim=2)
        real_plan = (
            source.unsqueeze(2)
            * (1.0 - dummy_fraction).unsqueeze(2)
            * real_conditional
        )
        dummy_plan = (source * dummy_fraction).unsqueeze(2)
        projected_log_v = torch.cat(
            (log_v[:, :-1], log_dummy_scale.unsqueeze(1)), dim=1
        )
        return (
            torch.cat((real_plan, dummy_plan), dim=2),
            projected_log_v,
            projection_iterations,
            projection_residual,
            projection_converged,
        )

    def cost_forward_batched(
        self,
        cost: torch.Tensor,
        final: bool = True,
        pred_order: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        return_full_plan: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Solve independent equal-shape OT problems in one batch.

        ``cost`` has shape ``[batch, n_windows, n_prototypes]``.  The scalar
        :meth:`cost_forward` remains the compatibility/reference path; this
        method is used only when OTSurv receives equal-length window sequences.
        No marginal is shared between batch members.
        """
        if (
            cost.ndim != 3
            or cost.shape[0] == 0
            or cost.shape[1] == 0
            or cost.shape[2] == 0
        ):
            raise ValueError(
                "cost must be a non-empty [batch, n_windows, n_prototypes] tensor"
            )
        if not torch.is_floating_point(cost):
            raise TypeError("cost must be a floating-point tensor")
        if not torch.isfinite(cost).all():
            raise ValueError("cost must contain only finite values")

        batch_size, n_windows, n_prototypes = cost.shape
        input_dtype = cost.dtype
        work_dtype = (
            torch.float64
            if cost.dtype in (torch.float16, torch.float32)
            else cost.dtype
        )
        work_cost = cost.to(dtype=work_dtype)
        source = torch.full(
            (batch_size, n_windows),
            1.0 / n_windows,
            device=cost.device,
            dtype=work_dtype,
        )
        prototype_prior = self._prototype_prior(
            n_prototypes, cost.device, work_dtype, pred_order
        )
        if self.use_dummy:
            expanded_cost = torch.cat(
                (work_cost, work_cost.new_zeros((batch_size, n_windows, 1))),
                dim=2,
            )
            target = torch.cat(
                (
                    self.rho * prototype_prior,
                    work_cost.new_tensor([1.0 - self.rho]),
                )
            )
        else:
            expanded_cost = work_cost
            target = prototype_prior

        log_source = self._safe_log(source)
        log_target = self._safe_log(target)
        log_kernel = -expanded_cost / self.epsilon
        if not torch.isfinite(log_kernel).all().detach().item():
            raise FloatingPointError(
                "scaled OT costs are non-finite; increase epsilon or rescale costs"
            )
        log_v = torch.zeros(
            (batch_size, target.numel()), device=cost.device, dtype=work_dtype
        )
        positive_target = target > 0
        shape_target = positive_target.clone()
        if self.semi_use and self.use_dummy:
            shape_target[-1] = False
        relaxation = self.gamma / (self.gamma + self.epsilon)
        requires_projected_early_stop = (
            self.semi_use and self.use_dummy and self.rho < 1.0
        )

        def target_update(
            current_log_v: torch.Tensor,
            current_log_kernel: torch.Tensor = log_kernel,
            current_log_source: torch.Tensor = log_source,
        ) -> torch.Tensor:
            current_log_u = current_log_source - torch.logsumexp(
                current_log_kernel + current_log_v.unsqueeze(1), dim=2
            )
            log_column_kernel = torch.logsumexp(
                current_log_kernel + current_log_u.unsqueeze(2), dim=1
            )
            updated_log_v = log_target.unsqueeze(0) - log_column_kernel
            if self.semi_use:
                if self.use_dummy:
                    updated_log_v = torch.cat(
                        (
                            relaxation * updated_log_v[:, :-1],
                            updated_log_v[:, -1:],
                        ),
                        dim=1,
                    )
                else:
                    updated_log_v = relaxation * updated_log_v
            return torch.where(
                positive_target.unsqueeze(0),
                updated_log_v,
                torch.full_like(updated_log_v, -torch.inf),
            )

        active = torch.ones(batch_size, device=cost.device, dtype=torch.bool)
        iteration_count = torch.zeros(
            batch_size, device=cost.device, dtype=torch.int64
        )
        potential_gauge_residual = work_cost.new_full((batch_size,), torch.inf)
        potential_shape_residual = work_cost.new_full((batch_size,), torch.inf)

        for iteration in range(1, self.iteration_budget + 1):
            candidate_log_v = target_update(log_v)
            delta = candidate_log_v - log_v
            all_finite_change = delta[:, positive_target]
            shape_change = delta[:, shape_target]
            potential_gauge_residual = all_finite_change.abs().max(dim=1).values
            potential_shape_residual = 0.5 * (
                shape_change.max(dim=1).values - shape_change.min(dim=1).values
            )
            iteration_count = torch.where(
                active,
                torch.full_like(iteration_count, iteration),
                iteration_count,
            )
            next_log_v = torch.where(active.unsqueeze(1), candidate_log_v, log_v)

            if self.stoperr > 0:
                stop_candidates = active & (
                    potential_shape_residual.detach() <= self.stoperr
                )
            else:
                stop_candidates = torch.zeros_like(active)

            if requires_projected_early_stop and stop_candidates.any().detach().item():
                selected = torch.nonzero(stop_candidates, as_tuple=False).squeeze(1)
                selected_kernel = log_kernel.index_select(0, selected)
                selected_source = source.index_select(0, selected)
                selected_log_source = log_source.index_select(0, selected)
                selected_log_v = next_log_v.index_select(0, selected)
                (
                    projected_plan,
                    projected_log_v,
                    _,
                    _,
                    _,
                ) = self._project_hard_dummy_batched(
                    log_kernel=selected_kernel,
                    log_v=selected_log_v,
                    source=selected_source,
                    dummy_target=target[-1],
                )
                projected_candidate_log_v = target_update(
                    projected_log_v,
                    current_log_kernel=selected_kernel,
                    current_log_source=selected_log_source,
                )
                next_projected_plan, _, _, _, _ = self._project_hard_dummy_batched(
                    log_kernel=selected_kernel,
                    log_v=projected_candidate_log_v,
                    source=selected_source,
                    dummy_target=target[-1],
                )
                projected_delta = projected_candidate_log_v - projected_log_v
                projected_shape_delta = projected_delta[:, shape_target]
                projected_shape_residual = 0.5 * (
                    projected_shape_delta.max(dim=1).values
                    - projected_shape_delta.min(dim=1).values
                )
                projected_conditionals = projected_plan / selected_source.unsqueeze(2)
                next_projected_conditionals = (
                    next_projected_plan / selected_source.unsqueeze(2)
                )
                projected_transport_residual = (
                    0.5
                    * (next_projected_conditionals - projected_conditionals)
                    .abs()
                    .sum(dim=2)
                ).max(dim=1).values
                projected_row_error = (
                    projected_plan.sum(dim=2) - selected_source
                ).abs().max(dim=1).values
                projected_dummy_error = (
                    projected_plan[:, :, -1].sum(dim=1) - target[-1]
                ).abs()
                projected_hard_error = torch.maximum(
                    projected_row_error, projected_dummy_error
                )
                projected_success = (
                    (projected_transport_residual.detach() <= self.stoperr)
                    & (projected_shape_residual.detach() <= self.stoperr)
                    & (projected_hard_error.detach() <= HARD_MARGINAL_TOLERANCE)
                )
                next_log_v = next_log_v.index_copy(0, selected, projected_log_v)
                selected_active = ~projected_success
                active = active.index_copy(0, selected, selected_active)
            else:
                active = active & ~stop_candidates

            log_v = next_log_v
            if not active.any().detach().item():
                break

        projection_iterations = torch.zeros(
            batch_size, device=cost.device, dtype=torch.int64
        )
        projection_residual = work_cost.new_zeros(batch_size)
        projection_converged = torch.ones(
            batch_size, device=cost.device, dtype=torch.bool
        )
        if self.use_dummy:
            (
                full_plan,
                log_v,
                projection_iterations,
                projection_residual,
                projection_converged,
            ) = self._project_hard_dummy_batched(
                log_kernel=log_kernel,
                log_v=log_v,
                source=source,
                dummy_target=target[-1],
            )
        else:
            log_u = log_source - torch.logsumexp(
                log_kernel + log_v.unsqueeze(1), dim=2
            )
            log_plan = log_kernel + log_u.unsqueeze(2) + log_v.unsqueeze(1)
            full_plan = torch.exp(log_plan)

        final_candidate_log_v = target_update(log_v)
        fixed_point_delta = final_candidate_log_v - log_v
        all_fixed_point_delta = fixed_point_delta[:, positive_target]
        shape_fixed_point_delta = fixed_point_delta[:, shape_target]
        potential_gauge_residual = all_fixed_point_delta.abs().max(dim=1).values
        potential_shape_residual = 0.5 * (
            shape_fixed_point_delta.max(dim=1).values
            - shape_fixed_point_delta.min(dim=1).values
        )

        if self.use_dummy:
            next_plan, _, _, _, _ = self._project_hard_dummy_batched(
                log_kernel=log_kernel,
                log_v=final_candidate_log_v,
                source=source,
                dummy_target=target[-1],
            )
        else:
            next_log_u = log_source - torch.logsumexp(
                log_kernel + final_candidate_log_v.unsqueeze(1), dim=2
            )
            next_log_plan = (
                log_kernel
                + next_log_u.unsqueeze(2)
                + final_candidate_log_v.unsqueeze(1)
            )
            next_plan = torch.exp(next_log_plan)

        row_conditionals = full_plan / source.unsqueeze(2)
        next_row_conditionals = next_plan / source.unsqueeze(2)
        transport_residual = (
            0.5 * (next_row_conditionals - row_conditionals).abs().sum(dim=2)
        ).max(dim=1).values
        row_marginal = full_plan.sum(dim=2)
        column_marginal = full_plan.sum(dim=1)
        prototype_mass = (
            column_marginal[:, :-1].sum(dim=1)
            if self.use_dummy
            else column_marginal.sum(dim=1)
        )
        dummy_mass = (
            column_marginal[:, -1]
            if self.use_dummy
            else full_plan.new_zeros(batch_size)
        )
        row_marginal_error = (row_marginal - source).abs().max(dim=1).values
        target_marginal_error = (
            column_marginal - target.unsqueeze(0)
        ).abs().max(dim=1).values
        enforced_marginal_error = row_marginal_error
        if self.use_dummy:
            enforced_marginal_error = torch.maximum(
                enforced_marginal_error, (dummy_mass - target[-1]).abs()
            )
        if (
            not torch.isfinite(full_plan).all().detach().item()
            or not torch.isfinite(enforced_marginal_error).all().detach().item()
            or (enforced_marginal_error.detach() > HARD_MARGINAL_TOLERANCE)
            .any()
            .item()
        ):
            max_error = enforced_marginal_error.detach().max().item()
            raise RuntimeError(
                "OT hard source/dummy marginal check failed: "
                f"max_error={max_error:.6e}, tolerance={HARD_MARGINAL_TOLERANCE:.6e}"
            )
        dummy_marginal_error = (
            (dummy_mass - target[-1]).abs()
            if self.use_dummy
            else full_plan.new_zeros(batch_size)
        )
        if self.semi_use:
            hard_marginal_error = torch.maximum(
                row_marginal_error, dummy_marginal_error
            )
        else:
            hard_marginal_error = torch.maximum(
                row_marginal_error, target_marginal_error
            )
        converged = (
            (transport_residual.detach() <= self.stoperr)
            & (potential_shape_residual.detach() <= self.stoperr)
            & projection_converged
            & (hard_marginal_error.detach() <= HARD_MARGINAL_TOLERANCE)
            if self.stoperr > 0
            else torch.zeros(batch_size, device=cost.device, dtype=torch.bool)
        )
        target_prototype_mass = (
            target[:-1].sum() if self.use_dummy else target.sum()
        ).to(input_dtype)
        diagnostics: Dict[str, torch.Tensor] = {
            "iterations": iteration_count,
            "converged": converged,
            "fixed_point_residual": transport_residual.to(input_dtype),
            "potential_shape_residual": potential_shape_residual.to(input_dtype),
            "potential_gauge_residual": potential_gauge_residual.to(input_dtype),
            "row_marginal_error": row_marginal_error.to(input_dtype),
            "target_marginal_error": target_marginal_error.to(input_dtype),
            "dummy_marginal_error": dummy_marginal_error.to(input_dtype),
            "hard_marginal_error": hard_marginal_error.to(input_dtype),
            "dummy_projection_iterations": projection_iterations,
            "dummy_projection_budget": torch.full_like(
                projection_iterations, self.numItermax
            ),
            "dummy_projection_converged": projection_converged,
            "dummy_projection_residual": projection_residual.to(input_dtype),
            "prototype_mass": prototype_mass.to(input_dtype),
            "dummy_mass": dummy_mass.to(input_dtype),
            "target_prototype_mass": target_prototype_mass.expand(batch_size),
        }
        self.last_diagnostics = diagnostics

        returned_plan = (
            full_plan
            if return_full_plan
            else (full_plan[:, :, :-1] if self.use_dummy else full_plan)
        )
        if final:
            returned_plan = returned_plan * n_windows
        returned_plan = returned_plan.to(dtype=input_dtype)
        if return_diagnostics:
            return returned_plan, diagnostics
        return returned_plan

    def cost_forward(
        self,
        cost: torch.Tensor,
        final: bool = True,
        count: bool = False,
        pred_order: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        return_full_plan: bool = False,
    ) -> SolverReturn:
        if cost.ndim != 2 or cost.shape[0] == 0 or cost.shape[1] == 0:
            raise ValueError("cost must be a non-empty [n_windows, n_prototypes] matrix")
        if not torch.is_floating_point(cost):
            raise TypeError("cost must be a floating-point tensor")
        if not torch.isfinite(cost).all():
            raise ValueError("cost must contain only finite values")

        input_dtype = cost.dtype
        work_dtype = torch.float64 if cost.dtype in (torch.float16, torch.float32) else cost.dtype
        work_cost = cost.to(dtype=work_dtype)
        expanded_cost, source, target = self._problem(work_cost, pred_order)

        log_source = self._safe_log(source)
        log_target = self._safe_log(target)
        log_kernel = -expanded_cost / self.epsilon
        if not torch.isfinite(log_kernel).all().detach().item():
            raise FloatingPointError(
                "scaled OT costs are non-finite; increase epsilon or rescale costs"
            )
        log_v = torch.zeros_like(target)
        positive_target = target > 0
        shape_target = positive_target.clone()
        if self.semi_use and self.use_dummy:
            # The dummy has a hard marginal and therefore a separate dual
            # update.  Its delta cannot be compared with the KL-relaxed real
            # prototype deltas when quotienting out their common gauge.
            shape_target[-1] = False
        relaxation = self.gamma / (self.gamma + self.epsilon)
        requires_projected_early_stop = (
            self.semi_use and self.use_dummy and self.rho < 1.0
        )

        def target_update(current_log_v: torch.Tensor) -> torch.Tensor:
            current_log_u = log_source - torch.logsumexp(
                log_kernel + current_log_v.unsqueeze(0), dim=1
            )
            log_column_kernel = torch.logsumexp(
                log_kernel + current_log_u.unsqueeze(1), dim=0
            )
            updated_log_v = log_target - log_column_kernel
            if self.semi_use:
                if self.use_dummy:
                    updated_log_v = torch.cat(
                        (
                            relaxation * updated_log_v[:-1],
                            updated_log_v[-1:],
                        )
                    )
                else:
                    updated_log_v = relaxation * updated_log_v
            return torch.where(
                positive_target,
                updated_log_v,
                torch.full_like(updated_log_v, -torch.inf),
            )

        converged = False
        iteration_count = 0
        potential_gauge_residual = work_cost.new_tensor(torch.inf)
        potential_shape_residual = work_cost.new_tensor(torch.inf)
        for iteration_count in range(1, self.iteration_budget + 1):
            candidate_log_v = target_update(log_v)

            all_finite_change = (candidate_log_v - log_v)[positive_target]
            shape_change = (candidate_log_v - log_v)[shape_target]
            potential_gauge_residual = all_finite_change.abs().max()
            potential_shape_residual = 0.5 * (
                shape_change.max() - shape_change.min()
            )
            log_v = candidate_log_v
            if (
                self.stoperr > 0
                and potential_shape_residual.detach().item() <= self.stoperr
            ):
                if not requires_projected_early_stop:
                    converged = True
                    break

                # With a positive hard dummy mass, a small raw potential
                # change is only a cheap trigger.  Verify the complete
                # projected transport map before stopping; if it fails, keep
                # iterating from the hard-feasible projected state.
                projected_plan, projected_log_v, _, _ = self._project_hard_dummy(
                    log_kernel=log_kernel,
                    log_v=log_v,
                    source=source,
                    dummy_target=target[-1],
                )
                projected_candidate_log_v = target_update(projected_log_v)
                next_projected_plan, _, _, _ = self._project_hard_dummy(
                    log_kernel=log_kernel,
                    log_v=projected_candidate_log_v,
                    source=source,
                    dummy_target=target[-1],
                )
                projected_delta = projected_candidate_log_v - projected_log_v
                projected_shape_delta = projected_delta[shape_target]
                projected_shape_residual = 0.5 * (
                    projected_shape_delta.max() - projected_shape_delta.min()
                )
                projected_conditionals = projected_plan / source.unsqueeze(1)
                next_projected_conditionals = (
                    next_projected_plan / source.unsqueeze(1)
                )
                projected_transport_residual = (
                    0.5
                    * (
                        next_projected_conditionals - projected_conditionals
                    ).abs().sum(dim=1)
                ).max()
                projected_row_error = (
                    projected_plan.sum(dim=1) - source
                ).abs().max()
                projected_dummy_error = (
                    projected_plan[:, -1].sum() - target[-1]
                ).abs()
                projected_hard_error = torch.maximum(
                    projected_row_error, projected_dummy_error
                )
                log_v = projected_log_v
                if (
                    projected_transport_residual.detach().item() <= self.stoperr
                    and projected_shape_residual.detach().item() <= self.stoperr
                    and projected_hard_error.detach().item()
                    <= HARD_MARGINAL_TOLERANCE
                ):
                    converged = True
                    break

        # Recompute the source potential so the required source marginal is the
        # final update even when the target marginal is deliberately relaxed.
        # Whenever a dummy is enabled its total mass is an exact hard
        # constraint, independently of the real-prototype relaxation switch.
        projection_iterations = 0
        projection_residual = work_cost.new_zeros(())
        projection_converged = True
        if self.use_dummy:
            (
                full_plan,
                log_v,
                projection_iterations,
                projection_residual,
            ) = self._project_hard_dummy(
                log_kernel=log_kernel,
                log_v=log_v,
                source=source,
                dummy_target=target[-1],
            )
        else:
            log_u = log_source - torch.logsumexp(
                log_kernel + log_v.unsqueeze(0), dim=1
            )
            log_plan = log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0)
            full_plan = torch.exp(log_plan)

        # Report the residual of the fixed-point map at the potentials that
        # produced the returned plan, including the hard dummy projection.
        final_candidate_log_v = target_update(log_v)
        fixed_point_delta = final_candidate_log_v - log_v
        all_fixed_point_delta = fixed_point_delta[positive_target]
        shape_fixed_point_delta = fixed_point_delta[shape_target]
        potential_gauge_residual = all_fixed_point_delta.abs().max()
        potential_shape_residual = 0.5 * (
            shape_fixed_point_delta.max() - shape_fixed_point_delta.min()
        )

        # Log-potentials have a numerically irrelevant common-scale mode: a
        # shared shift is cancelled by the source normalization and can keep
        # drifting after the transport itself is stable. Measure the fixed
        # point in transport space by applying one more complete update,
        # including the hard dummy projection when enabled.
        if self.use_dummy:
            next_plan, _, _, _ = self._project_hard_dummy(
                log_kernel=log_kernel,
                log_v=final_candidate_log_v,
                source=source,
                dummy_target=target[-1],
            )
        else:
            next_log_u = log_source - torch.logsumexp(
                log_kernel + final_candidate_log_v.unsqueeze(0), dim=1
            )
            next_log_plan = (
                log_kernel
                + next_log_u.unsqueeze(1)
                + final_candidate_log_v.unsqueeze(0)
            )
            next_plan = torch.exp(next_log_plan)
        row_conditionals = full_plan / source.unsqueeze(1)
        next_row_conditionals = next_plan / source.unsqueeze(1)
        transport_residual = (
            0.5 * (next_row_conditionals - row_conditionals).abs().sum(dim=1)
        ).max()
        row_marginal = full_plan.sum(dim=1)
        column_marginal = full_plan.sum(dim=0)
        prototype_mass = column_marginal[:-1].sum() if self.use_dummy else column_marginal.sum()
        dummy_mass = column_marginal[-1] if self.use_dummy else full_plan.new_zeros(())
        row_marginal_error = (row_marginal - source).abs().max()
        target_marginal_error = (column_marginal - target).abs().max()
        enforced_marginal_error = row_marginal_error
        if self.use_dummy:
            enforced_marginal_error = torch.maximum(
                enforced_marginal_error,
                (dummy_mass - target[-1]).abs(),
            )
        hard_tolerance = HARD_MARGINAL_TOLERANCE
        if (
            not torch.isfinite(full_plan).all().detach().item()
            or not torch.isfinite(enforced_marginal_error).detach().item()
            or enforced_marginal_error.detach().item() > hard_tolerance
        ):
            raise RuntimeError(
                "OT hard source/dummy marginal check failed: "
                f"error={enforced_marginal_error.detach().item():.6e}, "
                f"tolerance={hard_tolerance:.6e}"
            )
        if self.semi_use:
            dummy_marginal_error = (
                (dummy_mass - target[-1]).abs() if self.use_dummy else full_plan.new_zeros(())
            )
            hard_marginal_error = torch.maximum(row_marginal_error, dummy_marginal_error)
        else:
            dummy_marginal_error = (
                (dummy_mass - target[-1]).abs() if self.use_dummy else full_plan.new_zeros(())
            )
            hard_marginal_error = torch.maximum(row_marginal_error, target_marginal_error)
        converged = bool(
            self.stoperr > 0
            and transport_residual.detach().item() <= self.stoperr
            and potential_shape_residual.detach().item() <= self.stoperr
            and projection_converged
            and hard_marginal_error.detach().item() <= hard_tolerance
        )
        diagnostics: Dict[str, torch.Tensor] = {
            "iterations": torch.tensor(iteration_count, device=cost.device, dtype=torch.int64),
            "converged": torch.tensor(converged, device=cost.device, dtype=torch.bool),
            "fixed_point_residual": transport_residual.to(
                device=cost.device, dtype=input_dtype
            ),
            "potential_shape_residual": potential_shape_residual.to(
                device=cost.device, dtype=input_dtype
            ),
            "potential_gauge_residual": potential_gauge_residual.to(
                device=cost.device, dtype=input_dtype
            ),
            "row_marginal_error": row_marginal_error.to(input_dtype),
            "target_marginal_error": target_marginal_error.to(input_dtype),
            "dummy_marginal_error": dummy_marginal_error.to(input_dtype),
            "hard_marginal_error": hard_marginal_error.to(input_dtype),
            "dummy_projection_iterations": torch.tensor(
                projection_iterations, device=cost.device, dtype=torch.int64
            ),
            "dummy_projection_budget": torch.tensor(
                self.numItermax, device=cost.device, dtype=torch.int64
            ),
            "dummy_projection_converged": torch.tensor(
                projection_converged, device=cost.device, dtype=torch.bool
            ),
            "dummy_projection_residual": projection_residual.to(input_dtype),
            "prototype_mass": prototype_mass.to(input_dtype),
            "dummy_mass": dummy_mass.to(input_dtype),
            "target_prototype_mass": (
                target[:-1].sum() if self.use_dummy else target.sum()
            ).to(input_dtype),
        }
        self.last_diagnostics = diagnostics

        if return_full_plan:
            returned_plan = full_plan
        else:
            returned_plan = full_plan[:, :-1] if self.use_dummy else full_plan
        if final:
            returned_plan = returned_plan * cost.shape[0]
        returned_plan = returned_plan.to(dtype=input_dtype)

        if count and return_diagnostics:
            return returned_plan, iteration_count, diagnostics
        if count:
            return returned_plan, iteration_count
        if return_diagnostics:
            return returned_plan, diagnostics
        return returned_plan

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        cost = -torch.log_softmax(logits, dim=1)
        return self.cost_forward(cost)


class SemiCurrSinkhornKnopp_stable(SemiCurrSinkhornKnopp):
    """Backward-compatible name for the log-domain implementation."""
