from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optimal Transport loss
d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)

# Survival loss
class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, logits, times, censorships):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(logits=logits, y=times, c=censorships,
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


# TODO: document better and clean up
def nll_loss(logits, y, c, alpha=0.0, eps=1e-7, reduction='mean'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    logits: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(logits).
    y: (n_batches, )
        The true time bin index label.
    c: (n_batches, )
        The censoring status indicator.
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # print("h shape", h.shape)

    # make sure these are ints
    y = y.long()
    c = c.long()

    hazards = torch.sigmoid(logits)

    S = torch.cumprod(1 - hazards, dim=1)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # hazards[y] = hazards(1)
    # S[1] = S(1)

    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)

    neg_l = censored_loss + uncensored_loss
    
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
        censored_loss = censored_loss.mean()
        uncensored_loss = uncensored_loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
        censored_loss = censored_loss.sum()
        uncensored_loss = uncensored_loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return {'loss': loss, 'uncensored_loss': uncensored_loss, 'censored_loss': censored_loss}

def _flatten_cox_input(value, name):
    """Return a Cox input as ``[n_samples]`` without losing its graph."""
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == 1:
        return value
    if value.ndim == 2 and value.shape[1] == 1:
        return value[:, 0]
    raise ValueError(f"{name} must have shape [N] or [N, 1], got {tuple(value.shape)}")


def _validate_cox_inputs(lrisks, survival_times, event_indicators):
    lrisks_flat = _flatten_cox_input(lrisks, "lrisks")
    times_flat = _flatten_cox_input(survival_times, "survival_times")
    events_flat = _flatten_cox_input(event_indicators, "event_indicators")

    if not torch.is_floating_point(lrisks_flat):
        raise TypeError("lrisks must be a floating-point tensor")
    if lrisks_flat.numel() == 0:
        raise ValueError("Cox loss requires at least one sample")
    if times_flat.numel() != lrisks_flat.numel() or events_flat.numel() != lrisks_flat.numel():
        raise ValueError("lrisks, survival_times, and event_indicators must have equal length")

    times_flat = times_flat.to(device=lrisks_flat.device)
    events_flat = events_flat.to(device=lrisks_flat.device, dtype=lrisks_flat.dtype)
    if not torch.isfinite(lrisks_flat).all():
        raise ValueError("lrisks must contain only finite values")
    if not torch.isfinite(times_flat).all():
        raise ValueError("survival_times must contain only finite values")
    if not torch.isfinite(events_flat).all():
        raise ValueError("event_indicators must contain only finite values")
    if not torch.all((events_flat == 0) | (events_flat == 1)):
        raise ValueError("event_indicators must contain only 0 or 1")
    return lrisks_flat, times_flat, events_flat


def partial_ll_loss(lrisks, survival_times, event_indicators):
    """Mean negative Cox partial log-likelihood with Breslow ties.

    All samples supplied to this function form one risk set. For each distinct
    observed event time ``t``, Breslow's approximation contributes

    ``sum(log_risk[event at t]) - d_t * logsumexp(log_risk[time >= t])``.

    Args:
        lrisks: Log-risk scores with shape ``[N]`` or ``[N, 1]``.
        survival_times: Observed times with the same shape convention.
        event_indicators: Binary indicators where 1 denotes an observed event.

    Returns:
        A dictionary containing scalar ``loss``, averaged over observed events.
    """
    lrisks, survival_times, event_indicators = _validate_cox_inputs(
        lrisks, survival_times, event_indicators
    )
    num_events = event_indicators.sum()
    if num_events.detach().item() == 0:
        return {'loss': lrisks.sum() * 0.0}

    sort_index = torch.argsort(survival_times, descending=True)
    sorted_times = survival_times[sort_index]
    sorted_events = event_indicators[sort_index]
    sorted_lrisks = lrisks[sort_index]

    _, group_counts = torch.unique_consecutive(sorted_times, return_counts=True)
    group_end_indices = torch.cumsum(group_counts, dim=0) - 1
    group_ids = torch.repeat_interleave(
        torch.arange(group_counts.numel(), device=lrisks.device),
        group_counts,
    )

    group_event_counts = lrisks.new_zeros(group_counts.numel())
    group_event_counts.scatter_add_(0, group_ids, sorted_events)
    group_event_lrisks = lrisks.new_zeros(group_counts.numel())
    group_event_lrisks.scatter_add_(0, group_ids, sorted_lrisks * sorted_events)

    log_risk_set_sums = torch.logcumsumexp(sorted_lrisks, dim=0)[group_end_indices]
    log_likelihood = group_event_lrisks - group_event_counts * log_risk_set_sums
    return {'loss': -log_likelihood.sum() / num_events}


def _validate_aft_targets(true_times, true_events, *, device, dtype):
    """Validate explicit FPT targets without accepting metadata fallbacks."""

    times = torch.as_tensor(true_times, device=device, dtype=dtype).reshape(-1)
    events = torch.as_tensor(true_events, device=device, dtype=dtype).reshape(-1)
    if times.numel() == 0 or times.numel() != events.numel():
        raise ValueError("FPT times and events must be non-empty and aligned")
    if not torch.isfinite(times).all() or not torch.isfinite(events).all():
        raise ValueError("FPT times and events must be finite")
    if (times < 0.0).any():
        raise ValueError("FPT times must be non-negative")
    if not torch.all((events == 0) | (events == 1)):
        raise ValueError("FPT events must be binary")
    return times, events


def lognormal_aft_loss(mu, sigma, true_times, true_events):
    """Censoring-aware log-normal AFT negative log likelihood.

    ``true_events == 1`` uses the log-density of ``log1p(true_time)``;
    censored rows use the corresponding log survival probability.  The
    returned scalar remains connected to ``mu`` and ``sigma`` even when a
    batch contains no observed events.
    """

    if not torch.is_tensor(mu) or not torch.is_floating_point(mu):
        raise TypeError("mu must be a floating-point tensor")
    location = mu.reshape(-1)
    scale = torch.as_tensor(sigma, device=location.device, dtype=location.dtype)
    if scale.numel() != 1 or not torch.isfinite(scale).all() or (scale <= 0.0).any():
        raise ValueError("sigma must be one finite positive scalar")
    times, events = _validate_aft_targets(
        true_times, true_events, device=location.device, dtype=location.dtype
    )
    if location.numel() != times.numel():
        raise ValueError("mu and FPT targets must have equal length")
    log_time = torch.log1p(times)
    z = (log_time - location) / scale
    # Keep both branches finite before ``where`` selects the event/censoring
    # contribution.  This avoids an inf * 0 gradient when a very late
    # censored target drives the normal-tail calculation below machine range.
    z_safe = z.clamp(min=-1.0e6, max=1.0e6)
    log_pdf = -0.5 * z_safe.square() - scale.log() - 0.5 * torch.log(
        location.new_tensor(2.0 * torch.pi)
    )
    # log Phi(-z) is stable for both very early and very late failures.
    log_survival = torch.special.log_ndtr(-z_safe)
    log_survival = log_survival.clamp_min(
        torch.log(location.new_tensor(torch.finfo(location.dtype).tiny))
    )
    negative_log_likelihood = torch.where(events == 1, -log_pdf, -log_survival)
    result = negative_log_likelihood.mean()
    if not torch.isfinite(result):
        raise RuntimeError("log-normal AFT loss produced a non-finite value")
    return result


def fpt_smooth_l1_loss(mu, true_times, true_events, *, beta=1.0):
    """Bounded FPT location loss on observed true events only."""

    if not torch.is_tensor(mu) or not torch.is_floating_point(mu):
        raise TypeError("mu must be a floating-point tensor")
    location = mu.reshape(-1)
    times, events = _validate_aft_targets(
        true_times, true_events, device=location.device, dtype=location.dtype
    )
    if location.numel() != times.numel():
        raise ValueError("mu and FPT targets must have equal length")
    observed = events == 1
    if not bool(observed.any()):
        return location.sum() * 0.0
    return F.smooth_l1_loss(
        location[observed], torch.log1p(times[observed]), beta=float(beta), reduction="mean"
    )


def r13_teacher_pairwise_rank_loss(risk, teacher_risk):
    """Distil only the frozen R13 pair ordering into a new risk score.

    Ties in the teacher are omitted.  No survival/FPT labels are consumed by
    this term, which makes its provenance easy to audit independently.
    """

    if not torch.is_tensor(risk) or not torch.is_floating_point(risk):
        raise TypeError("risk must be a floating-point tensor")
    scores = risk.reshape(-1)
    # The teacher is a frozen target.  Detach even when a caller accidentally
    # supplies a tensor that still has an autograd history; otherwise the
    # distillation term would backpropagate into the teacher graph.
    teacher = torch.as_tensor(
        teacher_risk, device=scores.device, dtype=scores.dtype
    ).detach().reshape(-1)
    if scores.numel() != teacher.numel() or scores.numel() < 1:
        raise ValueError("risk and teacher_risk must have equal non-empty length")
    if not torch.isfinite(scores).all() or not torch.isfinite(teacher).all():
        raise ValueError("risk and teacher_risk must be finite")
    if scores.numel() < 2:
        return scores.sum() * 0.0
    delta_teacher = teacher[:, None] - teacher[None, :]
    upper = torch.triu(torch.ones_like(delta_teacher, dtype=torch.bool), diagonal=1)
    comparable = upper & (delta_teacher != 0.0)
    if not bool(comparable.any()):
        return scores.sum() * 0.0
    delta_student = scores[:, None] - scores[None, :]
    signed_margin = delta_student[comparable] * torch.sign(delta_teacher[comparable])
    result = F.softplus(-signed_margin).mean()
    if not torch.isfinite(result):
        raise RuntimeError("teacher pairwise rank loss produced a non-finite value")
    return result


class LogNormalAFTLoss(nn.Module):
    """Module wrapper for :func:`lognormal_aft_loss`."""

    def forward(self, mu, sigma, true_times, true_events):
        return lognormal_aft_loss(mu, sigma, true_times, true_events)


class FPTSmoothL1Loss(nn.Module):
    """Module wrapper for :func:`fpt_smooth_l1_loss`."""

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = float(beta)

    def forward(self, mu, true_times, true_events):
        return fpt_smooth_l1_loss(
            mu, true_times, true_events, beta=self.beta
        )


class R13TeacherPairwiseRank(nn.Module):
    """Module wrapper for :func:`r13_teacher_pairwise_rank_loss`."""

    def forward(self, risk, teacher_risk):
        return r13_teacher_pairwise_rank_loss(risk, teacher_risk)


def cox_breslow_loss_and_logit_gradients(
    detached_full_lrisks,
    survival_times,
    event_indicators,
):
    """Compute full-risk-set loss and logit gradients for two-pass training.

    ``detached_full_lrisks`` should contain the first-pass logits for the entire
    risk set. The returned gradient tensor has the same shape and sample order.
    During a second, graph-building pass, each micro-batch can backpropagate
    ``(live_logits * logit_gradients_for_batch).sum()``. This yields the exact
    gradient of the full-risk-set Breslow loss without treating each micro-batch
    as an independent Cox risk set.

    Returns:
        ``(loss, logit_gradients)`` as detached tensors.
    """
    if not torch.is_tensor(detached_full_lrisks):
        raise TypeError("detached_full_lrisks must be a torch.Tensor")

    with torch.enable_grad():
        full_lrisks = detached_full_lrisks.detach().clone().requires_grad_(True)
        loss = partial_ll_loss(
            full_lrisks,
            survival_times,
            event_indicators,
        )['loss']
        logit_gradients, = torch.autograd.grad(loss, full_lrisks, create_graph=False)
    return loss.detach(), logit_gradients.detach()


def temporal_order_loss(lrisks, earlier_indices, later_indices):
    """Penalize a later acquisition whose log risk is not higher.

    This is an optional training-only inductive bias for a degradation
    trajectory.  ``earlier_indices`` and ``later_indices`` must be aligned
    pairs from the same bearing; construction and metadata validation belong
    to the caller.  The loss has no access to a failure time or future
    waveform, and it is never used as a model input at inference time.

    A softplus ranking penalty remains informative at ties and avoids the
    zero-gradient plateau of a hinge penalty.  It is intentionally expressed
    on full-risk-set logits so exact-Cox replay can apply its logit gradient
    without forming an invalid per-micro-batch temporal objective.
    """
    lrisks = _flatten_cox_input(lrisks, "lrisks")
    if not torch.is_floating_point(lrisks):
        raise TypeError("lrisks must be a floating-point tensor")
    if lrisks.numel() == 0:
        raise ValueError("temporal order loss requires at least one sample")
    if not torch.isfinite(lrisks).all():
        raise ValueError("lrisks must contain only finite values")

    earlier = torch.as_tensor(
        earlier_indices, device=lrisks.device, dtype=torch.long
    ).reshape(-1)
    later = torch.as_tensor(
        later_indices, device=lrisks.device, dtype=torch.long
    ).reshape(-1)
    if earlier.numel() == 0 or later.numel() == 0:
        raise ValueError("temporal order loss requires at least one pair")
    if earlier.shape != later.shape:
        raise ValueError("temporal order pair indices must have equal length")
    if (
        (earlier < 0).any()
        or (later < 0).any()
        or (earlier >= lrisks.numel()).any()
        or (later >= lrisks.numel()).any()
    ):
        raise ValueError("temporal order pair indices are out of bounds")
    if torch.equal(earlier, later) or (earlier == later).any():
        raise ValueError("temporal order pairs must contain distinct samples")

    later_minus_earlier = lrisks[later] - lrisks[earlier]
    penalties = torch.nn.functional.softplus(-later_minus_earlier)
    return {
        "loss": penalties.mean(),
        "pair_count": lrisks.new_tensor(float(earlier.numel())),
        "violation_rate": (later_minus_earlier <= 0.0).to(lrisks.dtype).mean(),
    }


def temporal_order_loss_and_logit_gradients(
    detached_full_lrisks,
    earlier_indices,
    later_indices,
):
    """Return full-trajectory temporal loss and exact logit gradients.

    This mirrors :func:`cox_breslow_loss_and_logit_gradients`: callers may
    replay the returned coefficients through memory-bounded forward passes
    while retaining the gradient of one full trajectory-level objective.
    """
    if not torch.is_tensor(detached_full_lrisks):
        raise TypeError("detached_full_lrisks must be a torch.Tensor")

    with torch.enable_grad():
        full_lrisks = detached_full_lrisks.detach().clone().requires_grad_(True)
        details = temporal_order_loss(
            full_lrisks,
            earlier_indices,
            later_indices,
        )
        logit_gradients, = torch.autograd.grad(
            details["loss"], full_lrisks, create_graph=False
        )
    return (
        details["loss"].detach(),
        logit_gradients.detach(),
        details["pair_count"].detach(),
        details["violation_rate"].detach(),
    )


class CoxLoss(nn.Module):
    """Cox negative partial log-likelihood using Breslow tie handling."""
    def __init__(self):
        super().__init__()

    def forward(self, logits, times, censorships):
        return partial_ll_loss(
            lrisks=logits,
            survival_times=times,
            event_indicators=(1 - censorships).to(dtype=logits.dtype),
        )


class SurvRankingLoss(nn.Module):
    """
    Implements the surivival ranking loss which approximates the negaive c-index; see Section 3.2 of (Luck et al, 2018) -- but be careful of the typo in their c-index formula.

    The c-index for risk scores z_1, ..., z_n is given by

    c_index = sum_{(a, b) are comparable} 1(z_a > z_b)

    where (a, b) are comparable if and only if a's event is observed and a has a strictly lower survival time than b. This ignores ties.

    We replace the indicator with a continous approximation

    1(z_a - z_b > 0 ) ~= phi(z_a - z_b)

    e.g. where phi(r) is a Relu or sigmoid function.

    The loss function we want to minimize is then

    - sum_{(a, b) are comparable} phi(z_a - z_b)

    where z_a, z_b are the risk scores output by the network.

    Parameters
    ----------
    phi: str
        Which indicator approximation to use. Must be one of ['relu', 'sigmoid'].

    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']

    References
    ----------
    Luck, M., Sylvain, T., Cohen, J.P., Cardinal, H., Lodi, A. and Bengio, Y., 2018. Learning to rank for censored survival data. arXiv preprint arXiv:1806.01984.
    """

    def __init__(self, phi='sigmoid', reduction='mean'):
        super().__init__()

        assert phi in ['sigmoid', 'relu']
        assert reduction in ['mean', 'sum']
        self.phi = phi
        self.reduction = reduction

    def forward(self, z, times, censorships):
        """
        Parameters
        ----------
        z: (batch_size, 1)
            The predicted risk scores.

        c_t: (batch_size, 2)
            first element: censorship
            second element: survival time
        """
        batch_size = z.shape[0]
        if batch_size == 1:
            # raise NotImplementedError("Batch size must be at least 2")
            return {'loss': torch.tensor(-1e5)}

        # censorship, times = c_t[:, 0], c_t[:, 1]
        events = 1 - censorships

        ##############################
        # determine comparable pairs #
        ##############################
        Z_more_risky = []
        Z_less_risky = []
        for (idx_a, idx_b) in combinations(range(batch_size), 2):
            time_a, event_a = times[idx_a], events[idx_a]
            time_b, event_b = times[idx_b], events[idx_b]

            if time_a < time_b and event_a:
                # a and b are comparable, a is more risky
                Z_more_risky.append(z[idx_a])
                Z_less_risky.append(z[idx_b])

            elif time_b < time_a and event_b:
                # a and b are comparable, b is more risky
                Z_more_risky.append(z[idx_b])
                Z_less_risky.append(z[idx_a])

        # if there are no comparable pairs then just return zero
        if len(Z_less_risky) == 0:
            # TODO: perhaps return None?
            return {'loss': None}

        Z_more_risky = torch.stack(Z_more_risky)
        Z_less_risky = torch.stack(Z_less_risky)

        # compute approximate c indices
        r = Z_more_risky - Z_less_risky
        if self.phi == 'sigmoid':
            approx_c_indices = torch.sigmoid(r)

        elif self.phi == 'relu':
            approx_c_indices = torch.relu(r)

        # negative mean/sum of c-indices
        if self.reduction == 'mean':
            return {'loss': - approx_c_indices.mean()}
        if self.reduction == 'sum':
            return {'loss': -approx_c_indices.sum()}
