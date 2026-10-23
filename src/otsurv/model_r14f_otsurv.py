"""R14-F joint-training dual-endpoint container.

R14-F keeps the R13 ranking endpoint and the R14-B AFT time endpoint as
separate modules.  Unlike R14-E, the R13 branch remains trainable throughout
the fixed-budget fit.  Its current risk is exposed to R14-B only as a detached
teacher target; there is no reverse gradient path from the time loss.
"""

from __future__ import annotations

import torch
from torch import nn

from .model_r13_otsurv import R13OTSurv
from .model_r14b_otsurv import R14BOTSurv


class R14FOTSurv(nn.Module):
    """Compose a trainable R13 rank branch with an R14-B time branch."""

    MODE = "r14f"

    def __init__(self, rank_anchor: R13OTSurv, time_expert: R14BOTSurv) -> None:
        super().__init__()
        if not isinstance(rank_anchor, R13OTSurv):
            raise TypeError("R14-F rank_anchor must be an R13OTSurv instance")
        if not isinstance(time_expert, R14BOTSurv):
            raise TypeError("R14-F time_expert must be an R14BOTSurv instance")
        self.rank_anchor = rank_anchor
        self.time_expert = time_expert
        self.mode = self.MODE
        self.native_risk_score_definition_ = "trainable_r13_cox_log_risk"
        self.survival_curve_definition_ = "r14b_lognormal_aft_time_expert"

    def forward(
        self,
        raw_signals,
        bearing_ids,
        time_steps,
        *,
        return_components: bool = False,
        evaluation_times=None,
    ):
        """Return both endpoint outputs when the branches share a device.

        The R14-F trainer can deliberately place the two branches on different
        GPUs.  In that setup callers must use branch-specific prediction
        methods on the regressor, because an ordinary module forward has only
        one input-device contract.
        """

        rank_device = next(self.rank_anchor.parameters()).device
        time_device = next(self.time_expert.parameters()).device
        if rank_device != time_device:
            raise RuntimeError(
                "multi-device R14-F requires branch-specific regressor prediction"
            )
        rank_output = self.rank_anchor(
            raw_signals,
            bearing_ids,
            time_steps,
            return_components=return_components,
        )
        time_output = self.time_expert(
            raw_signals,
            bearing_ids,
            time_steps,
            return_components=return_components,
            evaluation_times=evaluation_times,
        )
        output = {
            "logits": rank_output["logits"],
            "risk": rank_output["logits"],
            "r13_anchor_risk": rank_output["logits"],
            "time_mu": time_output["mu"],
            "time_sigma": time_output["sigma"],
            "time_expert_risk": time_output["risk"],
        }
        if "survival" in time_output:
            output["survival"] = time_output["survival"]
        if return_components:
            output["rank_anchor"] = rank_output
            output["time_expert"] = time_output
        return output
