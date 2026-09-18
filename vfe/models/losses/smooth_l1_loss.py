"""Box regression losses. Port of ``mmdet.models.losses.smooth_l1_loss``.

Smooth L1 is quadratic within ``beta`` of the target and linear outside it, so
a badly-placed anchor contributes a bounded gradient instead of dominating the
batch. The RPN uses ``beta=1/9`` (deltas are unnormalised, so errors are small)
and the RoI head ``beta=1`` (deltas are divided by ``target_stds=0.2``, so
errors are ~5x larger).
"""

from __future__ import annotations

import torch
from torch import nn

from vfe.models.builder import LOSSES
from vfe.models.losses.utils import weighted_loss

__all__ = ["SmoothL1Loss", "L1Loss", "smooth_l1_loss", "l1_loss"]


@weighted_loss
def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if target.numel() == 0:
        # Not `pred.new_zeros(...)`: multiplying by 0 keeps pred in the graph, so
        # an image with no positives still produces a gradient-connected zero
        # rather than a tensor DDP will flag as an unused parameter.
        return pred.sum() * 0

    if pred.size() != target.size():
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    diff = torch.abs(pred - target)
    return torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)


@weighted_loss
def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.numel() == 0:
        return pred.sum() * 0
    if pred.size() != target.size():
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    return torch.abs(pred - target)


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    """Args: ``beta`` the quadratic/linear crossover, ``reduction``, ``loss_weight``."""

    def __init__(self, beta: float = 1.0, reduction: str = "mean", loss_weight: float = 1.0):
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: float | None = None,
        reduction_override: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if reduction_override not in (None, "none", "mean", "sum"):
            raise ValueError(f"bad reduction_override {reduction_override!r}")
        reduction = reduction_override or self.reduction
        return self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )

    def extra_repr(self) -> str:
        return f"beta={self.beta}, loss_weight={self.loss_weight}"


@LOSSES.register_module()
class L1Loss(nn.Module):
    """Plain L1. Not used by the VID configs, but cheap to keep alongside."""

    def __init__(self, reduction: str = "mean", loss_weight: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: float | None = None,
        reduction_override: str | None = None,
    ) -> torch.Tensor:
        if reduction_override not in (None, "none", "mean", "sum"):
            raise ValueError(f"bad reduction_override {reduction_override!r}")
        reduction = reduction_override or self.reduction
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )

    def extra_repr(self) -> str:
        return f"loss_weight={self.loss_weight}"
