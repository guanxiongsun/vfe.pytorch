"""Sigmoid focal loss. Port of ``mmdet.models.losses.focal_loss``.

mmdet ran mmcv's compiled kernel on CUDA and this formula on CPU; the two agree
to float noise, so only the formula is kept. Like the rest of the one-stage
losses it takes ``(N, C)`` logits against ``(N,)`` class indices where
background is ``C`` (mmdet's label convention since v2.5), and turns the
indices into one-hot rows with an all-zero row for background.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vfe.models.builder import LOSSES
from vfe.models.losses.utils import weight_reduce_loss

__all__ = ["FocalLoss", "sigmoid_focal_loss"]


def sigmoid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean",
    avg_factor: float | None = None,
) -> torch.Tensor:
    """``pred`` logits ``(N, C)``, ``target`` one-hot ``(N, C)``."""
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # One weight per prior, shared across classes.
                weight = weight.view(-1, 1)
            else:
                if weight.numel() != loss.numel():
                    raise ValueError("weight must have one entry per prior or per element")
                weight = weight.view(loss.size(0), -1)
    return weight_reduce_loss(loss, weight, reduction, avg_factor)


@LOSSES.register_module()
class FocalLoss(nn.Module):
    """Args:
        use_sigmoid: only ``True`` exists (softmax focal loss is not in mmdet).
        gamma: exponent of the modulating factor.
        alpha: weight of the positive class.
    """

    def __init__(
        self,
        use_sigmoid: bool = True,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super().__init__()
        if not use_sigmoid:
            raise NotImplementedError("only the sigmoid focal loss exists")
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
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
        """``target`` is ``(N,)`` class indices (background = ``C``) or an
        already one-hot ``(N, C)`` tensor."""
        if reduction_override not in (None, "none", "mean", "sum"):
            raise ValueError(f"bad reduction_override {reduction_override!r}")
        reduction = reduction_override or self.reduction
        if target.dim() == 1:
            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1)[:, :num_classes]
        return self.loss_weight * sigmoid_focal_loss(
            pred, target, weight, gamma=self.gamma, alpha=self.alpha, reduction=reduction,
            avg_factor=avg_factor,
        )

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, alpha={self.alpha}, loss_weight={self.loss_weight}"
