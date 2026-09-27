"""IoU loss on decoded boxes. Port of ``mmdet.models.losses.iou_loss.IoULoss``.

FCOS regresses on the box itself rather than on deltas, so the loss compares
two sets of ``(x1, y1, x2, y2)`` boxes by their overlap: ``-log(IoU)`` by
default. mmdet's bounded / GIoU / DIoU / CIoU variants are not ported; no
config here uses them.
"""

from __future__ import annotations

import torch
from torch import nn

from vfe.core.bbox.iou import bbox_overlaps
from vfe.models.builder import LOSSES
from vfe.models.losses.utils import weighted_loss

__all__ = ["IoULoss", "iou_loss"]


@weighted_loss
def iou_loss(
    pred: torch.Tensor, target: torch.Tensor, mode: str = "log", eps: float = 1e-6
) -> torch.Tensor:
    """Elementwise IoU loss between aligned box sets ``(n, 4)``."""
    if mode not in ("linear", "square", "log"):
        raise ValueError(f"mode must be 'linear', 'square' or 'log', got {mode!r}")
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    if mode == "linear":
        return 1 - ious
    if mode == "square":
        return 1 - ious**2
    return -ious.log()


@LOSSES.register_module()
class IoULoss(nn.Module):
    """Args:
        mode: ``'log'`` (``-log IoU``), ``'linear'`` (``1 - IoU``) or ``'square'``.
        eps: floor on the IoU before the log.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        mode: str = "log",
    ):
        super().__init__()
        if mode not in ("linear", "square", "log"):
            raise ValueError(f"mode must be 'linear', 'square' or 'log', got {mode!r}")
        self.mode = mode
        self.eps = eps
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
        if weight is not None and not torch.any(weight > 0) and reduction != "none":
            # Every box weighted zero: a zero that still carries the graph.
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        if weight is not None and weight.dim() > 1:
            # A per-coordinate (n, 4) weight collapses to one per box.
            if weight.shape != pred.shape:
                raise ValueError("a 2-d weight must match pred's shape")
            weight = weight.mean(-1)
        return self.loss_weight * iou_loss(
            pred, target, weight, mode=self.mode, eps=self.eps, reduction=reduction,
            avg_factor=avg_factor, **kwargs,
        )

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}, eps={self.eps}, loss_weight={self.loss_weight}"
