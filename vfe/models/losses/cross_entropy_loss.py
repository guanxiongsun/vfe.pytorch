"""Classification loss. Port of ``mmdet.models.losses.cross_entropy_loss``.

The same class covers two quite different jobs, selected by ``use_sigmoid``:

* ``use_sigmoid=False`` -- softmax cross entropy over ``num_classes + 1``
  channels, used by the RoI head where the extra channel is background.
* ``use_sigmoid=True`` -- per-channel binary cross entropy, used by the RPN
  where there is a single objectness logit and "background" is just a 0 label.

mmdet's ``use_mask`` path (``mask_cross_entropy``, for Mask R-CNN) is not
ported; no config here has a mask head.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from vfe.models.builder import LOSSES
from vfe.models.losses.utils import weight_reduce_loss

__all__ = ["CrossEntropyLoss", "cross_entropy", "binary_cross_entropy"]


def cross_entropy(
    pred: torch.Tensor,
    label: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    avg_factor: float | None = None,
    class_weight: torch.Tensor | None = None,
    ignore_index: int | None = -100,
) -> torch.Tensor:
    """Softmax cross entropy over ``pred`` of shape ``(N, C)``."""
    # -100 is F.cross_entropy's own default sentinel.
    ignore_index = -100 if ignore_index is None else ignore_index
    loss = F.cross_entropy(
        pred, label, weight=class_weight, reduction="none", ignore_index=ignore_index
    )
    if weight is not None:
        weight = weight.float()
    return weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)


def _expand_onehot_labels(
    labels: torch.Tensor,
    label_weights: torch.Tensor | None,
    label_channels: int,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(N,)`` class indices -> ``(N, label_channels)`` one-hot, plus matching weights.

    Labels that are negative, equal to ``ignore_index``, or out of range get an
    all-zero row and zero weight -- that is how "ignore this anchor" is
    expressed once the loss is elementwise.
    """
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0), label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights = bin_label_weights * valid_mask

    return bin_labels, bin_label_weights


def binary_cross_entropy(
    pred: torch.Tensor,
    label: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    avg_factor: float | None = None,
    class_weight: torch.Tensor | None = None,
    ignore_index: int | None = -100,
) -> torch.Tensor:
    """Sigmoid BCE. ``label`` may be ``(N,)`` class indices or already ``(N, C)``."""
    ignore_index = -100 if ignore_index is None else ignore_index
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1), ignore_index)

    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction="none"
    )
    return weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """Configurable softmax or sigmoid classification loss.

    Args:
        use_sigmoid: use per-channel BCE instead of softmax cross entropy.
        reduction: ``'none'``, ``'mean'`` or ``'sum'``.
        class_weight: per-class rescaling, as ``F.cross_entropy``'s ``weight``.
        ignore_index: label value to skip. ``None`` means ``-100``.
        loss_weight: scalar multiplier on the reduced loss.
    """

    def __init__(
        self,
        use_sigmoid: bool = False,
        reduction: str = "mean",
        class_weight: Sequence[float] | None = None,
        ignore_index: int | None = None,
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.cls_criterion = binary_cross_entropy if use_sigmoid else cross_entropy

    def forward(
        self,
        cls_score: torch.Tensor,
        label: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: float | None = None,
        reduction_override: str | None = None,
        ignore_index: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if reduction_override not in (None, "none", "mean", "sum"):
            raise ValueError(f"bad reduction_override {reduction_override!r}")
        reduction = reduction_override or self.reduction
        if ignore_index is None:
            ignore_index = self.ignore_index

        class_weight = (
            None if self.class_weight is None else cls_score.new_tensor(self.class_weight)
        )
        return self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            **kwargs,
        )

    def extra_repr(self) -> str:
        return f"use_sigmoid={self.use_sigmoid}, loss_weight={self.loss_weight}"
