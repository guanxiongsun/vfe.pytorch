"""Top-k accuracy, reported alongside the RoI head's loss. Port of
``mmdet.models.losses.accuracy``."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["accuracy", "Accuracy"]


def accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    topk: int | tuple[int, ...] = 1,
    thresh: float | None = None,
) -> torch.Tensor | list[torch.Tensor]:
    """Percentage of ``pred`` rows whose top-k predictions include ``target``.

    Args:
        pred: ``(N, num_classes)`` scores.
        target: ``(N,)`` class indices.
        topk: single int, or a tuple to get several cutoffs at once.
        thresh: if set, a prediction scoring below it never counts as correct.

    Returns:
        A scalar tensor, or a list of them if ``topk`` is a tuple.
    """
    return_single = isinstance(topk, int)
    topk_tuple = (topk,) if return_single else tuple(topk)

    maxk = max(topk_tuple)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.0) for _ in topk_tuple]
        return accu[0] if return_single else accu
    if pred.ndim != 2 or target.ndim != 1:
        raise ValueError(f"expected (N, C) and (N,), got {pred.shape} and {target.shape}")
    if pred.size(0) != target.size(0):
        raise ValueError(f"count mismatch: {pred.size(0)} vs {target.size(0)}")
    if maxk > pred.size(1):
        raise ValueError(f"maxk {maxk} exceeds pred dimension {pred.size(1)}")

    pred_value, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()  # (maxk, N)
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    if thresh is not None:
        correct = correct & (pred_value > thresh).t()

    res = [
        correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / pred.size(0))
        for k in topk_tuple
    ]
    return res[0] if return_single else res


class Accuracy(nn.Module):
    def __init__(self, topk: tuple[int, ...] = (1,), thresh: float | None = None):
        super().__init__()
        self.topk = topk
        self.thresh = thresh

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        return accuracy(pred, target, self.topk, self.thresh)
