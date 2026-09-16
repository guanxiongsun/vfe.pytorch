"""Loss weighting and reduction. Port of ``mmdet.models.losses.utils``.

Detection losses are computed elementwise, then weighted per sample (to mask
out ignored anchors, or to zero the regression loss on negatives), then
reduced. ``avg_factor`` exists because the natural denominator is usually not
the element count: RPN divides by the number of *sampled* anchors, the RoI head
by the number of *positives*, and the elementwise tensor is neither.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch

__all__ = ["reduce_loss", "weight_reduce_loss", "weighted_loss"]


def reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"reduction must be 'none', 'mean' or 'sum', got {reduction!r}")


def weight_reduce_loss(
    loss: torch.Tensor,
    weight: torch.Tensor | None = None,
    reduction: str = "mean",
    avg_factor: float | None = None,
) -> torch.Tensor:
    """Apply ``weight`` elementwise, then reduce.

    With ``avg_factor`` set and ``reduction='mean'``, the denominator is
    ``avg_factor`` rather than ``loss.numel()``.
    """
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        return reduce_loss(loss, reduction)
    if reduction == "mean":
        return loss.sum() / avg_factor
    if reduction == "none":
        return loss
    raise ValueError('avg_factor cannot be used with reduction="sum"')


def weighted_loss(loss_func: Callable) -> Callable:
    """Give an elementwise ``loss_func(pred, target, **kwargs)`` the standard
    ``weight`` / ``reduction`` / ``avg_factor`` arguments.

    >>> @weighted_loss
    ... def l1_loss(pred, target):
    ...     return (pred - target).abs()
    >>> l1_loss(torch.tensor([0., 2., 3.]), torch.ones(3),
    ...         weight=torch.tensor([1., 0., 1.]))
    tensor(1.)
    """

    @functools.wraps(loss_func)
    def wrapper(
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
        avg_factor: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        loss = loss_func(pred, target, **kwargs)
        return weight_reduce_loss(loss, weight, reduction, avg_factor)

    return wrapper
