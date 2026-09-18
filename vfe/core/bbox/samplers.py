"""Positive/negative sampling. Port of ``mmdet.core.bbox.samplers``.

An image has orders of magnitude more negatives than positives, so the heads
train on a fixed-size subset: ``num`` boxes, at most ``pos_fraction`` of them
positive, the rest negative.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

import torch

from vfe.core.bbox.assigners import AssignResult
from vfe.core.builder import BBOX_SAMPLERS

__all__ = ["SamplingResult", "BaseSampler", "RandomSampler", "PseudoSampler"]


class SamplingResult:
    """The sampled boxes and what they were matched to.

    ``pos_assigned_gt_inds`` is zero-based here -- the one-based convention of
    :class:`AssignResult` ends at this boundary.
    """

    def __init__(
        self,
        pos_inds: torch.Tensor,
        neg_inds: torch.Tensor,
        bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        assign_result: AssignResult,
        gt_flags: torch.Tensor,
    ):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # An image with no annotations still needs a well-shaped (0, 4).
            if self.pos_assigned_gt_inds.numel() != 0:
                raise AssertionError("positives assigned despite there being no gt boxes")
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]

        self.pos_gt_labels = (
            None if assign_result.labels is None else assign_result.labels[pos_inds]
        )

    @property
    def bboxes(self) -> torch.Tensor:
        """Positives then negatives, in that order -- the heads rely on it."""
        return torch.cat([self.pos_bboxes, self.neg_bboxes])

    def to(self, device: Any) -> SamplingResult:
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__dict__[key] = value.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"<SamplingResult(num_gts={self.num_gts}, "
            f"num_pos={len(self.pos_inds)}, num_neg={len(self.neg_inds)})>"
        )


class BaseSampler(metaclass=ABCMeta):
    """Draw ``num`` boxes, splitting the budget between positives and negatives."""

    def __init__(
        self,
        num: int,
        pos_fraction: float,
        neg_pos_ub: float = -1,
        add_gt_as_proposals: bool = True,
        **kwargs: Any,
    ):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals

    @abstractmethod
    def _sample_pos(self, assign_result: AssignResult, num_expected: int, **kwargs: Any):
        """Pick at most ``num_expected`` indices from the positives."""

    @abstractmethod
    def _sample_neg(self, assign_result: AssignResult, num_expected: int, **kwargs: Any):
        """Pick at most ``num_expected`` indices from the negatives."""

    def sample(
        self,
        assign_result: AssignResult,
        bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SamplingResult:
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]
        # Proposals arrive as (x1, y1, x2, y2, score); drop the score.
        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0],), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError("gt_labels must be given when add_gt_as_proposals is True")
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs
        )
        # mmdet found sampled indices occasionally repeat; unique() also sorts,
        # which the RoI head's ordering assumptions quietly depend on.
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()

        # Negatives take whatever the positives left, capped by neg_pos_ub.
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            neg_upper_bound = int(self.neg_pos_ub * max(1, num_sampled_pos))
            num_expected_neg = min(num_expected_neg, neg_upper_bound)
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs
        ).unique()

        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)

    # mmdet allows a sampler to delegate its two halves to different objects
    # (OHEM does). Nothing here does, so both point back at self.
    @property
    def pos_sampler(self) -> BaseSampler:
        return self

    @property
    def neg_sampler(self) -> BaseSampler:
        return self


@BBOX_SAMPLERS.register_module()
class RandomSampler(BaseSampler):
    """Uniform random subsampling of positives and negatives."""

    def random_choice(self, gallery: torch.Tensor, num: int) -> torch.Tensor:
        """``num`` distinct entries drawn uniformly from ``gallery``."""
        if len(gallery) < num:
            raise ValueError(f"cannot draw {num} from a gallery of {len(gallery)}")
        # randperm on CPU then move: mmdet's workaround for a torch bug where the
        # CUDA randperm returned out-of-range values. Also makes the draw
        # reproducible from the CPU RNG, which the parity harness depends on.
        perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
        return gallery[perm]

    def _sample_pos(
        self, assign_result: AssignResult, num_expected: int, **kwargs: Any
    ) -> torch.Tensor:
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        return self.random_choice(pos_inds, num_expected)

    def _sample_neg(
        self, assign_result: AssignResult, num_expected: int, **kwargs: Any
    ) -> torch.Tensor:
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        return self.random_choice(neg_inds, num_expected)


@BBOX_SAMPLERS.register_module()
class PseudoSampler(BaseSampler):
    """Keep every assigned box. Used when the head does its own balancing."""

    def __init__(self, **kwargs: Any):
        super().__init__(num=-1, pos_fraction=-1, add_gt_as_proposals=False)

    def _sample_pos(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def _sample_neg(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def sample(
        self,
        assign_result: AssignResult,
        bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SamplingResult:
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        )
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        return SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
