"""Ground-truth assignment. Port of ``mmdet.core.bbox.assigners.MaxIoUAssigner``.

An assignment labels every candidate box (anchor or proposal) with one of:

* ``-1`` -- ignored, counts as neither positive nor negative
* ``0``  -- negative (background)
* ``i+1`` -- positive, matched to ground truth ``i``

The off-by-one is mmdet's: index 0 is reserved for "background", so gt indices
are stored one-based throughout and only decremented when boxes are gathered.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vfe.core.bbox.iou import BboxOverlaps2D
from vfe.core.builder import BBOX_ASSIGNERS, build_iou_calculator

__all__ = ["AssignResult", "MaxIoUAssigner"]


class AssignResult:
    """Outcome of assigning ground truths to candidate boxes.

    Attributes:
        num_gts: number of ground-truth boxes considered.
        gt_inds: ``(n,)`` one-based gt index per candidate, or 0 / -1 (see module docs).
        max_overlaps: ``(n,)`` IoU of each candidate with its best ground truth.
        labels: ``(n,)`` class label per candidate, ``-1`` where not positive.
    """

    def __init__(
        self,
        num_gts: int,
        gt_inds: torch.Tensor,
        max_overlaps: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels

    @property
    def num_preds(self) -> int:
        return len(self.gt_inds)

    def add_gt_(self, gt_labels: torch.Tensor) -> None:
        """Prepend the ground truths themselves as candidates, each matched to itself.

        Used by the RoI head's sampler (``add_gt_as_proposals=True``) so that
        early in training, when the RPN proposes nothing useful, there is still
        a positive to learn from.
        """
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device
        )
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps]
        )
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])

    def __repr__(self) -> str:
        return (
            f"<AssignResult(num_gts={self.num_gts}, num_preds={self.num_preds}, "
            f"num_pos={int((self.gt_inds > 0).sum())})>"
        )


@BBOX_ASSIGNERS.register_module()
class MaxIoUAssigner:
    """Assign by IoU with a positive/negative threshold pair.

    Args:
        pos_iou_thr: candidates whose best IoU is at least this are positive.
        neg_iou_thr: float, or a ``(low, high)`` band; candidates inside it are
            negative. Everything between ``neg_iou_thr`` and ``pos_iou_thr``
            stays ``-1`` and is ignored.
        min_pos_iou: floor for the low-quality match in step 4 below.
        gt_max_assign_all: when a ground truth's best IoU is tied across several
            candidates, take all of them rather than the first.
        ignore_iof_thr: candidates overlapping an ignore-region by more than
            this (as IoF) are forced to ``-1``. ``-1`` disables.
        ignore_wrt_candidates: measure that IoF over the candidate's area
            rather than the ignore region's.
        match_low_quality: run step 4. On by default for the RPN; mmdet turns it
            off for RoI heads because it can reassign a candidate away from its
            own best ground truth.

    Assignment runs in order, each step able to overwrite the last:

    1. everything starts at ``-1``
    2. best IoU below ``neg_iou_thr`` -> 0
    3. best IoU at or above ``pos_iou_thr`` -> that ground truth
    4. (``match_low_quality``) every ground truth claims its own best
       candidate, provided that IoU clears ``min_pos_iou`` -- so a small or
       oddly-shaped object still gets at least one positive.
    """

    def __init__(
        self,
        pos_iou_thr: float,
        neg_iou_thr: float | Sequence[float],
        min_pos_iou: float = 0.0,
        gt_max_assign_all: bool = True,
        ignore_iof_thr: float = -1,
        ignore_wrt_candidates: bool = True,
        match_low_quality: bool = True,
        gpu_assign_thr: int = -1,
        iou_calculator: dict | None = None,
    ):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.match_low_quality = match_low_quality
        # mmdet falls back to CPU above `gpu_assign_thr` ground truths to bound the
        # (num_gts x num_bboxes) overlap matrix. Kept so config parity holds, but
        # ImageNet VID never has enough boxes per frame to trigger it.
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = (
            BboxOverlaps2D() if iou_calculator is None else build_iou_calculator(iou_calculator)
        )

    def assign(
        self,
        bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_bboxes_ignore: torch.Tensor | None = None,
        gt_labels: torch.Tensor | None = None,
    ) -> AssignResult:
        assign_on_cpu = 0 < self.gpu_assign_thr < gt_bboxes.shape[0]
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        if (
            self.ignore_iof_thr > 0
            and gt_bboxes_ignore is not None
            and gt_bboxes_ignore.numel() > 0
            and bboxes.numel() > 0
        ):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(bboxes, gt_bboxes_ignore, mode="iof")
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(gt_bboxes_ignore, bboxes, mode="iof")
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            # -1 here makes step 2's `max_overlaps >= 0` test skip these columns.
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_overlaps(
        self, overlaps: torch.Tensor, gt_labels: torch.Tensor | None = None
    ) -> AssignResult:
        """Assign from a precomputed ``(num_gts, num_bboxes)`` overlap matrix."""
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        assigned_gt_inds = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No ground truth at all: every candidate is background, not ignored.
                assigned_gt_inds[:] = 0
            assigned_labels = (
                None if gt_labels is None else overlaps.new_full((num_bboxes,), -1, dtype=torch.long)
            )
            return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # For each candidate, its best gt; for each gt, its best candidate.
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. negatives
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            if len(self.neg_iou_thr) != 2:
                raise ValueError(f"neg_iou_thr tuple must have 2 entries, got {self.neg_iou_thr}")
            assigned_gt_inds[
                (max_overlaps >= self.neg_iou_thr[0]) & (max_overlaps < self.neg_iou_thr[1])
            ] = 0

        # 3. positives
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. low-quality matches
        if self.match_low_quality:
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
