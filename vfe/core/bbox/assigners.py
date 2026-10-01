"""Ground-truth assignment. Port of ``mmdet.core.bbox.assigners.MaxIoUAssigner``
and (for YOLOX) ``SimOTAAssigner``.

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
import torch.nn.functional as F

from vfe.core.bbox.iou import BboxOverlaps2D, bbox_overlaps
from vfe.core.builder import BBOX_ASSIGNERS, build_iou_calculator

__all__ = ["AssignResult", "MaxIoUAssigner", "SimOTAAssigner"]


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


@BBOX_ASSIGNERS.register_module()
class SimOTAAssigner:
    """YOLOX's SimOTA. Port of ``mmdet.core.bbox.assigners.SimOTAAssigner``.

    Candidates are the priors inside a ground-truth box or within
    ``center_radius`` strides of its centre. Each ground truth takes the
    ``k`` cheapest candidates, where the cost is the classification BCE of
    ``sqrt(score)`` plus ``iou_weight x -log IoU`` (plus a huge penalty
    outside box-and-centre) and ``k`` is the sum of its ``candidate_topk``
    best IoUs, at least 1. A prior claimed twice keeps its cheapest.

    ``max_overlaps`` of a positive is its IoU with the matched box, which the
    head uses as the classification target.
    """

    def __init__(self, center_radius: float = 2.5, candidate_topk: int = 10,
                 iou_weight: float = 3.0, cls_weight: float = 1.0):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

    def assign(self, pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels,
               gt_bboxes_ignore=None, eps: float = 1e-7) -> AssignResult:
        """``pred_scores (n, C)`` (sigmoid, times objectness), ``priors
        (n, 4)`` as ``(cx, cy, stride_w, stride_h)``, ``decoded_bboxes (n, 4)``."""
        try:
            return self._assign(pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels, eps)
        except torch.cuda.OutOfMemoryError:
            # mmdet's fallback: an image with very many ground truths can
            # exhaust the GPU; redo it on the CPU.
            device = pred_scores.device
            torch.cuda.empty_cache()
            result = self._assign(pred_scores.cpu(), priors.cpu(), decoded_bboxes.cpu(),
                                  gt_bboxes.cpu().float(), gt_labels.cpu(), eps)
            result.gt_inds = result.gt_inds.to(device)
            result.max_overlaps = result.max_overlaps.to(device)
            result.labels = result.labels.to(device)
            return result

    def _assign(self, pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels, eps):
        inf = 100000000
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(priors, gt_bboxes)
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = (None if gt_labels is None else
                               decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long))
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + eps)
        gt_onehot_label = F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1]).float() \
            .unsqueeze(0).repeat(num_valid, 1, 1)
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        cls_cost = F.binary_cross_entropy(valid_pred_scores.sqrt_(), gt_onehot_label,
                                          reduction="none").sum(-1)
        cost_matrix = (cls_cost * self.cls_weight + iou_cost * self.iou_weight
                       + (~is_in_boxes_and_center) * inf)
        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(
            cost_matrix, pairwise_ious, num_gt, valid_mask)

        # valid_mask now marks the matched priors only.
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -inf, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        """``(is_in_gts_or_centers (n,), is_in_boxes_and_centers (m, num_gt))``,
        the second over the priors the first selects."""
        num_gt = gt_bboxes.size(0)
        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        deltas = torch.stack([repeated_x - gt_bboxes[:, 0], repeated_y - gt_bboxes[:, 1],
                              gt_bboxes[:, 2] - repeated_x, gt_bboxes[:, 3] - repeated_y], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y
        ct_deltas = torch.stack([repeated_x - ct_box_l, repeated_y - ct_box_t,
                                 ct_box_r - repeated_x, ct_box_b - repeated_y], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all
        is_in_boxes_and_centers = (is_in_gts[is_in_gts_or_centers, :]
                                   & is_in_cts[is_in_gts_or_centers, :])
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        """Match each ground truth to its dynamic-k cheapest candidates;
        updates ``valid_mask`` in place to the matched priors."""
        matching_matrix = torch.zeros_like(cost)
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            _, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0.0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
