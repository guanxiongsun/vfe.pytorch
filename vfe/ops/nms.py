"""NMS, mmcv-compatible, on top of ``torchvision.ops.nms``."""

from __future__ import annotations

import torch
from torchvision.ops import nms as tv_nms

__all__ = ["nms", "batched_nms"]


def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    offset: int = 0,
    score_threshold: float = 0.0,
    max_num: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for ``mmcv.ops.nms``.

    Args:
        boxes: ``(N, 4)`` in ``(x1, y1, x2, y2)``.
        scores: ``(N,)``.
        iou_threshold: boxes with IoU above this against a kept box are dropped.
        offset: 0 or 1 -- box width is ``x2 - x1 + offset``. mmcv's legacy
            Pascal-VOC convention; torchvision only implements ``offset=0``, so
            ``offset=1`` is emulated by growing ``x2``/``y2`` by one, which
            gives an identical IoU matrix.
        score_threshold: pre-filter boxes scoring at or below this.
        max_num: keep at most this many boxes (``-1`` = unlimited).

    Returns:
        ``(dets, inds)`` where ``dets`` is ``(K, 5)`` of ``[x1, y1, x2, y2, score]``
        and ``inds`` indexes into the *original* ``boxes``.

    Unlike mmcv's, this does not accept numpy arrays -- no call site needs it.
    """
    if boxes.shape[-1] != 4:
        raise ValueError(f"boxes must be (N, 4), got {tuple(boxes.shape)}")
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(f"boxes/scores length mismatch: {boxes.shape[0]} vs {scores.shape[0]}")
    if offset not in (0, 1):
        raise ValueError(f"offset must be 0 or 1, got {offset}")

    valid_inds = None
    if score_threshold > 0:
        valid_mask = scores > score_threshold
        valid_inds = torch.nonzero(valid_mask, as_tuple=False).squeeze(dim=1)
        boxes, scores = boxes[valid_mask], scores[valid_mask]

    boxes_for_nms = boxes
    if offset == 1:
        boxes_for_nms = boxes + boxes.new_tensor([0.0, 0.0, 1.0, 1.0])

    inds = tv_nms(boxes_for_nms, scores, float(iou_threshold))

    if max_num > 0:
        inds = inds[:max_num]
    if valid_inds is not None:
        dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
        return dets, valid_inds[inds]

    dets = torch.cat((boxes[inds], scores[inds].reshape(-1, 1)), dim=1)
    return dets, inds


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    nms_cfg: dict | None,
    class_agnostic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for ``mmcv.ops.batched_nms``.

    NMS is applied independently per ``idxs`` group by shifting each group's
    boxes into a disjoint coordinate range. ``nms_cfg`` mirrors mmcv's: it may
    carry ``type`` (only ``'nms'`` is supported here -- this repo's configs use
    nothing else), ``iou_threshold``, ``class_agnostic``, ``max_num`` and
    ``split_thr``.

    ``split_thr`` exists because offsetting hundreds of thousands of boxes into
    one NMS call is an OOM risk; above the threshold we loop per group instead
    and re-sort by score at the end, exactly as mmcv does.
    """
    nms_cfg_ = dict(nms_cfg or {})
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)

    nms_type = nms_cfg_.pop("type", "nms")
    if nms_type != "nms":
        raise NotImplementedError(f"nms type {nms_type!r} is not ported (configs only use 'nms')")

    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

    split_thr = nms_cfg_.pop("split_thr", 10000)
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        # NMS may reweight scores (e.g. soft-NMS), so take them back from `dets`.
        scores = dets[:, 4]
    else:
        max_num = nms_cfg_.pop("max_num", -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        scores_after_nms = scores.new_zeros(scores.size())
        for group_id in torch.unique(idxs):
            mask = (idxs == group_id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    return torch.cat([boxes, scores[:, None]], -1), keep
