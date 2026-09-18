"""Detection post-processing. Port of ``mmdet.core.post_processing.bbox_nms``.

mmdet's ONNX/TensorRT branches are dropped, as are ``fast_nms`` (YOLACT) and
``merge_aug_*`` (test-time augmentation) -- no config here uses them.
"""

from __future__ import annotations

import torch

from vfe.ops import batched_nms

__all__ = ["multiclass_nms"]


def multiclass_nms(
    multi_bboxes: torch.Tensor,
    multi_scores: torch.Tensor,
    score_thr: float,
    nms_cfg: dict,
    max_num: int = -1,
    score_factors: torch.Tensor | None = None,
    return_inds: bool = False,
):
    """Score-threshold then class-wise NMS.

    Args:
        multi_bboxes: ``(n, 4)`` shared across classes, or ``(n, num_classes * 4)``
            for class-specific regression.
        multi_scores: ``(n, num_classes + 1)``. The **last** column is
            background and is dropped -- an off-by-one here silently shifts
            every predicted label.
        score_thr: drop boxes scoring at or below this, per class.
        nms_cfg: e.g. ``dict(type='nms', iou_threshold=0.5)``.
        max_num: keep at most this many detections overall. ``-1`` for no cap.
        score_factors: ``(n,)`` multiplied into the scores *after* thresholding
            (mmdet found applying it before costs ~1% mAP on YOLOv3).
        return_inds: also return, for each kept detection, its index into the
            flattened ``(n * num_classes,)`` candidate list.

    Returns:
        ``(dets, labels)``, or ``(dets, labels, inds)``. ``dets`` is ``(k, 5)``
        as ``[x1, y1, x2, y2, score]``; ``labels`` is 0-based and excludes
        background.
    """
    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]
    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    valid_mask = scores > score_thr
    if score_factors is not None:
        score_factors = score_factors.view(-1, 1).expand(multi_scores.size(0), num_classes)
        scores = scores * score_factors.reshape(-1)

    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    if bboxes.numel() == 0:
        dets = torch.cat([bboxes, scores[:, None]], -1)
        return (dets, labels, inds) if return_inds else (dets, labels)

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], inds[keep]
    return dets, labels[keep]
