"""IoU / IoF / GIoU between box sets. Port of ``mmdet.core.bbox.iou_calculators``."""

from __future__ import annotations

import torch

from vfe.core.builder import IOU_CALCULATORS

__all__ = ["bbox_overlaps", "BboxOverlaps2D"]


def bbox_overlaps(
    bboxes1: torch.Tensor,
    bboxes2: torch.Tensor,
    mode: str = "iou",
    is_aligned: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Overlap between two box sets in ``(x1, y1, x2, y2)`` format.

    Args:
        bboxes1: ``(..., m, 4)``.
        bboxes2: ``(..., n, 4)``. Leading batch dims must match ``bboxes1``.
        mode: ``'iou'`` (over union), ``'iof'`` (over ``bboxes1``'s area), or
            ``'giou'`` (generalized IoU).
        is_aligned: pair up rows instead of taking the full cross product, in
            which case ``m == n`` and the result is ``(..., m)``.
        eps: floor on the denominator, so zero-area boxes give 0 rather than NaN.

    Returns:
        ``(..., m, n)``, or ``(..., m)`` when ``is_aligned``.
    """
    if mode not in ("iou", "iof", "giou"):
        raise ValueError(f"Unsupported mode {mode}")
    if not (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0):
        raise ValueError(f"bboxes1 must be (..., 4), got {tuple(bboxes1.shape)}")
    if not (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0):
        raise ValueError(f"bboxes2 must be (..., 4), got {tuple(bboxes2.shape)}")
    if bboxes1.shape[:-2] != bboxes2.shape[:-2]:
        raise ValueError(
            f"batch dims differ: {bboxes1.shape[:-2]} vs {bboxes2.shape[:-2]}"
        )
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned and rows != cols:
        raise ValueError(f"is_aligned needs equal counts, got {rows} and {cols}")

    if rows * cols == 0:
        shape = batch_shape + ((rows,) if is_aligned else (rows, cols))
        return bboxes1.new_empty(shape)

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        union = area1 + area2 - overlap if mode in ("iou", "giou") else area1
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
        rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ("iou", "giou"):
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])

    eps_t = union.new_tensor([eps])
    union = torch.max(union, eps_t)
    ious = overlap / union
    if mode in ("iou", "iof"):
        return ious

    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
    enclose_area = torch.max(enclose_wh[..., 0] * enclose_wh[..., 1], eps_t)
    return ious - (enclose_area - union) / enclose_area


@IOU_CALCULATORS.register_module()
class BboxOverlaps2D:
    """Callable wrapper around :func:`bbox_overlaps`, so assigners can be configured.

    mmdet's version also has an fp16 path that downcasts to save memory on
    dense detectors; dropped, since nothing here runs at a scale that needs it.
    """

    def __call__(
        self,
        bboxes1: torch.Tensor,
        bboxes2: torch.Tensor,
        mode: str = "iou",
        is_aligned: bool = False,
    ) -> torch.Tensor:
        # Tolerate (x1, y1, x2, y2, score) so proposals can be passed straight in.
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
