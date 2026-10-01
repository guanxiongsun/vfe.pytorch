"""Box format conversions. Port of ``mmdet.core.bbox.transforms``."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

__all__ = [
    "bbox_flip",
    "bbox_mapping",
    "bbox_mapping_back",
    "bbox2roi",
    "roi2bbox",
    "bbox2result",
    "distance2bbox",
    "bbox2distance",
    "bbox_xyxy_to_cxcywh",
]


def distance2bbox(
    points: torch.Tensor, distance: torch.Tensor, max_shape: Sequence[int] | None = None
) -> torch.Tensor:
    """Points ``(N, 2)`` plus ``(left, top, right, bottom)`` distances ``(N, 4)``
    -> boxes ``(N, 4)``, clipped to ``max_shape`` ``(H, W[, C])`` if given.

    The anchor-free decode: FCOS predicts how far each box edge is from the
    point that predicts it.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    bboxes = torch.stack([x1, y1, x2, y2], -1)
    if max_shape is not None:
        if bboxes.dim() != 2:
            raise ValueError("batched decoding with max_shape is not ported")
        bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
        bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
    return bboxes


def bbox2distance(
    points: torch.Tensor, bbox: torch.Tensor, max_dis: float | None = None, eps: float = 0.1
) -> torch.Tensor:
    """Inverse of :func:`distance2bbox`; with ``max_dis`` the distances are
    clamped to ``[0, max_dis - eps]``."""
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


def bbox_flip(
    bboxes: torch.Tensor, img_shape: Sequence[int], direction: str = "horizontal"
) -> torch.Tensor:
    """Mirror boxes within ``img_shape``. ``bboxes`` is ``(..., 4*k)``."""
    if bboxes.shape[-1] % 4 != 0:
        raise ValueError(f"last dim must be a multiple of 4, got {bboxes.shape[-1]}")
    if direction not in ("horizontal", "vertical", "diagonal"):
        raise ValueError(f"unknown flip direction {direction!r}")
    flipped = bboxes.clone()
    if direction in ("horizontal", "diagonal"):
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
    if direction in ("vertical", "diagonal"):
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    return flipped


def bbox_mapping(
    bboxes: torch.Tensor,
    img_shape: Sequence[int],
    scale_factor: float | Sequence[float],
    flip: bool,
    flip_direction: str = "horizontal",
) -> torch.Tensor:
    """Original image coordinates -> network input coordinates."""
    new_bboxes = bboxes * bboxes.new_tensor(scale_factor)
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape, flip_direction)
    return new_bboxes


def bbox_mapping_back(
    bboxes: torch.Tensor,
    img_shape: Sequence[int],
    scale_factor: float | Sequence[float],
    flip: bool,
    flip_direction: str = "horizontal",
) -> torch.Tensor:
    """Network input coordinates -> original image coordinates."""
    new_bboxes = bbox_flip(bboxes, img_shape, flip_direction) if flip else bboxes
    new_bboxes = new_bboxes.view(-1, 4) / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def bbox2roi(bbox_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """Per-image box lists -> one ``(n, 5)`` tensor of ``[batch_ind, x1, y1, x2, y2]``.

    This is the format ``roi_align`` wants: the batch index rides along in
    column 0 so boxes from every image can be cropped in a single call.
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    return torch.cat(rois_list, 0)


def roi2bbox(rois: torch.Tensor) -> list[torch.Tensor]:
    """Inverse of :func:`bbox2roi`, splitting on the batch-index column."""
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = rois[:, 0] == img_id.item()
        bbox_list.append(rois[inds, 1:])
    return bbox_list


def bbox2result(
    bboxes: torch.Tensor | np.ndarray, labels: torch.Tensor | np.ndarray, num_classes: int
) -> list[np.ndarray]:
    """Detections -> one ``(n_i, 5)`` numpy array per class, which is what eval expects."""
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes)]
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    return [bboxes[labels == i, :] for i in range(num_classes)]


def bbox_xyxy_to_cxcywh(bbox: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` corners -> centre, width, height."""
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    return torch.cat([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], dim=-1)
