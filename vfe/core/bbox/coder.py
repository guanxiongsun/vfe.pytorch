"""R-CNN box parameterisation. Port of ``mmdet.core.bbox.coder.delta_xywh_bbox_coder``.

Boxes are encoded relative to a reference box (anchor or proposal) as
``(dx, dy, dw, dh)``: centre offsets in units of the reference's width/height,
and log-space size ratios. The deltas are then whitened by ``target_means`` /
``target_stds`` so the regression head sees roughly unit-scale targets.

mmdet's ONNX batch-decoding path (``onnx_delta2bbox``) is not ported -- it was
already deprecated in 2.19 for non-ONNX use, and nothing here exports to ONNX.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from vfe.core.builder import BBOX_CODERS

__all__ = ["DeltaXYWHBBoxCoder", "bbox2delta", "delta2bbox"]


def bbox2delta(
    proposals: torch.Tensor,
    gt: torch.Tensor,
    means: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    stds: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Regression targets taking ``proposals`` to ``gt``. Inverse of :func:`delta2bbox`."""
    if proposals.size() != gt.size():
        raise ValueError(f"shape mismatch: {tuple(proposals.shape)} vs {tuple(gt.shape)}")

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    deltas = torch.stack(
        [(gx - px) / pw, (gy - py) / ph, torch.log(gw / pw), torch.log(gh / ph)], dim=-1
    )
    means_t = deltas.new_tensor(means).unsqueeze(0)
    stds_t = deltas.new_tensor(stds).unsqueeze(0)
    return (deltas - means_t) / stds_t


def delta2bbox(
    rois: torch.Tensor,
    deltas: torch.Tensor,
    means: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    stds: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    max_shape: Sequence[int] | None = None,
    wh_ratio_clip: float = 16 / 1000,
    clip_border: bool = True,
    add_ctr_clamp: bool = False,
    ctr_clamp: int = 32,
) -> torch.Tensor:
    """Apply ``deltas`` to ``rois``. Inverse of :func:`bbox2delta`.

    Args:
        rois: ``(N, 4)`` reference boxes.
        deltas: ``(N, 4)`` or ``(N, num_classes * 4)`` for class-specific regression.
        max_shape: ``(H, W)``; with ``clip_border`` the result is clamped into it.
        wh_ratio_clip: caps ``|dw|``/``|dh|`` at ``|log(wh_ratio_clip)|``, so a
            wild prediction cannot produce an ``exp`` overflow.
        add_ctr_clamp, ctr_clamp: YOLOF's extra centre-shift clamp; unused here.

    Returns:
        Same shape as ``deltas``.
    """
    num_bboxes, num_classes = deltas.size(0), deltas.size(1) // 4
    if num_bboxes == 0:
        return deltas

    deltas = deltas.reshape(-1, 4)
    means_t = deltas.new_tensor(means).view(1, -1)
    stds_t = deltas.new_tensor(stds).view(1, -1)
    denorm_deltas = deltas * stds_t + means_t

    dxy = denorm_deltas[:, :2]
    dwh = denorm_deltas[:, 2:]

    rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
    pxy = (rois_[:, :2] + rois_[:, 2:]) * 0.5
    pwh = rois_[:, 2:] - rois_[:, :2]

    dxy_wh = pwh * dxy

    max_ratio = abs(math.log(wh_ratio_clip))
    if add_ctr_clamp:
        dxy_wh = torch.clamp(dxy_wh, max=ctr_clamp, min=-ctr_clamp)
        dwh = torch.clamp(dwh, max=max_ratio)
    else:
        dwh = dwh.clamp(min=-max_ratio, max=max_ratio)

    gxy = pxy + dxy_wh
    gwh = pwh * dwh.exp()
    bboxes = torch.cat([gxy - gwh * 0.5, gxy + gwh * 0.5], dim=-1)
    if clip_border and max_shape is not None:
        bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
        bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
    return bboxes.reshape(num_bboxes, -1)


@BBOX_CODERS.register_module()
class DeltaXYWHBBoxCoder:
    """Configurable front end for :func:`bbox2delta` / :func:`delta2bbox`."""

    def __init__(
        self,
        target_means: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        target_stds: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        clip_border: bool = True,
        add_ctr_clamp: bool = False,
        ctr_clamp: int = 32,
    ):
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp

    def encode(self, bboxes: torch.Tensor, gt_bboxes: torch.Tensor) -> torch.Tensor:
        if bboxes.size(0) != gt_bboxes.size(0):
            raise ValueError(f"count mismatch: {bboxes.size(0)} vs {gt_bboxes.size(0)}")
        if bboxes.size(-1) != 4 or gt_bboxes.size(-1) != 4:
            raise ValueError("encode expects (..., 4) boxes")
        return bbox2delta(bboxes, gt_bboxes, self.means, self.stds)

    def decode(
        self,
        bboxes: torch.Tensor,
        pred_bboxes: torch.Tensor,
        max_shape: Sequence[int] | None = None,
        wh_ratio_clip: float = 16 / 1000,
    ) -> torch.Tensor:
        if pred_bboxes.size(0) != bboxes.size(0):
            raise ValueError(f"count mismatch: {pred_bboxes.size(0)} vs {bboxes.size(0)}")
        if pred_bboxes.ndim != 2:
            raise ValueError(
                f"decode expects 2D deltas, got {pred_bboxes.ndim}D. mmdet's batched "
                "(B, N, 4) path existed only for ONNX export and is not ported."
            )
        return delta2bbox(
            bboxes,
            pred_bboxes,
            self.means,
            self.stds,
            max_shape,
            wh_ratio_clip,
            self.clip_border,
            self.add_ctr_clamp,
            self.ctr_clamp,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(target_means={self.means}, "
            f"target_stds={self.stds}, clip_border={self.clip_border})"
        )
