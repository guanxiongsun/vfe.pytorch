"""RoIAlign, mmcv-compatible, on top of ``torchvision.ops.roi_align``.

mmcv's ``aligned=True`` is the same half-pixel-shifted sampling as
torchvision's ``aligned=True`` (mmcv even documents it as the Detectron2
convention), so this is a straight delegation. Only ``pool_mode='avg'`` is
supported -- ``'max'`` has no torchvision equivalent and no config here uses it.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import roi_align as tv_roi_align

__all__ = ["roi_align", "RoIAlign"]


def roi_align(
    input: torch.Tensor,
    rois: torch.Tensor,
    output_size: int | tuple[int, int],
    spatial_scale: float = 1.0,
    sampling_ratio: int = 0,
    pool_mode: str = "avg",
    aligned: bool = True,
) -> torch.Tensor:
    """Drop-in for ``mmcv.ops.roi_align``.

    Args:
        input: ``(N, C, H, W)`` feature map.
        rois: ``(K, 5)`` of ``[batch_index, x1, y1, x2, y2]``.
        output_size: pooled ``(h, w)``.
        spatial_scale: multiplier mapping roi coords onto ``input`` coords.
        sampling_ratio: samples per bin per axis; ``0`` means ``ceil(roi_size/out_size)``.
        pool_mode: only ``'avg'``.
        aligned: half-pixel shift (Detectron2 convention). mmcv defaults to True.

    Returns:
        ``(K, C, h, w)``.
    """
    if pool_mode != "avg":
        raise NotImplementedError(f"pool_mode={pool_mode!r} is not ported; only 'avg' is used")
    h, w = _pair(output_size)
    return tv_roi_align(
        input,
        rois,
        output_size=(h, w),
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio,
        aligned=aligned,
    )


class RoIAlign(nn.Module):
    """Drop-in for ``mmcv.ops.RoIAlign``; see :func:`roi_align`."""

    def __init__(
        self,
        output_size: int | tuple[int, int],
        spatial_scale: float = 1.0,
        sampling_ratio: int = 0,
        pool_mode: str = "avg",
        aligned: bool = True,
        use_torchvision: bool = False,  # accepted for config compat; ignored
    ):
        super().__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.pool_mode = pool_mode
        self.aligned = aligned

    def forward(self, input: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        return roi_align(
            input,
            rois,
            self.output_size,
            self.spatial_scale,
            self.sampling_ratio,
            self.pool_mode,
            self.aligned,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(output_size={self.output_size}, "
            f"spatial_scale={self.spatial_scale}, sampling_ratio={self.sampling_ratio}, "
            f"pool_mode={self.pool_mode}, aligned={self.aligned})"
        )
