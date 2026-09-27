"""Grid points for anchor-free heads. Port of
``mmdet.core.anchor.point_generator.MlvlPointGenerator``.

Where an anchor generator lays several boxes on every feature-map cell, a point
generator lays one point at the cell centre, ``(i + offset) * stride``. FCOS
regresses each box as four distances from such a point.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.nn.modules.utils import _pair

from .builder import PRIOR_GENERATORS

__all__ = ["MlvlPointGenerator"]


@PRIOR_GENERATORS.register_module()
class MlvlPointGenerator:
    """Args:
        strides: one stride per level, an int or an ``(w, h)`` pair.
        offset: where in the cell the point sits, as a fraction of the stride.
    """

    def __init__(self, strides: Sequence[int | tuple[int, int]], offset: float = 0.5):
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    @property
    def num_base_priors(self) -> list[int]:
        """One point per cell on every level."""
        return [1 for _ in range(len(self.strides))]

    @staticmethod
    def _meshgrid(x: torch.Tensor, y: torch.Tensor, row_major: bool = True):
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        if row_major:
            return xx.reshape(-1), yy.reshape(-1)
        return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(
        self,
        featmap_sizes: Sequence[tuple[int, int]],
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
        with_stride: bool = False,
    ) -> list[torch.Tensor]:
        """Points of every level: one ``(H*W, 2)`` tensor of ``(x, y)`` per
        level, or ``(H*W, 4)`` with ``(stride_w, stride_h)`` appended."""
        if self.num_levels != len(featmap_sizes):
            raise ValueError(
                f"expected {self.num_levels} feature maps, got {len(featmap_sizes)}"
            )
        return [
            self.single_level_grid_priors(
                featmap_sizes[i], level_idx=i, dtype=dtype, device=device, with_stride=with_stride
            )
            for i in range(self.num_levels)
        ]

    def single_level_grid_priors(
        self,
        featmap_size: tuple[int, int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
        with_stride: bool = False,
    ) -> torch.Tensor:
        """Points of one level, row-major over the ``(h, w)`` feature map."""
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        # Same operation order as mmdet, so the values are bit-identical:
        # integer arange, plus the offset, times the stride, then cast.
        shift_x = ((torch.arange(0, feat_w, device=device) + self.offset) * stride_w).to(dtype)
        shift_y = ((torch.arange(0, feat_h, device=device) + self.offset) * stride_h).to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            return torch.stack([shift_xx, shift_yy], dim=-1)
        stride_ws = shift_xx.new_full((shift_xx.shape[0],), stride_w).to(dtype)
        stride_hs = shift_yy.new_full((shift_yy.shape[0],), stride_h).to(dtype)
        return torch.stack([shift_xx, shift_yy, stride_ws, stride_hs], dim=-1)

    def valid_flags(
        self,
        featmap_sizes: Sequence[tuple[int, int]],
        pad_shape: Sequence[int],
        device: str | torch.device = "cuda",
    ) -> list[torch.Tensor]:
        """Which points of each level fall inside the padded image."""
        if self.num_levels != len(featmap_sizes):
            raise ValueError(
                f"expected {self.num_levels} feature maps, got {len(featmap_sizes)}"
            )
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            multi_level_flags.append(
                self.single_level_valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w), device=device
                )
            )
        return multi_level_flags

    def single_level_valid_flags(
        self,
        featmap_size: tuple[int, int],
        valid_size: tuple[int, int],
        device: str | torch.device = "cuda",
    ) -> torch.Tensor:
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        if valid_h > feat_h or valid_w > feat_w:
            raise ValueError(f"valid size {valid_size} exceeds the feature map {featmap_size}")
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        return valid_xx & valid_yy

    def sparse_priors(
        self,
        prior_idxs: torch.Tensor,
        featmap_size: tuple[int, int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
    ) -> torch.Tensor:
        """Points for the flat cell indices ``prior_idxs`` of one level."""
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height + self.offset) * self.strides[level_idx][1]
        return torch.stack([x, y], 1).to(dtype).to(device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(strides={self.strides}, offset={self.offset})"
