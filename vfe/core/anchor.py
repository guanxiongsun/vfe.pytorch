"""Anchor generation. Port of ``mmdet.core.anchor``.

Anchors are built once per level as ``base_anchors`` -- every (scale, ratio)
combination centred on the origin -- then broadcast over the feature grid by
adding the stride-scaled cell coordinates.

Dropped from mmdet's version: the deprecated ``grid_anchors`` /
``single_level_grid_anchors`` aliases, ``sparse_priors`` (used only by sparse
detectors), and the SSD/Legacy/YOLO generator subclasses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.nn.modules.utils import _pair

from vfe.core.builder import PRIOR_GENERATORS

__all__ = ["AnchorGenerator", "images_to_levels", "anchor_inside_flags"]


@PRIOR_GENERATORS.register_module()
class AnchorGenerator:
    """Standard anchor generator for 2D anchor-based detectors.

    Args:
        strides: per level, the stride of the feature map in ``(w, h)``.
        ratios: height/width ratios within a level.
        scales: anchor scales within a level. Mutually exclusive with
            ``octave_base_scale``/``scales_per_octave``.
        base_sizes: per level, the side length a scale of 1 corresponds to.
            Defaults to the level's smallest stride.
        scale_major: vary scale fastest when flattening the (ratio, scale)
            grid. True since mmdet 2.0; the channel order of a pretrained
            ``rpn_cls`` conv depends on it.
        octave_base_scale, scales_per_octave: RetinaNet's way of expressing
            ``scales`` as a geometric series.
        centers: per level, an explicit anchor centre within the cell.
        center_offset: anchor centre as a fraction of ``base_size``. 0 since
            mmdet 2.0, which is what puts anchors on cell corners rather than
            cell centres.
    """

    def __init__(
        self,
        strides: Sequence[int | tuple[int, int]],
        ratios: Sequence[float],
        scales: Sequence[float] | None = None,
        base_sizes: Sequence[float] | None = None,
        scale_major: bool = True,
        octave_base_scale: float | None = None,
        scales_per_octave: int | None = None,
        centers: Sequence[tuple[float, float]] | None = None,
        center_offset: float = 0.0,
    ):
        if center_offset != 0 and centers is not None:
            raise ValueError(f"centers cannot be set when center_offset != 0, got {centers}")
        if not 0 <= center_offset <= 1:
            raise ValueError(f"center_offset should be in [0, 1], got {center_offset}")

        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = (
            [min(stride) for stride in self.strides] if base_sizes is None else list(base_sizes)
        )
        if len(self.base_sizes) != len(self.strides):
            raise ValueError(
                f"strides and base_sizes must match, got {self.strides} and {self.base_sizes}"
            )
        if centers is not None and len(centers) != len(strides):
            raise ValueError(f"strides and centers must match, got {strides} and {centers}")

        has_octave = octave_base_scale is not None and scales_per_octave is not None
        if has_octave == (scales is not None):
            raise ValueError(
                "set exactly one of `scales` or `octave_base_scale` + `scales_per_octave`"
            )
        if scales is not None:
            self.scales = torch.Tensor(scales)
        else:
            octave_scales = [2 ** (i / scales_per_octave) for i in range(scales_per_octave)]
            self.scales = torch.Tensor([s * octave_base_scale for s in octave_scales])

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_priors(self) -> list[int]:
        """Anchors per feature-map cell, per level."""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_base_anchors(self) -> list[int]:
        return self.num_base_priors

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    def gen_base_anchors(self) -> list[torch.Tensor]:
        return [
            self.gen_single_level_base_anchors(
                base_size,
                scales=self.scales,
                ratios=self.ratios,
                center=None if self.centers is None else self.centers[i],
            )
            for i, base_size in enumerate(self.base_sizes)
        ]

    def gen_single_level_base_anchors(
        self,
        base_size: float,
        scales: torch.Tensor,
        ratios: torch.Tensor,
        center: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """``(num_scales * num_ratios, 4)`` anchors for one level.

        Ratio is applied area-preservingly: ``h *= sqrt(r)``, ``w /= sqrt(r)``.
        """
        w = h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = torch.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).view(-1)

        return torch.stack(
            [
                x_center - 0.5 * ws,
                y_center - 0.5 * hs,
                x_center + 0.5 * ws,
                y_center + 0.5 * hs,
            ],
            dim=-1,
        )

    @staticmethod
    def _meshgrid(
        x: torch.Tensor, y: torch.Tensor, row_major: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xx = x.repeat(y.shape[0])
        yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
        return (xx, yy) if row_major else (yy, xx)

    def grid_priors(
        self,
        featmap_sizes: Sequence[tuple[int, int]],
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
    ) -> list[torch.Tensor]:
        """Anchors for every cell of every level, one ``(H*W*A, 4)`` tensor per level."""
        if self.num_levels != len(featmap_sizes):
            raise ValueError(
                f"expected {self.num_levels} feature maps, got {len(featmap_sizes)}"
            )
        return [
            self.single_level_grid_priors(featmap_sizes[i], i, dtype=dtype, device=device)
            for i in range(self.num_levels)
        ]

    def single_level_grid_priors(
        self,
        featmap_size: tuple[int, int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda",
    ) -> torch.Tensor:
        """Anchors for one level, ordered cell-major: all ``A`` anchors of cell
        ``(0, 0)``, then ``(0, 1)``, and so on."""
        base_anchors = self.base_anchors[level_idx].to(device).to(dtype)
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = torch.arange(0, feat_w, device=device).to(dtype) * stride_w
        shift_y = torch.arange(0, feat_h, device=device).to(dtype) * stride_h

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)

        # (1, A, 4) + (K, 1, 4) -> (K, A, 4)
        return (base_anchors[None, :, :] + shifts[:, None, :]).view(-1, 4)

    def valid_flags(
        self,
        featmap_sizes: Sequence[tuple[int, int]],
        pad_shape: Sequence[int],
        device: str | torch.device = "cuda",
    ) -> list[torch.Tensor]:
        """Mark anchors whose cell falls inside the *unpadded* image.

        Batches are padded to a common size, so the bottom/right of a feature
        map can correspond to nothing but padding.
        """
        if self.num_levels != len(featmap_sizes):
            raise ValueError(
                f"expected {self.num_levels} feature maps, got {len(featmap_sizes)}"
            )
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(math.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(math.ceil(w / anchor_stride[0])), feat_w)
            multi_level_flags.append(
                self.single_level_valid_flags(
                    (feat_h, feat_w),
                    (valid_feat_h, valid_feat_w),
                    self.num_base_anchors[i],
                    device=device,
                )
            )
        return multi_level_flags

    def single_level_valid_flags(
        self,
        featmap_size: tuple[int, int],
        valid_size: tuple[int, int],
        num_base_anchors: int,
        device: str | torch.device = "cuda",
    ) -> torch.Tensor:
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        if not (valid_h <= feat_h and valid_w <= feat_w):
            raise ValueError(f"valid_size {valid_size} exceeds featmap_size {featmap_size}")
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid[:, None].expand(valid.size(0), num_base_anchors).contiguous().view(-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(strides={self.strides}, ratios={self.ratios.tolist()}, "
            f"scales={self.scales.tolist()}, base_sizes={self.base_sizes}, "
            f"scale_major={self.scale_major}, center_offset={self.center_offset})"
        )


def images_to_levels(
    target: Sequence[torch.Tensor], num_levels: Sequence[int]
) -> list[torch.Tensor]:
    """``[per-image, all levels]`` -> ``[per-level, all images]``.

    The heads compute targets per image over the concatenated anchors of every
    level; the losses want them back per level, to line up with the per-level
    predictions.
    """
    stacked = torch.stack(list(target), 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        level_targets.append(stacked[:, start:end])
        start = end
    return level_targets


def anchor_inside_flags(
    flat_anchors: torch.Tensor,
    valid_flags: torch.Tensor,
    img_shape: Sequence[int],
    allowed_border: int = 0,
) -> torch.Tensor:
    """Narrow ``valid_flags`` to anchors lying within ``allowed_border`` of the image.

    ``allowed_border < 0`` disables the check, keeping anchors that hang off
    the edge.
    """
    img_h, img_w = img_shape[:2]
    if allowed_border < 0:
        return valid_flags
    return (
        valid_flags
        & (flat_anchors[:, 0] >= -allowed_border)
        & (flat_anchors[:, 1] >= -allowed_border)
        & (flat_anchors[:, 2] < img_w + allowed_border)
        & (flat_anchors[:, 3] < img_h + allowed_border)
    )
