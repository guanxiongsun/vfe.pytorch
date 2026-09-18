"""``FPN`` -- Feature Pyramid Network, the neck used by the Swin-T STPN configs.

Ported from mmdet unchanged in behaviour. Note the two ways extra pyramid
levels are produced: ``add_extra_convs=False`` (what the STPN config uses, via
``num_outs=5`` over 4 backbone levels) appends a stride-2 *max pool* of the last
output, not a conv -- P6 in those configs therefore has no parameters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...layers import ConvModule, xavier_init
from ..builder import NECKS

__all__ = ["FPN"]


@NECKS.register_module()
class FPN(nn.Module):
    """Feature Pyramid Network (https://arxiv.org/abs/1612.03144).

    Args:
        add_extra_convs: ``False`` -> extra levels come from max pooling.
            ``True`` is an alias for ``'on_input'``; the string forms select
            which tensor the first extra conv consumes.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: bool | str = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        upsample_cfg: dict | None = None,
    ):
        super().__init__()
        if not isinstance(in_channels, (list, tuple)):
            raise TypeError(f"in_channels must be a list, got {type(in_channels)}")
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = dict(upsample_cfg) if upsample_cfg else {"mode": "nearest"}

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            if num_outs < self.num_ins - start_level:
                raise ValueError("num_outs is smaller than the number of used backbone levels")
        else:
            # With an explicit end_level no extra level may be synthesised.
            self.backbone_end_level = end_level
            if end_level > self.num_ins:
                raise ValueError("end_level exceeds the number of input levels")
            if num_outs != end_level - start_level:
                raise ValueError("num_outs must equal end_level - start_level")
        self.start_level = start_level
        self.end_level = end_level

        if isinstance(add_extra_convs, str):
            if add_extra_convs not in ("on_input", "on_lateral", "on_output"):
                raise ValueError(f"invalid add_extra_convs {add_extra_convs!r}")
        elif add_extra_convs:
            add_extra_convs = "on_input"
        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            self.lateral_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=None if no_norm_on_lateral else norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
            )
            self.fpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
            )

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                extra_in = (
                    self.in_channels[self.backbone_end_level - 1]
                    if i == 0 and self.add_extra_convs == "on_input"
                    else out_channels
                )
                self.fpn_convs.append(
                    ConvModule(
                        extra_in,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False,
                    )
                )

    def init_weights(self) -> None:
        """mmdet's ``init_cfg=dict(type='Xavier', layer='Conv2d', ...)``."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if len(inputs) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} input levels, got {len(inputs)}")

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Top-down path. `scale_factor` and `size` cannot both be given to
        # F.interpolate, so the config picks one.
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if "scale_factor" in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg
                )
            else:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=laterals[i - 1].shape[2:], **self.upsample_cfg
                )

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                # Parameter-free extra levels (Faster/Mask R-CNN).
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == "on_input":
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == "on_lateral":
                    extra_source = laterals[-1]
                else:  # 'on_output'
                    extra_source = outs[-1]
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    src = F.relu(outs[-1]) if self.relu_before_extra_convs else outs[-1]
                    outs.append(self.fpn_convs[i](src))
        return tuple(outs)
