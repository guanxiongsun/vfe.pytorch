"""``ChannelMapper`` -- the neck used by the DC5 (ResNet) VID configs.

A 1:1 per-level conv that only changes the channel count; unlike FPN there is
no cross-level fusion. The DC5 configs feed it a single 2048-channel stride-16
map and take a single 512-channel map out.
"""

from __future__ import annotations

import torch
from torch import nn

from ...layers import ConvModule, xavier_init
from ...layers.conv_module import DEFAULT_ACT_CFG
from ..builder import NECKS

__all__ = ["ChannelMapper"]


@NECKS.register_module()
class ChannelMapper(nn.Module):
    """Map each input level to ``out_channels``.

    Args:
        num_outs: if larger than ``len(in_channels)``, extra stride-2 3x3 convs
            are stacked on the last input level to synthesise coarser levels.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        kernel_size: int = 3,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = DEFAULT_ACT_CFG,
        num_outs: int | None = None,
    ):
        super().__init__()
        if not isinstance(in_channels, (list, tuple)):
            raise TypeError(f"in_channels must be a list, got {type(in_channels)}")
        self.extra_convs = None
        if num_outs is None:
            num_outs = len(in_channels)

        self.convs = nn.ModuleList(
            ConvModule(
                in_channel,
                out_channels,
                kernel_size,
                padding=(kernel_size - 1) // 2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            for in_channel in in_channels
        )

        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels), num_outs):
                in_channel = in_channels[-1] if i == len(in_channels) else out_channels
                self.extra_convs.append(
                    ConvModule(
                        in_channel,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                    )
                )

    def init_weights(self) -> None:
        """mmdet's ``init_cfg=dict(type='Xavier', layer='Conv2d', ...)``."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if len(inputs) != len(self.convs):
            raise ValueError(f"expected {len(self.convs)} input levels, got {len(inputs)}")
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        if self.extra_convs:
            for i, conv in enumerate(self.extra_convs):
                outs.append(conv(inputs[-1] if i == 0 else outs[-1]))
        return tuple(outs)
