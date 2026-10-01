"""The Cross Stage Partial block of YOLOv5 / YOLOX. Port of
``mmdet.models.utils.csp_layer``.

A ``CSPLayer`` splits its input into two 1x1 projections: one runs through a
stack of Darknet bottlenecks, the other skips them, and a final 1x1 conv fuses
the concatenation. The attribute names (``main_conv``, ``short_conv``,
``final_conv``, ``blocks``) are mmdet's, so its checkpoints load unchanged.

Depthwise-separable convs (YOLOX-Nano's ``use_depthwise``) are not ported:
no config here uses them.
"""

from __future__ import annotations

import torch
from torch import nn

from vfe.layers.conv_module import ConvModule

__all__ = ["DarknetBottleneck", "CSPLayer"]

YOLOX_NORM = dict(type="BN", momentum=0.03, eps=0.001)
YOLOX_ACT = dict(type="Swish")


def _no_depthwise(use_depthwise: bool) -> None:
    if use_depthwise:
        raise NotImplementedError("depthwise-separable convs (YOLOX-Nano) are not ported")


class DarknetBottleneck(nn.Module):
    """A 1x1 conv to ``expansion`` of the output width, then a 3x3 conv,
    with an identity shortcut when ``add_identity`` and the widths match."""

    def __init__(self, in_channels: int, out_channels: int, expansion: float = 0.5,
                 add_identity: bool = True, use_depthwise: bool = False, conv_cfg=None,
                 norm_cfg: dict = YOLOX_NORM, act_cfg: dict = YOLOX_ACT):
        super().__init__()
        _no_depthwise(use_depthwise)
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvModule(in_channels, hidden_channels, 1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(hidden_channels, out_channels, 3, stride=1, padding=1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    """Args:
        expand_ratio: width of each path, as a fraction of ``out_channels``.
        num_blocks: Darknet bottlenecks on the main path.
    """

    def __init__(self, in_channels: int, out_channels: int, expand_ratio: float = 0.5,
                 num_blocks: int = 1, add_identity: bool = True, use_depthwise: bool = False,
                 conv_cfg=None, norm_cfg: dict = YOLOX_NORM, act_cfg: dict = YOLOX_ACT):
        super().__init__()
        _no_depthwise(use_depthwise)
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(in_channels, mid_channels, 1, conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.short_conv = ConvModule(in_channels, mid_channels, 1, conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.final_conv = ConvModule(2 * mid_channels, out_channels, 1, conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.blocks = nn.Sequential(*[
            DarknetBottleneck(mid_channels, mid_channels, 1.0, add_identity, use_depthwise,
                              conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        return self.final_conv(torch.cat((x_main, x_short), dim=1))
