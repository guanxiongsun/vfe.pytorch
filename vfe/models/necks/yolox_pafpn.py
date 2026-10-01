"""YOLOX's path-aggregation neck. Port of ``mmdet.models.necks.yolox_pafpn``.

Top-down, each coarser map is reduced by a 1x1 conv, upsampled and fused
with the next finer backbone map by a CSP block; bottom-up, each finer output
is downsampled by a strided 3x3 conv and fused with the next coarser one;
finally a 1x1 conv brings every level to ``out_channels``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from vfe.layers.conv_module import ConvModule
from vfe.layers.csp_layer import YOLOX_ACT, YOLOX_NORM, CSPLayer, _no_depthwise
from vfe.models.backbones.csp_darknet import init_yolox_convs
from vfe.models.builder import NECKS

__all__ = ["YOLOXPAFPN"]


@NECKS.register_module()
class YOLOXPAFPN(nn.Module):
    """Args:
        in_channels: the backbone's widths, finest first.
        out_channels: the width of every output level.
        num_csp_blocks: bottlenecks in each CSP block.
    """

    def __init__(self, in_channels: Sequence[int], out_channels: int, num_csp_blocks: int = 3,
                 use_depthwise: bool = False, upsample_cfg: dict | None = None, conv_cfg=None,
                 norm_cfg: dict = YOLOX_NORM, act_cfg: dict = YOLOX_ACT, init_cfg=None):
        super().__init__()
        _no_depthwise(use_depthwise)
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.upsample = nn.Upsample(**(upsample_cfg or dict(scale_factor=2, mode="nearest")))
        kw = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(ConvModule(in_channels[idx], in_channels[idx - 1], 1, **kw))
            self.top_down_blocks.append(CSPLayer(in_channels[idx - 1] * 2, in_channels[idx - 1],
                                                 num_blocks=num_csp_blocks, add_identity=False,
                                                 **kw))
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(ConvModule(in_channels[idx], in_channels[idx], 3, stride=2,
                                               padding=1, **kw))
            self.bottom_up_blocks.append(CSPLayer(in_channels[idx] * 2, in_channels[idx + 1],
                                                  num_blocks=num_csp_blocks, add_identity=False,
                                                  **kw))
        self.out_convs = nn.ModuleList(
            [ConvModule(in_channels[i], out_channels, 1, **kw) for i in range(len(in_channels))])

    def init_weights(self) -> None:
        init_yolox_convs(self)

    def forward(self, inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(inputs) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} inputs, got {len(inputs)}")
        n = len(self.in_channels)
        # Top-down. inner_outs[0] is replaced by its reduced form, which the
        # bottom-up path fuses with.
        inner_outs = [inputs[-1]]
        for idx in range(n - 1, 0, -1):
            feat_high = self.reduce_layers[n - 1 - idx](inner_outs[0])
            inner_outs[0] = feat_high
            inner_out = self.top_down_blocks[n - 1 - idx](
                torch.cat([self.upsample(feat_high), inputs[idx - 1]], 1))
            inner_outs.insert(0, inner_out)
        # Bottom-up.
        outs = [inner_outs[0]]
        for idx in range(n - 1):
            downsampled = self.downsamples[idx](outs[-1])
            outs.append(self.bottom_up_blocks[idx](torch.cat([downsampled, inner_outs[idx + 1]], 1)))
        return tuple(conv(out) for conv, out in zip(self.out_convs, outs, strict=True))
