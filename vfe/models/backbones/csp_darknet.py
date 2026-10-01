"""CSP-Darknet, the backbone of YOLOv5 and YOLOX. Port of
``mmdet.models.backbones.csp_darknet``.

A ``Focus`` stem folds each 2x2 patch into channels, then four stages each
halve the resolution with a strided 3x3 conv and refine with a ``CSPLayer``;
the last stage adds an SPP block before its CSP layer. ``deepen_factor`` and
``widen_factor`` scale the block counts and widths: YOLOX-S is 0.33 / 0.5,
-M 0.67 / 0.75, -L 1.0 / 1.0, -X 1.33 / 1.25. The default outputs are stages
2-4, strides 8 / 16 / 32.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from vfe.layers.conv_module import ConvModule
from vfe.layers.csp_layer import YOLOX_ACT, YOLOX_NORM, CSPLayer, _no_depthwise
from vfe.layers.weight_init import kaiming_init
from vfe.models.builder import BACKBONES

__all__ = ["CSPDarknet", "Focus", "SPPBottleneck", "init_yolox_convs"]


def init_yolox_convs(module: nn.Module) -> None:
    """mmdet's YOLOX ``init_cfg``: every ``Conv2d`` Kaiming-uniform with
    ``a = sqrt(5)`` over ``fan_in`` (torch's own conv default) and a zero
    bias. Norm layers keep torch's defaults."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=math.sqrt(5), mode="fan_in", nonlinearity="leaky_relu",
                         distribution="uniform")


class Focus(nn.Module):
    """Space-to-depth: the four pixels of every 2x2 patch become channels
    (top-left, bottom-left, top-right, bottom-right), then one conv."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 stride: int = 1, conv_cfg=None, norm_cfg: dict = YOLOX_NORM,
                 act_cfg: dict = YOLOX_ACT):
        super().__init__()
        self.conv = ConvModule(in_channels * 4, out_channels, kernel_size, stride,
                               padding=(kernel_size - 1) // 2, conv_cfg=conv_cfg,
                               norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]),
                      dim=1)
        return self.conv(x)


class SPPBottleneck(nn.Module):
    """YOLOv3-SPP's spatial pyramid pooling: max pools of several sizes,
    stride 1, concatenated with their input between two 1x1 convs."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_sizes: Sequence[int] = (5, 9, 13), conv_cfg=None,
                 norm_cfg: dict = YOLOX_NORM, act_cfg: dict = YOLOX_ACT):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid_channels, 1, stride=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.poolings = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        self.conv2 = ConvModule(mid_channels * (len(kernel_sizes) + 1), out_channels, 1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        return self.conv2(x)


@BACKBONES.register_module()
class CSPDarknet(nn.Module):
    """Args:
        arch: ``'P5'`` (four stages) or ``'P6'`` (five).
        deepen_factor / widen_factor: block-count and width multipliers.
        out_indices: which of stem (0) and stages (1..) to return.
        frozen_stages: freeze the stem and stages up to this index (-1: none).
        norm_eval: keep every BatchNorm in eval mode while training.
        spp_kernal_sizes: SPP pool sizes (mmdet's spelling, kept for configs).
    """

    arch_settings = {
        # in, out, blocks, add_identity, use_spp
        "P5": [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, False, True]],
        "P6": [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]],
    }

    def __init__(self, arch: str = "P5", deepen_factor: float = 1.0, widen_factor: float = 1.0,
                 out_indices: Sequence[int] = (2, 3, 4), frozen_stages: int = -1,
                 use_depthwise: bool = False, arch_ovewrite=None,
                 spp_kernal_sizes: Sequence[int] = (5, 9, 13), conv_cfg=None,
                 norm_cfg: dict = YOLOX_NORM, act_cfg: dict = YOLOX_ACT,
                 norm_eval: bool = False, init_cfg=None):
        super().__init__()
        _no_depthwise(use_depthwise)
        arch_setting = arch_ovewrite or self.arch_settings[arch]
        if not set(out_indices) <= set(range(len(arch_setting) + 1)):
            raise ValueError(f"out_indices {out_indices} out of range")
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError(f"frozen_stages must be in range(-1, {len(arch_setting) + 1}), "
                             f"got {frozen_stages}")
        self.out_indices = tuple(out_indices)
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.stem = Focus(3, int(arch_setting[0][0] * widen_factor), kernel_size=3,
                          conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.layers = ["stem"]
        for i, (in_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(
                arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = [ConvModule(in_channels, out_channels, 3, stride=2, padding=1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)]
            if use_spp:
                stage.append(SPPBottleneck(out_channels, out_channels,
                                           kernel_sizes=spp_kernal_sizes, conv_cfg=conv_cfg,
                                           norm_cfg=norm_cfg, act_cfg=act_cfg))
            stage.append(CSPLayer(out_channels, out_channels, num_blocks=num_blocks,
                                  add_identity=add_identity, conv_cfg=conv_cfg,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{i + 1}")

    def init_weights(self) -> None:
        init_yolox_convs(self)

    def _freeze_stages(self) -> None:
        for i in range(self.frozen_stages + 1):
            m = getattr(self, self.layers[i])
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outs = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
