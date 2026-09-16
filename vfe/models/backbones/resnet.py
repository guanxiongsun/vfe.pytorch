"""ResNet backbone, ported from ``mmdet.models.backbones.resnet``.

Structure and parameter names follow mmdet exactly so released checkpoints load
without remapping -- and, because mmdet's naming coincides with torchvision's
(``conv1/bn1/layer1.0.conv1/...``), so do the ``torchvision://resnet101``
pretrained weights the configs start from.

One subtlety worth stating, because it silently changes results: mmdet applies
a stage's ``dilation`` to **every** block in that stage, whereas
``torchvision.models.resnet`` with ``replace_stride_with_dilation`` gives the
stage's *first* block the previous stage's dilation. For the DC5 configs here
(``strides=(1,2,2,1), dilations=(1,1,1,2)``) that means layer4's first block is
dilated in mmdet but not in torchvision. This port follows mmdet.

Dropped from mmdet's version: DCN, plugins, ``deep_stem``/``avg_down``
(ResNetV1d), and the mmcv init-cfg machinery -- no config in this repo uses
them, and weights come from checkpoints rather than from init rules.
"""

from __future__ import annotations

import torch
import torch.utils.checkpoint as cp
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from ...layers import build_conv_layer, build_norm_layer
from ..builder import BACKBONES

__all__ = ["BasicBlock", "Bottleneck", "ResLayer", "ResNet"]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        style: str = "pytorch",
        with_cp: bool = False,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
    ):
        super().__init__()
        norm_cfg = norm_cfg or {"type": "BN"}

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg, inplanes, planes, 3, stride=stride,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self) -> nn.Module:
        return getattr(self, self.norm1_name)

    @property
    def norm2(self) -> nn.Module:
        return getattr(self, self.norm2_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _inner_forward(x):
            identity = x
            out = self.relu(self.norm1(self.conv1(x)))
            out = self.norm2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            out = _inner_forward(x)
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        style: str = "pytorch",
        with_cp: bool = False,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
    ):
        """If ``style='pytorch'`` the stride-2 layer is the 3x3 conv; if
        ``'caffe'``, it is the first 1x1 conv."""
        super().__init__()
        if style not in ("pytorch", "caffe"):
            raise ValueError(f"style must be 'pytorch' or 'caffe', got {style!r}")
        norm_cfg = norm_cfg or {"type": "BN"}

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp

        if style == "pytorch":
            self.conv1_stride, self.conv2_stride = 1, stride
        else:
            self.conv1_stride, self.conv2_stride = stride, 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg, inplanes, planes, 1, stride=self.conv1_stride, bias=False
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, stride=self.conv2_stride,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg, planes, planes * self.expansion, 1, bias=False
        )
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self) -> nn.Module:
        return getattr(self, self.norm1_name)

    @property
    def norm2(self) -> nn.Module:
        return getattr(self, self.norm2_name)

    @property
    def norm3(self) -> nn.Module:
        return getattr(self, self.norm3_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _inner_forward(x):
            identity = x
            out = self.relu(self.norm1(self.conv1(x)))
            out = self.relu(self.norm2(self.conv2(out)))
            out = self.norm3(self.conv3(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            out = _inner_forward(x)
        return self.relu(out)


class ResLayer(nn.Sequential):
    """One ResNet stage: ``num_blocks`` blocks, downsampling in the first.

    ``dilation`` reaches every block through ``**kwargs`` -- see the module
    docstring for why that differs from torchvision.
    """

    def __init__(
        self,
        block: type[nn.Module],
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int = 1,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        **kwargs,
    ):
        norm_cfg = norm_cfg or {"type": "BN"}
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    conv_cfg, inplanes, planes * block.expansion, 1, stride=stride, bias=False
                ),
                build_norm_layer(norm_cfg, planes * block.expansion)[1],
            )

        layers = [
            block(
                inplanes=inplanes, planes=planes, stride=stride, downsample=downsample,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs,
            )
        ]
        inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes, planes=planes, stride=1,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, **kwargs,
                )
            )
        super().__init__(*layers)


@BACKBONES.register_module()
class ResNet(nn.Module):
    """ResNet backbone returning a tuple of stage features.

    Args:
        depth: one of 18/34/50/101/152.
        strides / dilations: per-stage; DC5 uses ``(1, 2, 2, 1)`` and
            ``(1, 1, 1, 2)`` to keep stride 16 with a dilated final stage.
        out_indices: which stages to return. The VID configs use ``(3,)``.
        frozen_stages: freeze the stem and this many stages (-1 = none).
        norm_eval: keep all BatchNorms in eval mode during training. The VID
            configs rely on this -- batch size 1 would otherwise wreck the
            running statistics.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth: int,
        in_channels: int = 3,
        stem_channels: int | None = None,
        base_channels: int = 64,
        num_stages: int = 4,
        strides: tuple[int, ...] = (1, 2, 2, 2),
        dilations: tuple[int, ...] = (1, 1, 1, 1),
        out_indices: tuple[int, ...] = (0, 1, 2, 3),
        style: str = "pytorch",
        frozen_stages: int = -1,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        norm_eval: bool = True,
        with_cp: bool = False,
        zero_init_residual: bool = True,
        init_cfg: dict | None = None,
    ):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet")
        if not 1 <= num_stages <= 4:
            raise ValueError(f"num_stages must be in [1, 4], got {num_stages}")
        if len(strides) != num_stages or len(dilations) != num_stages:
            raise ValueError("strides and dilations must both have length num_stages")
        if max(out_indices) >= num_stages:
            raise ValueError(f"out_indices {out_indices} exceed num_stages {num_stages}")

        norm_cfg = norm_cfg or {"type": "BN", "requires_grad": True}
        stem_channels = stem_channels if stem_channels is not None else base_channels

        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        # Retained so the detector can find pretrained weights; see
        # vfe.models.checkpoint.load_pretrained.
        self.init_cfg = init_cfg

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            planes = base_channels * 2**i
            res_layer = ResLayer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=strides[i],
                dilation=dilations[i],
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)

    @property
    def norm1(self) -> nn.Module:
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        self.conv1 = build_conv_layer(
            self.conv_cfg, in_channels, stem_channels, 7, stride=2, padding=3, bias=False
        )
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, stem_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in (self.conv1, self.norm1):
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            x = getattr(self, layer_name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        return self
