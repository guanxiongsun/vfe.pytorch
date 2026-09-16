"""Transformer building blocks, replacing ``mmcv.cnn.bricks.transformer`` and
the ``PatchEmbed``/``PatchMerging``/``AdaptivePadding`` trio from
``mmdet.models.utils.transformer``.

Only what Swin needs is here. ``PatchMerging`` in particular is mmdet's
``nn.Unfold``-based rewrite, not the original Swin implementation -- the two
order the 2x2 neighbourhood's channels differently, which is exactly why
released Swin checkpoints need ``swin_converter`` before they will load.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .builders import build_activation_layer, build_conv_layer, build_norm_layer
from .drop import build_dropout

__all__ = ["AdaptivePadding", "PatchEmbed", "PatchMerging", "FFN", "to_2tuple"]


def to_2tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class AdaptivePadding(nn.Module):
    """Pad so a strided kernel covers the input exactly.

    ``'corner'`` pads bottom-right only; ``'same'`` splits the padding around
    the input (TensorFlow's SAME). Swin uses ``'corner'``.
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding: str = "corner"):
        super().__init__()
        if padding not in ("same", "corner"):
            raise ValueError(f"padding must be 'same' or 'corner', got {padding!r}")
        self.padding = padding
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)

    def get_pad_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = F.pad(x, [0, pad_w, 0, pad_h])
            else:
                x = F.pad(
                    x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
                )
        return x


class PatchEmbed(nn.Module):
    """Image -> patch tokens, implemented as a strided conv.

    Returns ``(tokens, (out_h, out_w))``; the spatial shape has to travel
    alongside the tokens because everything downstream reshapes back to 2D.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 768,
        conv_type: str = "Conv2d",
        kernel_size: int = 16,
        stride: int | None = 16,
        padding: int | tuple | str = "corner",
        dilation: int = 1,
        bias: bool = True,
        norm_cfg: dict | None = None,
        input_size: int | tuple | None = None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding
            )
            padding = 0  # the conv itself must not pad as well
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1] if norm_cfg is not None else None

        if input_size:
            input_size = to_2tuple(input_size)
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_size = (input_size[0] + pad_h, input_size[1] + pad_w)
            h_out = (
                input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
            ) // stride[0] + 1
            w_out = (
                input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
            ) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.adap_padding:
            x = self.adap_padding(x)
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class PatchMerging(nn.Module):
    """Downsample tokens 2x by concatenating each 2x2 neighbourhood.

    Implemented with ``nn.Unfold`` (mmdet's version, ~25% faster than the
    original Swin gather). The unfold traversal order differs from the original,
    so ``reduction.weight`` from an upstream Swin checkpoint must be permuted --
    see ``vfe.models.backbones.swin_convert``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple = 2,
        stride: int | tuple | None = None,
        padding: int | tuple | str = "corner",
        dilation: int | tuple = 1,
        bias: bool = False,
        norm_cfg: dict | None = None,
    ):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = {"type": "LN"}
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = stride if stride else kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding
            )
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)

        self.sampler = nn.Unfold(
            kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride
        )
        sample_dim = kernel_size[0] * kernel_size[1] * in_channels
        self.norm = build_norm_layer(norm_cfg, sample_dim)[1] if norm_cfg is not None else None
        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(
        self, x: torch.Tensor, input_size: Sequence[int]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        B, L, C = x.shape
        if not isinstance(input_size, Sequence):
            raise TypeError(f"input_size must be a Sequence, got {input_size!r}")
        H, W = input_size
        if L != H * W:
            raise ValueError(f"input has {L} tokens but input_size {input_size} implies {H * W}")

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x)  # B, kh*kw*C, out_h*out_w
        out_h = (
            H
            + 2 * self.sampler.padding[0]
            - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1)
            - 1
        ) // self.sampler.stride[0] + 1
        out_w = (
            W
            + 2 * self.sampler.padding[1]
            - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1)
            - 1
        ) // self.sampler.stride[1] + 1

        x = x.transpose(1, 2)
        x = self.norm(x) if self.norm else x
        return self.reduction(x), (out_h, out_w)


class FFN(nn.Module):
    """Feed-forward network with an identity (residual) connection.

    The nesting of ``self.layers`` matters: mmcv wraps each hidden block in its
    own ``Sequential``, giving keys ``layers.0.0.{weight,bias}`` and
    ``layers.1.{weight,bias}``. Flattening it would silently break checkpoints.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        act_cfg: dict | None = None,
        ffn_drop: float = 0.0,
        dropout_layer: dict | None = None,
        add_identity: bool = True,
    ):
        super().__init__()
        if num_fcs < 2:
            raise ValueError(f"num_fcs must be >= 2, got {num_fcs}")
        if act_cfg is None:
            act_cfg = {"type": "ReLU", "inplace": True}
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()
        self.add_identity = add_identity

    def forward(self, x: torch.Tensor, identity: torch.Tensor | None = None) -> torch.Tensor:
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)
