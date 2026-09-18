"""``ConvModule`` -- conv + norm + activation, replacing ``mmcv.cnn.ConvModule``.

Dropped from mmcv's version: spectral norm, explicit padding layers
(``padding_mode`` other than ``'zeros'``/``'circular'``), and the custom weight
init -- checkpoints are always loaded here, so mmcv's Kaiming-on-construct is
dead weight. The ``order`` argument and the ``bias='auto'`` rule are kept
because both affect parameter *names* and *shapes*, and therefore checkpoint
compatibility.
"""

from __future__ import annotations

from types import MappingProxyType

import torch
from torch import nn

from .builders import build_activation_layer, build_conv_layer, build_norm_layer

__all__ = ["ConvModule", "DEFAULT_ACT_CFG"]

# mmcv's default is `act_cfg=dict(type='ReLU')`, so `act_cfg=None` means "no
# activation" -- a distinction FPN depends on (its lateral and output convs are
# activation-free). Expressing the default as a sentinel rather than coercing
# None keeps both meanings available. Read-only so the shared instance cannot
# be mutated by a caller.
DEFAULT_ACT_CFG = MappingProxyType({"type": "ReLU"})


class ConvModule(nn.Module):
    """A conv layer bundled with optional norm and activation.

    Args:
        bias: ``'auto'`` means ``False`` when a norm layer follows the conv
            (the norm's own bias makes the conv's redundant), else ``True``.
        act_cfg: defaults to ReLU; pass ``None`` for no activation.
        order: the three stages in application order. Norm channel count
            follows from whether it sits before or after the conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool | str = "auto",
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = DEFAULT_ACT_CFG,
        inplace: bool = True,
        order: tuple[str, str, str] = ("conv", "norm", "act"),
    ):
        super().__init__()
        act_cfg = dict(act_cfg) if act_cfg is not None else None
        if set(order) != {"conv", "norm", "act"}:
            raise ValueError(f"order must be a permutation of conv/norm/act, got {order}")

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.order = tuple(order)

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bool(bias)

        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=self.with_bias,
        )
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.groups = self.conv.groups

        if self.with_norm:
            norm_channels = out_channels if order.index("norm") > order.index("conv") else in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        if self.with_activation:
            act_cfg_ = dict(act_cfg)
            # Activations without an `inplace` kwarg (e.g. GELU) must not get one.
            if act_cfg_.get("type") not in ("Tanh", "PReLU", "Sigmoid", "GELU"):
                act_cfg_.setdefault("inplace", inplace)
            self.activate = build_activation_layer(act_cfg_)

    @property
    def norm(self) -> nn.Module | None:
        return getattr(self, self.norm_name) if self.norm_name else None

    def forward(self, x: torch.Tensor, activate: bool = True, norm: bool = True) -> torch.Tensor:
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x
