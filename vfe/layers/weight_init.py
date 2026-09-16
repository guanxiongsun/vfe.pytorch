"""Weight initialisers, replacing ``mmcv.cnn.{xavier,normal,constant}_init``.

mmcv drives initialisation declaratively through ``init_cfg`` dicts and
``BaseModule.init_weights``. That indirection is dropped: each module here
writes an explicit ``init_weights`` that calls these functions directly, so the
scheme a module uses is readable in the module itself.

The defaults match mmcv's (``bias=0``, ``gain=1``, uniform/normal switch) --
they decide the starting point of any run trained from scratch, and diverging
would silently change training curves while inference parity still passed.
"""

from __future__ import annotations

import numpy as np
from torch import nn

__all__ = [
    "constant_init",
    "normal_init",
    "trunc_normal_init",
    "xavier_init",
    "kaiming_init",
    "bias_init_with_prob",
]


def _init_bias(module: nn.Module, bias: float) -> None:
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    _init_bias(module, bias)


def normal_init(module: nn.Module, mean: float = 0, std: float = 1, bias: float = 0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    _init_bias(module, bias)


def trunc_normal_init(
    module: nn.Module,
    mean: float = 0,
    std: float = 1,
    a: float = -2,
    b: float = 2,
    bias: float = 0,
) -> None:
    """Truncated normal init, as Swin uses. ``nn.init.trunc_normal_`` is the
    same erfinv-based algorithm mmcv vendored, so values match."""
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.trunc_normal_(module.weight, mean, std, a, b)
    _init_bias(module, bias)


def xavier_init(
    module: nn.Module, gain: float = 1, bias: float = 0, distribution: str = "normal"
) -> None:
    if distribution not in ("normal", "uniform"):
        raise ValueError(f"distribution must be 'normal' or 'uniform', got {distribution!r}")
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    _init_bias(module, bias)


def kaiming_init(
    module: nn.Module,
    a: float = 0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0,
    distribution: str = "normal",
) -> None:
    if distribution not in ("normal", "uniform"):
        raise ValueError(f"distribution must be 'normal' or 'uniform', got {distribution!r}")
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    _init_bias(module, bias)


def bias_init_with_prob(prior_prob: float) -> float:
    """Bias that makes a sigmoid output ``prior_prob`` at zero activation."""
    return float(-np.log((1 - prior_prob) / prior_prob))
