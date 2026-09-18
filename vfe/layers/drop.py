"""Stochastic depth, replacing ``mmcv.cnn.bricks.drop``.

``build_dropout`` exists because the Swin configs pass ``drop_path_rate``
through as a ``dict(type='DropPath', drop_prob=...)`` config; the two registered
types are the only ones those configs use.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["DropPath", "Dropout", "build_dropout", "drop_path"]


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Zero out whole samples with probability ``drop_prob``.

    Follows timm/mmcv: the surviving samples are scaled by ``1/keep_prob`` so
    the expectation is unchanged.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * random_tensor.floor()


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class Dropout(nn.Dropout):
    """``nn.Dropout`` with mmcv's ``drop_prob`` argument name."""

    def __init__(self, drop_prob: float = 0.5, inplace: bool = False):
        super().__init__(p=drop_prob, inplace=inplace)


DROPOUT_LAYERS: dict[str, type[nn.Module]] = {
    "DropPath": DropPath,
    "Dropout": Dropout,
}


def build_dropout(cfg: dict) -> nn.Module:
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg)}")
    cfg_ = dict(cfg)
    layer_type = cfg_.pop("type", None)
    if layer_type not in DROPOUT_LAYERS:
        raise KeyError(f"unsupported dropout {layer_type!r}; known: {sorted(DROPOUT_LAYERS)}")
    return DROPOUT_LAYERS[layer_type](**cfg_)
