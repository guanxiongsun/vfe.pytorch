"""``Scale`` -- a single learnable multiplier, replacing ``mmcv.cnn.Scale``.

FCOS shares one head across pyramid levels and gives each level its own
``Scale`` on the regression output, so the shared convs can predict distances
in a level-independent range.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["Scale"]


class Scale(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

    def extra_repr(self) -> str:
        return f"scale={self.scale.item():g}"
