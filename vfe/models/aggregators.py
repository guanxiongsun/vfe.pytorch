"""Feature aggregators for the VID detectors. Port of
``mmdet.models.aggregators.mamba_aggregator``.

An aggregator enhances each key-frame RoI feature with a weighted sum of
reference RoI features, the weights coming from multi-head dot-product
attention between the two sets (SELSA's design). MAMBA's version adds where
the references come from: explicit reference frames on the first frame of a
video, the memory bank on every frame after that.
"""

from __future__ import annotations

import torch
from torch import nn

from vfe.models.builder import AGGREGATORS
from vfe.models.memory import MemoryBank

__all__ = ["MambaAggregator"]


@AGGREGATORS.register_module()
class MambaAggregator(nn.Module):
    """Args:
        in_channels: RoI feature width; must divide by ``num_attention_blocks``.
        num_attention_blocks: attention heads.
        memory_cfg: keyword arguments for :class:`~vfe.models.memory.MemoryBank`.
    """

    def __init__(self, in_channels: int, num_attention_blocks: int = 16,
                 memory_cfg: dict | None = None):
        super().__init__()
        if in_channels % num_attention_blocks:
            raise ValueError(
                f"in_channels={in_channels} is not divisible by "
                f"num_attention_blocks={num_attention_blocks}"
            )
        self.fc_embed = nn.Linear(in_channels, in_channels)
        self.ref_fc_embed = nn.Linear(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, in_channels)
        self.ref_fc = nn.Linear(in_channels, in_channels)
        self.num_attention_blocks = num_attention_blocks
        self.memory_bank = MemoryBank(**(memory_cfg or {}))

    def forward(self, x: torch.Tensor, ref_x: torch.Tensor | None) -> torch.Tensor:
        """With ``ref_x``: (re)start the memory from it and attend to it -- the
        first frame, and every training step. Without: attend to a sample of
        the memory."""
        if ref_x is not None:
            self.memory_bank.init_memory(ref_x)
        else:
            ref_x = self.memory_bank.sample()
        return self.forward_with_ref_x(x, ref_x)

    def reset_memory_bank(self) -> None:
        self.memory_bank.reset()

    def update_memory_bank(self, x: torch.Tensor) -> None:
        self.memory_bank.update(x)

    def forward_with_ref_x(self, x: torch.Tensor, ref_x: torch.Tensor) -> torch.Tensor:
        """``(N, C)`` key features, ``(M, C)`` references -> ``(N, C)``.

        Each of the ``B = num_attention_blocks`` heads computes softmax
        attention over the M references from ``C/B``-wide embeddings, then sums
        the (separately projected) references with those weights.
        """
        roi_n = x.shape[0]
        ref_roi_n = ref_x.shape[0]

        # (B, N, C/B)
        x_embed = self.fc_embed(x).view(roi_n, self.num_attention_blocks, -1).permute(1, 0, 2)
        # (B, C/B, M)
        ref_x_embed = (
            self.ref_fc_embed(ref_x).view(ref_roi_n, self.num_attention_blocks, -1).permute(1, 2, 0)
        )

        # (B, N, M)
        weights = torch.bmm(x_embed, ref_x_embed) / (x_embed.shape[-1] ** 0.5)
        weights = weights.softmax(dim=2)

        # (B, M, C/B)
        ref_x_new = self.ref_fc(ref_x).view(ref_roi_n, self.num_attention_blocks, -1).permute(1, 0, 2)
        # (N, B, C/B) -> (N, C)
        x_new = torch.bmm(weights, ref_x_new).permute(1, 0, 2).contiguous()
        return self.fc(x_new.view(roi_n, -1))
