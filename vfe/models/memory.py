"""Instance-level feature memory for MAMBA. Port of ``mmdet.models.memory``.

At test time MAMBA does not re-extract reference frames for every key frame.
The first frame of a video fills the memory with RoI features of its reference
frames; every later frame *reads* a random subset of the memory as its
references and *writes* its own top-k enhanced RoI features back. The memory
is therefore a growing, randomly thinned sample of the whole video so far.

Behaviour worth knowing, all kept as in the original:

* ``sample()`` returns the whole memory while it holds fewer than
  ``key_length`` features, and a random ``key_length``-subset after that.
* ``update()`` appends while the memory is under ``max_length`` -- so it can
  overshoot by up to one update's worth -- and after that keeps a random
  subset of the old features, *in random order*, followed by the new ones.
* Both draw from the global **CPU** RNG (``torch.randperm`` without a device),
  whatever device the features live on.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["MemoryBank"]


class MemoryBank(nn.Module):
    """Args:
        max_length: size past which updates replace rather than append.
        key_length: size past which reads return a random subset.
        sampling_policy / updating_policy: only ``'random'`` exists.
    """

    def __init__(
        self,
        max_length: int = 20000,
        key_length: int = 2000,
        sampling_policy: str = "random",
        updating_policy: str = "random",
    ):
        super().__init__()
        if sampling_policy != "random" or updating_policy != "random":
            raise NotImplementedError("only the 'random' sampling/updating policies exist")
        self.max_length = max_length
        self.key_length = key_length
        self.sampling_policy = sampling_policy
        self.updating_policy = updating_policy
        # A plain attribute, not a buffer: the memory is per-video inference
        # state and must never end up in a checkpoint.
        self.feat: torch.Tensor | None = None

    def reset(self) -> None:
        self.feat = None

    def init_memory(self, feat: torch.Tensor) -> None:
        """Replace the memory with ``feat`` of shape ``(n, c)``."""
        self.feat = feat

    def sample(self) -> torch.Tensor:
        if self.feat is None:
            # mmdet returned [] here, which then failed obscurely inside the
            # aggregator. Reaching this means a video was not started with its
            # first frame, so say that instead.
            raise RuntimeError(
                "MemoryBank.sample() before any features were written; the first frame "
                "of a video must be processed (with its reference frames) first"
            )
        if len(self.feat) < self.key_length:
            return self.feat
        sampled_ind = torch.randperm(len(self.feat))[: self.key_length]
        return self.feat[sampled_ind]

    def update(self, new_feat: torch.Tensor) -> None:
        if self.feat is None:
            self.feat = new_feat
            return
        if len(self.feat) < self.max_length:
            self.feat = torch.cat([self.feat, new_feat], dim=0)
            return
        reserved_ind = torch.randperm(len(self.feat))[: -len(new_feat)]
        self.feat = torch.cat([self.feat[reserved_ind], new_feat], dim=0)

    def __len__(self) -> int:
        return 0 if self.feat is None else len(self.feat)
