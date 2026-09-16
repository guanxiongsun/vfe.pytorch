"""Samplers. Port of ``mmdet.datasets.samplers.distributed_video_sampler``."""

from __future__ import annotations

import logging

from torch.utils.data import Sampler

__all__ = ["DistributedVideoSampler"]

logger = logging.getLogger(__name__)


class DistributedVideoSampler(Sampler[int]):
    """Split a test set across ranks without splitting any video.

    Video detectors with cross-frame state (MAMBA's memory) must see each video
    on one rank, in order. Each rank gets a contiguous range of dataset indices
    starting at a video's first frame; boundaries are placed after roughly
    ``len(dataset) / num_replicas`` frames. Concatenating the ranks' results in
    rank order therefore restores dataset order, which evaluation relies on.

    Args:
        dataset: must have ``data_infos`` with ``frame_id``, and each video's
            first frame (``frame_id == 0``) must come first in that video.
    """

    def __init__(self, dataset, num_replicas: int, rank: int):
        first_frames = [i for i, info in enumerate(dataset.data_infos) if info["frame_id"] == 0]
        if len(first_frames) < num_replicas:
            raise ValueError(f"only {len(first_frames)} videos for {num_replicas} ranks")

        num_samples = len(dataset)
        frames_per_rank = num_samples // num_replicas
        split_flags = [0]
        for first in first_frames:
            if first - split_flags[-1] > frames_per_rank:
                split_flags.append(first)
            if len(split_flags) == num_replicas:
                break
        split_flags.append(num_samples)
        if len(split_flags) != num_replicas + 1:
            raise ValueError(f"could not split {len(first_frames)} videos over {num_replicas} "
                             "ranks without leaving a rank empty")

        self.indices = [list(range(split_flags[i], split_flags[i + 1]))
                        for i in range(num_replicas)]
        self.rank = rank
        if rank == 0:
            logger.info("frames per rank: %s", [len(ix) for ix in self.indices])

    def __iter__(self):
        return iter(self.indices[self.rank])

    def __len__(self) -> int:
        return len(self.indices[self.rank])
