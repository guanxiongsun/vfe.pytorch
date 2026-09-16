"""Samplers. Port of ``mmdet.datasets.samplers.{group_sampler,distributed_video_sampler}``.

Training draws from ``GroupSampler`` (one process) or ``DistributedGroupSampler``
(DDP). Both keep landscape and portrait images (the dataset's ``flag``) in
separate batches and pad each group to fill whole batches. The distributed
one shuffles from a generator seeded with ``epoch + seed``, so every rank
agrees on the permutation; ``set_epoch`` must be called each epoch.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
from torch.utils.data import Sampler

__all__ = ["GroupSampler", "DistributedGroupSampler", "DistributedVideoSampler"]

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


class GroupSampler(Sampler[int]):
    """Single-process training sampler; shuffles with numpy's global RNG."""

    def __init__(self, dataset, samples_per_gpu: int = 1):
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = sum(
            int(np.ceil(size / samples_per_gpu)) * samples_per_gpu for size in self.group_sizes
        )

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)) * self.samples_per_gpu - size
            indices.append(np.concatenate([indice, np.random.choice(indice, num_extra)]))
        indices = np.concatenate(indices)
        batches = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(batches).astype(np.int64).tolist()
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


class DistributedGroupSampler(Sampler[int]):
    """DDP training sampler.

    ``num_samples`` per rank is the sum over groups of
    ``ceil(size / samples_per_gpu / num_replicas) * samples_per_gpu``; groups are
    padded by repeating their own shuffled indices. With the MAMBA/STPN data
    and 8 ranks at one image each, that is 13,711, the iterations per epoch in
    the original logs.
    """

    def __init__(self, dataset, samples_per_gpu: int = 1, num_replicas: int = 1, rank: int = 0,
                 seed: int | None = 0):
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0
        self.flag = dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = sum(
            int(math.ceil(size * 1.0 / samples_per_gpu / num_replicas)) * samples_per_gpu
            for size in self.group_sizes
        )
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            indice = indice[list(torch.randperm(int(size), generator=g).numpy())].tolist()
            extra = int(math.ceil(size * 1.0 / self.samples_per_gpu / self.num_replicas)) \
                * self.samples_per_gpu * self.num_replicas - len(indice)
            tmp = indice.copy()
            for _ in range(extra // size):
                indice.extend(tmp)
            indice.extend(tmp[:extra % size])
            indices.extend(indice)
        if len(indices) != self.total_size:
            raise AssertionError(f"{len(indices)} indices, expected {self.total_size}")

        indices = [
            indices[j]
            for i in list(torch.randperm(len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu)
        ]
        offset = self.num_samples * self.rank
        return iter(indices[offset:offset + self.num_samples])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
