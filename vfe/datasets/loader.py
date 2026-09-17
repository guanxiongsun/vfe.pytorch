"""Training data loaders. Port of ``mmdet.datasets.builder.build_dataloader``
for the epoch-based training the configs use, extended for gradient
accumulation.

The original data stream
------------------------
The released models trained on 8 GPUs with one image each. Rank ``r``'s
loader drew indices from ``DistributedGroupSampler(num_replicas=8, rank=r,
seed=seed)`` and ran ``workers_per_gpu`` worker processes; worker ``w`` seeded
``random`` and numpy with ``workers_per_gpu * r + w + seed``. Reference-frame
sampling and flips draw from those generators, so a sample's augmentation
depends on which worker of which rank loaded it and in what order. Workers
are not persistent: they restart every epoch, replaying the same streams over
that epoch's reshuffled indices.

Virtual ranks
-------------
With gradient accumulation, ``world_size * accumulate`` *virtual ranks* stand
in for the original GPUs. Process ``p`` owns virtual ranks ``p * accumulate``
to ``(p + 1) * accumulate - 1`` and runs one loader for each, built exactly as
the original rank's loader was, so every sample is drawn and augmented as in
the original run. This needs ``workers_per_gpu > 0``: without worker processes
all of a process's loaders would share its one random stream.
"""

from __future__ import annotations

import random
from functools import partial

import numpy as np
from torch.utils.data import DataLoader

from vfe.datasets.collate import collate_video_train
from vfe.datasets.samplers import DistributedGroupSampler, GroupSampler

__all__ = ["build_train_loaders", "virtual_ranks", "worker_init_fn"]


def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int) -> None:
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def virtual_ranks(rank: int, accumulate: int) -> list[int]:
    """The virtual ranks process ``rank`` runs, one per micro-step, in order."""
    return list(range(rank * accumulate, (rank + 1) * accumulate))


def build_train_loaders(dataset, samples_per_gpu: int, workers_per_gpu: int,
                        world_size: int = 1, rank: int = 0, accumulate: int = 1,
                        seed: int | None = None) -> list[DataLoader]:
    """One loader per micro-step of process ``rank``.

    With a single virtual rank (one process, no accumulation) this is mmdet's
    non-distributed loader: ``GroupSampler``, shuffling with numpy's global
    generator in the main process.
    """
    if accumulate < 1:
        raise ValueError(f"accumulate must be >= 1, got {accumulate}")
    virtual_world = world_size * accumulate
    if accumulate > 1 and workers_per_gpu == 0:
        raise ValueError("gradient accumulation needs workers_per_gpu > 0 to reproduce "
                         "each virtual rank's random stream")
    loaders = []
    for vrank in virtual_ranks(rank, accumulate):
        if virtual_world > 1:
            sampler = DistributedGroupSampler(dataset, samples_per_gpu, virtual_world, vrank,
                                              seed=seed)
        else:
            sampler = GroupSampler(dataset, samples_per_gpu)
        init_fn = (partial(worker_init_fn, num_workers=workers_per_gpu, rank=vrank, seed=seed)
                   if seed is not None else None)
        loaders.append(DataLoader(
            dataset,
            batch_size=samples_per_gpu,
            sampler=sampler,
            num_workers=workers_per_gpu,
            collate_fn=collate_video_train,
            pin_memory=False,
            worker_init_fn=init_fn,
            persistent_workers=False,
        ))
    return loaders
