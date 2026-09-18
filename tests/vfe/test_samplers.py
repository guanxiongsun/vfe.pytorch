"""Training samplers and the virtual-rank layout of gradient accumulation."""

import math

import numpy as np
import pytest

from vfe.datasets.loader import build_train_loaders, virtual_ranks, worker_init_fn
from vfe.datasets.samplers import DistributedGroupSampler, GroupSampler


class FakeDataset:
    """Only what the samplers read: a length and the aspect-ratio group flags."""

    def __init__(self, flag):
        self.flag = np.asarray(flag, dtype=np.uint8)

    def __len__(self):
        return len(self.flag)

    def __getitem__(self, idx):
        return idx


@pytest.fixture
def dataset():
    rng = np.random.RandomState(0)
    return FakeDataset(rng.randint(0, 2, size=103))


def test_distributed_group_sampler_length(dataset):
    world = 8
    sizes = np.bincount(dataset.flag)
    expected = sum(math.ceil(size / world) for size in sizes)
    for rank in range(world):
        assert len(DistributedGroupSampler(dataset, 1, world, rank, seed=5)) == expected


def test_distributed_group_sampler_covers_every_index_and_is_seeded(dataset):
    world = 8
    samplers = [DistributedGroupSampler(dataset, 1, world, rank, seed=5) for rank in range(world)]
    for epoch in (0, 1):
        for sampler in samplers:
            sampler.set_epoch(epoch)
        shards = [list(sampler) for sampler in samplers]
        assert set().union(*shards) == set(range(len(dataset)))  # padding only repeats indices
        assert sum(len(s) for s in shards) == world * len(samplers[0])
    samplers[0].set_epoch(0)
    first = list(samplers[0])
    samplers[0].set_epoch(1)
    assert list(samplers[0]) != first  # reshuffled per epoch
    samplers[0].set_epoch(0)
    assert list(samplers[0]) == first  # and reproducible


def test_group_sampler_keeps_groups_together():
    dataset = FakeDataset([0, 1] * 10 + [1] * 3)
    np.random.seed(0)
    indices = list(GroupSampler(dataset, samples_per_gpu=2))
    assert len(indices) % 2 == 0
    for i in range(0, len(indices), 2):
        assert dataset.flag[indices[i]] == dataset.flag[indices[i + 1]]


def test_virtual_ranks():
    assert [virtual_ranks(p, 2) for p in range(4)] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert virtual_ranks(3, 1) == [3]


def test_accumulation_loaders_use_the_original_ranks_samplers(dataset):
    loaders = build_train_loaders(dataset, 1, 1, world_size=4, rank=2, accumulate=2, seed=7)
    assert [loader.sampler.rank for loader in loaders] == [4, 5]
    assert all(loader.sampler.num_replicas == 8 for loader in loaders)


def test_accumulation_needs_worker_processes(dataset):
    with pytest.raises(ValueError):
        build_train_loaders(dataset, 1, 0, world_size=1, rank=0, accumulate=2, seed=7)


def test_worker_seeds_follow_mmdet():
    worker_init_fn(worker_id=3, num_workers=4, rank=5, seed=100)
    after = np.random.rand()
    np.random.seed(4 * 5 + 3 + 100)
    assert after == np.random.rand()
