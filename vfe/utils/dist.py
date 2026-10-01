"""Distributed-process helpers (replace ``mmcv.runner.{init_dist,get_dist_info}``).

Processes are started by ``torchrun``, which sets ``RANK``, ``WORLD_SIZE``,
``LOCAL_RANK``, ``MASTER_ADDR`` and ``MASTER_PORT``; each process binds to the
GPU numbered by its local rank.
"""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist

__all__ = ["get_dist_info", "init_dist", "is_main_process"]

# How long a collective may wait. While rank 0 scores an evaluation (COCO-style
# AP over ImageNet VID's 176K frames takes about 17 minutes), the other ranks
# wait at a barrier. PyTorch 2's NCCL default (10 minutes, then abort) killed
# such training runs at their final evaluation; torch 1.10, which the
# original stack used, waited.
COLLECTIVE_TIMEOUT = timedelta(hours=2)


def get_dist_info() -> tuple[int, int]:
    """``(rank, world_size)``; ``(0, 1)`` when not distributed."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def is_main_process() -> bool:
    return get_dist_info()[0] == 0


def init_dist(backend: str = "nccl") -> None:
    """Join the process group described by torchrun's environment. With NCCL,
    each process binds to the GPU numbered by its local rank; ``gloo`` runs
    on CPU (local tests)."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("init_dist expects to run under torchrun (RANK/WORLD_SIZE unset)")
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend=backend, timeout=COLLECTIVE_TIMEOUT)
