"""Test loops. Port of ``mmdet.apis.test.{single_gpu_test,multi_gpu_test}``.

Both run ``model(return_loss=False, rescale=True, **batch)`` over the loader in
order and return one entry per sample. For video detectors the order is the
protocol: frames of a video must arrive sequentially on the same process.
"""

from __future__ import annotations

import os
import os.path as osp
import pickle
import shutil
import tempfile
import time

import torch
import torch.distributed as dist

from vfe.utils import get_dist_info

__all__ = ["single_gpu_test", "multi_gpu_test", "to_device"]


def to_device(batch: dict, device: torch.device) -> dict:
    """Move every tensor in a collated batch to ``device``; metas stay put."""
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, list):
            return [move(v) for v in value]
        return value
    return {key: move(value) for key, value in batch.items()}


def _log_progress(done: int, total: int, start: float, every: int = 1000) -> None:
    if done % every == 0 or done == total:
        rate = done / max(time.time() - start, 1e-9)
        eta = (total - done) / max(rate, 1e-9)
        print(f"[test] {done}/{total} frames, {rate:.1f}/s, eta {eta / 60:.1f} min", flush=True)


def single_gpu_test(model: torch.nn.Module, data_loader, device: torch.device) -> list:
    model.eval()
    results, start = [], time.time()
    total = len(data_loader.dataset) if not hasattr(data_loader.sampler, "indices") \
        else len(data_loader.sampler)
    for batch in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **to_device(batch, device))
        results.extend(result)
        _log_progress(len(results), total, start)
    return results


def multi_gpu_test(model: torch.nn.Module, data_loader, device: torch.device,
                   tmpdir: str) -> list | None:
    """Run on every rank; returns the gathered results on rank 0, None elsewhere.

    Each rank pickles its part to ``tmpdir`` (which must be on storage shared
    by all ranks), and rank 0 concatenates the parts in rank order. That is the
    original's collection, and with :class:`DistributedVideoSampler` it restores
    dataset order.
    """
    model.eval()
    rank, world_size = get_dist_info()
    results, start = [], time.time()
    total = len(data_loader.sampler)
    for batch in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **to_device(batch, device))
        results.extend(result)
        if rank == 0:
            _log_progress(len(results), total, start)

    os.makedirs(tmpdir, exist_ok=True)
    with open(osp.join(tmpdir, f"part_{rank}.pkl"), "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    dist.barrier()
    if rank != 0:
        return None
    gathered = []
    for i in range(world_size):
        with open(osp.join(tmpdir, f"part_{i}.pkl"), "rb") as f:
            gathered.extend(pickle.load(f))
    shutil.rmtree(tmpdir)
    return gathered


def make_tmpdir(work_dir: str) -> str:
    """A fresh directory under ``work_dir`` (shared storage), agreed on by all ranks."""
    rank, _ = get_dist_info()
    holder = [tempfile.mkdtemp(prefix=".dist_test_", dir=work_dir) if rank == 0 else None]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(holder, src=0)
    return holder[0]
