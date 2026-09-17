"""Epoch-based training. Replaces ``mmdet.apis.train_detector`` together with
the mmcv ``EpochBasedRunner`` and hooks it assembled. Each epoch runs in the
order those hooks did:

* start: every LR to the epoch's regular value (``LrUpdaterHook``),
  ``sampler.set_epoch`` (``DistSamplerSeedHook``), log history cleared;
* each iteration: warmup LR (``LrUpdaterHook``); forward, ``forward_train`` then
  ``parse_losses`` (``train_step``); zero_grad, backward, gradient clipping and
  the optimiser step (``OptimizerHook``); every ``log_config.interval``
  iterations, a log entry averaging the interval (``TextLoggerHook``);
* end: a checkpoint every ``checkpoint_config.interval`` epochs and after the
  last (``CheckpointHook``), then evaluation every ``evaluation.interval``
  epochs, logged as a ``val`` entry (``DistEvalHook``).

Gradient accumulation (``accumulate > 1``) runs each original iteration's
``world_size * accumulate`` samples as micro-steps; :mod:`vfe.datasets.loader`
keeps every sample and its augmentation as in the original run. Each
micro-step's loss is divided by ``accumulate`` and DDP synchronises on the
last micro-step only, so the optimiser receives the mean gradient over all of
them: the original batch's gradient, provided no layer uses batch statistics.
MAMBA and STPN freeze every BatchNorm, and the loop refuses to accumulate
otherwise.

Each original process also drew from its own torch generators (CPU and CUDA),
seeded alike: RPN and RoI sampling, and the data loader's base seed at the
start of every epoch. :class:`RngStreams` keeps one generator state per virtual
rank and swaps it in around that rank's micro-step and iterator creation, so
one process replays every virtual rank's random decisions. With identical
kernels, ``n`` processes with ``k`` micro-steps then reproduce ``n * k``
processes exactly (halving a loss halves every gradient exactly).

Two deliberate departures from mmcv, both for exactness: checkpoints also store
every process's generator states, so resuming continues the random streams
instead of restarting them; and evaluation runs on a forked generator, so it
never shifts the training streams.
"""

from __future__ import annotations

import contextlib
import logging
import os.path as osp
import random
import time
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from vfe.apis.test import (
    evaluation_kwargs,
    make_tmpdir,
    multi_gpu_test,
    single_gpu_test,
    to_device,
)
from vfe.datasets import build_dataset, collate_video_test
from vfe.datasets.loader import build_train_loaders
from vfe.datasets.samplers import DistributedVideoSampler
from vfe.engine import build_lr_scheduler, build_optimizer, clip_grads
from vfe.engine.checkpoint import resume_checkpoint, save_checkpoint
from vfe.engine.train_log import LogBuffer, TrainLogger
from vfe.models.checkpoint import load_checkpoint
from vfe.models.detectors.base import parse_losses
from vfe.utils import get_dist_info

__all__ = ["RngStreams", "train_detector", "train_step"]

# mmdet's NumClassCheckHook only asserts that heads match the dataset's classes;
# check_num_classes does the same once, up front.
SUPPORTED_CUSTOM_HOOKS = ("NumClassCheckHook",)


def check_runtime_config(cfg) -> None:
    """Refuse runtime features this loop does not implement instead of ignoring them."""
    if cfg.get("fp16") is not None:
        raise NotImplementedError("fp16 training is not ported (the released models are fp32)")
    if cfg.get("momentum_config") is not None:
        raise NotImplementedError("momentum_config is not ported")
    workflow = [tuple(w) for w in cfg.get("workflow", [("train", 1)])]
    if workflow != [("train", 1)]:
        raise NotImplementedError(f"workflow {workflow}: only [('train', 1)] is ported")
    runner_type = (cfg.get("runner") or {}).get("type", "EpochBasedRunner")
    if runner_type != "EpochBasedRunner":
        raise NotImplementedError(f"runner {runner_type!r}: only EpochBasedRunner is ported")
    for hook in cfg.get("custom_hooks") or []:
        if hook.get("type") not in SUPPORTED_CUSTOM_HOOKS:
            raise NotImplementedError(f"custom hook {hook.get('type')!r} is not ported")
    unknown = set(cfg.get("optimizer_config") or {}) - {"grad_clip", "_delete_"}
    if unknown:
        raise NotImplementedError(f"optimizer_config keys {sorted(unknown)} are not ported")


def check_num_classes(model: nn.Module, classes) -> None:
    """mmdet's ``NumClassCheckHook``: every head with ``num_classes`` except the
    RPN must predict the dataset's classes."""
    for name, module in model.named_modules():
        if hasattr(module, "num_classes") and type(module).__name__ != "RPNHead" \
                and module.num_classes != len(classes):
            raise ValueError(f"{name}.num_classes = {module.num_classes}, but the dataset has "
                             f"{len(classes)} classes")


def _require_frozen_batch_norm(model: nn.Module) -> None:
    training = [name for name, m in model.named_modules()
                if isinstance(m, _BatchNorm) and m.training]
    if training:
        raise ValueError("gradient accumulation matches the original batch only when no "
                         f"BatchNorm uses batch statistics; in training mode: {training[:3]}...")


class RngStreams:
    """Torch generator states (CPU, and CUDA on ``device``), one per virtual
    rank, all starting from the current state."""

    def __init__(self, num_streams: int, device: torch.device):
        self.device = device if device.type == "cuda" else None
        self.cpu = [torch.get_rng_state() for _ in range(num_streams)]
        self.cuda = ([torch.cuda.get_rng_state(device) for _ in range(num_streams)]
                     if self.device is not None else None)

    def state_dict(self) -> dict[str, Any]:
        """Every stream's torch states, plus this process's numpy and Python
        generators, in types a ``weights_only`` load accepts."""
        if len(self.cpu) == 1:  # the global generators are the stream
            cpu = [torch.get_rng_state()]
            cuda = [torch.cuda.get_rng_state(self.device)] if self.device is not None else None
        else:
            cpu, cuda = list(self.cpu), list(self.cuda) if self.cuda is not None else None
        name, keys, pos, has_gauss, cached = np.random.get_state()
        return {"cpu": cpu, "cuda": cuda,
                "numpy": (name, torch.from_numpy(keys.astype(np.int64)), pos, has_gauss, cached),
                "python": random.getstate()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if len(state["cpu"]) != len(self.cpu) or (state["cuda"] is None) != (self.cuda is None):
            raise ValueError("generator states were saved for a different layout")
        if len(self.cpu) == 1:
            torch.set_rng_state(state["cpu"][0])
            if self.device is not None:
                torch.cuda.set_rng_state(state["cuda"][0], self.device)
        else:
            self.cpu = list(state["cpu"])
            self.cuda = list(state["cuda"]) if self.cuda is not None else None
        name, keys, pos, has_gauss, cached = state["numpy"]
        np.random.set_state((name, keys.numpy().astype(np.uint32), pos, has_gauss, cached))
        random.setstate(state["python"])

    @contextlib.contextmanager
    def use(self, stream: int):
        if len(self.cpu) == 1:  # nothing to swap
            yield
            return
        torch.set_rng_state(self.cpu[stream])
        if self.cuda is not None:
            torch.cuda.set_rng_state(self.cuda[stream], self.device)
        try:
            yield
        finally:
            self.cpu[stream] = torch.get_rng_state()
            if self.cuda is not None:
                self.cuda[stream] = torch.cuda.get_rng_state(self.device)


def train_step(model: nn.Module, batches: list[dict], optimizer: torch.optim.Optimizer,
               device: torch.device, rng: RngStreams | None = None) -> OrderedDict[str, float]:
    """Forward and backward over one iteration's micro-batches (one per virtual
    rank, drawing from its own generators); the averaged gradient is left in
    ``.grad``. Returns the log variables averaged over the micro-steps (each
    already averaged over processes by ``parse_losses``)."""
    optimizer.zero_grad(set_to_none=True)
    k = len(batches)
    log_vars: OrderedDict[str, float] = OrderedDict()
    for m, batch in enumerate(batches):
        skip_sync = isinstance(model, DistributedDataParallel) and m < k - 1
        with rng.use(m) if rng is not None else contextlib.nullcontext(), \
                model.no_sync() if skip_sync else contextlib.nullcontext():
            loss, step_vars = parse_losses(model(**to_device(batch, device)))
            (loss / k).backward()
        for key, value in step_vars.items():
            log_vars[key] = log_vars.get(key, 0.0) + value / k
    return log_vars


def train_detector(model: nn.Module, dataset, cfg, *, work_dir: str, timestamp: str,
                   meta: dict[str, Any], logger: logging.Logger, device: torch.device,
                   seed: int | None, distributed: bool, validate: bool = True,
                   accumulate: int = 1, resume_from: str | None = None,
                   load_from: str | None = None, max_epochs: int | None = None,
                   max_iters_per_epoch: int | None = None) -> None:
    check_runtime_config(cfg)
    check_num_classes(model, dataset.CLASSES)
    rank, world_size = get_dist_info()
    data_cfg = cfg.data

    loaders = build_train_loaders(dataset, data_cfg.samples_per_gpu, data_cfg.workers_per_gpu,
                                  world_size, rank, accumulate, seed)
    iters_per_epoch = len(loaders[0])
    if max_iters_per_epoch is not None:
        iters_per_epoch = min(iters_per_epoch, max_iters_per_epoch)
    max_epochs = max_epochs or cfg.runner.max_epochs
    samples_per_iter = data_cfg.samples_per_gpu * accumulate

    model = model.to(device)
    ddp = model
    if distributed:
        ddp = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=cfg.get("find_unused_parameters", False))
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_lr_scheduler(optimizer, cfg.lr_config)
    grad_clip = (cfg.get("optimizer_config") or {}).get("grad_clip")

    rng = RngStreams(accumulate, device)
    start_epoch = global_iter = 0
    if resume_from:
        checkpoint = resume_checkpoint(model, resume_from, optimizer)
        start_epoch, global_iter = checkpoint["meta"]["epoch"], checkpoint["meta"]["iter"]
        # As mmcv's runner: an epoch holds fewer iterations on more GPUs, so the
        # iteration count is rescaled when the (virtual) GPU count changes. The
        # published MAMBA run did this when it resumed its 4-GPU epoch 3 on 8.
        saved_world = checkpoint["meta"].get("virtual_world_size")
        if saved_world and saved_world != world_size * accumulate:
            global_iter = int(global_iter * saved_world / (world_size * accumulate))
            logger.info("the iteration number is changed due to change of GPU number "
                        "(%d -> %d)", saved_world, world_size * accumulate)
        logger.info("resumed epoch %d, iter %d from %s", start_epoch, global_iter, resume_from)
        states = checkpoint.get("rng_states")
        try:
            if states is None or len(states) != world_size:
                raise ValueError("no generator states for this number of processes")
            rng.load_state_dict(states[rank])
        except ValueError as err:
            logger.warning("random streams restart from the seed (%s)", err)
        del checkpoint
    elif load_from:
        load_checkpoint(model, load_from, map_location="cpu", log=logger.warning)
        logger.info("loaded weights from %s", load_from)

    eval_cfg = dict(cfg.get("evaluation") or {})
    eval_interval = eval_cfg.get("interval", 1)
    val_dataset = val_loader = None
    if validate:
        val_dataset = build_dataset(cfg.data.val)
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            sampler=DistributedVideoSampler(val_dataset, world_size, rank) if distributed else None,
            num_workers=data_cfg.workers_per_gpu, collate_fn=collate_video_test)

    log_buffer = LogBuffer()
    log_interval = cfg.log_config.interval
    ckpt_interval = cfg.checkpoint_config.interval
    train_log = TrainLogger(logger, osp.join(work_dir, f"{timestamp}.log.json"), rank,
                            iters_per_epoch, max_epochs * iters_per_epoch, log_interval,
                            start_iter=global_iter)
    train_log.dump(meta)
    logger.info("Start running, work_dir: %s; %d epochs of %d iterations; %d process(es) x %d "
                "micro-step(s) x %d sample(s)", work_dir, max_epochs, iters_per_epoch,
                world_size, accumulate, data_cfg.samples_per_gpu)

    for epoch in range(start_epoch, max_epochs):
        ddp.train()
        if accumulate > 1:
            _require_frozen_batch_norm(model)
        scheduler.before_epoch(epoch)
        for loader in loaders:
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
        log_buffer.clear()
        iterators = []
        for m, loader in enumerate(loaders):
            with rng.use(m):  # the iterator draws the workers' base seed
                iterators.append(iter(loader))
        t = time.time()
        for inner_iter in range(iters_per_epoch):
            scheduler.before_iter(global_iter)
            batches = [next(it) for it in iterators]
            log_buffer.update({"data_time": time.time() - t})

            log_buffer.update(train_step(ddp, batches, optimizer, device, rng), samples_per_iter)
            if grad_clip is not None:
                grad_norm = clip_grads(model.parameters(), **grad_clip)
                if grad_norm is not None:
                    log_buffer.update({"grad_norm": float(grad_norm)}, samples_per_iter)
            optimizer.step()

            log_buffer.update({"time": time.time() - t})
            t = time.time()
            if (inner_iter + 1) % log_interval == 0:
                log_buffer.average(log_interval)
                train_log.log_train(epoch + 1, inner_iter + 1, global_iter,
                                    optimizer.param_groups[0]["lr"],
                                    train_log.max_memory_mb(device), log_buffer.output)
                log_buffer.output.clear()
            global_iter += 1
        del iterators  # stop this epoch's workers before checkpointing and evaluating

        epoch_done = epoch + 1
        if epoch_done % ckpt_interval == 0 or epoch_done == max_epochs:
            rng_states = [rng.state_dict()]
            if distributed:
                rng_states = [None] * world_size
                dist.all_gather_object(rng_states, rng.state_dict())
            if rank == 0:
                logger.info("Saving checkpoint at %d epochs", epoch_done)
                save_checkpoint(model, osp.join(work_dir, f"epoch_{epoch_done}.pth"), optimizer,
                                meta={**meta, "epoch": epoch_done, "iter": global_iter,
                                      "virtual_world_size": world_size * accumulate},
                                extra={"rng_states": rng_states})
            if distributed:
                dist.barrier()

        if validate and epoch_done % eval_interval == 0:
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                if distributed:
                    results = multi_gpu_test(model, val_loader, device, make_tmpdir(work_dir))
                else:
                    results = single_gpu_test(model, val_loader, device)
            if rank == 0:
                metrics = val_dataset.evaluate(results, **evaluation_kwargs(eval_cfg))
                train_log.log_val(epoch_done, len(val_dataset), optimizer.param_groups[0]["lr"],
                                  metrics)
            if distributed:
                dist.barrier()
