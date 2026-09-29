"""The training hooks YOLOX's configs name in ``custom_hooks``. Port of
mmdet's ``ExpMomentumEMAHook``, ``YOLOXModeSwitchHook`` and ``SyncNormHook``.

The loop in :mod:`vfe.engine.trainer` calls them where mmcv's runner did, in
mmdet's priority order: at an epoch's start the mode switch and norm sync
(48), then the EMA swap (49); after every optimiser step the EMA update; at
an epoch's end the norm sync, then the EMA swap, then the checkpoint and the
evaluation -- which therefore see the averaged weights.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

__all__ = ["ExpMomentumEMAHook", "YOLOXModeSwitchHook", "SyncNormHook", "build_hooks"]


def _head(model: nn.Module) -> nn.Module:
    """The dense head of a detector, or of the detector a video model wraps."""
    detector = getattr(model, "detector", model)
    return detector.bbox_head


class ExpMomentumEMAHook:
    """An exponential moving average of every state-dict entry (parameters
    and buffers), kept as ``ema_*`` buffers of the model so checkpoints carry
    it. The momentum decays from 1 towards ``momentum``:
    ``(1 - m) * exp(-(1 + i) / total_iter) + m`` at iteration ``i``.

    The averaged and the live weights swap places at every epoch's end (so
    the checkpoint and the evaluation get the average) and swap back at the
    next epoch's start.
    """

    def __init__(self, momentum: float = 0.0002, total_iter: int = 2000, interval: int = 1,
                 skip_buffers: bool = False, resume_from=None, priority: Any = None):
        if not 0 < momentum < 1:
            raise ValueError("momentum must be in (0, 1)")
        if resume_from is not None:
            raise NotImplementedError("resume through the trainer's --resume-from instead")
        self.momentum = momentum
        self.total_iter = total_iter
        self.interval = interval
        self.skip_buffers = skip_buffers

    def before_run(self, model: nn.Module) -> None:
        """Register the averages; call on the bare model before DDP and before
        loading a checkpoint that holds them."""
        self.model_parameters = (dict(model.named_parameters()) if self.skip_buffers
                                 else model.state_dict())
        self.param_ema_buffer = {}
        for name, value in self.model_parameters.items():
            buffer_name = f"ema_{name.replace('.', '_')}"
            self.param_ema_buffer[name] = buffer_name
            model.register_buffer(buffer_name, value.data.clone())
        self.model_buffers = dict(model.named_buffers())

    def momentum_at(self, global_iter: int) -> float:
        return ((1 - self.momentum) * math.exp(-(1 + global_iter) / self.total_iter)
                + self.momentum)

    def after_train_iter(self, global_iter: int) -> None:
        if (global_iter + 1) % self.interval != 0:
            return
        momentum = self.momentum_at(global_iter)
        for name, parameter in self.model_parameters.items():
            if parameter.dtype.is_floating_point:
                ema = self.model_buffers[self.param_ema_buffer[name]]
                ema.mul_(1 - momentum).add_(parameter.data, alpha=momentum)

    def _swap(self) -> None:
        for name, value in self.model_parameters.items():
            ema = self.model_buffers[self.param_ema_buffer[name]]
            live = value.data.clone()
            value.data.copy_(ema.data)
            ema.data.copy_(live)

    def before_train_epoch(self, epoch: int, max_epochs: int, model, dataset, logger) -> None:
        self._swap()

    def after_train_epoch(self, epoch: int, model) -> None:
        self._swap()


class YOLOXModeSwitchHook:
    """At the epoch ``max_epochs - num_last_epochs`` (1-based; mmdet's
    counting), switch the training pipeline's ``skip_type_keys`` transforms
    off and the head's L1 loss on. Unlike mmdet's, it also applies to runs
    resumed past that epoch."""

    def __init__(self, num_last_epochs: int = 15,
                 skip_type_keys=("Mosaic", "RandomAffine", "MixUp"), priority: Any = None):
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = tuple(skip_type_keys)
        self._switched = False

    def before_train_epoch(self, epoch: int, max_epochs: int, model, dataset, logger) -> None:
        if self._switched or epoch + 1 < max_epochs - self.num_last_epochs:
            return
        logger.info("No mosaic and mixup aug now! Add additional L1 loss now!")
        dataset.update_skip_type_keys(self.skip_type_keys)
        _head(model).use_l1 = True
        self._switched = True

    def after_train_epoch(self, epoch: int, model) -> None:
        pass


class SyncNormHook:
    """Average every norm layer's state across processes at the end of every
    ``interval``-th epoch, and of every epoch once the last
    ``num_last_epochs`` begin."""

    def __init__(self, num_last_epochs: int = 15, interval: int = 1, priority: Any = None):
        self.num_last_epochs = num_last_epochs
        self.interval = interval

    def before_train_epoch(self, epoch: int, max_epochs: int, model, dataset, logger) -> None:
        if epoch + 1 >= max_epochs - self.num_last_epochs:
            self.interval = 1

    def after_train_epoch(self, epoch: int, model) -> None:
        if (epoch + 1) % self.interval or not (dist.is_available() and dist.is_initialized()):
            return
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        states = OrderedDict()
        for name, child in model.named_modules():
            if isinstance(child, nn.modules.batchnorm._NormBase):
                for key, value in child.state_dict().items():
                    states[f"{name}.{key}"] = value
        if not states:
            return
        flat = torch.cat([v.flatten().float() for v in states.values()])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= world_size
        averaged = OrderedDict()
        for (key, value), part in zip(states.items(), flat.split([v.numel() for v in
                                                                  states.values()]),
                                      strict=True):
            averaged[key] = part.reshape(value.shape)
        model.load_state_dict(averaged, strict=False)


HOOKS = {"ExpMomentumEMAHook": ExpMomentumEMAHook, "YOLOXModeSwitchHook": YOLOXModeSwitchHook,
         "SyncNormHook": SyncNormHook}


def build_hooks(custom_hooks) -> list:
    """The hooks of a config's ``custom_hooks``, in mmdet's priority order
    (the mode switch and norm sync before the EMA). mmdet's
    ``NumClassCheckHook`` is not a hook here: the trainer always checks."""
    hooks = []
    for cfg in custom_hooks or []:
        cfg = dict(cfg)
        kind = cfg.pop("type")
        if kind == "NumClassCheckHook":
            continue
        hooks.append(HOOKS[kind](**cfg))
    order = {"YOLOXModeSwitchHook": 0, "SyncNormHook": 0, "ExpMomentumEMAHook": 1}
    return sorted(hooks, key=lambda h: order[type(h).__name__])
