"""Learning-rate schedules. Port of mmcv's ``StepLrUpdaterHook`` (``by_epoch=True``)
and mmdet's ``YOLOXLrUpdaterHook`` (below).

The LR is decided at two moments, exactly as mmcv's hook decided it:

* ``before_epoch(epoch)`` (0-based): every group goes to its regular value,
  ``base_lr * gamma ** (number of steps <= epoch)``, clipped at ``min_lr``.
* ``before_iter(global_iter)`` (0-based, counted across epochs): while
  ``global_iter < warmup_iters``, the warmup value; at ``global_iter ==
  warmup_iters``, the regular value; after that nothing changes until the next
  epoch.

With ``warmup='linear'`` the warmup value is
``regular * (1 - (1 - i / warmup_iters) * (1 - warmup_ratio))``, so it starts
at ``warmup_ratio * regular``. Epoch boundaries come from the data loader, so
the schedule depends on the global batch size: MAMBA's 13,711 iterations per
epoch assume 8 images per step.
"""

from __future__ import annotations

import math
from typing import Any

import torch

__all__ = ["StepLrScheduler", "YOLOXLrScheduler", "build_lr_scheduler"]


class StepLrScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, step: int | list[int],
                 gamma: float = 0.1, min_lr: float | None = None, warmup: str | None = None,
                 warmup_iters: int = 0, warmup_ratio: float = 0.1, by_epoch: bool = True,
                 warmup_by_epoch: bool = False):
        if not by_epoch or warmup_by_epoch:
            raise NotImplementedError("only by_epoch=True with iteration warmup is ported")
        if warmup not in (None, "constant", "linear", "exp"):
            raise ValueError(f"unknown warmup {warmup!r}")
        if warmup is not None and (warmup_iters <= 0 or not 0 < warmup_ratio <= 1):
            raise ValueError("warmup needs warmup_iters > 0 and 0 < warmup_ratio <= 1")
        if isinstance(step, list) and not all(isinstance(s, int) and s > 0 for s in step):
            raise ValueError(f"step must be positive ints, got {step}")
        self.optimizer = optimizer
        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lr = [group["initial_lr"] for group in optimizer.param_groups]
        self.regular_lr: list[float] = []

    def _lr(self, epoch: int, base_lr: float) -> float:
        if isinstance(self.step, int):
            exp = epoch // self.step
        else:
            exp = next((i for i, s in enumerate(self.step) if epoch < s), len(self.step))
        lr = base_lr * (self.gamma ** exp)
        return lr if self.min_lr is None else max(lr, self.min_lr)

    def _set(self, lrs: list[float]) -> None:
        for group, lr in zip(self.optimizer.param_groups, lrs, strict=True):
            group["lr"] = lr

    def _warmup_lr(self, cur_iter: int) -> list[float]:
        if self.warmup == "constant":
            return [lr * self.warmup_ratio for lr in self.regular_lr]
        if self.warmup == "linear":
            k = (1 - cur_iter / self.warmup_iters) * (1 - self.warmup_ratio)
            return [lr * (1 - k) for lr in self.regular_lr]
        k = self.warmup_ratio ** (1 - cur_iter / self.warmup_iters)
        return [lr * k for lr in self.regular_lr]

    def before_epoch(self, epoch: int) -> None:
        self.regular_lr = [self._lr(epoch, base) for base in self.base_lr]
        self._set(self.regular_lr)

    def before_iter(self, global_iter: int) -> None:
        if self.warmup is None or global_iter > self.warmup_iters:
            return
        if global_iter == self.warmup_iters:
            self._set(self.regular_lr)
        else:
            self._set(self._warmup_lr(global_iter))

    def state_dict(self) -> dict[str, Any]:
        return {"base_lr": list(self.base_lr), "regular_lr": list(self.regular_lr)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.base_lr = list(state["base_lr"])
        self.regular_lr = list(state["regular_lr"])


class YOLOXLrScheduler:
    """mmdet's ``YOLOXLrUpdaterHook`` (mmcv's cosine annealing, ``by_epoch=
    False``, epoch-counted warmup), decided before every iteration ``i``
    (0-based, across epochs) of ``T = iters_per_epoch * max_epochs``:

    * ``i < W`` (``W = warmup_iters`` epochs of iterations): the ``'exp'``
      warmup, ``base * warmup_ratio * ((i + 1) / W) ** 2``;
    * then, with ``L = num_last_epochs`` epochs of iterations and ``p = i + 1``:
      ``base * min_lr_ratio`` once ``p >= T - L``, else cosine annealing from
      ``base`` to that value over ``(p - W) / (T - W - L)``.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, num_last_epochs: int,
                 iters_per_epoch: int, max_epochs: int, min_lr_ratio: float = 0.05,
                 warmup: str = "exp", warmup_iters: int = 5, warmup_ratio: float = 1.0,
                 warmup_by_epoch: bool = True, by_epoch: bool = False):
        if by_epoch or not warmup_by_epoch or warmup != "exp":
            raise NotImplementedError("the YOLOX policy is ported with by_epoch=False, "
                                      "warmup='exp' and warmup_by_epoch=True only")
        self.optimizer = optimizer
        self.min_lr_ratio = min_lr_ratio
        self.warmup_ratio = warmup_ratio
        self.warmup_iters = warmup_iters * iters_per_epoch
        self.last_iters = num_last_epochs * iters_per_epoch
        self.max_iters = max_epochs * iters_per_epoch
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lr = [group["initial_lr"] for group in optimizer.param_groups]

    def _regular(self, base_lr: float, global_iter: int) -> float:
        target = base_lr * self.min_lr_ratio
        progress = global_iter + 1
        if progress >= self.max_iters - self.last_iters:
            return target
        factor = (progress - self.warmup_iters) / (
            self.max_iters - self.warmup_iters - self.last_iters)
        return target + 0.5 * (base_lr - target) * (math.cos(math.pi * factor) + 1)

    def before_epoch(self, epoch: int) -> None:
        pass

    def before_iter(self, global_iter: int) -> None:
        if global_iter < self.warmup_iters:
            k = self.warmup_ratio * ((global_iter + 1) / self.warmup_iters) ** 2
            lrs = [base * k for base in self.base_lr]
        else:
            lrs = [self._regular(base, global_iter) for base in self.base_lr]
        for group, lr in zip(self.optimizer.param_groups, lrs, strict=True):
            group["lr"] = lr

    def state_dict(self) -> dict[str, Any]:
        return {"base_lr": list(self.base_lr)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.base_lr = list(state["base_lr"])


def build_lr_scheduler(optimizer: torch.optim.Optimizer, lr_config: dict[str, Any],
                       iters_per_epoch: int | None = None, max_epochs: int | None = None):
    """``lr_config`` is a config's ``lr_config`` dict; the YOLOX policy also
    needs the epoch length and count."""
    cfg = dict(lr_config)
    cfg.pop("_delete_", None)
    policy = cfg.pop("policy")
    if policy.lower() == "step":
        return StepLrScheduler(optimizer, **cfg)
    if policy == "YOLOX":
        if iters_per_epoch is None or max_epochs is None:
            raise ValueError("the YOLOX policy needs iters_per_epoch and max_epochs")
        return YOLOXLrScheduler(optimizer, iters_per_epoch=iters_per_epoch,
                                max_epochs=max_epochs, **cfg)
    raise NotImplementedError(f"lr policy {policy!r} is not ported")
