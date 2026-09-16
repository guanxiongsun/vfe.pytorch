"""Learning-rate schedule. Port of mmcv's ``StepLrUpdaterHook`` (``by_epoch=True``).

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

from typing import Any

import torch

__all__ = ["StepLrScheduler", "build_lr_scheduler"]


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


def build_lr_scheduler(optimizer: torch.optim.Optimizer, lr_config: dict[str, Any]
                       ) -> StepLrScheduler:
    """``lr_config`` is a config's ``lr_config`` dict."""
    cfg = dict(lr_config)
    policy = cfg.pop("policy")
    if policy.lower() != "step":
        raise NotImplementedError(f"lr policy {policy!r} is not ported")
    return StepLrScheduler(optimizer, **cfg)
