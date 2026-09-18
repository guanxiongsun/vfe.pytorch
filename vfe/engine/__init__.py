"""Training machinery that replaces mmcv's runner: optimiser construction,
gradient clipping and the LR schedule so far (Phase 6)."""

from .lr_scheduler import StepLrScheduler, build_lr_scheduler
from .optimizer import build_optimizer, clip_grads, paramwise_settings

__all__ = ["build_optimizer", "clip_grads", "paramwise_settings", "StepLrScheduler",
           "build_lr_scheduler"]
