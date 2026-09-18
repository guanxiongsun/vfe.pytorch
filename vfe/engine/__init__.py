"""Everything that runs a model: the training and test loops that replace
mmcv's runner and its hooks, plus the optimiser, LR schedule, gradient
clipping, checkpoints and logging they are built from."""

from .lr_scheduler import StepLrScheduler, build_lr_scheduler
from .optimizer import build_optimizer, clip_grads, paramwise_settings

# Imported after the pieces above: trainer reads them from their own modules.
from .evaluator import (  # isort: skip
    evaluation_kwargs,
    make_tmpdir,
    multi_gpu_test,
    single_gpu_test,
    to_device,
)
from .trainer import RngStreams, rescaled_iteration, train_detector, train_step  # isort: skip

__all__ = ["build_optimizer", "clip_grads", "paramwise_settings", "StepLrScheduler",
           "build_lr_scheduler", "single_gpu_test", "multi_gpu_test", "make_tmpdir",
           "to_device", "evaluation_kwargs", "train_detector", "train_step", "RngStreams",
           "rescaled_iteration"]
