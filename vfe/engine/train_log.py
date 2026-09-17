"""Training logs in mmcv's formats: ``<timestamp>.log`` (text) and
``<timestamp>.log.json`` (one JSON object per line), so curves from new runs
overlay the original runs' logs with the same parser.

The JSON file starts with the run's meta (environment, config, seed), then one
line per ``log_config.interval`` training iterations::

    {"mode": "train", "epoch": 4, "iter": 50, "lr": 0.001, "memory": 5240,
     "data_time": 2.25163, "loss_rpn_cls": 0.01472, ..., "loss": 0.16645,
     "grad_norm": 3.07772, "time": 2.42548}

and one ``"mode": "val"`` line per evaluation. As in mmcv: ``iter`` counts
within the epoch; values average the interval; floats are rounded to 5
decimals; ``lr`` is the first parameter group's; ``memory`` is the peak CUDA
allocation in MiB since the start, the maximum over ranks; the partial
interval at the end of an epoch is not logged.
"""

from __future__ import annotations

import datetime
import json
import logging
import os.path as osp
import subprocess
import sys
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

__all__ = ["LogBuffer", "TrainLogger", "collect_env", "get_logger"]


class LogBuffer:
    """mmcv's ``LogBuffer``: per-key history, averaged over the last ``n``
    entries weighted by each entry's sample count."""

    def __init__(self):
        self.val_history: OrderedDict[str, list[float]] = OrderedDict()
        self.n_history: OrderedDict[str, list[int]] = OrderedDict()
        self.output: OrderedDict[str, float] = OrderedDict()

    def clear(self) -> None:
        self.val_history.clear()
        self.n_history.clear()
        self.output.clear()

    def update(self, values: dict[str, float], count: int = 1) -> None:
        for key, value in values.items():
            self.val_history.setdefault(key, []).append(value)
            self.n_history.setdefault(key, []).append(count)

    def average(self, n: int = 0) -> None:
        """Averages into ``output``; ``n=0`` averages the whole history."""
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            self.output[key] = float(np.sum(values * nums) / np.sum(nums))


def get_logger(log_file: str | None, rank: int) -> logging.Logger:
    """The ``vfe`` logger: INFO to stdout (and ``log_file``) on rank 0, errors
    only elsewhere."""
    logger = logging.getLogger("vfe")
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s - vfe - %(levelname)s - %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if rank == 0 and log_file is not None:
        handlers.append(logging.FileHandler(log_file, mode="w"))
    for handler in handlers:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if rank == 0 else logging.ERROR)
    return logger


def _round_float(value: Any) -> Any:
    if isinstance(value, list):
        return [_round_float(v) for v in value]
    if isinstance(value, float):
        return round(value, 5)
    return value


class TrainLogger:
    """Writes the text and JSON logs (rank 0 only; other ranks call it too so
    the collective in :meth:`max_memory_mb` lines up)."""

    SKIP_IN_TEXT = ("mode", "epoch", "iter", "lr", "time", "data_time", "memory")

    def __init__(self, logger: logging.Logger, json_path: str, rank: int, iters_per_epoch: int,
                 max_iters: int, interval: int, start_iter: int = 0):
        self.logger = logger
        self.json_path = json_path
        self.rank = rank
        self.iters_per_epoch = iters_per_epoch
        self.max_iters = max_iters
        self.interval = interval
        self.start_iter = start_iter
        self.time_sec_tot = 0.0

    def dump(self, entry: dict[str, Any]) -> None:
        if self.rank != 0:
            return
        with open(self.json_path, "a+") as f:
            f.write(json.dumps(OrderedDict((k, _round_float(v)) for k, v in entry.items())))
            f.write("\n")

    @staticmethod
    def max_memory_mb(device: torch.device) -> int | None:
        if device.type != "cuda":
            return None
        mem = torch.tensor([torch.cuda.max_memory_allocated(device) / (1024 * 1024)],
                           dtype=torch.int, device=device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.reduce(mem, 0, op=torch.distributed.ReduceOp.MAX)
        return int(mem.item())

    def log_train(self, epoch: int, inner_iter: int, global_iter: int, lr: float,
                  memory: int | None, output: dict[str, float]) -> None:
        """``epoch`` and ``inner_iter`` 1-based; ``global_iter`` 0-based, as mmcv's runner."""
        entry: OrderedDict[str, Any] = OrderedDict(mode="train", epoch=epoch, iter=inner_iter,
                                                   lr=lr)
        if memory is not None:
            entry["memory"] = memory
        entry.update(output)

        self.time_sec_tot += output["time"] * self.interval
        time_sec_avg = self.time_sec_tot / (global_iter - self.start_iter + 1)
        eta = datetime.timedelta(seconds=int(time_sec_avg * (self.max_iters - global_iter - 1)))
        text = (f"Epoch [{epoch}][{inner_iter}/{self.iters_per_epoch}]\tlr: {lr:.3e}, eta: {eta}, "
                f"time: {output['time']:.3f}, data_time: {output['data_time']:.3f}, ")
        if memory is not None:
            text += f"memory: {memory}, "
        text += ", ".join(f"{k}: {v:.4f}" for k, v in output.items() if k not in self.SKIP_IN_TEXT)
        self.logger.info(text)
        self.dump(entry)

    def log_val(self, epoch: int, num_samples: int, lr: float, metrics: dict[str, float]) -> None:
        entry = OrderedDict(mode="val", epoch=epoch, iter=num_samples, lr=lr)
        entry.update((k, float(v)) for k, v in metrics.items())
        text = ", ".join(f"{k}: {float(v):.4f}" for k, v in metrics.items())
        self.logger.info(f"Epoch(val) [{epoch}][{num_samples}]\t{text}")
        self.dump(entry)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=osp.dirname(__file__),
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_env() -> str:
    """Environment summary for the log header (mmcv's ``collect_env`` subset)."""
    import cv2
    import torchvision

    env = OrderedDict()
    env["sys.platform"] = sys.platform
    env["Python"] = sys.version.replace("\n", "")
    env["CUDA available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        by_name: dict[str, list[str]] = {}
        for i in range(torch.cuda.device_count()):
            by_name.setdefault(torch.cuda.get_device_name(i), []).append(str(i))
        for name, ids in by_name.items():
            env[f"GPU {','.join(ids)}"] = name
    env["PyTorch"] = torch.__version__
    env["CUDA runtime"] = torch.version.cuda
    env["cuDNN"] = torch.backends.cudnn.version()
    env["TF32 (matmul / cuDNN)"] = (f"{torch.backends.cuda.matmul.allow_tf32} / "
                                    f"{torch.backends.cudnn.allow_tf32}")
    env["TorchVision"] = torchvision.__version__
    env["OpenCV"] = cv2.__version__
    env["NumPy"] = np.__version__
    env["vfe"] = _git_hash()
    return "\n".join(f"{k}: {v}" for k, v in env.items())
