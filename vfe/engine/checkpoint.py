"""Training checkpoints in mmcv's layout: ``{"meta", "state_dict", "optimizer"}``.

``meta["epoch"]`` counts completed epochs and ``meta["iter"]`` completed
iterations, as mmcv's ``EpochBasedRunner.save_checkpoint`` recorded them, so
resuming continues with the next epoch. Weights load through
:func:`vfe.models.checkpoint.load_checkpoint`, which also reads the originals.

Beyond mmcv, the training loop adds ``rng_states``: every process's random
generator states, so a resumed run continues exactly as an uninterrupted one
would (mmcv restarted the streams from the seed). Other readers ignore the key.
"""

from __future__ import annotations

import os
import os.path as osp
import time
from collections import OrderedDict
from typing import Any

import torch
from torch import nn

from vfe.models.checkpoint import load_checkpoint

__all__ = ["save_checkpoint", "resume_checkpoint"]


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model: nn.Module, filename: str, optimizer: torch.optim.Optimizer | None = None,
                    meta: dict[str, Any] | None = None, link_latest: bool = True,
                    extra: dict[str, Any] | None = None) -> None:
    """Write atomically (a crash mid-save leaves the previous file intact), and
    point ``latest.pth`` next to it at the new file. ``extra`` adds top-level
    entries."""
    model = _unwrap(model)
    meta = dict(meta or {})
    meta["time"] = time.asctime()
    if getattr(model, "CLASSES", None) is not None:
        meta["CLASSES"] = model.CLASSES
    checkpoint = {
        "meta": meta,
        "state_dict": OrderedDict((k, v.detach().cpu()) for k, v in model.state_dict().items()),
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint.update(extra or {})

    tmp = f"{filename}.tmp"
    torch.save(checkpoint, tmp)
    os.replace(tmp, filename)
    if link_latest:
        latest = osp.join(osp.dirname(filename), "latest.pth")
        tmp_link = f"{latest}.tmp"
        if osp.lexists(tmp_link):
            os.remove(tmp_link)
        os.symlink(osp.basename(filename), tmp_link)
        os.replace(tmp_link, latest)


def resume_checkpoint(model: nn.Module, filename: str,
                      optimizer: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    """Load weights (strictly) and optimiser state; returns the whole checkpoint."""
    checkpoint = load_checkpoint(_unwrap(model), filename, map_location="cpu", strict=True)
    if optimizer is not None:
        if "optimizer" not in checkpoint:
            raise KeyError(f"{filename} has no optimizer state to resume from")
        optimizer.load_state_dict(checkpoint["optimizer"])
    meta = checkpoint.get("meta", {})
    for key in ("epoch", "iter"):
        if key not in meta:
            raise KeyError(f"{filename}: meta has no {key!r}; not a training checkpoint")
    return checkpoint
