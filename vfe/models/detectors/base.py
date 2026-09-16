"""Detector base class and loss parsing. Port of ``mmdet.models.detectors.base``.

What survives is the part the training loop and the VID wrappers rely on: the
``forward`` dispatch between training and inference, and ``parse_losses``,
which turns a head's loss dict into the scalar to backpropagate plus the
values to log. mmcv's runner hooks (``train_step`` / ``val_step``), the async
and ONNX paths, test-time augmentation and ``show_result`` are gone; Phase 6's
training loop calls ``parse_losses`` directly.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
import torch.distributed as dist
from torch import nn

__all__ = ["BaseDetector", "parse_losses"]


def parse_losses(
    losses: Mapping[str, torch.Tensor | list[torch.Tensor]],
) -> tuple[torch.Tensor, OrderedDict[str, float]]:
    """``(loss, log_vars)`` from a head's raw loss dict.

    Each entry is a tensor or a list of them (one per FPN level); lists are
    summed after taking each element's mean. **Only keys containing "loss"
    count towards the total** -- ``acc`` is logged but never backpropagated,
    and a new head whose loss key lacks "loss" would silently not train.

    Under DDP every rank must produce the same keys (otherwise the all-reduce
    below hangs), and the logged values are averaged across ranks. The
    returned ``loss`` is *not* averaged: DDP averages the gradients instead.
    """
    log_vars: OrderedDict[str, torch.Tensor | float] = OrderedDict()
    for name, value in losses.items():
        if isinstance(value, torch.Tensor):
            log_vars[name] = value.mean()
        elif isinstance(value, list):
            log_vars[name] = sum(v.mean() for v in value)
        else:
            raise TypeError(f"{name} is not a tensor or list of tensors")

    loss = sum(value for key, value in log_vars.items() if "loss" in key)

    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        num_vars = torch.tensor(len(log_vars), device=loss.device)
        dist.all_reduce(num_vars)
        if num_vars != len(log_vars) * dist.get_world_size():
            raise RuntimeError(
                f"loss log variables differ across ranks; rank {dist.get_rank()} has "
                f"{len(log_vars)}: {','.join(log_vars)}"
            )

    log_vars["loss"] = loss
    for name, value in log_vars.items():
        if distributed:
            value = value.data.clone()
            dist.all_reduce(value.div_(dist.get_world_size()))
        log_vars[name] = value.item()
    return loss, log_vars


class BaseDetector(nn.Module):
    """``forward`` dispatch shared by the still-image detectors."""

    @property
    def with_neck(self) -> bool:
        return getattr(self, "neck", None) is not None

    def extract_feat(self, img: torch.Tensor):
        raise NotImplementedError

    def forward_train(self, img, img_metas, **kwargs):
        raise NotImplementedError

    def simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    def forward(self, img, img_metas, return_loss: bool = True, **kwargs):
        """Losses when ``return_loss``, else detections.

        mmdet's test-time call nests ``img`` and ``img_metas`` one level deeper,
        one entry per test-time augmentation. That form is still accepted for a
        single augmentation; more than one raises, since TTA is not ported.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)

        if isinstance(img, list):
            if len(img) != 1 or len(img_metas) != 1:
                raise NotImplementedError(
                    f"test-time augmentation is not ported; got {len(img)} augmentations"
                )
            img, img_metas = img[0], img_metas[0]
            if "proposals" in kwargs:
                kwargs["proposals"] = kwargs["proposals"][0]
        return self.simple_test(img, img_metas, **kwargs)
