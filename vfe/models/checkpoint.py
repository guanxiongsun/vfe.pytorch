"""Checkpoint loading, replacing ``mmcv.runner.load_checkpoint``.

Handles the three checkpoint sources this repo's configs actually name:

* ``torchvision://resnet101`` -- the ImageNet backbone MAMBA/SELSA start from.
  mmcv resolved these through torchvision's old ``model_urls`` table; modern
  torchvision exposes the same files through the ``Weights`` enums, and the
  URLs are byte-identical (verified for resnet50/resnet101), so checkpoints
  downloaded under either stack are interchangeable.
* an ``https://`` URL -- how the Swin configs fetch their pretrained weights.
* a local ``.pth`` path -- the released MAMBA/STPN detector checkpoints.

Dropped from mmcv: the ``open-mmlab://`` and ``mmcls://`` model zoos (no config
here targets them; they would need mmcv's JSON index to resolve).
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

__all__ = ["load_checkpoint", "load_state_dict", "resolve_checkpoint_uri"]

logger = logging.getLogger(__name__)

TORCHVISION_PREFIX = "torchvision://"


def resolve_checkpoint_uri(filename: str) -> str:
    """Expand a ``torchvision://name`` shorthand to a real URL."""
    if not filename.startswith(TORCHVISION_PREFIX):
        return filename

    from torchvision.models import get_model_weights

    name = filename[len(TORCHVISION_PREFIX) :]
    try:
        weights = get_model_weights(name)
    except (ValueError, KeyError) as e:
        raise ValueError(f"unknown torchvision model {name!r} in {filename!r}") from e
    # mmcv's table held torchvision's original ImageNet weights, which are the
    # V1 entry; `.DEFAULT` may point at newer recipes and would change results.
    if "IMAGENET1K_V1" not in weights.__members__:
        raise ValueError(f"{name!r} has no IMAGENET1K_V1 weights to match mmcv's model zoo")
    return weights["IMAGENET1K_V1"].url


def _read_checkpoint(filename: str, map_location: str | torch.device = "cpu") -> Any:
    uri = resolve_checkpoint_uri(filename)
    if uri.startswith(("http://", "https://")):
        return torch.hub.load_state_dict_from_url(uri, map_location=map_location)
    return torch.load(uri, map_location=map_location, weights_only=True)


def _extract_state_dict(checkpoint: Any) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must be a dict, got {type(checkpoint)}")
    for key in ("state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], Mapping):
            return OrderedDict(checkpoint[key])
    return OrderedDict(checkpoint)


def load_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    strict: bool = False,
    log: Callable[[str], None] | None = None,
) -> tuple[list[str], list[str]]:
    """Load ``state_dict`` into ``module``, reporting what did not line up.

    Unlike ``nn.Module.load_state_dict``, a non-strict load here still *logs*
    missing/unexpected/shape-mismatched keys. Silent partial loads are the
    classic way a ported backbone ends up running on random weights.
    """
    log = log or logger.warning
    own = module.state_dict()

    mismatched = [
        f"{k}: checkpoint {tuple(v.shape)} vs model {tuple(own[k].shape)}"
        for k, v in state_dict.items()
        if k in own and v.shape != own[k].shape
    ]
    filtered = {k: v for k, v in state_dict.items() if k not in own or v.shape == own[k].shape}

    incompatible = module.load_state_dict(filtered, strict=False)
    missing = [k for k in incompatible.missing_keys]
    unexpected = list(incompatible.unexpected_keys)

    problems = []
    if mismatched:
        problems.append("shape mismatch, skipped:\n  " + "\n  ".join(mismatched))
    if unexpected:
        problems.append(f"unexpected keys ({len(unexpected)}): {unexpected}")
    if missing:
        problems.append(f"missing keys ({len(missing)}): {missing}")

    if problems:
        message = "\n".join(problems)
        if strict:
            raise RuntimeError(f"error loading state_dict:\n{message}")
        log(message)
    return missing, unexpected


def load_checkpoint(
    model: nn.Module,
    filename: str,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
    revise_keys: tuple[tuple[str, str], ...] = ((r"^module\.", ""),),
    log: Callable[[str], None] | None = None,
) -> Any:
    """Load weights from ``filename`` into ``model``; returns the raw checkpoint.

    ``revise_keys`` is a list of ``(pattern, replacement)`` regex substitutions
    applied to every key, defaulting to stripping the ``module.`` prefix that
    ``DistributedDataParallel`` adds when saving.
    """
    checkpoint = _read_checkpoint(filename, map_location)
    state_dict = _extract_state_dict(checkpoint)
    for pattern, replacement in revise_keys:
        state_dict = OrderedDict(
            (re.sub(pattern, replacement, k), v) for k, v in state_dict.items()
        )
    load_state_dict(model, state_dict, strict=strict, log=log)
    return checkpoint
