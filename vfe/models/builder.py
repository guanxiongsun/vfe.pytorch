"""Model registries and ``build_*`` helpers, replacing ``mmdet.models.builder``.

mmdet aliases *every* component registry (``BACKBONES``, ``NECKS``, ``HEADS``,
...) to one shared ``MODELS`` registry, so a name registered as a backbone is
equally findable as a neck. That is kept here deliberately: the configs were
written against it, and collapsing to a single namespace also means a typo'd
``type`` fails with one clear "not registered" error rather than depending on
which builder happened to be called.
"""

from __future__ import annotations

from typing import Any

from ..registry import Registry, build_from_cfg

__all__ = [
    "MODELS",
    "BACKBONES",
    "NECKS",
    "ROI_EXTRACTORS",
    "SHARED_HEADS",
    "HEADS",
    "LOSSES",
    "DETECTORS",
    "AGGREGATORS",
    "MEMORY",
    "build_backbone",
    "build_neck",
    "build_roi_extractor",
    "build_shared_head",
    "build_head",
    "build_loss",
    "build_aggregator",
    "build_memory",
    "build_detector",
    "build_model",
]

MODELS = Registry("models")

# Aliases, not separate registries -- see the module docstring.
BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS
AGGREGATORS = MODELS
MEMORY = MODELS


def build_backbone(cfg: dict) -> Any:
    return build_from_cfg(cfg, BACKBONES)


def build_neck(cfg: dict) -> Any:
    return build_from_cfg(cfg, NECKS)


def build_roi_extractor(cfg: dict) -> Any:
    return build_from_cfg(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg: dict) -> Any:
    return build_from_cfg(cfg, SHARED_HEADS)


def build_head(cfg: dict) -> Any:
    return build_from_cfg(cfg, HEADS)


def build_loss(cfg: dict) -> Any:
    return build_from_cfg(cfg, LOSSES)


def build_aggregator(cfg: dict) -> Any:
    return build_from_cfg(cfg, AGGREGATORS)


def build_memory(cfg: dict) -> Any:
    return build_from_cfg(cfg, MEMORY)


def build_detector(cfg: dict, train_cfg: dict | None = None, test_cfg: dict | None = None) -> Any:
    """Build a detector.

    mmdet accepted ``train_cfg``/``test_cfg`` as outer arguments and warned
    about it; the configs in this repo all nest them inside ``model``, so the
    arguments are kept only to reject the deprecated form loudly.
    """
    if train_cfg is not None and cfg.get("train_cfg") is not None:
        raise ValueError("train_cfg given both as an argument and inside the model cfg")
    if test_cfg is not None and cfg.get("test_cfg") is not None:
        raise ValueError("test_cfg given both as an argument and inside the model cfg")
    default_args = {}
    if train_cfg is not None:
        default_args["train_cfg"] = train_cfg
    if test_cfg is not None:
        default_args["test_cfg"] = test_cfg
    return build_from_cfg(cfg, DETECTORS, default_args or None)


def build_model(cfg: dict, train_cfg: dict | None = None, test_cfg: dict | None = None) -> Any:
    return build_detector(cfg, train_cfg, test_cfg)
