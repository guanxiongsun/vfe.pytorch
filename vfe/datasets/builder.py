"""Registries for datasets and pipeline transforms, so data configs' ``type``
names resolve as they did under mmdet."""

from __future__ import annotations

from typing import Any

from vfe.registry import Registry, build_from_cfg

__all__ = ["DATASETS", "PIPELINES", "build_dataset"]

DATASETS = Registry("dataset")
PIPELINES = Registry("pipeline")


def build_dataset(cfg: dict, default_args: dict | None = None) -> Any:
    return build_from_cfg(cfg, DATASETS, default_args)
