"""Registries for datasets and pipeline transforms, so data configs' ``type``
names resolve as they did under mmdet."""

from __future__ import annotations

from typing import Any

from vfe.registry import Registry, build_from_cfg

__all__ = ["DATASETS", "PIPELINES", "build_dataset"]

DATASETS = Registry("dataset")
PIPELINES = Registry("pipeline")


def build_dataset(cfg: dict | list | tuple, default_args: dict | None = None) -> Any:
    """A dataset config, or a list of them (concatenated, as mmdet did)."""
    if isinstance(cfg, (list, tuple)):
        from vfe.datasets.dataset_wrappers import ConcatDataset

        return ConcatDataset([build_dataset(c, default_args) for c in cfg])
    return build_from_cfg(cfg, DATASETS, default_args)
