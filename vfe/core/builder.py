"""Registries for the non-``nn.Module`` pieces of the detection pipeline.

Unlike ``vfe.models.builder`` -- where every registry is an alias of one
``MODELS`` registry, because mmdet does it that way and config ``type`` names
must resolve identically -- mmdet genuinely keeps these separate. Kept separate
here too, so a typo in an assigner name reports the assigners rather than every
class in the project.
"""

from __future__ import annotations

from typing import Any

from vfe.registry import Registry, build_from_cfg

__all__ = [
    "BBOX_ASSIGNERS",
    "BBOX_SAMPLERS",
    "BBOX_CODERS",
    "IOU_CALCULATORS",
    "PRIOR_GENERATORS",
    "ANCHOR_GENERATORS",
    "build_assigner",
    "build_sampler",
    "build_bbox_coder",
    "build_iou_calculator",
    "build_prior_generator",
    "build_anchor_generator",
]

BBOX_ASSIGNERS = Registry("bbox_assigner")
BBOX_SAMPLERS = Registry("bbox_sampler")
BBOX_CODERS = Registry("bbox_coder")
IOU_CALCULATORS = Registry("iou_calculator")
PRIOR_GENERATORS = Registry("prior_generator")
# mmdet's pre-2.18 name for the same registry; configs still use `AnchorGenerator`.
ANCHOR_GENERATORS = PRIOR_GENERATORS


def build_assigner(cfg: dict, **default_args: Any) -> Any:
    return build_from_cfg(cfg, BBOX_ASSIGNERS, default_args or None)


def build_sampler(cfg: dict, **default_args: Any) -> Any:
    return build_from_cfg(cfg, BBOX_SAMPLERS, default_args or None)


def build_bbox_coder(cfg: dict, **default_args: Any) -> Any:
    return build_from_cfg(cfg, BBOX_CODERS, default_args or None)


def build_iou_calculator(cfg: dict, **default_args: Any) -> Any:
    return build_from_cfg(cfg, IOU_CALCULATORS, default_args or None)


def build_prior_generator(cfg: dict, **default_args: Any) -> Any:
    return build_from_cfg(cfg, PRIOR_GENERATORS, default_args or None)


build_anchor_generator = build_prior_generator
