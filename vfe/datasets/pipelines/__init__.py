"""Data pipelines: each transform takes and returns a results dict (or a list of
them, one per frame, for the ``Seq*`` transforms). Importing this package
registers them in :data:`vfe.datasets.builder.PIPELINES`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from vfe.datasets.builder import PIPELINES
from vfe.registry import build_from_cfg

from .formatting import ConcatVideoReferences, MultiImagesToTensor, ToList, VideoCollect
from .loading import LoadImageFromFile, LoadMultiImagesFromFile
from .transforms import (
    Normalize,
    Pad,
    RandomFlip,
    Resize,
    SeqNormalize,
    SeqPad,
    SeqRandomFlip,
    SeqResize,
)

__all__ = [
    "Compose",
    "LoadImageFromFile",
    "LoadMultiImagesFromFile",
    "Resize",
    "SeqResize",
    "RandomFlip",
    "SeqRandomFlip",
    "Normalize",
    "SeqNormalize",
    "Pad",
    "SeqPad",
    "VideoCollect",
    "ConcatVideoReferences",
    "MultiImagesToTensor",
    "ToList",
]


class Compose:
    """Apply transforms in order; a transform returning ``None`` drops the sample."""

    def __init__(self, transforms: Sequence[dict | Callable]):
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
            elif not callable(transform):
                raise TypeError(f"transform must be a dict or callable, got {type(transform)}")
            self.transforms.append(transform)

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
            if data is None:
                return None
        return data
