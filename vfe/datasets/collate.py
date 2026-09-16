"""Batching for the video test loop.

Replaces ``mmcv.parallel.collate`` + ``scatter`` for the test path. The model's
``simple_test`` was written against the nesting those produced, and
``parity_vid_pipeline`` checks this reproduces it exactly: for one sample,
tensors gain a batch axis inside a one-item list, and metas gain two list
levels (reference metas three, since they are already a list).
"""

from __future__ import annotations

from typing import Any

__all__ = ["collate_video_test"]

META_KEYS = ("img_metas", "ref_img_metas")


def collate_video_test(samples: list[dict[str, list[Any]]]) -> dict[str, list[Any]]:
    """Collate one test sample (as produced by a pipeline ending in ``ToList``).

    Returns ``img=[Tensor(1, 3, H, W)]``, ``img_metas=[[meta]]`` and, on frames
    with references, ``ref_img=[Tensor(1, R, 3, H, W)]``,
    ``ref_img_metas=[[[meta, ...]]]``.
    """
    if len(samples) != 1:
        raise ValueError(f"video testing is one frame per batch; got {len(samples)} samples")
    (sample,) = samples
    batch = {}
    for key, value in sample.items():
        if not isinstance(value, list) or len(value) != 1:
            raise ValueError(f"{key}: expected a one-item list (the pipeline's ToList), "
                             f"got {type(value).__name__}")
        item = value[0]
        batch[key] = [[item]] if key in META_KEYS else [item.unsqueeze(0)]
    return batch
