"""Batching for video training and testing.

Replaces ``mmcv.parallel.collate`` + ``scatter``. The models were written
against the nesting those produced, and the pipeline parity checks confirm
these functions reproduce it exactly.

mmcv chose per value through ``DataContainer`` flags; here the same rules are
applied by key:

* images (``img``, ``ref_img``) are padded bottom/right to the batch's largest
  size and stacked;
* metas (``*img_metas``) stay plain Python lists;
* every other tensor (ground truth, proposals) becomes a per-sample list.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

__all__ = ["collate_video_test", "collate_video_train"]

STACKED_KEYS = ("img", "ref_img")

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


def collate_video_train(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate training samples (as produced by a pipeline ending in
    ``SeqDefaultFormatBundle``) into the arguments of ``forward_train``."""
    batch: dict[str, Any] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        if key in STACKED_KEYS:
            max_h = max(v.shape[-2] for v in values)
            max_w = max(v.shape[-1] for v in values)
            batch[key] = torch.stack(
                [F.pad(v, (0, max_w - v.shape[-1], 0, max_h - v.shape[-2])) for v in values]
            )
        else:
            batch[key] = values
    return batch
