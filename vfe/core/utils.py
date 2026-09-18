"""Small helpers the detection heads share. Ports of ``mmdet.core.utils.misc``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch

__all__ = ["multi_apply", "unmap", "select_single_mlvl", "filter_scores_and_topk"]


def multi_apply(func: Callable, *args: Any, **kwargs: Any) -> tuple[list, ...]:
    """Map ``func`` over parallel argument lists, transposing the results.

    ``func`` returning an n-tuple over m inputs gives an n-tuple of m-lists.
    The heads use this to compute per-image targets and get back one list per
    target kind.
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results, strict=True)))


def unmap(data: torch.Tensor, count: int, inds: torch.Tensor, fill: float = 0) -> torch.Tensor:
    """Scatter ``data`` back into a ``count``-long tensor at ``inds``, ``fill`` elsewhere.

    The inverse of the "keep only anchors inside the image" filtering the
    anchor head does before computing targets.
    """
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds.type(torch.bool)] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.type(torch.bool), :] = data
    return ret


def select_single_mlvl(
    mlvl_tensors: Sequence[torch.Tensor], batch_id: int, detach: bool = True
) -> list[torch.Tensor]:
    """Take image ``batch_id`` out of every level of a multi-level batch tensor.

    ``detach`` defaults to True as in mmdet: a two-stage detector must not
    backpropagate the RoI head's loss through the proposals.
    """
    if not isinstance(mlvl_tensors, (list, tuple)):
        raise TypeError(f"mlvl_tensors must be a list or tuple, got {type(mlvl_tensors)}")
    if detach:
        return [t[batch_id].detach() for t in mlvl_tensors]
    return [t[batch_id] for t in mlvl_tensors]


def filter_scores_and_topk(
    scores: torch.Tensor,
    score_thr: float,
    topk: int,
    results: dict | list | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    """Threshold ``scores`` then keep the ``topk`` highest, indexing ``results`` alongside.

    Returns ``(scores, labels, keep_idxs, filtered_results)``. ``scores`` is
    (num_bboxes, K); ``labels`` is the column index each kept score came from.
    """
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    # mmdet notes torch.sort outruns .topk on GPU; kept for numerical identity too,
    # since the two can order ties differently.
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results: Any = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(
                f"Only supports dict, list or Tensor, but got {type(results)}."
            )
    return scores, labels, keep_idxs, filtered_results
