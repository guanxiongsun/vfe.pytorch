"""Replay the fixed-shape parts of single-stage inference as CUDA graphs.

At batch 1 on a fast GPU the FCOS head is bound by kernel launches, not
arithmetic: each pyramid level costs about the same whatever its size (on a
GH200, ~1.2 ms of forward per level for P3's 76x128 cells and P7's 5x8
alike). A CUDA graph replays a captured sequence of kernels with one launch.

:func:`enable_cuda_graphs` wraps these, and only when gradients are off:

* the backbone and the neck, each one graph per input shape (graphed as
  modules rather than ``extract_feat``, so models that call them apart --
  EOVOD aggregating between them -- are covered too);
* ``bbox_head.forward_single``, one graph per level and feature shape (FCOS's
  and YOLOX's heads).

Everything data-dependent -- EOVOD's aggregation, post-processing, NMS --
stays eager. A graph is captured the first time a shape is seen (after a few
warm-up calls on a side stream) and its outputs are copied out on every
replay, so callers may keep them. Numerics are those of the eager kernels.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils._pytree import tree_map

__all__ = ["GraphedCallable", "enable_cuda_graphs"]


class GraphedCallable:
    """``fn(*tensors)`` replayed from a CUDA graph per distinct input shape."""

    def __init__(self, fn: Callable[..., Any], warmup: int = 3):
        self.fn = fn
        self.warmup = warmup
        self.graphs: dict[tuple, tuple[torch.cuda.CUDAGraph, list[torch.Tensor], Any]] = {}

    def __call__(self, *tensors: torch.Tensor):
        if torch.is_grad_enabled() or not all(t.is_cuda for t in tensors):
            return self.fn(*tensors)
        key = tuple((tuple(t.shape), t.dtype, t.device) for t in tensors)
        entry = self.graphs.get(key)
        if entry is None:
            entry = self._capture(tensors)
            self.graphs[key] = entry
        graph, inputs, outputs = entry
        for static, t in zip(inputs, tensors, strict=True):
            static.copy_(t)
        graph.replay()
        return tree_map(lambda o: o.clone() if isinstance(o, torch.Tensor) else o, outputs)

    def _capture(self, tensors):
        inputs = [t.clone() for t in tensors]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                self.fn(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        # A private memory pool per graph: EOVOD interleaves shapes (frames,
        # reference chunks), and graphs sharing a pool must replay in the
        # order they were captured.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = self.fn(*inputs)
        return graph, inputs, outputs


def enable_cuda_graphs(detector: torch.nn.Module) -> None:
    """Graph the backbone, the neck and the dense head's per-level forward in
    place. ``detector`` is a single-stage detector (e.g. EOVOD's
    ``model.detector``); call after moving it to the GPU and ``eval()``."""
    backbone = detector.backbone
    backbone.forward = GraphedCallable(backbone.forward)
    neck = getattr(detector, "neck", None)
    if neck is not None:
        neck_forward = neck.forward
        # The neck takes its levels as one sequence; the graph takes tensors.
        graphed_neck = GraphedCallable(lambda *levels: neck_forward(list(levels)))
        neck.forward = lambda inputs: graphed_neck(*inputs)
    head = detector.bbox_head
    original = head.forward_single
    per_level: dict[Any, GraphedCallable] = {}
    if type(head).__name__ == "YOLOXHead":
        # YOLOX passes each level's own modules; the class tower identifies it.
        def forward_single(x, cls_convs, reg_convs, conv_cls, conv_reg, conv_obj, reg_x=None):
            key = id(cls_convs)
            if key not in per_level:
                modules = (cls_convs, reg_convs, conv_cls, conv_reg, conv_obj)
                per_level[key] = GraphedCallable(lambda a, b, m=modules: original(a, *m, b))
            return per_level[key](x, x if reg_x is None else reg_x)
    else:
        def forward_single(x, scale, stride, reg_x=None):
            if stride not in per_level:
                per_level[stride] = GraphedCallable(
                    lambda a, b, s=scale, st=stride: original(a, s, st, b))
            # The graph needs a tensor for both towers; plain FCOS feeds x to both.
            return per_level[stride](x, x if reg_x is None else reg_x)

    head.forward_single = forward_single
