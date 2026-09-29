"""Single-stage detectors. Port of ``mmdet.models.detectors.{single_stage,fcos,yolox}``.

Backbone (+ neck) -> one dense head that predicts boxes at every cell. The
detector only routes: the whole ``train_cfg`` / ``test_cfg`` goes to the head,
unlike the two-stage split into ``rpn`` and ``rcnn`` slices.

EOVOD wraps one of these and calls its parts directly (``extract_feat``,
``bbox_head.simple_test`` on a subset of levels), as MAMBA does with its
two-stage detector.
"""

from __future__ import annotations

from typing import Any

import torch

from vfe.core import bbox2result
from vfe.models.builder import DETECTORS, build_backbone, build_head, build_neck
from vfe.models.detectors.base import BaseDetector

__all__ = ["SingleStageDetector", "FCOS", "YOLOX"]


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    def __init__(
        self,
        backbone: dict,
        neck: dict | None = None,
        bbox_head: dict | None = None,
        train_cfg: Any = None,
        test_cfg: Any = None,
    ):
        super().__init__()
        if bbox_head is None:
            raise ValueError("a single-stage detector needs a bbox_head")
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.bbox_head = build_head(dict(bbox_head, train_cfg=train_cfg, test_cfg=test_cfg))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self) -> None:
        """Each part's own scheme; call before loading a full checkpoint, not after."""
        self.backbone.init_weights()
        if self.with_neck:
            self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img: torch.Tensor):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None,
                      **kwargs) -> dict[str, Any]:
        x = self.extract_feat(img)
        return self.bbox_head.forward_train(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)

    def simple_test(self, img, img_metas, rescale=False):
        """Per-image detections, as ``bbox2result``'s per-class arrays."""
        results_list = self.bbox_head.simple_test(self.extract_feat(img), img_metas, rescale=rescale)
        return [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]


@DETECTORS.register_module()
class FCOS(SingleStageDetector):
    """`FCOS <https://arxiv.org/abs/1904.01355>`_: an anchor-free single-stage
    detector; the head is :class:`~vfe.models.dense_heads.FCOSHead`."""

    def __init__(self, backbone, neck, bbox_head, train_cfg=None, test_cfg=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg)


@DETECTORS.register_module()
class YOLOX(SingleStageDetector):
    """`YOLOX <https://arxiv.org/abs/2107.08430>`_. Port of
    ``mmdet.models.detectors.yolox``: the head is
    :class:`~vfe.models.dense_heads.YOLOXHead`, and training is multi-scale.

    Every ``random_size_interval`` training steps, process 0 draws a new
    input size -- ``size_multiplier`` times a random integer in
    ``random_size_range``, at the default size's aspect ratio -- and
    broadcasts it; each batch (already padded to ``input_size``) is resized
    to the current size with its boxes. Inference is at the pipeline's size.
    """

    def __init__(self, backbone, neck, bbox_head, train_cfg=None, test_cfg=None,
                 input_size=(640, 640), size_multiplier=32, random_size_range=(15, 25),
                 random_size_interval=10, init_cfg=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg)
        self._default_input_size = tuple(input_size)
        self._input_size = tuple(input_size)
        self._random_size_range = tuple(random_size_range)
        self._random_size_interval = random_size_interval
        self._size_multiplier = size_multiplier
        self._progress_in_iter = 0

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None,
                      **kwargs) -> dict[str, Any]:
        img, gt_bboxes = self._preprocess(img, gt_bboxes)
        losses = super().forward_train(img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore)
        if (self._progress_in_iter + 1) % self._random_size_interval == 0:
            self._input_size = self._random_resize(img.device)
        self._progress_in_iter += 1
        return losses

    def _preprocess(self, img, gt_bboxes):
        """The batch and its boxes at the current input size. The boxes are
        scaled in place, as in mmdet."""
        scale_y = self._input_size[0] / self._default_input_size[0]
        scale_x = self._input_size[1] / self._default_input_size[1]
        if scale_x != 1 or scale_y != 1:
            img = torch.nn.functional.interpolate(img, size=self._input_size, mode="bilinear",
                                                  align_corners=False)
            for gt_bbox in gt_bboxes:
                gt_bbox[..., 0::2] = gt_bbox[..., 0::2] * scale_x
                gt_bbox[..., 1::2] = gt_bbox[..., 1::2] * scale_y
        return img, gt_bboxes

    def _random_resize(self, device) -> tuple[int, int]:
        import random

        import torch.distributed as dist

        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        # NCCL broadcasts need a GPU tensor; gloo (CPU tests) a CPU one.
        tensor = torch.zeros(2, dtype=torch.long, device=device)
        if rank == 0:
            size = random.randint(*self._random_size_range)
            aspect_ratio = float(self._default_input_size[1]) / self._default_input_size[0]
            tensor[0] = self._size_multiplier * size
            tensor[1] = self._size_multiplier * int(aspect_ratio * size)
        if distributed and dist.get_world_size() > 1:
            dist.barrier()
            dist.broadcast(tensor, 0)
        return int(tensor[0]), int(tensor[1])
