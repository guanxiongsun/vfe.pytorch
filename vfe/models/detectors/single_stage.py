"""Single-stage detectors. Port of ``mmdet.models.detectors.{single_stage,fcos}``.

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

__all__ = ["SingleStageDetector", "FCOS"]


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
