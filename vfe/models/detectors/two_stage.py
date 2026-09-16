"""Two-stage detectors. Port of ``mmdet.models.detectors.{two_stage,faster_rcnn}``.

Backbone (+ neck) -> RPN proposals -> RoI head. The detector itself holds
almost no logic; what it does own is *config routing*, which is easy to get
subtly wrong: the model config has one ``train_cfg`` / ``test_cfg`` pair, and
each head receives its own slice of it --

* ``rpn_head``  <- ``train_cfg.rpn`` (assigner, sampler) and ``test_cfg.rpn``
* ``roi_head``  <- ``train_cfg.rcnn`` and ``test_cfg.rcnn``
* proposals fed to the RoI head *during training* use ``train_cfg.rpn_proposal``
  (e.g. 600 per image for MAMBA), not ``test_cfg.rpn`` (300) -- falling back to
  the latter only if the former is absent.

In the VID configs this detector is not the top-level model: MAMBA, SELSA and
STPN each wrap one and call its parts directly (``extract_feat``,
``rpn_head.simple_test_rpn``, ``roi_head.forward_train``, ...).
"""

from __future__ import annotations

from typing import Any

import torch

from vfe.models.builder import DETECTORS, build_backbone, build_head, build_neck
from vfe.models.detectors.base import BaseDetector

__all__ = ["TwoStageDetector", "FasterRCNN"]


@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    def __init__(
        self,
        backbone: dict,
        neck: dict | None = None,
        rpn_head: dict | None = None,
        roi_head: dict | None = None,
        train_cfg: Any = None,
        test_cfg: Any = None,
    ):
        super().__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = build_head(
                dict(
                    rpn_head,
                    train_cfg=train_cfg["rpn"] if train_cfg is not None else None,
                    test_cfg=test_cfg["rpn"],
                )
            )
        if roi_head is not None:
            self.roi_head = build_head(
                dict(
                    roi_head,
                    train_cfg=train_cfg["rcnn"] if train_cfg is not None else None,
                    test_cfg=test_cfg["rcnn"],
                )
            )

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self) -> bool:
        return getattr(self, "rpn_head", None) is not None

    @property
    def with_roi_head(self) -> bool:
        return getattr(self, "roi_head", None) is not None

    def init_weights(self) -> None:
        """Each part's own scheme: the backbone loads its pretrained checkpoint
        (or random-inits), and the neck and heads apply mmdet's defaults.

        Call this *before* loading a full detector checkpoint, not after -- it
        would overwrite the loaded weights.
        """
        self.backbone.init_weights()
        if self.with_neck:
            self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights()

    def extract_feat(self, img: torch.Tensor):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None,
                      proposals=None, **kwargs) -> dict[str, Any]:
        """Losses from both stages, merged into one dict. The RPN's keys are
        prefixed ``loss_rpn_*`` so they cannot collide with the RoI head's."""
        x = self.extract_feat(img)

        losses: dict[str, Any] = {}
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg["rpn"])
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs,
            )
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        losses.update(
            self.roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, **kwargs
            )
        )
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Per-image detections, as ``bbox2result``'s per-class arrays."""
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        return self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)


@DETECTORS.register_module()
class FasterRCNN(TwoStageDetector):
    """`Faster R-CNN <https://arxiv.org/abs/1506.01497>`_: a two-stage detector
    whose RPN and RoI head are both required."""

    def __init__(self, backbone, rpn_head, roi_head, train_cfg, test_cfg, neck=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
