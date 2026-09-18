"""The second stage of a two-stage detector. Port of
``mmdet.models.roi_heads.standard_roi_head`` (with ``base_roi_head`` and the
parts of ``test_mixins.BBoxTestMixin`` it uses folded in).

Training: assign each proposal to a ground-truth box, sample a class-balanced
subset, pool a feature per sampled RoI, and score it with the bbox head.
Inference: pool a feature for every proposal, decode, and NMS per class.

Dropped from mmdet, since no config here uses them: the mask branch, the
shared head (C4-style ``ResLayer``), test-time augmentation, the async and ONNX
paths, and ``forward_dummy``. The VID RoI heads override ``_bbox_forward`` and
friends to take reference-frame features, and reuse everything else.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vfe.core import bbox2result, bbox2roi, build_assigner, build_sampler
from vfe.models.builder import HEADS, build_head, build_roi_extractor

__all__ = ["StandardRoIHead"]


@HEADS.register_module()
class StandardRoIHead(nn.Module):
    """Args:
        bbox_roi_extractor: config for the RoI feature extractor.
        bbox_head: config for the box head.
        train_cfg: the ``rcnn`` training config (assigner, sampler, pos_weight).
        test_cfg: the ``rcnn`` test config (score_thr, nms, max_per_img).
    """

    def __init__(
        self,
        bbox_roi_extractor: dict,
        bbox_head: dict,
        train_cfg: Any = None,
        test_cfg: Any = None,
    ):
        super().__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

        self.bbox_assigner = None
        self.bbox_sampler = None
        if train_cfg:
            self.bbox_assigner = build_assigner(train_cfg["assigner"])
            self.bbox_sampler = build_sampler(train_cfg["sampler"], context=self)

    def init_weights(self) -> None:
        # The extractor has no parameters; only the box head needs initialising.
        self.bbox_head.init_weights()

    # ---- training ----------------------------------------------------------

    def _assign_and_sample(self, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        """One ``SamplingResult`` per image. Shared with the VID RoI heads,
        which differ only in what happens after sampling."""
        num_imgs = len(proposal_list)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None] * num_imgs
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_results.append(
                self.bbox_sampler.sample(
                    assign_result, proposal_list[i], gt_bboxes[i], gt_labels[i]
                )
            )
        return sampling_results

    def forward_train(self, x, img_metas, proposal_list, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, **kwargs) -> dict[str, torch.Tensor]:
        """Returns ``dict(loss_cls=..., acc=..., loss_bbox=...)``.

        Args:
            x: multi-level features, each ``(B, C, H, W)``.
            img_metas: per-image meta dicts.
            proposal_list: per-image ``(n, 5)`` proposals from the RPN.
            gt_bboxes / gt_labels: per-image ground truth.
        """
        sampling_results = self._assign_and_sample(
            proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore
        )
        bbox_results = self._bbox_forward_train(
            x, sampling_results, gt_bboxes, gt_labels, img_metas
        )
        return dict(bbox_results["loss_bbox"])

    def _bbox_forward(self, x, rois):
        """Pool RoI features and run the box head; used by train and test."""
        bbox_feats = self.bbox_roi_extractor(x[: self.bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        return dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas):
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg
        )
        loss_bbox = self.bbox_head.loss(
            bbox_results["cls_score"], bbox_results["bbox_pred"], rois, *bbox_targets
        )
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    # ---- inference ---------------------------------------------------------

    def simple_test(self, x, proposal_list, img_metas, proposals=None, rescale=False):
        """Per-image detections as ``bbox2result`` lists: one ``(k, 5)`` numpy
        array per class."""
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale
        )
        return [
            bbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

    def _empty_result(self, rois: torch.Tensor, rcnn_test_cfg):
        """What an image with no proposals decodes to. Without a test config
        the caller wants pre-NMS boxes and scores, so the shapes differ."""
        if rcnn_test_cfg is None:
            return rois.new_zeros(0, 4), rois.new_zeros((0, self.bbox_head.fc_cls.out_features))
        return rois.new_zeros(0, 5), rois.new_zeros((0,), dtype=torch.long)

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False):
        """``(det_bboxes, det_labels)``, each a per-image list.

        All images' RoIs go through the head as one batch, then the predictions
        are split back per image for decoding, since image sizes and scale
        factors differ.
        """
        rois = bbox2roi(proposals)
        if rois.shape[0] == 0:
            det_bbox, det_label = self._empty_result(rois, rcnn_test_cfg)
            return [det_bbox] * len(proposals), [det_label] * len(proposals)

        bbox_results = self._bbox_forward(x, rois)
        return self._decode_per_image(
            rois, bbox_results, proposals, img_metas, rcnn_test_cfg, rescale
        )

    def _decode_per_image(self, rois, bbox_results, proposals, img_metas, rcnn_test_cfg,
                          rescale):
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = bbox_results["cls_score"].split(num_proposals_per_img, 0)
        bbox_pred = bbox_results["bbox_pred"]
        if bbox_pred is not None:
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
        else:
            bbox_pred = (None,) * len(proposals)

        det_bboxes, det_labels = [], []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                det_bbox, det_label = self._empty_result(rois[i], rcnn_test_cfg)
            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_metas[i]["img_shape"],
                    img_metas[i]["scale_factor"],
                    rescale=rescale,
                    cfg=rcnn_test_cfg,
                )
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
