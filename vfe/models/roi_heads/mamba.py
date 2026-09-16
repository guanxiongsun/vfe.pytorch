"""MAMBA's RoI head and box head. Port of
``mmdet.models.roi_heads.vid.{mamba_roi_head,bbox_heads.mamba_bbox_head}``.

Structurally a ``Shared2FCBBoxHead`` with an aggregator after each shared FC:
``x = fc(x); x = x + aggregator(x, refs); x = relu(x)``. What varies is where
the references come from:

* **First frame of a video, and every training step:** reference RoI features
  are passed in explicitly (pooled from the reference frames' top-k
  proposals). Each aggregator's memory is reset and re-initialised with them.
* **Every later test frame:** no references are passed. Each aggregator reads
  a sample of its memory, and afterwards writes back the top-k rows of the
  enhanced features (the rows of the highest-scoring proposals, since RPN
  output is sorted by score).

Features written to or initialising the memory are taken *before* the ReLU;
that asymmetry with the returned features is the original's and is kept.
"""

from __future__ import annotations

import torch
from torch import nn

from vfe.core import bbox2result, bbox2roi
from vfe.models.builder import HEADS, build_aggregator
from vfe.models.roi_heads.bbox_heads import ConvFCBBoxHead
from vfe.models.roi_heads.standard_roi_head import StandardRoIHead

__all__ = ["MambaBBoxHead", "MambaRoIHead"]


@HEADS.register_module()
class MambaBBoxHead(ConvFCBBoxHead):
    """Args:
        aggregator: config for one :class:`~vfe.models.aggregators.MambaAggregator`;
            one is built per shared FC layer.
        topk: how many of a frame's RoIs (best-scoring first) are written to
            memory, and how many proposals per reference frame are used.
    """

    def __init__(self, aggregator: dict, topk: int = 300, **kwargs):
        super().__init__(**kwargs)
        self.aggregator = nn.ModuleList(
            [build_aggregator(aggregator) for _ in range(self.num_shared_fcs)]
        )
        # Not self.relu: that one is in-place, and the pre-activation features
        # were just handed to the memory bank.
        self.inplace_false_relu = nn.ReLU(inplace=False)
        self.topk = topk

    # init_weights is inherited unchanged. mmdet's init recursion never reached
    # the aggregators (they sit in a ModuleList, which has no init_weights),
    # so they keep torch's default nn.Linear init -- as they do here.

    def forward(self, x: torch.Tensor, ref_x: torch.Tensor | None):
        if ref_x is not None:
            return self._first_frame_forward(x, ref_x)

        for conv in self.shared_convs:
            x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for i, fc in enumerate(self.shared_fcs):
                x = fc(x)
                x = x + self.aggregator[i](x, ref_x=None)
                self.aggregator[i].update_memory_bank(x[: self.topk])
                x = self.inplace_false_relu(x)
        return self._forward_branches(x, x)

    def _first_frame_forward(self, x: torch.Tensor, ref_x: torch.Tensor):
        """Key features ``(N, C, H, W)``, reference features ``(M, C, H, W)``.
        The references pass through the same shared layers as the key frame."""
        for conv in self.shared_convs:
            x = conv(x)
            ref_x = conv(ref_x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
                ref_x = self.avg_pool(ref_x)
            x = x.flatten(1)
            ref_x = ref_x.flatten(1)
            for i, fc in enumerate(self.shared_fcs):
                x = fc(x)
                ref_x = fc(ref_x)
                self.aggregator[i].reset_memory_bank()
                x = x + self.aggregator[i](x, ref_x)
                ref_x = self.inplace_false_relu(ref_x)
                x = self.inplace_false_relu(x)
        return self._forward_branches(x, x)


@HEADS.register_module()
class MambaRoIHead(StandardRoIHead):
    """A ``StandardRoIHead`` whose box head also sees reference-frame RoIs."""

    def forward_train(self, x, ref_x, img_metas, proposal_list, ref_proposal_list, gt_bboxes,
                      gt_labels, gt_bboxes_ignore=None, **kwargs) -> dict[str, torch.Tensor]:
        sampling_results = self._assign_and_sample(
            proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore
        )
        rois = bbox2roi([res.bboxes for res in sampling_results])
        ref_rois = bbox2roi(ref_proposal_list)
        bbox_results = self._bbox_forward(x, ref_x, rois, ref_rois)
        bbox_targets = self.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.train_cfg
        )
        return self.bbox_head.loss(
            bbox_results["cls_score"], bbox_results["bbox_pred"], rois, *bbox_targets
        )

    def _bbox_forward(self, x, ref_x, rois, ref_rois):
        """``ref_x`` is None on every test frame after the first; the box head
        then reads its memory instead."""
        bbox_feats = self.bbox_roi_extractor(x[: self.bbox_roi_extractor.num_inputs], rois)
        ref_bbox_feats = None
        if ref_x is not None:
            ref_bbox_feats = self.bbox_roi_extractor(
                ref_x[: self.bbox_roi_extractor.num_inputs], ref_rois
            )
        cls_score, bbox_pred = self.bbox_head(bbox_feats, ref_bbox_feats)
        return dict(cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

    def simple_test(self, x, ref_x, proposals_list, ref_proposals_list, img_metas,
                    proposals=None, rescale=False):
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, ref_x, proposals_list, ref_proposals_list, img_metas, self.test_cfg,
            rescale=rescale,
        )
        return [
            bbox2result(det_bboxes[i], det_labels[i], self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

    def simple_test_bboxes(self, x, ref_x, proposals, ref_proposals, img_metas, rcnn_test_cfg,
                           rescale=False):
        # No whole-batch early exit for empty proposals, unlike
        # StandardRoIHead: the box head must still run so the memory update
        # happens, as in the original.
        rois = bbox2roi(proposals)
        ref_rois = bbox2roi(ref_proposals) if ref_x is not None else None
        bbox_results = self._bbox_forward(x, ref_x, rois, ref_rois)
        return self._decode_per_image(
            rois, bbox_results, proposals, img_metas, rcnn_test_cfg, rescale
        )
