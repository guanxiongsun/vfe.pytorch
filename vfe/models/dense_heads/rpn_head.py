"""Region proposal network head. Port of ``mmdet.models.dense_heads.rpn_head``.

A 3x3 conv over the neck output, then two 1x1 convs for objectness and box
deltas. ``num_classes`` is fixed to 1 -- the RPN only asks "object or not".

The decode path differs from a normal detection head in one way worth noting:
NMS is run *per feature level*, not globally, by handing ``batched_nms`` the
level index as the class label. Proposals from different levels describe
different scales and should not suppress each other.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from vfe.layers import ConvModule
from vfe.models.builder import HEADS
from vfe.models.dense_heads.anchor_head import AnchorHead
from vfe.ops import batched_nms

__all__ = ["RPNHead"]


@HEADS.register_module()
class RPNHead(AnchorHead):
    """Args: ``in_channels``, ``num_convs`` (stacked 3x3 convs, 1 by default),
    plus everything :class:`AnchorHead` takes."""

    def __init__(self, in_channels: int, num_convs: int = 1, **kwargs):
        # Set before super().__init__, which calls _init_layers().
        self.num_convs = num_convs
        super().__init__(1, in_channels, **kwargs)

    def _init_layers(self) -> None:
        if self.num_convs > 1:
            # inplace=False: with several stacked convs the ReLU output is
            # needed for the backward pass, and an in-place ReLU would clobber
            # it.
            self.rpn_conv = nn.Sequential(
                *[
                    ConvModule(
                        self.in_channels if i == 0 else self.feat_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        inplace=False,
                    )
                    for i in range(self.num_convs)
                ]
            )
        else:
            self.rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(
            self.feat_channels, self.num_base_priors * self.cls_out_channels, 1
        )
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * 4, 1)

    def forward_single(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        return self.rpn_cls(x), self.rpn_reg(x)

    def loss(self, cls_scores, bbox_preds, gt_bboxes, img_metas, gt_bboxes_ignore=None):
        """Same as :meth:`AnchorHead.loss` but with ``gt_labels=None`` (there is
        only one class) and the keys renamed so they survive being merged into
        the detector's loss dict alongside the RoI head's."""
        losses = super().loss(
            cls_scores, bbox_preds, gt_bboxes, None, img_metas, gt_bboxes_ignore=gt_bboxes_ignore
        )
        return dict(loss_rpn_cls=losses["loss_cls"], loss_rpn_bbox=losses["loss_bbox"])

    def _get_bboxes_single(self, cls_score_list, bbox_pred_list, mlvl_anchors, img_meta, cfg,
                           rescale=False, with_nms=True, **kwargs):
        """Proposals for one image, as ``(n, 5)`` = box + score."""
        cfg = copy.deepcopy(self.test_cfg if cfg is None else cfg)
        img_shape = img_meta["img_shape"]

        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get("nms_pre", -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            if rpn_cls_score.size()[-2:] != rpn_bbox_pred.size()[-2:]:
                raise ValueError("cls and reg predictions disagree on spatial size")
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                scores = rpn_cls_score.reshape(-1).sigmoid()
            else:
                # Column 0 is foreground, column 1 background -- the v2.5 label
                # convention, not the older "background first" one.
                scores = rpn_cls_score.reshape(-1, 2).softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            anchors = mlvl_anchors[level_idx]
            if 0 < nms_pre < scores.shape[0]:
                # A full sort rather than topk: mmdet found it faster here, and
                # it fixes the order of equal scores, which topk does not.
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(scores.new_full((scores.size(0),), level_idx, dtype=torch.long))

        return self._bbox_post_process(
            mlvl_scores, mlvl_bbox_preds, mlvl_valid_anchors, level_ids, cfg, img_shape
        )

    def _bbox_post_process(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors, level_ids, cfg,
                           img_shape, **kwargs):
        """Decode, drop degenerate boxes, then per-level NMS."""
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
        proposals = self.bbox_coder.decode(anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg["min_bbox_size"] >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > cfg["min_bbox_size"]) & (h > cfg["min_bbox_size"])
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() == 0:
            return proposals.new_zeros(0, 5)
        dets, _ = batched_nms(proposals, scores, ids, cfg["nms"])
        return dets[: cfg["max_per_img"]]
