"""Anchor-based dense head. Port of ``mmdet.models.dense_heads.anchor_head``
(merged with the parts of ``base_dense_head`` that it actually uses).

The head owns three things: the anchors it predicts relative to, the
*assignment* of those anchors to ground truth, and the two losses. Everything
the mmdet version carries for other detectors is dropped -- the generic
``_get_bboxes_single`` (RetinaNet-style, score-factor aware), ``BBoxTestMixin``
and TTA, and ONNX export. ``RPNHead`` is the only dense head any config in this
repo uses, and it overrides the decode path anyway.

Label convention, which is easy to get backwards: since mmdet v2.5 foreground
classes are ``[0, num_classes - 1]`` and *background is ``num_classes``*. For
the RPN, ``num_classes == 1``, so a positive anchor gets label 0 and a negative
gets 1 -- the opposite of what "objectness" suggests.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vfe.core import (
    anchor_inside_flags,
    build_assigner,
    build_bbox_coder,
    build_prior_generator,
    build_sampler,
    images_to_levels,
    multi_apply,
    select_single_mlvl,
    unmap,
)
from vfe.layers import normal_init
from vfe.models.builder import HEADS, build_loss

__all__ = ["AnchorHead"]


@HEADS.register_module()
class AnchorHead(nn.Module):
    """Args:
        num_classes: categories *excluding* background.
        in_channels: channels of the incoming feature map.
        feat_channels: hidden channels; used by subclasses, not by this class.
        anchor_generator / bbox_coder / loss_cls / loss_bbox: config dicts.
        reg_decoded_bbox: regress on decoded boxes instead of deltas. Needed
            for IoU-family losses; false for the SmoothL1 configs here.
        train_cfg / test_cfg: the ``rpn`` / ``rpn_proposal`` sub-configs.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        anchor_generator: dict | None = None,
        bbox_coder: dict | None = None,
        reg_decoded_bbox: bool = False,
        loss_cls: dict | None = None,
        loss_bbox: dict | None = None,
        train_cfg: Any = None,
        test_cfg: Any = None,
    ):
        super().__init__()
        anchor_generator = anchor_generator or dict(
            type="AnchorGenerator",
            scales=[8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        )
        bbox_coder = bbox_coder or dict(
            type="DeltaXYWHBBoxCoder",
            clip_border=True,
            target_means=(0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0),
        )
        loss_cls = loss_cls or dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0)
        loss_bbox = loss_bbox or dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        # With a sigmoid loss there is no explicit background channel: an anchor
        # scoring low on every class *is* background.
        self.cls_out_channels = num_classes if self.use_sigmoid_cls else num_classes + 1
        if self.cls_out_channels <= 0:
            raise ValueError(f"num_classes={num_classes} is too small")
        self.reg_decoded_bbox = reg_decoded_bbox

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
            sampler_cfg = train_cfg.get("sampler")
            # `sampling` decides the loss denominator further down: with a real
            # sampler it is pos + neg (the sampled anchors), without one it is
            # just the positives.
            self.sampling = sampler_cfg is not None and sampler_cfg["type"] != "PseudoSampler"
            self.sampler = build_sampler(
                sampler_cfg if self.sampling else dict(type="PseudoSampler"), context=self
            )

        self.prior_generator = build_prior_generator(anchor_generator)
        # An int, not a list: only SSD varies the anchor count per level, and
        # SSD is not ported.
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self._init_layers()

    def _init_layers(self) -> None:
        self.conv_cls = nn.Conv2d(
            self.in_channels, self.num_base_priors * self.cls_out_channels, 1
        )
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_base_priors * 4, 1)

    def init_weights(self) -> None:
        """mmdet's ``init_cfg=dict(type='Normal', layer='Conv2d', std=0.01)``."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

    # ---- forward -----------------------------------------------------------

    def forward_single(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.conv_cls(x), self.conv_reg(x)

    def forward(self, feats):
        """Returns ``(cls_scores, bbox_preds)``, each a list over scale levels."""
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device="cuda"):
        """Anchors and validity flags per image.

        The anchors themselves are identical for every image in the batch (same
        padded size), so they are computed once and shared by reference. The
        *flags* are not: they depend on each image's ``pad_shape``.
        """
        multi_level_anchors = self.prior_generator.grid_priors(featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(len(img_metas))]
        valid_flag_list = [
            self.prior_generator.valid_flags(featmap_sizes, meta["pad_shape"], device)
            for meta in img_metas
        ]
        return anchor_list, valid_flag_list

    # ---- targets -----------------------------------------------------------

    def _get_targets_single(
        self,
        flat_anchors,
        valid_flags,
        gt_bboxes,
        gt_bboxes_ignore,
        gt_labels,
        img_meta,
        label_channels=1,
        unmap_outputs=True,
    ):
        """Assign, sample and build regression/classification targets for one image."""
        inside_flags = anchor_inside_flags(
            flat_anchors, valid_flags, img_meta["img_shape"][:2], self.train_cfg["allowed_border"]
        )
        if not inside_flags.any():
            return (None,) * 7
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors,
            gt_bboxes,
            gt_bboxes_ignore,
            # With a sampler in play the labels are looked up afterwards from
            # `pos_assigned_gt_inds`, so the assigner does not need them.
            None if self.sampling else gt_labels,
        )
        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        # Everything starts as background with zero weight; assignment then
        # turns on the anchors that were actually sampled.
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if self.reg_decoded_bbox:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes
                )
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only the RPN passes gt_labels=None: it has a single foreground
                # class, which is 0.
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            pos_weight = self.train_cfg["pos_weight"]
            label_weights[pos_inds] = 1.0 if pos_weight <= 0 else pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        if unmap_outputs:
            # Scatter back over the full anchor set; anchors outside the image
            # get the background label and zero weight, so they cost nothing.
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds,
                sampling_result)

    def get_targets(
        self,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        img_metas,
        gt_bboxes_ignore_list=None,
        gt_labels_list=None,
        label_channels=1,
        unmap_outputs=True,
        return_sampling_results=False,
    ):
        """Per-image targets, regrouped per scale level.

        Targets are computed per image over the concatenated anchor set, then
        ``images_to_levels`` transposes them back so each entry lines up with
        one feature map -- which is the layout ``loss_single`` needs.
        """
        num_imgs = len(img_metas)
        if not len(anchor_list) == len(valid_flag_list) == num_imgs:
            raise ValueError("anchor_list, valid_flag_list and img_metas must agree in length")

        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = [torch.cat(anchor_list[i]) for i in range(num_imgs)]
        concat_valid_flag_list = [torch.cat(valid_flag_list[i]) for i in range(num_imgs)]

        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None] * num_imgs
        if gt_labels_list is None:
            gt_labels_list = [None] * num_imgs

        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs,
        )
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list,
         neg_inds_list, sampling_results_list) = results[:7]
        if any(labels is None for labels in all_labels):
            return None

        # max(..., 1) guards against a zero denominator on an image with no
        # positives; it biases the average slightly rather than producing NaN.
        num_total_pos = sum(max(inds.numel(), 1) for inds in pos_inds_list)
        num_total_neg = sum(max(inds.numel(), 1) for inds in neg_inds_list)

        res = (
            images_to_levels(all_labels, num_level_anchors),
            images_to_levels(all_label_weights, num_level_anchors),
            images_to_levels(all_bbox_targets, num_level_anchors),
            images_to_levels(all_bbox_weights, num_level_anchors),
            num_total_pos,
            num_total_neg,
        )
        if return_sampling_results:
            res = res + (sampling_results_list,)
        return res

    # ---- loss --------------------------------------------------------------

    def loss_single(
        self,
        cls_score,
        bbox_pred,
        anchors,
        labels,
        label_weights,
        bbox_targets,
        bbox_weights,
        num_total_samples,
    ):
        """Loss for one scale level. ``num_total_samples`` is the shared
        ``avg_factor``, so levels are weighted by how many anchors they own
        rather than each contributing equally."""
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_total_samples)

        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=num_total_samples
        )
        return loss_cls, loss_bbox

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        """Returns ``dict(loss_cls=[...], loss_bbox=[...])``, one entry per level."""
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        if len(featmap_sizes) != self.prior_generator.num_levels:
            raise ValueError(
                f"got {len(featmap_sizes)} feature levels but the anchor generator "
                f"is configured for {self.prior_generator.num_levels}"
            )
        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
        )
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        num_total_samples = num_total_pos + num_total_neg if self.sampling else num_total_pos

        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = [torch.cat(anchors) for anchors in anchor_list]
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
        )
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    # ---- inference ---------------------------------------------------------

    def get_bboxes(self, cls_scores, bbox_preds, img_metas=None, cfg=None, rescale=False,
                   with_nms=True, **kwargs):
        """Decode a batch of predictions into per-image boxes."""
        if len(cls_scores) != len(bbox_preds):
            raise ValueError("cls_scores and bbox_preds must have the same number of levels")
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(len(cls_scores))]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device
        )
        return [
            self._get_bboxes_single(
                select_single_mlvl(cls_scores, img_id),
                select_single_mlvl(bbox_preds, img_id),
                mlvl_priors,
                img_metas[img_id],
                cfg,
                rescale,
                with_nms,
                **kwargs,
            )
            for img_id in range(len(img_metas))
        ]

    def _get_bboxes_single(self, cls_score_list, bbox_pred_list, mlvl_priors, img_meta, cfg,
                           rescale=False, with_nms=True, **kwargs):
        raise NotImplementedError(
            "AnchorHead has no generic decode path; mmdet's was written for the "
            "RetinaNet family, which is not ported. Use RPNHead."
        )

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=None,
                      proposal_cfg=None, **kwargs):
        """Losses, plus proposals when ``proposal_cfg`` is given (two-stage training)."""
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        return losses, self.get_bboxes(*outs, img_metas=img_metas, cfg=proposal_cfg)

    def simple_test_rpn(self, x, img_metas):
        """Proposals only, using ``test_cfg``. The two-stage detector's entry point."""
        return self.get_bboxes(*self(x), img_metas=img_metas)
