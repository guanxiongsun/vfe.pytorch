"""YOLOX's decoupled, anchor-free head. Port of
``mmdet.models.dense_heads.yolox_head``.

Per level, a classification tower predicts class scores and a regression
tower predicts the box -- centre offset and log size, in units of the stride
-- and an objectness score. A detection's score is ``sigmoid(cls) x
sigmoid(obj)``. Training assigns priors to objects with SimOTA; the
classification target is the IoU of the predicted box, the box loss is
``1 - IoU^2``, and an L1 loss on the raw offsets joins for the last epochs
(``use_l1``, set by the training schedule).

Additions for EOVOD, absent from mmdet (as in :mod:`.fcos_head`): ``forward``,
``get_bboxes`` and ``simple_test`` take ``level_ids``, the pyramid levels to
run; ``with_levels`` / ``with_cls_scores`` report each detection's level and
its class score before objectness; ``reg_feats`` feed the regression tower
(and objectness) instead of the maps the classification tower reads. Also:
an image without detections returns ``(0, 5)`` boxes (mmdet's are ``(0, 4)``),
and the test config may cap ``max_per_img``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vfe.core import (
    bbox_xyxy_to_cxcywh,
    build_assigner,
    build_sampler,
    multi_apply,
    reduce_mean,
)
from vfe.core.point_generator import MlvlPointGenerator
from vfe.layers.conv_module import ConvModule
from vfe.layers.csp_layer import YOLOX_ACT, YOLOX_NORM, _no_depthwise
from vfe.layers.weight_init import bias_init_with_prob
from vfe.models.backbones.csp_darknet import init_yolox_convs
from vfe.models.builder import HEADS, build_loss
from vfe.ops import batched_nms

__all__ = ["YOLOXHead"]


@HEADS.register_module()
class YOLOXHead(nn.Module):
    """Args:
        num_classes: foreground classes (no background channel).
        in_channels: width of the neck's levels.
        feat_channels: width of the towers.
        stacked_convs: 3x3 convs per tower.
        strides: the levels' strides.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 2,
        strides: Sequence[int] = (8, 16, 32),
        use_depthwise: bool = False,
        dcn_on_last_conv: bool = False,
        conv_bias: bool | str = "auto",
        conv_cfg: dict | None = None,
        norm_cfg: dict = YOLOX_NORM,
        act_cfg: dict = YOLOX_ACT,
        loss_cls: dict | None = None,
        loss_bbox: dict | None = None,
        loss_obj: dict | None = None,
        loss_l1: dict | None = None,
        train_cfg: Any = None,
        test_cfg: Any = None,
        init_cfg: Any = None,
    ):
        super().__init__()
        _no_depthwise(use_depthwise)
        if dcn_on_last_conv:
            raise NotImplementedError("deformable convs are not ported")
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = list(strides)
        self.num_levels = len(self.strides)
        self.conv_bias = conv_bias
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.use_sigmoid_cls = True

        self.loss_cls = build_loss(loss_cls or dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0))
        self.loss_bbox = build_loss(loss_bbox or dict(
            type="IoULoss", mode="square", eps=1e-16, reduction="sum", loss_weight=5.0))
        self.loss_obj = build_loss(loss_obj or dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0))
        self.loss_l1 = build_loss(loss_l1 or dict(type="L1Loss", reduction="sum",
                                                  loss_weight=1.0))
        # Off until the last epochs; the training schedule turns it on.
        self.use_l1 = False

        self.prior_generator = MlvlPointGenerator(self.strides, offset=0)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if train_cfg:
            self.assigner = build_assigner(train_cfg["assigner"])
            self.sampler = build_sampler(dict(type="PseudoSampler"))
        self._init_layers()

    def _init_layers(self) -> None:
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            self.multi_level_conv_cls.append(nn.Conv2d(self.feat_channels, self.cls_out_channels,
                                                       1))
            self.multi_level_conv_reg.append(nn.Conv2d(self.feat_channels, 4, 1))
            self.multi_level_conv_obj.append(nn.Conv2d(self.feat_channels, 1, 1))

    def _build_stacked_convs(self) -> nn.Sequential:
        return nn.Sequential(*[
            ConvModule(self.in_channels if i == 0 else self.feat_channels, self.feat_channels, 3,
                       stride=1, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
                       act_cfg=self.act_cfg, bias=self.conv_bias)
            for i in range(self.stacked_convs)
        ])

    def init_weights(self) -> None:
        init_yolox_convs(self)
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.multi_level_conv_cls, self.multi_level_conv_obj,
                                      strict=True):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    # ---- forward -----------------------------------------------------------

    def _levels(self, level_ids: Sequence[int] | None, num_feats: int) -> list[int]:
        levels = list(range(self.num_levels)) if level_ids is None else list(level_ids)
        if len(levels) != num_feats:
            raise ValueError(f"{num_feats} feature maps for levels {levels}")
        if any(lvl < 0 or lvl >= self.num_levels for lvl in levels):
            raise ValueError(f"level ids {levels} outside 0..{self.num_levels - 1}")
        return levels

    def forward(self, feats: Sequence[torch.Tensor], level_ids: Sequence[int] | None = None,
                reg_feats: Sequence[torch.Tensor] | None = None):
        """``(cls_scores, bbox_preds, objectnesses)``, each a list over the
        levels run (every level when ``level_ids`` is None); ``reg_feats``,
        one per level if given, feed the regression tower."""
        levels = self._levels(level_ids, len(feats))
        if reg_feats is None:
            reg_feats = [None] * len(feats)
        elif len(reg_feats) != len(feats):
            raise ValueError(f"{len(reg_feats)} regression maps for {len(feats)} levels")
        return multi_apply(
            self.forward_single, feats,
            [self.multi_level_cls_convs[lvl] for lvl in levels],
            [self.multi_level_reg_convs[lvl] for lvl in levels],
            [self.multi_level_conv_cls[lvl] for lvl in levels],
            [self.multi_level_conv_reg[lvl] for lvl in levels],
            [self.multi_level_conv_obj[lvl] for lvl in levels],
            reg_feats,
        )

    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg, conv_obj, reg_x=None):
        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x if reg_x is None else reg_x)
        return conv_cls(cls_feat), conv_reg(reg_feat), conv_obj(reg_feat)

    # ---- decoding ----------------------------------------------------------

    def _priors(self, featmap_sizes, levels, dtype, device) -> list[torch.Tensor]:
        """``(H*W, 4)`` priors ``(x, y, stride_w, stride_h)`` per level run."""
        return [self.prior_generator.single_level_grid_priors(
                    size, level_idx=lvl, dtype=dtype, device=device, with_stride=True)
                for size, lvl in zip(featmap_sizes, levels, strict=True)]

    @staticmethod
    def _bbox_decode(priors: torch.Tensor, bbox_preds: torch.Tensor) -> torch.Tensor:
        xys = bbox_preds[..., :2] * priors[:, 2:] + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]
        tl_x = xys[..., 0] - whs[..., 0] / 2
        tl_y = xys[..., 1] - whs[..., 1] / 2
        br_x = xys[..., 0] + whs[..., 0] / 2
        br_y = xys[..., 1] + whs[..., 1] / 2
        return torch.stack([tl_x, tl_y, br_x, br_y], -1)

    def _flatten(self, maps: Sequence[torch.Tensor], num_imgs: int, channels: int):
        return torch.cat([m.permute(0, 2, 3, 1).reshape(num_imgs, -1, channels) for m in maps],
                         dim=1)

    def get_bboxes(self, cls_scores, bbox_preds, objectnesses, img_metas=None, cfg=None,
                   rescale=False, with_nms=True, level_ids=None, with_levels=False,
                   with_cls_scores=False):
        """Per image ``(det_bboxes (n, 5), det_labels (n,))``, then
        ``det_levels (n,)`` when ``with_levels`` and ``det_cls_scores (n,)``
        when ``with_cls_scores``. ``with_nms`` is accepted for the API; mmdet's
        YOLOX always applies NMS."""
        if not len(cls_scores) == len(bbox_preds) == len(objectnesses):
            raise ValueError("one prediction of each kind per level")
        cfg = self.test_cfg if cfg is None else cfg
        levels = self._levels(level_ids, len(cls_scores))
        num_imgs = len(img_metas)
        priors = self._priors([c.shape[2:] for c in cls_scores], levels, cls_scores[0].dtype,
                              cls_scores[0].device)
        flatten_cls_scores = self._flatten(cls_scores, num_imgs, self.cls_out_channels).sigmoid()
        flatten_bbox_preds = self._flatten(bbox_preds, num_imgs, 4)
        flatten_objectness = self._flatten(objectnesses, num_imgs, 1)[..., 0].sigmoid()
        flatten_bboxes = self._bbox_decode(torch.cat(priors), flatten_bbox_preds)
        if rescale:
            scale_factors = [meta["scale_factor"] for meta in img_metas]
            flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(scale_factors).unsqueeze(1)
        prior_levels = None
        if with_levels:
            prior_levels = torch.cat([torch.full((len(p),), lvl, dtype=torch.long,
                                                 device=p.device)
                                      for p, lvl in zip(priors, levels, strict=True)])
        return [self._bboxes_nms(flatten_cls_scores[i], flatten_bboxes[i], flatten_objectness[i],
                                 cfg, prior_levels, with_levels, with_cls_scores)
                for i in range(num_imgs)]

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg, prior_levels=None,
                    with_levels=False, with_cls_scores=False):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = score_factor * max_scores >= cfg["score_thr"]
        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]
        extras = []
        if with_levels:
            extras.append(prior_levels[valid_mask])
        if with_cls_scores:
            extras.append(max_scores[valid_mask])
        if labels.numel() == 0:
            return (bboxes.new_zeros((0, 5)), labels, *extras)
        dets, keep = batched_nms(bboxes, scores, labels, cfg["nms"])
        max_per_img = cfg.get("max_per_img") if hasattr(cfg, "get") else None
        if max_per_img is not None:
            dets, keep = dets[:max_per_img], keep[:max_per_img]
        return (dets, labels[keep], *(extra[keep] for extra in extras))

    # ---- loss --------------------------------------------------------------

    def loss(self, cls_scores, bbox_preds, objectnesses, gt_bboxes, gt_labels, img_metas,
             gt_bboxes_ignore=None) -> dict[str, torch.Tensor]:
        num_imgs = len(img_metas)
        featmap_sizes = [c.shape[2:] for c in cls_scores]
        priors = self._priors(featmap_sizes, range(self.num_levels), cls_scores[0].dtype,
                              cls_scores[0].device)
        flatten_cls_preds = self._flatten(cls_scores, num_imgs, self.cls_out_channels)
        flatten_bbox_preds = self._flatten(bbox_preds, num_imgs, 4)
        flatten_objectness = self._flatten(objectnesses, num_imgs, 1)[..., 0]
        flatten_priors = torch.cat(priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets, num_fg_imgs = multi_apply(
            self._get_target_single, flatten_cls_preds.detach(), flatten_objectness.detach(),
            flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1), flatten_bboxes.detach(),
            gt_bboxes, gt_labels)

        num_pos = torch.tensor(sum(num_fg_imgs), dtype=torch.float,
                               device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)
        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)

        loss_bbox = self.loss_bbox(flatten_bboxes.view(-1, 4)[pos_masks],
                                   bbox_targets) / num_total_samples
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1), obj_targets) / num_total_samples
        loss_cls = self.loss_cls(flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
                                 cls_targets) / num_total_samples
        losses = dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
            losses["loss_l1"] = self.loss_l1(flatten_bbox_preds.view(-1, 4)[pos_masks],
                                             l1_targets) / num_total_samples
        return losses

    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes, gt_bboxes,
                           gt_labels):
        """One image's targets over all ``(n,)`` priors: the foreground mask,
        then classification, objectness, box and L1 targets and the number of
        positives."""
        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        if num_gts == 0:
            return (cls_preds.new_zeros(num_priors).bool(),
                    cls_preds.new_zeros((0, self.num_classes)),
                    cls_preds.new_zeros((num_priors, 1)),
                    cls_preds.new_zeros((0, 4)), cls_preds.new_zeros((0, 4)), 0)

        # SimOTA wants the priors' centres, not their top-left corners.
        offset_priors = torch.cat([priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)
        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(), offset_priors,
            decoded_bboxes, gt_bboxes, gt_labels)
        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target, priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return foreground_mask, cls_target, obj_target, bbox_target, l1_target, num_pos_per_img

    @staticmethod
    def _get_l1_target(l1_target, gt_bboxes, priors, eps=1e-8):
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target

    # ---- entry points ------------------------------------------------------

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None,
                      reg_feats=None, **kwargs) -> dict[str, torch.Tensor]:
        outs = self(x, reg_feats=reg_feats)
        return self.loss(*outs, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=gt_bboxes_ignore)

    def simple_test(self, feats, img_metas, rescale=False, level_ids=None, with_levels=False,
                    with_cls_scores=False, reg_feats=None):
        """Detections for the levels in ``level_ids`` (all when None)."""
        outs = self(feats, level_ids=level_ids, reg_feats=reg_feats)
        return self.get_bboxes(*outs, img_metas=img_metas, rescale=rescale, level_ids=level_ids,
                               with_levels=with_levels, with_cls_scores=with_cls_scores)
