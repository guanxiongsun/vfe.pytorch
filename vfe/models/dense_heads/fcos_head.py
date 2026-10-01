"""FCOS, the anchor-free dense head. Port of
``mmdet.models.dense_heads.{anchor_free_head,fcos_head}`` together with the
score-factor-aware decode path of ``base_dense_head`` that :class:`AnchorHead`
deliberately left out (it was written for this family of heads).

FCOS predicts, at every cell of every pyramid level, a class score, the four
distances to the box edges, and a *centerness* that down-weights cells far
from an object's centre at test time. Each level owns a range of box sizes
(``regress_ranges``), which is how one object ends up on one level.

Label convention: foreground classes are ``[0, num_classes - 1]`` and
background is ``num_classes``.

One addition for EOVOD, absent from mmdet: ``forward``, ``get_bboxes`` and
``simple_test`` take ``level_ids``, the subset of pyramid levels to run. The
size prior skips the head on low levels for frames predicted to hold no small
objects -- where most of a one-stage detector's head time goes -- and
``with_levels=True`` reports which level each detection came from, which the
size prior needs to decide that. ``with_cls_scores=True`` also reports each
detection's class score before centerness is multiplied in, which EOVOD may
validate on. ``loss`` always sees every level.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vfe.core import (
    build_bbox_coder,
    filter_scores_and_topk,
    multi_apply,
    reduce_mean,
    select_single_mlvl,
)
from vfe.core.point_generator import MlvlPointGenerator
from vfe.layers import ConvModule, Scale, bias_init_with_prob, normal_init
from vfe.models.builder import HEADS, build_loss
from vfe.ops import batched_nms

__all__ = ["AnchorFreeHead", "FCOSHead"]

INF = 1e8


@HEADS.register_module()
class AnchorFreeHead(nn.Module):
    """Two conv towers (classification, regression) shared across levels, each
    ``stacked_convs`` deep, followed by a 3x3 predictor.

    Args:
        num_classes: categories *excluding* background.
        in_channels / feat_channels: input and tower width.
        strides: one per pyramid level, low to high.
        conv_bias: ``'auto'`` -> no conv bias when a norm layer follows.
        norm_cfg: e.g. ``dict(type='GN', num_groups=32)`` for FCOS.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        strides: Sequence[int] = (4, 8, 16, 32, 64),
        conv_bias: bool | str = "auto",
        loss_cls: dict | None = None,
        loss_bbox: dict | None = None,
        bbox_coder: dict | None = None,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        train_cfg: Any = None,
        test_cfg: Any = None,
    ):
        super().__init__()
        loss_cls = loss_cls or dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        )
        loss_bbox = loss_bbox or dict(type="IoULoss", loss_weight=1.0)
        bbox_coder = bbox_coder or dict(type="DistancePointBBoxCoder")
        if conv_bias != "auto" and not isinstance(conv_bias, bool):
            raise ValueError(f"conv_bias must be 'auto' or a bool, got {conv_bias!r}")

        self.num_classes = num_classes
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        self.cls_out_channels = num_classes if self.use_sigmoid_cls else num_classes + 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = tuple(strides)
        self.conv_bias = conv_bias
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.prior_generator = MlvlPointGenerator(self.strides)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # Post-processing with one GPU->CPU synchronisation per image instead
        # of one per level (same detections; see _get_bboxes_single_one_sync).
        self.one_sync_postprocess = False
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    def _init_layers(self) -> None:
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()

    def _tower(self) -> nn.ModuleList:
        convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            convs.append(
                ConvModule(
                    chn, self.feat_channels, 3, stride=1, padding=1, conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg, bias=self.conv_bias,
                )
            )
        return convs

    def _init_cls_convs(self) -> None:
        self.cls_convs = self._tower()

    def _init_reg_convs(self) -> None:
        self.reg_convs = self._tower()

    def _init_predictor(self) -> None:
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

    def init_weights(self) -> None:
        """mmdet's ``init_cfg``: every conv N(0, 0.01), and ``conv_cls``'s bias
        set so a fresh head predicts a 1% prior for every class."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_init_with_prob(0.01))

    def forward_single(self, x: torch.Tensor, reg_x: torch.Tensor | None = None):
        """``reg_x``, when given, feeds the regression tower instead of ``x``."""
        cls_feat = x
        reg_feat = x if reg_x is None else reg_x
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        return cls_score, bbox_pred, cls_feat, reg_feat

    def _levels(self, level_ids: Sequence[int] | None, num_feats: int) -> list[int]:
        levels = list(range(self.num_levels)) if level_ids is None else list(level_ids)
        if len(levels) != num_feats:
            raise ValueError(f"{num_feats} feature maps for levels {levels}")
        if any(lvl < 0 or lvl >= self.num_levels for lvl in levels):
            raise ValueError(f"level ids {levels} outside 0..{self.num_levels - 1}")
        return levels


@HEADS.register_module()
class FCOSHead(AnchorFreeHead):
    """Args (beyond :class:`AnchorFreeHead`):
        regress_ranges: the ``(min, max)`` of ``max(l, t, r, b)`` each level
            is responsible for; objects outside every range on a level are
            background there.
        center_sampling / center_sample_radius: count only cells within
            ``radius * stride`` of the box centre as positives.
        norm_on_bbox: predict distances in units of the stride (ReLU) instead
            of ``exp``; the official repo's later recipe.
        centerness_on_reg: centerness from the regression tower instead of
            the classification tower.
        loss_centerness: BCE on the centerness logit.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        regress_ranges: Sequence[tuple[float, float]] = (
            (-1, 64), (64, 128), (128, 256), (256, 512), (512, INF)
        ),
        center_sampling: bool = False,
        center_sample_radius: float = 1.5,
        norm_on_bbox: bool = False,
        centerness_on_reg: bool = False,
        loss_cls: dict | None = None,
        loss_bbox: dict | None = None,
        loss_centerness: dict | None = None,
        norm_cfg: dict | None = None,
        **kwargs,
    ):
        self.regress_ranges = tuple(tuple(r) for r in regress_ranges)
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        if norm_cfg is None:
            norm_cfg = dict(type="GN", num_groups=32, requires_grad=True)
        super().__init__(
            num_classes, in_channels, loss_cls=loss_cls, loss_bbox=loss_bbox, norm_cfg=norm_cfg,
            **kwargs,
        )
        if len(self.regress_ranges) != self.num_levels:
            raise ValueError(
                f"{len(self.regress_ranges)} regress ranges for {self.num_levels} strides"
            )
        loss_centerness = loss_centerness or dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
        )
        self.loss_centerness = build_loss(loss_centerness)

    def _init_layers(self) -> None:
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    # ---- forward -----------------------------------------------------------

    def forward(self, feats: Sequence[torch.Tensor], level_ids: Sequence[int] | None = None,
                reg_feats: Sequence[torch.Tensor] | None = None):
        """``(cls_scores, bbox_preds, centernesses)``, each a list over the
        levels run. ``feats`` holds one map per entry of ``level_ids`` (every
        level when None); ``reg_feats``, if given, one per level as well, feed
        the regression tower instead."""
        levels = self._levels(level_ids, len(feats))
        scales = [self.scales[lvl] for lvl in levels]
        strides = [self.strides[lvl] for lvl in levels]
        if reg_feats is None:
            reg_feats = [None] * len(feats)
        elif len(reg_feats) != len(feats):
            raise ValueError(f"{len(reg_feats)} regression maps for {len(feats)} levels")
        return multi_apply(self.forward_single, feats, scales, strides, reg_feats)

    def forward_single(self, x: torch.Tensor, scale: Scale, stride: int,
                       reg_x: torch.Tensor | None = None):
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x, reg_x)
        centerness = self.conv_centerness(reg_feat if self.centerness_on_reg else cls_feat)
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred = bbox_pred * stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness

    # ---- loss --------------------------------------------------------------

    def loss(self, cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels, img_metas,
             gt_bboxes_ignore=None) -> dict[str, torch.Tensor]:
        """Focal loss over every cell, centerness-weighted IoU loss and BCE on
        centerness over the positives; all normalised by the positive count."""
        if not len(cls_scores) == len(bbox_preds) == len(centernesses) == self.num_levels:
            raise ValueError("loss needs the outputs of every level")
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes, dtype=bbox_preds[0].dtype, device=bbox_preds[0].device
        )
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ])
        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds
        ])
        flatten_centerness = torch.cat([
            centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses
        ])
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds, pos_decoded_target_preds, weight=pos_centerness_targets,
                avg_factor=centerness_denorm,
            )
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos
            )
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_centerness=loss_centerness)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Per-level labels and ``(l, t, r, b)`` targets, images concatenated
        within each level (the layout ``loss`` flattens)."""
        num_levels = len(points)
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i])
            for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        num_points = [center.size(0) for center in points]

        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single, gt_bboxes_list, gt_labels_list, points=concat_points,
            regress_ranges=concat_regress_ranges, num_points_per_lvl=num_points,
        )
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list
        ]

        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """One image: every point gets the smallest ground-truth box it lies
        inside whose size falls in its level's range, or background."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return (gt_labels.new_full((num_points,), self.num_classes),
                    gt_bboxes.new_zeros((num_points, 4)))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end
            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0], x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1], y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2], gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3], gt_bboxes[..., 3], y_maxs)
            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1
            )
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1])
        )

        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        return labels, bbox_targets

    @staticmethod
    def centerness_target(pos_bbox_targets: torch.Tensor) -> torch.Tensor:
        """``sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))``: 1 at the
        centre, 0 on the edge."""
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]
            ) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    # ---- inference ---------------------------------------------------------

    def get_bboxes(self, cls_scores, bbox_preds, score_factors=None, img_metas=None, cfg=None,
                   rescale=False, with_nms=True, level_ids=None, with_levels=False,
                   with_cls_scores=False):
        """Per-image detections from the outputs of the levels in ``level_ids``.

        Each result is ``(det_bboxes (n, 5), det_labels (n,))``, plus
        ``det_levels (n,)`` when ``with_levels`` and then ``det_cls_scores
        (n,)`` (the class score before centerness) when ``with_cls_scores``.
        With ``with_nms=False`` the pre-NMS candidates come back instead.
        """
        if not len(cls_scores) == len(bbox_preds) == len(score_factors):
            raise ValueError("cls_scores, bbox_preds and score_factors must agree in length")
        levels = self._levels(level_ids, len(cls_scores))
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(len(cls_scores))]
        mlvl_priors = [
            self.prior_generator.single_level_grid_priors(
                featmap_sizes[i], lvl, dtype=cls_scores[0].dtype, device=cls_scores[0].device
            )
            for i, lvl in enumerate(levels)
        ]
        return [
            self._get_bboxes_single(
                select_single_mlvl(cls_scores, img_id),
                select_single_mlvl(bbox_preds, img_id),
                select_single_mlvl(score_factors, img_id),
                mlvl_priors,
                levels,
                img_metas[img_id],
                cfg,
                rescale,
                with_nms,
                with_levels,
                with_cls_scores,
            )
            for img_id in range(len(img_metas))
        ]

    def _get_bboxes_single(self, cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors,
                           levels, img_meta, cfg, rescale=False, with_nms=True,
                           with_levels=False, with_cls_scores=False):
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta["img_shape"]
        nms_pre = cfg.get("nms_pre", -1)

        if self.one_sync_postprocess:
            return self._get_bboxes_single_one_sync(
                cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors, levels,
                img_meta, cfg, rescale, with_nms, with_levels, with_cls_scores,
            )
        mlvl_bboxes, mlvl_scores, mlvl_labels, mlvl_score_factors, mlvl_levels = [], [], [], [], []
        for cls_score, bbox_pred, score_factor, priors, lvl in zip(
            cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors, levels, strict=True
        ):
            if cls_score.size()[-2:] != bbox_pred.size()[-2:]:
                raise ValueError("cls_score and bbox_pred sizes disagree")
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            # Threshold and top-k on the class scores alone (mmdet >= 2.19),
            # then bring the centerness in afterwards.
            scores, labels, keep_idxs, filtered = filter_scores_and_topk(
                scores, cfg["score_thr"], nms_pre, dict(bbox_pred=bbox_pred, priors=priors)
            )
            bboxes = self.bbox_coder.decode(
                filtered["priors"], filtered["bbox_pred"], max_shape=img_shape
            )
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_score_factors.append(score_factor[keep_idxs])
            mlvl_levels.append(labels.new_full(labels.shape, lvl))

        return self._bbox_post_process(
            mlvl_scores, mlvl_labels, mlvl_bboxes, mlvl_levels, img_meta["scale_factor"], cfg,
            rescale, with_nms, mlvl_score_factors, with_levels, with_cls_scores,
        )

    def _get_bboxes_single_one_sync(self, cls_score_list, bbox_pred_list, score_factor_list,
                                    mlvl_priors, levels, img_meta, cfg, rescale, with_nms,
                                    with_levels, with_cls_scores):
        """The same candidates as the per-level path, with one GPU->CPU
        synchronisation per image instead of one per level: each level keeps
        its ``nms_pre`` best class scores by a stable sort over all of them
        (those at or below ``score_thr`` set to -inf, so they sort last and
        valid ties keep index order, as mmdet's sort of the valid subset
        does), and the invalid ones are dropped once, after concatenation."""
        img_shape = img_meta["img_shape"]
        nms_pre = cfg.get("nms_pre", -1)
        thr = cfg["score_thr"]
        cols = self.cls_out_channels if self.use_sigmoid_cls else self.cls_out_channels - 1
        out = {k: [] for k in ("bboxes", "scores", "labels", "factors", "levels", "valid")}
        for cls_score, bbox_pred, score_factor, priors, lvl in zip(
            cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors, levels, strict=True
        ):
            if cls_score.size()[-2:] != bbox_pred.size()[-2:]:
                raise ValueError("cls_score and bbox_pred sizes disagree")
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            scores = cls_score.sigmoid() if self.use_sigmoid_cls else cls_score.softmax(-1)[:, :-1]
            flat = scores.reshape(-1)
            k = flat.numel() if nms_pre is None or nms_pre < 0 else min(nms_pre, flat.numel())
            ranked, order = flat.masked_fill(flat <= thr, float("-inf")).sort(
                descending=True, stable=True)
            order = order[:k]
            keep_idxs = torch.div(order, cols, rounding_mode="floor")
            labels = order % cols
            out["bboxes"].append(self.bbox_coder.decode(
                priors[keep_idxs], bbox_pred[keep_idxs], max_shape=img_shape))
            out["scores"].append(flat[order])
            out["labels"].append(labels)
            out["factors"].append(score_factor[keep_idxs])
            out["levels"].append(labels.new_full(labels.shape, lvl))
            out["valid"].append(ranked[:k] > float("-inf"))
        valid = torch.cat(out["valid"])
        kept = {name: [torch.cat(v)[valid]] for name, v in out.items() if name != "valid"}
        return self._bbox_post_process(
            kept["scores"], kept["labels"], kept["bboxes"], kept["levels"],
            img_meta["scale_factor"], cfg, rescale, with_nms, kept["factors"], with_levels,
            with_cls_scores,
        )

    @staticmethod
    def _bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes, mlvl_levels, scale_factor, cfg,
                           rescale, with_nms, mlvl_score_factors, with_levels,
                           with_cls_scores=False):
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= torch.as_tensor(
                scale_factor, dtype=mlvl_bboxes.dtype, device=mlvl_bboxes.device
            )
        mlvl_cls_scores = torch.cat(mlvl_scores)
        mlvl_scores = mlvl_cls_scores * torch.cat(mlvl_score_factors)
        mlvl_labels = torch.cat(mlvl_labels)
        mlvl_levels = torch.cat(mlvl_levels)

        def extras(keep=None):
            out = ()
            if with_levels:
                out += (mlvl_levels if keep is None else mlvl_levels[keep],)
            if with_cls_scores:
                out += (mlvl_cls_scores if keep is None else mlvl_cls_scores[keep],)
            return out

        if not with_nms:
            return (mlvl_bboxes, mlvl_scores, mlvl_labels) + extras()

        if mlvl_bboxes.numel() == 0:
            det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)
            return (det_bboxes, mlvl_labels) + extras()

        det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores, mlvl_labels, cfg["nms"])
        max_per_img = cfg["max_per_img"]
        keep_idxs = keep_idxs[:max_per_img]
        return (det_bboxes[:max_per_img], mlvl_labels[keep_idxs]) + extras(keep_idxs)

    # ---- entry points ------------------------------------------------------

    def forward_train(self, x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None,
                      reg_feats=None, **kwargs) -> dict[str, torch.Tensor]:
        outs = self(x, reg_feats=reg_feats)
        return self.loss(*outs, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=gt_bboxes_ignore)

    def simple_test(self, feats, img_metas, rescale=False, level_ids=None, with_levels=False,
                    with_cls_scores=False, reg_feats=None):
        """Detections for the levels in ``level_ids`` (all when None)."""
        outs = self(feats, level_ids=level_ids, reg_feats=reg_feats)
        return self.get_bboxes(
            *outs, img_metas=img_metas, rescale=rescale, level_ids=level_ids,
            with_levels=with_levels, with_cls_scores=with_cls_scores,
        )
