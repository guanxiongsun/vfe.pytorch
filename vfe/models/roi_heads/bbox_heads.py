"""Second-stage box heads. Port of
``mmdet.models.roi_heads.bbox_heads.{bbox_head,convfc_bbox_head}``.

Given a pooled feature per proposal, predict a class distribution over
``num_classes + 1`` (the last is background) and, per foreground class, a
refinement of the proposal box. ``Shared2FCBBoxHead`` -- two shared FC layers,
then the two linear predictors -- is the one every config here uses, and the
MAMBA and SELSA heads subclass ``ConvFCBBoxHead``.

Dropped from mmdet: ``refine_bboxes`` / ``regress_by_class`` (only Cascade-style
heads call them), the ``custom_cls_channels`` / ``custom_activation`` hooks for
Seesaw loss, ONNX export, and ``Shared4Conv1FCBBoxHead``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair

from vfe.core import build_bbox_coder, multi_apply, multiclass_nms
from vfe.layers import ConvModule, normal_init, xavier_init
from vfe.models.builder import HEADS, build_loss
from vfe.models.losses import accuracy

__all__ = ["BBoxHead", "ConvFCBBoxHead", "Shared2FCBBoxHead"]


def _build_linear(cfg: dict | None, in_features: int, out_features: int) -> nn.Linear:
    # mmdet's `build_linear_layer` also knows `NormedLinear`; nothing here uses it.
    layer_type = (cfg or {}).get("type", "Linear")
    if layer_type != "Linear":
        raise NotImplementedError(f"predictor type {layer_type!r} is not ported")
    return nn.Linear(in_features, out_features)


@HEADS.register_module()
class BBoxHead(nn.Module):
    """The minimal head: optional average pool, then one linear layer each for
    classification and regression.

    Args:
        with_avg_pool: pool the RoI feature to 1x1 instead of flattening it.
        with_cls / with_reg: which of the two predictors to build.
        roi_feat_size: spatial size of the incoming RoI feature.
        in_channels: channels of the incoming RoI feature.
        num_classes: foreground classes; the classifier has one more output.
        bbox_coder / loss_cls / loss_bbox: config dicts.
        reg_class_agnostic: predict one box for all classes rather than one
            per class.
        reg_decoded_bbox: apply the regression loss to decoded boxes.
    """

    def __init__(
        self,
        with_avg_pool: bool = False,
        with_cls: bool = True,
        with_reg: bool = True,
        roi_feat_size: int | tuple[int, int] = 7,
        in_channels: int = 256,
        num_classes: int = 80,
        bbox_coder: dict | None = None,
        reg_class_agnostic: bool = False,
        reg_decoded_bbox: bool = False,
        reg_predictor_cfg: dict | None = None,
        cls_predictor_cfg: dict | None = None,
        loss_cls: dict | None = None,
        loss_bbox: dict | None = None,
    ):
        super().__init__()
        if not (with_cls or with_reg):
            raise ValueError("a BBoxHead needs at least one of with_cls / with_reg")
        bbox_coder = bbox_coder or dict(
            type="DeltaXYWHBBoxCoder",
            clip_border=True,
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
        )
        loss_cls = loss_cls or dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0)
        loss_bbox = loss_bbox or dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0)

        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        in_features = in_channels
        if with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_features *= self.roi_feat_area
        self._build_predictors(in_features, in_features)

    def _build_predictors(self, cls_in_features: int, reg_in_features: int) -> None:
        """(Re)build ``fc_cls`` / ``fc_reg``. ``ConvFCBBoxHead`` calls this a
        second time once it knows what its branches output."""
        if self.with_cls:
            # +1: the background class.
            self.fc_cls = _build_linear(
                self.cls_predictor_cfg, cls_in_features, self.num_classes + 1
            )
        if self.with_reg:
            out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
            self.fc_reg = _build_linear(self.reg_predictor_cfg, reg_in_features, out_dim_reg)

    def init_weights(self) -> None:
        """mmdet's default ``init_cfg``: Normal(0.01) on ``fc_cls``, Normal(0.001)
        on ``fc_reg``. The regressor starts 10x smaller so early training does
        not move proposals far before the classifier has learned anything."""
        if self.with_cls:
            normal_init(self.fc_cls, std=0.01)
        if self.with_reg:
            normal_init(self.fc_reg, std=0.001)

    def forward(self, x: torch.Tensor):
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x).view(x.size(0), -1)
            else:
                # AvgPool2d rejects an empty batch; the mean is equivalent.
                x = torch.mean(x, dim=(-1, -2))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    # ---- targets -----------------------------------------------------------

    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes, pos_gt_labels, cfg):
        """Targets for one image's sampled RoIs, laid out positives first --
        the order ``SamplingResult.bboxes`` concatenates them in, which is what
        lets the two line up without an index."""
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        labels = pos_bboxes.new_full((num_samples,), self.num_classes, dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = cfg["pos_weight"]
            label_weights[:num_pos] = 1.0 if pos_weight <= 0 else pos_weight
            if self.reg_decoded_bbox:
                pos_bbox_targets = pos_gt_bboxes
            else:
                pos_bbox_targets = self.bbox_coder.encode(pos_bboxes, pos_gt_bboxes)
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0
        return labels, label_weights, bbox_targets, bbox_weights

    def get_targets(self, sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg, concat=True):
        """Targets for a batch; concatenated across images by default, which
        matches ``bbox2roi``'s layout of the RoIs themselves."""
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            [res.pos_bboxes for res in sampling_results],
            [res.neg_bboxes for res in sampling_results],
            [res.pos_gt_bboxes for res in sampling_results],
            [res.pos_gt_labels for res in sampling_results],
            cfg=rcnn_train_cfg,
        )
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights

    # ---- loss --------------------------------------------------------------

    def loss(self, cls_score, bbox_pred, rois, labels, label_weights, bbox_targets, bbox_weights,
             reduction_override=None) -> dict[str, torch.Tensor]:
        """``loss_cls``, ``acc`` and ``loss_bbox``.

        The two losses have different denominators, and both are mmdet's:
        classification averages over RoIs with non-zero weight, regression over
        *all* sampled RoIs (``bbox_targets.size(0)``) even though only positives
        contribute to the sum. The regression loss is therefore diluted by the
        negative fraction -- deliberate or not, it is what the released models
        were trained with.
        """
        losses: dict[str, torch.Tensor] = {}
        if cls_score is not None:
            # .item() syncs with the GPU; kept because the loss takes a Python
            # float here and changing that would change rounding.
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)
            if cls_score.numel() > 0:
                losses["loss_cls"] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override,
                )
                losses["acc"] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            # Foreground is [0, num_classes); background is num_classes and has
            # no regression target.
            pos_inds = (labels >= 0) & (labels < self.num_classes)
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
                else:
                    # Pick, for each positive RoI, the 4 deltas of its own class.
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[
                        pos_inds, labels[pos_inds]
                    ]
                losses["loss_bbox"] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds],
                    bbox_weights[pos_inds],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override,
                )
            else:
                # Gradient-connected zero; see SmoothL1Loss for why.
                losses["loss_bbox"] = bbox_pred[pos_inds].sum()
        return losses

    # ---- inference ---------------------------------------------------------

    def get_bboxes(self, rois, cls_score, bbox_pred, img_shape, scale_factor, rescale=False,
                   cfg: Any = None):
        """Decode one image's predictions.

        Returns ``(bboxes, scores)`` when ``cfg`` is None -- one box per class
        per RoI, before NMS -- else ``(det_bboxes, det_labels)`` after
        ``multiclass_nms``.
        """
        scores = F.softmax(cls_score, dim=-1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                # mmdet wrote `bboxes[:, [0, 2]].clamp_(...)`, which clamps a
                # copy (advanced indexing) and so did nothing. No config here
                # has with_reg=False, so fixing it changes no result.
                bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clamp(min=0, max=img_shape[1])
                bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clamp(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(bboxes.size(0), -1)

        if cfg is None:
            return bboxes, scores
        return multiclass_nms(bboxes, scores, cfg["score_thr"], cfg["nms"], cfg["max_per_img"])


@HEADS.register_module()
class ConvFCBBoxHead(BBoxHead):
    r"""Optional shared convs and FCs, then optional separate cls / reg branches::

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """

    def __init__(
        self,
        num_shared_convs: int = 0,
        num_shared_fcs: int = 0,
        num_cls_convs: int = 0,
        num_cls_fcs: int = 0,
        num_reg_convs: int = 0,
        num_reg_fcs: int = 0,
        conv_out_channels: int = 256,
        fc_out_channels: int = 1024,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_shared_convs + num_shared_fcs + num_cls_convs + num_cls_fcs + num_reg_convs \
                + num_reg_fcs <= 0:
            raise ValueError("ConvFCBBoxHead needs at least one conv or fc layer")
        if (num_cls_convs > 0 or num_reg_convs > 0) and num_shared_fcs != 0:
            # A branch conv needs a spatial input, which shared FCs have destroyed.
            raise ValueError("branch convs cannot follow shared fcs")
        if not self.with_cls and (num_cls_convs or num_cls_fcs):
            raise ValueError("cls branch layers given but with_cls=False")
        if not self.with_reg and (num_reg_convs or num_reg_fcs):
            raise ValueError("reg branch layers given but with_reg=False")

        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.shared_convs, self.shared_fcs, last_layer_dim = self._add_conv_fc_branch(
            num_shared_convs, num_shared_fcs, self.in_channels, True
        )
        self.shared_out_channels = last_layer_dim
        self.cls_convs, self.cls_fcs, self.cls_last_dim = self._add_conv_fc_branch(
            num_cls_convs, num_cls_fcs, self.shared_out_channels
        )
        self.reg_convs, self.reg_fcs, self.reg_last_dim = self._add_conv_fc_branch(
            num_reg_convs, num_reg_fcs, self.shared_out_channels
        )

        if num_shared_fcs == 0 and not self.with_avg_pool:
            if num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        self._build_predictors(self.cls_last_dim, self.reg_last_dim)

    def _add_conv_fc_branch(self, num_branch_convs, num_branch_fcs, in_channels,
                            is_shared=False):
        """convs -> (flatten) -> fcs. Returns the two ModuleLists and the
        output width."""
        last_layer_dim = in_channels
        branch_convs = nn.ModuleList()
        for i in range(num_branch_convs):
            branch_convs.append(
                ConvModule(
                    last_layer_dim if i == 0 else self.conv_out_channels,
                    self.conv_out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
        if num_branch_convs > 0:
            last_layer_dim = self.conv_out_channels

        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # The first FC sees a flattened C*H*W input, unless something
            # upstream already reduced it to C: an avg pool, or (for a separate
            # branch) the shared FCs.
            if (is_shared or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                branch_fcs.append(
                    nn.Linear(
                        last_layer_dim if i == 0 else self.fc_out_channels,
                        self.fc_out_channels,
                    )
                )
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self) -> None:
        """The base predictors' Normal init, then Xavier-uniform on every FC in
        the three branches.

        Branch convs, if any, keep torch's default init rather than mmcv
        ConvModule's Kaiming init -- ``vfe.layers.ConvModule`` has no
        ``init_weights``. No config here has branch convs.
        """
        super().init_weights()
        for branch in (self.shared_fcs, self.cls_fcs, self.reg_fcs):
            for m in branch.modules():
                xavier_init(m, distribution="uniform")

    def forward(self, x: torch.Tensor):
        for conv in self.shared_convs:
            x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        return self._forward_branches(x, x)

    def _forward_branches(self, x_cls: torch.Tensor, x_reg: torch.Tensor):
        """The separate cls / reg branches and the two predictors. Split out so
        the VID heads, which only change the shared part, can reuse it."""
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


@HEADS.register_module()
class Shared2FCBBoxHead(ConvFCBBoxHead):
    """Two shared FCs, no branch layers -- the Faster R-CNN default."""

    def __init__(self, fc_out_channels: int = 1024, **kwargs):
        super().__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            **kwargs,
        )
