"""Detectors on a TDViT backbone: the video models of TDViT's paper
(https://arxiv.org/abs/2402.09257), Table 2's Faster R-CNN and Table 3's SELSA.

TDViT puts the temporal modelling in the backbone
(:class:`~vfe.models.backbones.TDViT`), so around it a still-image detector
works unchanged: the neck, the RPN and the RoI head see one frame's features.
What this class adds is the backbone's video protocol:

* training: the key frame goes through the backbone with its reference
  frames, one per stage (the first ``num_stages`` of ``ref_img``,
  ``(B, R, 3, H, W)``); they only feed the TDTBs and have no losses;
* inference: frames arrive in video order, one per call; the backbone's
  memories are reset at each video's first frame (``frame_id == 0``) and
  filled as the video goes on.

The RoI head picks the detector, as in MAMBA. A ``StandardRoIHead`` is Faster
R-CNN (Table 2), and no reference frames are loaded at test time. A
``MambaRoIHead`` given reference RoIs on every frame is SELSA with RDN's
top-k reference proposals, the paper's "SELSA*" (Table 3): its references
are the frames of ``ref_img`` after the backbone's, seen on their own (every
TDTB attending to its frame, :meth:`TDViT.forward_spatial`). At test time a
video's references come with its first frame (``test_with_adaptive_stride``,
spread over the whole video), are extracted once, and every frame aggregates
them with its own top-k RoIs.

With a plain :class:`~vfe.models.backbones.SwinTransformer` backbone either
detector is the single-frame-backbone baseline (Table 2's Swin rows).
"""

from __future__ import annotations

import torch

from vfe.models.backbones.tdvit import TDViT
from vfe.models.builder import MODELS, build_detector
from vfe.models.roi_heads.mamba import MambaRoIHead
from vfe.models.vid.base import BaseVideoDetector

__all__ = ["TDViTDetector"]


@MODELS.register_module()
class TDViTDetector(BaseVideoDetector):
    """Args:
        detector: config of the wrapped two-stage detector; its backbone is a
            ``TDViT`` (or a ``SwinTransformer`` for the baseline), its RoI head
            a ``StandardRoIHead`` (Faster R-CNN) or a ``MambaRoIHead`` (SELSA).
        backbone_refs: how many of a key frame's reference frames, first in
            ``ref_img``, are the backbone's: a TDViT's stage count (the
            default), 0 for a Swin -- or TDViT's count for a Swin baseline
            loading TDViT's references, which then skips them.
        online: at test time a TDViT reads its memory; False sees every frame
            on its own (:meth:`TDViT.forward_spatial`) -- an ablation that
            keeps the weights and drops the temporal attention.
        frozen_modules: submodule name(s) to freeze at construction.
        train_cfg / test_cfg: unused (the detector carries its own), accepted
            because configs pass them.
    """

    def __init__(self, detector: dict, backbone_refs: int | None = None, online: bool = True,
                 frozen_modules=None, train_cfg=None, test_cfg=None):
        super().__init__()
        self.detector = build_detector(detector)
        if not hasattr(self.detector, "roi_head"):
            raise TypeError("TDViTDetector supports two-stage detectors")
        self.temporal = isinstance(self.detector.backbone, TDViT)
        self.selsa = isinstance(self.detector.roi_head, MambaRoIHead)
        self.online = online
        num_stages = len(self.detector.backbone.stages) if self.temporal else 0
        self.backbone_refs = num_stages if backbone_refs is None else backbone_refs
        if self.temporal and self.backbone_refs != num_stages:
            raise ValueError(f"a TDViT takes one reference per stage ({num_stages}), "
                             f"not backbone_refs={backbone_refs}")
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._started = False  # whether the current video's first frame was seen
        self._refs: tuple | None = None  # SELSA: the video's reference features and RoIs
        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def extract_feat(self, img: torch.Tensor, ref_img: torch.Tensor | None = None):
        """The key frame's features: from its references in training, from the
        backbone's memory at test time (unless not ``online``)."""
        backbone = self.detector.backbone
        if not self.temporal:
            x = backbone(img)
        elif ref_img is None and not self.online and not self.training:
            x = backbone.forward_spatial(img)
        else:
            x = backbone(img, ref_img)
        return self.detector.neck(x) if self.detector.with_neck else x

    def extract_still_feat(self, img: torch.Tensor):
        """Features of frames seen on their own: SELSA's references."""
        backbone = self.detector.backbone
        x = backbone.forward_spatial(img) if self.temporal else backbone(img)
        return self.detector.neck(x) if self.detector.with_neck else x

    def _topk(self, proposals_list):
        topk = self.detector.roi_head.bbox_head.topk
        return [proposals[:topk] for proposals in proposals_list]

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, ref_img=None,
                      ref_img_metas=None, gt_bboxes_ignore=None, gt_masks=None, proposals=None,
                      **kwargs) -> dict:
        """Losses for the key frames ``img``. ``ref_img`` holds their
        references: one per backbone stage, then SELSA's. Other pipeline
        outputs (``ref_gt_*``) are accepted and unused."""
        if gt_masks is not None:
            raise NotImplementedError("mask annotations are not ported")
        n = self.backbone_refs
        if (self.temporal or self.selsa) and ref_img is None:
            raise ValueError("this detector trains with reference frames (ref_img)")
        x = self.extract_feat(img, ref_img[:, :n] if self.temporal else None)

        detector = self.detector
        losses = {}
        if detector.with_rpn:
            proposal_cfg = detector.train_cfg.get("rpn_proposal", detector.test_cfg["rpn"])
            rpn_losses, proposal_list = detector.rpn_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
            )
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        if not self.selsa:
            losses.update(detector.roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore))
            return losses
        if len(img) != 1:
            raise ValueError("SELSA trains one key frame per GPU")
        if ref_img.shape[1] <= n:
            raise ValueError(f"SELSA needs reference frames after the backbone's {n}")
        # SELSA's references: with gradients, as in SELSA, through the
        # backbone frame by frame; the reference proposals use the test RPN.
        ref_x = self.extract_still_feat(ref_img[0, n:])
        ref_proposals_list = self._topk(
            detector.rpn_head.simple_test_rpn(ref_x, ref_img_metas[0][n:]))
        losses.update(detector.roi_head.forward_train(
            x, ref_x, img_metas, proposal_list, ref_proposals_list, gt_bboxes, gt_labels,
            gt_bboxes_ignore))
        return losses

    def simple_test(self, img, img_metas, ref_img=None, ref_img_metas=None, proposals=None,
                    ref_proposals=None, rescale: bool = False):
        """Detections for one frame, as ``bbox2result`` lists. Frames of a
        video must come in order, its first (``frame_id == 0``) first; for
        SELSA it brings the video's reference frames (``ref_img``
        ``[Tensor(1, R, C, H, W)]``, ``ref_img_metas`` ``[[[dict, ...]]]``,
        the test pipeline's nesting)."""
        if len(img) != 1:
            raise ValueError("TDViTDetector tests one frame at a time")
        if self.temporal or self.selsa:
            frame_id = img_metas[0].get("frame_id", -1)
            if frame_id < 0:
                raise KeyError("img_metas must carry 'frame_id' at test time")
            if frame_id == 0:
                self._start_video(ref_img, ref_img_metas)
            elif not self._started:
                raise RuntimeError("the first frame of a video (frame_id 0) must come first")
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.detector.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        roi_head = self.detector.roi_head
        if not self.selsa:
            return roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
        # The video's references, then the frame itself, as SELSA tests.
        ref_feats, ref_rois = self._refs
        ref_x = [torch.cat((ref, key), dim=0) for ref, key in zip(ref_feats, x, strict=True)]
        return roi_head.simple_test(x, ref_x, proposal_list, ref_rois + self._topk(proposal_list),
                                    img_metas, rescale=rescale)

    def _start_video(self, ref_img, ref_img_metas) -> None:
        if self.temporal:
            self.detector.backbone.reset_memory()
        if self.selsa:
            if ref_img is None:
                raise ValueError("SELSA needs the first frame of a video to bring its references")
            refs, metas = ref_img[0][0], ref_img_metas[0][0]
            ref_x = self.extract_still_feat(refs)
            self._refs = (ref_x, self._topk(self.detector.rpn_head.simple_test_rpn(ref_x, metas)))
        self._started = True

    def train(self, mode: bool = True):
        super().train(mode)
        self._started = False
        self._refs = None
        return self
