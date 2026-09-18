"""STPN: Spatio-Temporal Prompting Network for video object detection.
Port of ``mmdet.models.vid.stpn.{stpn,dvp_predictor}``.

Reference frames go through the backbone without prompts; a predictor turns
the last stage's features into ``num_prompts`` prompt tokens, and the backbone
(:class:`~vfe.models.backbones.STPNSwinTransformer`) prepends them when it
processes the key frame. Neck, RPN and RoI head are the still-image detector's.

At test time (``test_with_adaptive_stride``, the released configs) the prompts
are computed once per video, from the references sampled at its first frame,
and reused for every later frame. The fixed-stride sampler is not supported,
as in the original.
"""

from __future__ import annotations

import torch
from torch import nn

from vfe.models.builder import MODELS, build_detector
from vfe.models.vid.base import BaseVideoDetector

__all__ = ["STPN", "AttentionPredictor", "AveragePredictor"]


class AttentionPredictor(nn.Module):
    """Prompts as ``num_prompts`` learned queries attending, in
    ``num_attention_blocks`` heads, to every reference-feature position.

    Initialisation happens at construction (torch's ``nn.Linear`` default, a
    standard normal ``query``): the original's ``init_weights`` recursion never
    reached this module, so a detector-wide ``init_weights`` leaves it as is.
    """

    def __init__(self, in_channels: int, num_attention_blocks: int = 16, num_prompts: int = 5,
                 prompt_dims: int = 96):
        super().__init__()
        self.reduction = nn.Linear(in_channels, prompt_dims)
        self.query = nn.Parameter(torch.zeros(num_prompts, prompt_dims))
        nn.init.normal_(self.query)
        self.fc_embed = nn.Linear(prompt_dims, prompt_dims)
        self.ref_fc_embed = nn.Linear(prompt_dims, prompt_dims)
        self.fc = nn.Linear(prompt_dims, prompt_dims)
        self.ref_fc = nn.Linear(prompt_dims, prompt_dims)
        self.num_attention_blocks = num_attention_blocks

    def forward(self, ref_x: torch.Tensor) -> torch.Tensor:
        """``ref_x``: (B, C, H, W) reference features; returns (num_prompts, prompt_dims)."""
        B, C, H, W = ref_x.shape
        ref_x = ref_x.view(B, C, -1).permute(0, 2, 1).reshape(-1, C)
        ref_x = self.reduction(ref_x)

        x = self.query
        roi_n, ref_roi_n = x.shape[0], ref_x.shape[0]
        # [blocks, roi_n, C / blocks]
        x_embed = self.fc_embed(x).view(roi_n, self.num_attention_blocks, -1).permute(1, 0, 2)
        # [blocks, C / blocks, ref_roi_n]
        ref_x_embed = (self.ref_fc_embed(ref_x)
                       .view(ref_roi_n, self.num_attention_blocks, -1).permute(1, 2, 0))
        # [blocks, roi_n, ref_roi_n]
        weights = torch.bmm(x_embed, ref_x_embed) / (x_embed.shape[-1] ** 0.5)
        weights = weights.softmax(dim=2)

        # [blocks, ref_roi_n, C / blocks]
        ref_x_new = self.ref_fc(ref_x).view(ref_roi_n, self.num_attention_blocks,
                                            -1).permute(1, 0, 2)
        # [roi_n, blocks, C / blocks]
        x_new = torch.bmm(weights, ref_x_new).permute(1, 0, 2).contiguous()
        return self.fc(x_new.view(roi_n, -1))


class AveragePredictor(nn.Module):
    """Prompts as the ``num_prompts`` reference positions with the largest L1
    norm, averaged over reference frames and projected."""

    def __init__(self, in_channels: int, num_prompts: int = 5, prompt_dims: int = 96):
        super().__init__()
        self.num_prompts = num_prompts
        self.reduction = nn.Linear(in_channels, prompt_dims)

    @staticmethod
    def get_topk(x: torch.Tensor, k: int = 100) -> torch.Tensor:
        """``x``: (B, N, C); the ``k`` rows of largest L1 norm per item, averaged over B."""
        result = []
        for feat in x:
            _, inds = feat.norm(1, dim=-1).topk(k)
            result.append(feat[inds])
        return torch.stack(result).mean(dim=0)

    def forward(self, ref_x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = ref_x.shape
        ref_x = ref_x.view(B, C, -1).permute(0, 2, 1)
        return self.reduction(self.get_topk(ref_x, k=self.num_prompts))


@MODELS.register_module()
class STPN(BaseVideoDetector):
    """Args:
        detector: the still-image detector; its backbone must accept prompts.
        predictor: ``'att'`` (:class:`AttentionPredictor`) or ``'avg'``.
        embed_dims: channels of the backbone's last stage (the predictor's input).
        num_prompts, prompt_dims: prompt count and width (the backbone's embed dims).
    """

    def __init__(self, detector: dict, pretrained=None, init_cfg=None, frozen_modules=None,
                 train_cfg=None, test_cfg=None, predictor: str = "att", embed_dims: int = 768,
                 num_prompts: int = 5, prompt_dims: int = 96):
        super().__init__()
        if pretrained is not None or init_cfg is not None:
            raise NotImplementedError("STPN-level pretrained/init_cfg are not ported; "
                                      "set init_cfg on the backbone")
        self.detector = build_detector(detector)
        if not hasattr(self.detector, "roi_head"):
            raise TypeError("STPN needs a two-stage detector")
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if predictor == "att":
            self.prompt_predictor = AttentionPredictor(embed_dims, num_prompts=num_prompts,
                                                       prompt_dims=prompt_dims)
        elif predictor == "avg":
            self.prompt_predictor = AveragePredictor(embed_dims, num_prompts=num_prompts,
                                                     prompt_dims=prompt_dims)
        else:
            raise ValueError(f"unknown predictor {predictor!r}")
        self.prompt: torch.Tensor | None = None  # the current test video's prompts

        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def init_weights(self) -> None:
        self.detector.init_weights()

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, ref_img,
                      gt_bboxes_ignore=None, gt_masks=None, proposals=None, **kwargs) -> dict:
        """Losses for one key frame (``img``, batch size 1) prompted from its
        reference frames (``ref_img``, (1, R, C, H, W)). Other pipeline outputs
        (``ref_img_metas``, ``ref_gt_*``) are accepted and unused."""
        if len(img) != 1:
            raise ValueError("STPN supports one key frame per GPU")
        if gt_masks is not None:
            raise NotImplementedError("mask annotations are not ported")
        detector = self.detector
        with torch.no_grad():
            ref_x = detector.backbone(ref_img[0])[-1]
        prompt = self.prompt_predictor(ref_x)

        x = detector.backbone(img, prompt)
        if detector.with_neck:
            x = detector.neck(x)

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

        losses.update(detector.roi_head.forward_train(
            x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, **kwargs))
        return losses

    def extract_feats(self, img, img_metas, ref_img):
        """Prompted features of the current frame; on a video's first frame the
        prompts are recomputed from ``ref_img`` ((1, R, C, H, W))."""
        frame_id = img_metas[0].get("frame_id", -1)
        if frame_id < 0:
            raise KeyError("img_metas must carry 'frame_id' at test time")
        if img_metas[0].get("frame_stride", -1) >= 1:
            raise NotImplementedError("STPN supports only test_with_adaptive_stride")
        if frame_id == 0:
            ref_x = self.detector.backbone(ref_img[0])[-1]
            self.prompt = self.prompt_predictor(ref_x)
        if self.prompt is None:
            raise RuntimeError("the first frame of a video (frame_id 0) must come first")
        x = self.detector.backbone(img, self.prompt)
        if self.detector.with_neck:
            x = self.detector.neck(x)
        return x

    def simple_test(self, img, img_metas, ref_img=None, ref_img_metas=None, proposals=None,
                    ref_proposals=None, rescale: bool = False):
        if ref_img is not None:
            ref_img = ref_img[0]
        x = self.extract_feats(img, img_metas, ref_img)
        if proposals is None:
            proposal_list = self.detector.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        return self.detector.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)
