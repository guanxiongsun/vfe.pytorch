"""MAMBA video object detector. Port of ``mmdet.models.vid.mamba``.

Training is SELSA-style: a key frame and a few reference frames go through the
backbone together, the top-k proposals of each reference frame supply
reference RoI features, and the box head aggregates them into the key frame's
features.

Testing is where MAMBA differs, and it is stateful: frames must arrive in
video order, one per call. The first frame (``frame_id == 0``) comes with its
reference frames, which seed the memory banks in the box head. Every later
frame arrives alone and reads from and writes to those memories. How reference
features are gathered depends on the sampler's ``frame_stride`` meta:

* ``frame_stride < 1`` -- *adaptive stride*, what the released configs use.
  Only the first frame extracts references.
* ``frame_stride >= 1`` -- *fixed stride*. A sliding window of reference
  features is kept in ``self.memo`` and advanced every ``frame_stride``
  frames.
"""

from __future__ import annotations

import torch

from vfe.models.builder import MODELS, build_detector
from vfe.models.vid.base import BaseVideoDetector

__all__ = ["MAMBA"]


@MODELS.register_module()
class MAMBA(BaseVideoDetector):
    """Args:
        detector: config of the wrapped two-stage detector.
        frozen_modules: submodule name(s) to freeze at construction.
        train_cfg / test_cfg: unused by MAMBA itself (the detector carries its
            own), accepted because the configs pass them.
    """

    def __init__(self, detector: dict, frozen_modules=None, train_cfg=None, test_cfg=None):
        super().__init__()
        self.detector = build_detector(detector)
        if not hasattr(self.detector, "roi_head"):
            raise TypeError("MAMBA only supports two-stage detectors")
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # Fixed-stride test state; see extract_feats.
        self.memo: dict | None = None
        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, ref_img, ref_img_metas,
                      ref_gt_bboxes=None, ref_gt_labels=None, gt_instance_ids=None,
                      gt_bboxes_ignore=None, gt_masks=None, proposals=None,
                      ref_gt_instance_ids=None, ref_gt_bboxes_ignore=None, ref_gt_masks=None,
                      ref_proposals=None, **kwargs) -> dict:
        """Losses for one key frame (``img``, batch size 1) and its reference
        frames (``ref_img``, shape ``(1, R, C, H, W)``). The ``ref_gt_*`` and
        ``*_instance_ids`` arguments come from the data pipeline and are unused."""
        if len(img) != 1:
            raise ValueError("MAMBA supports one key frame per GPU")

        all_x = self.detector.extract_feat(torch.cat((img, ref_img[0]), dim=0))
        x = [level[[0]] for level in all_x]
        ref_x = [level[1:] for level in all_x]

        losses = {}
        detector = self.detector
        if detector.with_rpn:
            proposal_cfg = detector.train_cfg.get("rpn_proposal", detector.test_cfg["rpn"])
            rpn_losses, proposal_list = detector.rpn_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
            )
            losses.update(rpn_losses)
            # Reference proposals use the *test* RPN config, then keep top-k.
            ref_proposals_list = self._topk(detector.rpn_head.simple_test_rpn(ref_x, ref_img_metas[0]))
        else:
            proposal_list = proposals
            ref_proposals_list = ref_proposals

        losses.update(
            detector.roi_head.forward_train(
                x, ref_x, img_metas, proposal_list, ref_proposals_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore, **kwargs,
            )
        )
        return losses

    def _topk(self, proposals_list):
        topk = self.detector.roi_head.bbox_head.topk
        return [proposals[:topk] for proposals in proposals_list]

    def extract_feats(self, img, img_metas, ref_img, ref_img_metas):
        """Features for the current test frame, and the reference features it
        should be aggregated with (None after the first adaptive-stride frame).

        Reads the sampler's ``frame_id``, ``num_left_ref_imgs`` and
        ``frame_stride`` from ``img_metas[0]``.
        """
        frame_id = img_metas[0].get("frame_id", -1)
        if frame_id < 0:
            raise KeyError("img_metas must carry 'frame_id' at test time")
        num_left_ref_imgs = img_metas[0].get("num_left_ref_imgs", -1)
        frame_stride = img_metas[0].get("frame_stride", -1)

        if frame_stride < 1:
            x = self.detector.extract_feat(img)
            if frame_id != 0:
                return x, img_metas, None, None
            # The key frame joins its own references, after them. (The
            # original also stashed these in self.memo, which this mode never
            # reads again; that is omitted.)
            ref_feats = self.detector.extract_feat(ref_img[0])
            ref_x = [torch.cat((ref_feats[i], x[i]), dim=0) for i in range(len(x))]
            ref_img_metas = list(ref_img_metas[0]) + list(img_metas)
            return x, img_metas, ref_x, ref_img_metas

        if frame_id == 0:
            ref_feats = self.detector.extract_feat(ref_img[0])
            self.memo = {"img_metas": list(ref_img_metas[0]), "feats": list(ref_feats)}
            # The key frame is one of its own references; reuse its features.
            x = [feats[[num_left_ref_imgs]] for feats in self.memo["feats"]]
        elif frame_id % frame_stride == 0:
            if ref_img is None:
                raise ValueError(f"frame {frame_id} is on the stride and needs its reference frame")
            ref_feats = self.detector.extract_feat(ref_img[0])
            x = []
            for i in range(len(ref_feats)):
                # Slide the window: append the new reference, drop the oldest.
                self.memo["feats"][i] = torch.cat((self.memo["feats"][i], ref_feats[i]), dim=0)[1:]
                x.append(self.memo["feats"][i][[num_left_ref_imgs]])
            self.memo["img_metas"] = (self.memo["img_metas"] + list(ref_img_metas[0]))[1:]
        else:
            if ref_img is not None:
                raise ValueError(f"frame {frame_id} is off the stride but got a reference frame")
            x = self.detector.extract_feat(img)

        ref_x = list(self.memo["feats"])
        for i in range(len(x)):
            # In place, as in the original: this writes the current frame into
            # the window's centre slot *persistently*, not just for this call.
            ref_x[i][num_left_ref_imgs] = x[i]
        ref_img_metas = list(self.memo["img_metas"])
        ref_img_metas[num_left_ref_imgs] = img_metas[0]
        return x, img_metas, ref_x, ref_img_metas

    def simple_test(self, img, img_metas, ref_img=None, ref_img_metas=None, proposals=None,
                    ref_proposals=None, rescale=False):
        """Detections for one frame, as ``bbox2result`` lists.

        ``ref_img`` is ``[Tensor(1, R, C, H, W)]`` and ``ref_img_metas``
        ``[[[dict, ...]]]`` -- the test pipeline's nesting.
        """
        if ref_img is not None:
            ref_img = ref_img[0]
        if ref_img_metas is not None:
            ref_img_metas = ref_img_metas[0]
        x, img_metas, ref_x, ref_img_metas = self.extract_feats(
            img, img_metas, ref_img, ref_img_metas
        )

        if proposals is None:
            proposal_list = self.detector.rpn_head.simple_test_rpn(x, img_metas)
            ref_proposals_list = None
            if ref_x is not None:
                ref_proposals_list = self._topk(
                    self.detector.rpn_head.simple_test_rpn(ref_x, ref_img_metas)
                )
        else:
            proposal_list = proposals
            ref_proposals_list = ref_proposals

        return self.detector.roi_head.simple_test(
            x, ref_x, proposal_list, ref_proposals_list, img_metas, rescale=rescale
        )
