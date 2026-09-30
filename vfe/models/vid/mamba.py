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

That is the *instance level*, and all the released model has. The paper's
full model (Table 3, "Ours": 84.6 against the instance-only 83.7) also
enhances the feature map before the RPN: the *pixel level*
(:class:`MambaPixelLevel`, built from the ``pixel`` option). Every pixel of
the frame's map attends over key pixels drawn from a second memory bank,
which is filled with pixels inside the detected boxes of the reference frames
and then of every frame. Pixel-level inference supports the adaptive-stride
protocol only.

The RoI head picks the instance level: ``MambaRoIHead`` aggregates, a plain
``StandardRoIHead`` does not. With ``pixel`` and a plain head the model is the
paper's pixel-only row ("Ours_pix"); with neither, it is the single-frame
Faster R-CNN baseline trained on the same key frames.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from vfe.core import bbox2result
from vfe.models.aggregators import MambaAggregator
from vfe.models.builder import MODELS, build_detector
from vfe.models.memory import MemoryBank
from vfe.models.roi_heads.mamba import MambaRoIHead
from vfe.models.vid.base import BaseVideoDetector

__all__ = ["MAMBA", "MambaPixelLevel"]


def _random_rows(x: torch.Tensor, n: int) -> torch.Tensor:
    """At most ``n`` rows of ``x``, chosen uniformly. Draws from the global
    CPU RNG, as the memory banks do, so a seeded evaluation repeats."""
    if len(x) <= n:
        return x
    return x[torch.randperm(len(x))[:n].to(x.device)]


class MambaPixelLevel(nn.Module):
    """MAMBA's pixel-level enhancement: every pixel of a one-level feature map
    ``(1, C, H, W)`` attends over key pixels, and the result is added to it --
    the instance level's operation (``x + aggregator(x, keys)``, SELSA's
    attention) with pixels for RoIs.

    The keys come from a memory bank (random reads and replacement, as the
    instance level's). At test time it is reset at a video's first frame,
    filled with the pixels inside the reference frames' detections, and after
    every frame takes the pixels inside that frame's. For training, the keys
    are drawn from the reference frames directly.

    The paper fixes ``pixels_per_box`` (K = 100) and the key-set size (the
    memory's ``key_length``, 2,000). The rest follows the released EOVOD
    code's ``MPN``, the same design on FCOS: detections validated at
    ``score_thr``, at most ``pixels_per_frame`` pixels per frame, and a frame
    without a box contributes its ``fallback_pixels`` highest-norm pixels.

    Args:
        in_channels: channels of the enhanced map.
        num_attention_blocks: attention heads.
        position: ``'backbone'`` enhances the backbone's map before the neck
            (``MPN``'s ``before_fpn``); ``'neck'`` enhances the neck's output.
            Either way the RPN and the RoI head see the enhanced map.
        stride: the map's stride in input pixels.
        train_keys: ``'random'`` takes ``random_keys`` random pixels of each
            reference frame (the released ``MPN`` configs); ``'gt'`` takes
            ``pixels_per_box`` random pixels inside each reference frame's
            ground-truth boxes (the test-time rule with ground truth for
            detections). ``'gt'`` leaks: on DET images the references are the
            key image itself, and at one epoch it cost 2.1 AP50 alone and 4.1
            in the full model (docs/mamba-pixel-plan.md). Training keys are
            capped at the memory's ``key_length``, as test-time reads are.
        memory_cfg: keyword arguments for :class:`~vfe.models.memory.MemoryBank`.
    """

    def __init__(self, in_channels: int, num_attention_blocks: int = 16,
                 position: str = "backbone", stride: int = 16, train_keys: str = "random",
                 random_keys: int = 2000, score_thr: float = 0.3, pixels_per_box: int = 100,
                 pixels_per_frame: int = 1000, fallback_pixels: int = 50,
                 memory_cfg: dict | None = None):
        super().__init__()
        if position not in ("backbone", "neck"):
            raise ValueError(f"position must be 'backbone' or 'neck', got {position!r}")
        if train_keys not in ("gt", "random"):
            raise ValueError(f"train_keys must be 'gt' or 'random', got {train_keys!r}")
        self.aggregator = MambaAggregator(in_channels, num_attention_blocks, memory_cfg)
        self.position = position
        self.stride = stride
        self.train_keys = train_keys
        self.random_keys = random_keys
        self.score_thr = score_thr
        self.pixels_per_box = pixels_per_box
        self.pixels_per_frame = pixels_per_frame
        self.fallback_pixels = fallback_pixels

    @property
    def memory(self) -> MemoryBank:
        return self.aggregator.memory_bank

    def forward(self, feat: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """``(1, C, H, W)`` map and ``(M, C)`` keys -> the enhanced map."""
        _, c, h, w = feat.shape
        query = feat.flatten(2)[0].t()  # (HW, C)
        query = query + self.aggregator.forward_with_ref_x(query, keys)
        return query.t().reshape(1, c, h, w)

    def box_cells(self, boxes: torch.Tensor, h: int, w: int) -> list[torch.Tensor]:
        """Per ``(n, 4)`` box in input pixels, the flat indices (on the CPU)
        of the ``(H, W)`` cells whose centres lie inside it; the cell holding
        its centre when none does, so no box is lost."""
        cells = []
        for x1, y1, x2, y2 in (boxes / self.stride).tolist():
            c0, c1 = max(math.ceil(x1 - 0.5), 0), min(math.floor(x2 - 0.5), w - 1)
            r0, r1 = max(math.ceil(y1 - 0.5), 0), min(math.floor(y2 - 0.5), h - 1)
            if c0 > c1 or r0 > r1:
                c0 = c1 = min(max(int((x1 + x2) / 2), 0), w - 1)
                r0 = r1 = min(max(int((y1 + y2) / 2), 0), h - 1)
            rows = torch.arange(r0, r1 + 1)
            cols = torch.arange(c0, c1 + 1)
            cells.append((rows[:, None] * w + cols[None, :]).flatten())
        return cells

    def pixels(self, feat: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """``(C, H, W)`` map and ``(n, 4)`` boxes, most important first ->
        ``(m, C)`` pixels: up to ``pixels_per_box`` random cells of each box,
        at most ``pixels_per_frame`` in all; the ``fallback_pixels``
        highest-norm cells when there is no box."""
        c, h, w = feat.shape
        flat = feat.reshape(c, h * w)
        chosen = []
        for cells in self.box_cells(boxes, h, w):
            if len(cells) > self.pixels_per_box:
                cells = cells[torch.randperm(len(cells))[: self.pixels_per_box]]
            chosen.append(cells)
        if chosen:
            index = torch.cat(chosen)[: self.pixels_per_frame].to(feat.device)
        else:
            index = flat.norm(dim=0).topk(min(self.fallback_pixels, h * w)).indices
        return flat[:, index].t()

    def training_keys(self, ref_feats: torch.Tensor, ref_boxes: torch.Tensor) -> torch.Tensor:
        """Keys from the reference maps ``(R, C, H, W)``; ``ref_boxes``
        ``(n, 5)`` are their ground truth, reference index first."""
        if self.train_keys == "random":
            keys = [_random_rows(f.flatten(1).t(), self.random_keys) for f in ref_feats]
        else:
            keys = [self.pixels(f, ref_boxes[ref_boxes[:, 0] == r, 1:])
                    for r, f in enumerate(ref_feats)]
        return _random_rows(torch.cat(keys), self.memory.key_length)

    def write(self, feat: torch.Tensor, det_bboxes: torch.Tensor) -> None:
        """Write the pixels of the ``(C, H, W)`` map inside the ``(n, 5)``
        detections (input pixels, score last) that pass ``score_thr``,
        highest-scoring first."""
        valid = det_bboxes[det_bboxes[:, 4] > self.score_thr]
        boxes = valid[valid[:, 4].argsort(descending=True), :4]
        self.memory.update(self.pixels(feat, boxes).detach())

    def sample(self) -> torch.Tensor:
        return self.memory.sample()

    def reset(self) -> None:
        self.memory.reset()


@MODELS.register_module()
class MAMBA(BaseVideoDetector):
    """Args:
        detector: config of the wrapped two-stage detector. A ``MambaRoIHead``
            adds the instance level; a plain ``StandardRoIHead`` leaves it out.
        pixel: config of the pixel level (:class:`MambaPixelLevel`'s
            arguments), or None -- the released model -- to leave it out.
        frozen_modules: submodule name(s) to freeze at construction.
        train_cfg / test_cfg: unused by MAMBA itself (the detector carries its
            own), accepted because the configs pass them.
    """

    def __init__(self, detector: dict, pixel: dict | None = None, frozen_modules=None,
                 train_cfg=None, test_cfg=None):
        super().__init__()
        self.detector = build_detector(detector)
        if not hasattr(self.detector, "roi_head"):
            raise TypeError("MAMBA only supports two-stage detectors")
        self.instance_level = isinstance(self.detector.roi_head, MambaRoIHead)
        self.pixel = None if pixel is None else MambaPixelLevel(**pixel)
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

        if self.pixel is not None:
            x, ref_x = self._pixel_train_features(img, ref_img[0], ref_gt_bboxes)
        elif self.instance_level:
            all_x = self.detector.extract_feat(torch.cat((img, ref_img[0]), dim=0))
            x = [level[[0]] for level in all_x]
            ref_x = [level[1:] for level in all_x]
        else:
            x, ref_x = self.detector.extract_feat(img), None

        losses = {}
        detector = self.detector
        if detector.with_rpn:
            proposal_cfg = detector.train_cfg.get("rpn_proposal", detector.test_cfg["rpn"])
            rpn_losses, proposal_list = detector.rpn_head.forward_train(
                x, img_metas, gt_bboxes, gt_labels=None, gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
            )
            losses.update(rpn_losses)
            if self.instance_level:
                # Reference proposals use the *test* RPN config, then keep top-k.
                ref_proposals_list = self._topk(
                    detector.rpn_head.simple_test_rpn(ref_x, ref_img_metas[0]))
        else:
            proposal_list = proposals
            ref_proposals_list = ref_proposals

        if not self.instance_level:
            losses.update(detector.roi_head.forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ignore, **kwargs))
            return losses
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

    # ---- the pixel level ---------------------------------------------------------

    def _stage_one(self, img: torch.Tensor) -> torch.Tensor:
        """The map the pixel level enhances, ``(B, C, H, W)``: the backbone's
        or the neck's, which must have one level (MAMBA's DC5)."""
        if self.pixel.position == "backbone":
            feats = self.detector.backbone(img)
        else:
            feats = self.detector.extract_feat(img)
        if len(feats) != 1:
            raise ValueError(f"the pixel level needs a one-level feature map, got {len(feats)}")
        return feats[0]

    def _stage_two(self, feat: torch.Tensor):
        """The feature levels the RPN and the RoI head read, from a stage-one map."""
        if self.pixel.position == "backbone" and self.detector.with_neck:
            return self.detector.neck((feat,))
        return (feat,)

    def _pixel_train_features(self, img, refs, ref_gt_bboxes):
        """The key frame's levels with its map enhanced over keys drawn from
        the reference frames ``(R, C, H, W)``, and the references' levels
        (plain, for the instance level; None without it)."""
        feat = self._stage_one(torch.cat((img, refs), dim=0))
        key, ref = feat[:1], feat[1:]
        if ref_gt_bboxes is not None:
            ref_boxes = ref_gt_bboxes[0]
        else:
            ref_boxes = feat.new_zeros((0, 5))
        key = self.pixel(key, self.pixel.training_keys(ref, ref_boxes))
        return self._stage_two(key), self._stage_two(ref) if self.instance_level else None

    def _detect(self, x, img_metas, rescale: bool, first_frame: bool = False):
        """``(det_bboxes, det_labels)`` per image. With the instance level,
        ``first_frame`` aggregates the images with their own top-k RoIs and
        restarts the box head's memory with those; otherwise the box head
        reads and writes its memory."""
        proposal_list = self.detector.rpn_head.simple_test_rpn(x, img_metas)
        roi_head = self.detector.roi_head
        if not self.instance_level:
            return roi_head.simple_test_bboxes(x, img_metas, proposal_list, roi_head.test_cfg,
                                               rescale=rescale)
        ref_x, ref_proposals = (x, self._topk(proposal_list)) if first_frame else (None, None)
        return roi_head.simple_test_bboxes(x, ref_x, proposal_list, ref_proposals, img_metas,
                                           roi_head.test_cfg, rescale=rescale)

    def _start_video(self, key: torch.Tensor, img_metas, refs: torch.Tensor, ref_metas) -> None:
        """Fill the pixel memory from a video's first frame and its reference
        frames ``(R, C, H, W)``: detect on all of them (the key frame joins its
        references, after them, as in the instance level), then write the
        pixels inside their detections."""
        self.pixel.reset()
        maps = torch.cat((self._stage_one(refs), key), dim=0)
        metas = list(ref_metas) + list(img_metas)
        det_bboxes, _ = self._detect(self._stage_two(maps), metas, rescale=False,
                                     first_frame=True)
        for feat, boxes in zip(maps, det_bboxes, strict=True):
            self.pixel.write(feat, boxes)

    def _pixel_simple_test(self, img, img_metas, ref_img, ref_img_metas, rescale: bool):
        if len(img) != 1:
            raise ValueError("MAMBA tests one frame at a time")
        meta = img_metas[0]
        frame_id = meta.get("frame_id", -1)
        if frame_id < 0:
            raise KeyError("img_metas must carry 'frame_id' at test time")
        if meta.get("frame_stride", -1) >= 1:
            raise NotImplementedError("the pixel level supports the adaptive-stride protocol only")

        key = self._stage_one(img)
        if frame_id == 0:
            if ref_img is None:
                raise ValueError("the first frame of a video needs its reference frames")
            # The test pipeline's nesting: [Tensor(1, R, C, H, W)] and [[[dict, ...]]].
            self._start_video(key, img_metas, ref_img[0][0], ref_img_metas[0][0])
        enhanced = self.pixel(key, self.pixel.sample())
        det_bboxes, det_labels = self._detect(self._stage_two(enhanced), img_metas, rescale)

        boxes = det_bboxes[0]
        if rescale:
            boxes = boxes.clone()
            boxes[:, :4] *= boxes.new_tensor(meta["scale_factor"])
        self.pixel.write(enhanced[0], boxes)
        return [bbox2result(det_bboxes[0], det_labels[0],
                            self.detector.roi_head.bbox_head.num_classes)]

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
        if self.pixel is not None:
            return self._pixel_simple_test(img, img_metas, ref_img, ref_img_metas, rescale)
        if not self.instance_level:
            return self.detector.simple_test(img, img_metas, proposals=proposals, rescale=rescale)
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
