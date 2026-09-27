"""EOVOD: efficient one-stage video object detection by exploiting temporal
consistency (Sun, Hua, Hu, Robertson; ECCV 2022, arXiv 2402.09241),
implemented from the paper.

The paper's analysis (its Section 3): the accurate VID methods aggregate
features with attention, affordable in two-stage detectors because it runs over
~300 proposals. A one-stage detector has no proposals; FCOS's pyramid holds
~13k pixels at 600 px, and attention over all of them takes 21.9 GB and runs
at 4.6 FPS (two reference frames, a V100). Separately, about 80% of a
one-stage detector's head time goes on the low levels -- 65% on stride 8 --
which exist for small objects. Objects move and resize gradually between
frames, so the previous frame's detections say where, and how large, this
frame's objects are.

Its two modules (Section 4), both read off the previous frame's *validated*
detections, those with classification score above 0.5:

* **Location prior network.** The validated boxes, resized by an adjustment
  ratio r (0.8), are projected to each pyramid level by dividing by its stride
  and turned into a binary foreground mask M. Where M is 1 the pixel is
  enhanced by attention over the key set and replaces the original (Eq. 2);
  elsewhere the feature map is unchanged, and the enhanced maps go to the
  heads. Without a validated box the aggregation is skipped. The attention is
  Eq. 1, ``A(q, K) = q + sum_j w_ij (W k_j)``, the SELSA form: multi-head
  here, as SELSA's implementation has it.
* **Size prior network.** After a full detection at time t, for the next T
  frames the heads run only on the levels the validated boxes came from; then
  a full detection follows. T = 7 means a full detection every eighth frame.
  A frame whose objects are all large skips the expensive low levels.

The keys (Section 4.1, "Training and Inference"): at inference, the pixels
within the detected boxes on the reference frames -- 14 frames chosen as
SELSA's implementation chooses them, which is what this repository's
``test_with_adaptive_stride`` sampler hands a video's first frame. They are
gathered once per video and kept. In training, two support frames are drawn
by FGFA's temporal dropout, and the ground-truth boxes generate the mask and
select both the query and the key pixels; the current frame's detection losses
train the whole network end to end.

Where the paper is silent, the defaults follow it and the alternatives are
options: the key set may be updated from every frame and capped, MAMBA-style
(``memory``); the size prior may keep every level above the lowest validated
one rather than only the levels boxes came from (``keep_higher_levels``); the
first frame of a video, which has no previous frame, may take its own plain
detections as its prior (``bootstrap_first_frame``); and the training prior
may be jittered to mimic the motion between frames (``train_jitter``).

Everything random draws from torch's generator on the feature device, so
exact gradient accumulation (``RngStreams``) covers it; nothing uses numpy.
The aggregator's ``nn.Linear`` layers keep torch's default initialisation, as
MAMBA's aggregator does.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from vfe.core import bbox2result
from vfe.models.builder import MODELS, build_detector
from vfe.models.vid.base import BaseVideoDetector

__all__ = ["EOVOD", "PixelAggregator", "PixelMemory", "boxes_to_level_masks", "scale_boxes",
           "random_subset"]


def scale_boxes(boxes: torch.Tensor, ratio: float) -> torch.Tensor:
    """Scale ``(n, 4)`` boxes about their centres: the paper's adjustment ratio r."""
    if ratio == 1.0 or boxes.numel() == 0:
        return boxes
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    half_w = (boxes[:, 2] - boxes[:, 0]) * ratio / 2
    half_h = (boxes[:, 3] - boxes[:, 1]) * ratio / 2
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=1)


def boxes_to_level_masks(
    boxes: torch.Tensor, featmap_sizes: Sequence[tuple[int, int]], strides: Sequence[int]
) -> list[torch.Tensor]:
    """The paper's foreground mask M: ``(n, 4)`` boxes in input-image pixels ->
    one ``(H, W)`` bool mask per level, true where the cell's centre lies
    inside a box (the box projected to the level by dividing by its stride).

    A box smaller than a cell still marks the cell holding its centre, so no
    box disappears on the coarse levels.
    """
    masks = []
    for (h, w), stride in zip(featmap_sizes, strides, strict=True):
        h, w = int(h), int(w)
        if boxes.numel() == 0:
            masks.append(torch.zeros(h, w, dtype=torch.bool, device=boxes.device))
            continue
        xs = (torch.arange(w, device=boxes.device, dtype=boxes.dtype) + 0.5) * stride
        ys = (torch.arange(h, device=boxes.device, dtype=boxes.dtype) + 0.5) * stride
        in_x = (xs[None, :] >= boxes[:, 0:1]) & (xs[None, :] <= boxes[:, 2:3])  # (n, W)
        in_y = (ys[None, :] >= boxes[:, 1:2]) & (ys[None, :] <= boxes[:, 3:4])  # (n, H)
        mask = (in_y[:, :, None] & in_x[:, None, :]).any(dim=0)
        cx = ((boxes[:, 0] + boxes[:, 2]) / (2 * stride)).long().clamp_(0, w - 1)
        cy = ((boxes[:, 1] + boxes[:, 3]) / (2 * stride)).long().clamp_(0, h - 1)
        mask[cy, cx] = True
        masks.append(mask)
    return masks


def random_subset(feats: torch.Tensor, n: int) -> torch.Tensor:
    """At most ``n`` rows of ``feats``, chosen uniformly (torch RNG, on device)."""
    if len(feats) <= n:
        return feats
    return feats[torch.randperm(len(feats), device=feats.device)[:n]]


class PixelAggregator(nn.Module):
    """The paper's Eq. 1 in SELSA's multi-head form: each of ``num_heads``
    heads scores the ``C / num_heads``-wide embeddings of every query against
    every key, softmaxes over the keys and sums their projected features with
    those weights; the result is projected and added to the query."""

    def __init__(self, channels: int, num_heads: int = 16):
        super().__init__()
        if channels % num_heads:
            raise ValueError(f"channels={channels} is not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.fc_embed = nn.Linear(channels, channels)
        self.ref_fc_embed = nn.Linear(channels, channels)
        self.fc = nn.Linear(channels, channels)
        self.ref_fc = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, ref_x: torch.Tensor) -> torch.Tensor:
        """``(N, C)`` queries, ``(M, C)`` keys -> ``(N, C)`` enhanced queries."""
        n, m = x.shape[0], ref_x.shape[0]
        q = self.fc_embed(x).view(n, self.num_heads, -1).permute(1, 0, 2)  # (B, N, d)
        k = self.ref_fc_embed(ref_x).view(m, self.num_heads, -1).permute(1, 2, 0)  # (B, d, M)
        weights = (torch.bmm(q, k) / (q.shape[-1] ** 0.5)).softmax(dim=2)  # (B, N, M)
        v = self.ref_fc(ref_x).view(m, self.num_heads, -1).permute(1, 0, 2)  # (B, M, d)
        out = torch.bmm(weights, v).permute(1, 0, 2).reshape(n, -1)
        return x + self.fc(out)


class PixelMemory:
    """Per-level key pixels for the attention: per-video inference state,
    never saved in a checkpoint.

    The paper's key set is fixed for a video: the pixels inside the detected
    boxes on its reference frames, gathered once at the first frame. That is
    the default (``update=False``, no caps). ``update=True`` writes every
    frame's validated pixels as well; ``capacity`` / ``num_keys`` /
    ``write_per_frame`` bound the bank MAMBA-style -- random replacement of
    old pixels once full, a random subset read per frame.
    """

    def __init__(self, num_levels: int, update: bool = False, capacity: int | None = None,
                 num_keys: int | None = None, write_per_frame: int | None = None):
        if capacity is not None and write_per_frame is not None and write_per_frame > capacity:
            raise ValueError("write_per_frame cannot exceed capacity")
        self.num_levels = num_levels
        self.update = update
        self.capacity = capacity
        self.num_keys = num_keys
        self.write_per_frame = write_per_frame
        self.banks: list[torch.Tensor | None] = [None] * num_levels

    def reset(self) -> None:
        self.banks = [None] * self.num_levels

    def sizes(self) -> list[int]:
        return [0 if bank is None else len(bank) for bank in self.banks]

    def write(self, level: int, feats: torch.Tensor) -> None:
        if len(feats) == 0:
            return
        feats = feats.detach()
        if self.write_per_frame is not None:
            feats = random_subset(feats, self.write_per_frame)
        if self.capacity is not None:
            feats = random_subset(feats, self.capacity)
        bank = self.banks[level]
        if bank is None:
            self.banks[level] = feats
            return
        if self.capacity is not None:
            room = self.capacity - len(feats)
            if len(bank) > room:
                bank = random_subset(bank, room)
        self.banks[level] = torch.cat([bank, feats], dim=0)

    def sample(self, level: int) -> torch.Tensor | None:
        bank = self.banks[level]
        if bank is None or len(bank) == 0:
            return None
        return bank if self.num_keys is None else random_subset(bank, self.num_keys)


def _with_defaults(name: str, given: dict | None, defaults: dict) -> dict:
    given = dict(given or {})
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(f"{name}: unknown keys {sorted(unknown)}; allowed {sorted(defaults)}")
    return {**defaults, **given}


@MODELS.register_module()
class EOVOD(BaseVideoDetector):
    """Args:
        detector: config of the wrapped single-stage detector (FCOS).
        location_prior: ``score_thr`` (a detection above it is *validated*;
            the paper's 0.5), ``box_ratio`` (the paper's adjustment ratio r,
            0.8), ``train_jitter`` (random shift and rescale of the
            ground-truth boxes standing in for the previous frame's
            detections, as a fraction of box size; the paper uses none) and
            ``bootstrap_first_frame`` (give a video's first frame its own
            plain detections as prior; the paper skips aggregation there).
        size_prior: ``interval`` -- the paper's T: after a full detection,
            this many frames run only the levels the validated boxes came
            from; 0 is a full detection on every frame. ``keep_higher_levels``
            also runs every level above the lowest validated one. ``None``
            disables the size prior.
        memory: see :class:`PixelMemory`; the default is the paper's fixed
            key set.
        aggregator: ``num_heads``; ``shared`` uses one aggregator for every
            level instead of one per level; ``query_chunk`` bounds the
            ``heads x queries x keys`` attention tensor by processing that
            many queries at a time (no effect on the result).
        ref_chunk_size: reference frames detected at once when gathering the
            key set at a video's first frame (bounds peak memory).
        frozen_modules / train_cfg / test_cfg: as for MAMBA.
    """

    def __init__(
        self,
        detector: dict,
        location_prior: dict | None = None,
        size_prior: dict | None = None,
        memory: dict | None = None,
        aggregator: dict | None = None,
        ref_chunk_size: int = 4,
        frozen_modules=None,
        train_cfg: Any = None,
        test_cfg: Any = None,
    ):
        super().__init__()
        self.detector = build_detector(detector)
        head = getattr(self.detector, "bbox_head", None)
        if head is None or not hasattr(head, "strides") or not hasattr(head, "simple_test"):
            raise TypeError("EOVOD wraps a single-stage detector with a multi-level dense head")
        self.strides = [s if isinstance(s, int) else int(s[0]) for s in head.strides]

        location_prior = _with_defaults(
            "location_prior", location_prior,
            dict(score_thr=0.5, box_ratio=0.8, train_jitter=0.0, bootstrap_first_frame=False),
        )
        self.score_thr = float(location_prior["score_thr"])
        self.box_ratio = float(location_prior["box_ratio"])
        self.train_jitter = float(location_prior["train_jitter"])
        self.bootstrap_first_frame = bool(location_prior["bootstrap_first_frame"])

        if size_prior is None:
            self.size_prior_interval = None
            self.keep_higher_levels = False
        else:
            size_prior = _with_defaults(
                "size_prior", size_prior, dict(interval=7, keep_higher_levels=False)
            )
            self.size_prior_interval = int(size_prior["interval"])
            if self.size_prior_interval < 0:
                raise ValueError("size_prior.interval must be at least 0")
            self.keep_higher_levels = bool(size_prior["keep_higher_levels"])

        memory = _with_defaults(
            "memory", memory,
            dict(update=False, capacity=None, num_keys=None, write_per_frame=None),
        )
        self.memory = PixelMemory(len(self.strides), **memory)

        aggregator = _with_defaults(
            "aggregator", aggregator, dict(num_heads=16, shared=True, query_chunk=1024)
        )
        num_aggregators = 1 if aggregator["shared"] else len(self.strides)
        self.aggregators = nn.ModuleList(
            PixelAggregator(head.in_channels, aggregator["num_heads"])
            for _ in range(num_aggregators)
        )
        self.query_chunk = aggregator["query_chunk"]
        self.ref_chunk_size = ref_chunk_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._reset_video_state()
        if frozen_modules is not None:
            self.freeze_module(frozen_modules)

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    def _aggregator(self, level: int) -> PixelAggregator:
        return self.aggregators[0 if len(self.aggregators) == 1 else level]

    def _reset_video_state(self) -> None:
        self.memory.reset()
        # Validated boxes of the previous frame, in network-input coordinates.
        self._prev_boxes: torch.Tensor | None = None
        # Size prior: levels to run until the next full frame, and how many
        # frames remain before it.
        self._active_levels: list[int] | None = None
        self._frames_until_full = 0

    # ---- shared machinery ----------------------------------------------------

    def _pixels_in_boxes(self, feat: torch.Tensor, boxes: torch.Tensor, level: int) -> torch.Tensor:
        """``feat`` ``(C, H, W)`` of one level -> ``(N, C)`` pixels inside ``boxes``."""
        mask = boxes_to_level_masks(boxes, [feat.shape[-2:]], [self.strides[level]])[0]
        return feat[:, mask].t()

    def _enhance(self, feats, masks, keys) -> list[torch.Tensor]:
        """Eq. 2: attend the masked cells of each ``(1, C, H, W)`` level over
        that level's keys and write them back; levels without a mask or keys
        pass through unchanged."""
        out = []
        for level, x in enumerate(feats):
            mask = None if masks is None else masks[level]
            key = keys[level]
            if mask is None or key is None or len(key) == 0 or not bool(mask.any()):
                out.append(x)
                continue
            queries = x[0][:, mask].t()  # (N, C)
            aggregator = self._aggregator(level)
            chunks = queries.split(self.query_chunk) if self.query_chunk else (queries,)
            enhanced = torch.cat([aggregator(q, key) for q in chunks], dim=0)
            x = x.clone()
            x[0][:, mask] = enhanced.t()
            out.append(x)
        return out

    # ---- training --------------------------------------------------------------

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, ref_img, ref_img_metas,
                      ref_gt_bboxes=None, ref_gt_labels=None, gt_bboxes_ignore=None,
                      **kwargs) -> dict:
        """Losses for one key frame (``img``, batch size 1) and its support
        frames (``ref_img``, ``(1, R, C, H, W)``). ``ref_gt_bboxes[0]`` is
        ``(n, 5)`` as ``[reference index, x1, y1, x2, y2]``."""
        if len(img) != 1:
            raise ValueError("EOVOD trains on one key frame per GPU")
        refs = ref_img[0]
        feats = self.detector.extract_feat(torch.cat((img, refs), dim=0))
        key_feats = [f[:1] for f in feats]
        ref_feats = [f[1:] for f in feats]

        if ref_gt_bboxes is not None:
            ref_boxes = ref_gt_bboxes[0]
        else:
            ref_boxes = gt_bboxes[0].new_zeros((0, 5))
        keys = [self._training_keys(ref_feats[lvl], ref_boxes, lvl) for lvl in range(self.num_levels)]
        prior = self._training_prior(gt_bboxes[0])
        masks = boxes_to_level_masks(prior, [f.shape[-2:] for f in key_feats], self.strides)
        enhanced = self._enhance(key_feats, masks, keys)
        return self.detector.bbox_head.forward_train(
            enhanced, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore
        )

    def _training_keys(self, ref_feat: torch.Tensor, ref_boxes: torch.Tensor, level: int):
        """Pixels inside the support frames' ground-truth boxes at one level,
        ``(M, C)``; random pixels when no support frame has a box, so the
        aggregator runs (and DDP finds its parameters used) on every step."""
        pixels = []
        for r in range(ref_feat.shape[0]):
            boxes = ref_boxes[ref_boxes[:, 0] == r, 1:]
            if boxes.numel():
                pixels.append(self._pixels_in_boxes(ref_feat[r], boxes, level))
        if pixels:
            keys = torch.cat(pixels, dim=0)
        else:
            keys = ref_feat.permute(0, 2, 3, 1).reshape(-1, ref_feat.shape[1])
        if self.memory.num_keys is not None:
            keys = random_subset(keys, self.memory.num_keys)
        return keys

    def _training_prior(self, gt_bboxes: torch.Tensor) -> torch.Tensor:
        """The key frame's ground-truth boxes as the paper's training mask,
        optionally jittered like a previous frame's detections would be, then
        resized by ``box_ratio``."""
        boxes = gt_bboxes
        if self.train_jitter > 0 and boxes.numel():
            n = boxes.shape[0]
            wh = torch.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], dim=1)
            ctr = torch.stack([boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]], dim=1) / 2
            shift = (torch.rand(n, 2, device=boxes.device) * 2 - 1) * self.train_jitter * wh
            scale = 1 + (torch.rand(n, 1, device=boxes.device) * 2 - 1) * self.train_jitter
            ctr = ctr + shift
            wh = wh * scale
            boxes = torch.cat([ctr - wh / 2, ctr + wh / 2], dim=1)
        return scale_boxes(boxes, self.box_ratio)

    # ---- inference ---------------------------------------------------------------

    def simple_test(self, img, img_metas, ref_img=None, ref_img_metas=None, rescale=False,
                    **kwargs):
        """Detections for one frame, as ``bbox2result`` lists. Frames of a
        video must arrive in order; ``frame_id == 0`` resets the state and
        comes with the reference frames (``ref_img`` ``[Tensor(1, R, C, H, W)]``,
        ``ref_img_metas`` ``[[[dict, ...]]]``) whose detections supply the keys."""
        if len(img) != 1:
            raise ValueError("EOVOD tests one frame at a time")
        meta = img_metas[0]
        frame_id = meta.get("frame_id", -1)
        if frame_id < 0:
            raise KeyError("img_metas must carry 'frame_id' at test time")
        if frame_id == 0:
            self._reset_video_state()
            if ref_img is not None:
                # The test pipeline's nesting: [Tensor(1, R, C, H, W)] and [[[dict, ...]]].
                self._gather_reference_keys(ref_img[0][0], ref_img_metas[0][0])

        feats = self.detector.extract_feat(img)
        levels, full = self._levels_to_run()
        masks = None
        if self._prev_boxes is not None and len(self._prev_boxes):
            masks = boxes_to_level_masks(
                scale_boxes(self._prev_boxes, self.box_ratio),
                [f.shape[-2:] for f in feats], self.strides,
            )
        keys = [self.memory.sample(lvl) for lvl in range(self.num_levels)]
        enhanced = self._enhance(feats, masks, keys)

        det_bboxes, det_labels, det_levels = self.detector.bbox_head.simple_test(
            [enhanced[lvl] for lvl in levels], img_metas, rescale=False, level_ids=levels,
            with_levels=True,
        )[0]
        self._after_frame(enhanced, det_bboxes, det_levels, full)

        if rescale:
            det_bboxes = det_bboxes.clone()
            det_bboxes[:, :4] /= torch.as_tensor(
                meta["scale_factor"], dtype=det_bboxes.dtype, device=det_bboxes.device
            )
        return [bbox2result(det_bboxes, det_labels, self.detector.bbox_head.num_classes)]

    def _levels_to_run(self) -> tuple[list[int], bool]:
        """The levels to run on this frame, and whether it is a full frame."""
        all_levels = list(range(self.num_levels))
        if (self.size_prior_interval is None or self._active_levels is None
                or self._frames_until_full <= 0):
            return all_levels, True
        self._frames_until_full -= 1
        return list(self._active_levels), False

    def _after_frame(self, feats, det_bboxes, det_levels, full: bool) -> None:
        """Keep the validated boxes as the next frame's location prior, set
        the size prior on a full frame, and (if updating) write the memory."""
        valid = det_bboxes[:, 4] > self.score_thr
        boxes = det_bboxes[valid, :4].detach()
        self._prev_boxes = boxes
        if self.size_prior_interval is not None and full:
            if bool(valid.any()):
                levels = det_levels[valid].unique(sorted=True).tolist()
                if self.keep_higher_levels:
                    levels = list(range(levels[0], self.num_levels))
                self._active_levels = levels
                self._frames_until_full = self.size_prior_interval
            else:
                # Nothing detected: no evidence about sizes, so keep every level.
                self._active_levels = None
        if self.memory.update:
            self._write_memory(feats, boxes)

    def _write_memory(self, feats, boxes: torch.Tensor) -> None:
        if boxes.numel() == 0:
            return
        for level, feat in enumerate(feats):
            self.memory.write(level, self._pixels_in_boxes(feat[0], boxes, level))

    def _gather_reference_keys(self, refs: torch.Tensor, ref_metas: list[dict]) -> None:
        """Plain detection on the reference frames ``(R, C, H, W)``: the pixels
        within their validated boxes are the video's key set. With
        ``bootstrap_first_frame``, the reference that is the first frame itself
        also lends its detections as that frame's location prior."""
        head = self.detector.bbox_head
        start = 0
        for chunk in refs.split(self.ref_chunk_size):
            metas = list(ref_metas[start:start + len(chunk)])
            start += len(chunk)
            feats = self.detector.extract_feat(chunk)
            results = head.simple_test(feats, metas, rescale=False, with_levels=True)
            for i, (det_bboxes, _, _) in enumerate(results):
                boxes = det_bboxes[det_bboxes[:, 4] > self.score_thr, :4]
                self._write_memory([f[i:i + 1] for f in feats], boxes)
                if (self.bootstrap_first_frame and metas[i].get("frame_id") == 0
                        and self._prev_boxes is None):
                    self._prev_boxes = boxes
