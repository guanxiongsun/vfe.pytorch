"""EOVOD: efficient one-stage video object detection by exploiting temporal
consistency (Sun, Hua, Hu, Robertson; ECCV 2022, arXiv 2402.09241). Built from
the paper's method, not ported from its released code.

The paper's analysis of one-stage detectors on video:

* Attention-based feature aggregation is what makes two-stage VID methods
  (SELSA, MEGA, MAMBA) accurate, and it is affordable there because it runs
  over ~300 proposals. A one-stage detector has no proposals; its pyramid holds
  ~13k pixels at 600 px, and attention over all of them is not affordable.
* Objects move and resize gradually between frames -- "temporal consistency"
  -- so the previous frame's detections say where, and how large, this frame's
  objects are.
* About 80% of a one-stage detector's head time goes on the lowest pyramid
  levels, which exist for small objects.

Hence two priors, both read off the *previous frame's validated detections*
(score above ``score_thr``):

* **Location prior.** The validated boxes, shrunk about their centres by
  ``box_ratio``, are projected onto each level's grid and mark the foreground
  cells. Only those cells are enhanced: each attends over pixel features from
  other frames (SELSA-style multi-head attention) and adds the result to
  itself. Without a validated box the frame is detected plainly.
* **Size prior.** FCOS assigns objects to levels by size, so the level a
  validated box came from says how small this frame's objects are. Every
  ``interval`` frames the head runs on all levels; in between, only from the
  lowest level that produced a validated box upward. A frame with only large
  objects skips the expensive low levels.

The features the foreground cells attend to live in a **memory**: per level,
a random-replacement bank of pixel features written from inside every frame's
validated boxes (the enhanced features, as MAMBA's memory holds enhanced RoI
features). At a video's first frame the bank is seeded from the reference
frames the sampler supplies -- one of which is the first frame itself, whose
plain detections then serve as its own location prior.

Training mirrors inference with ground truth in place of detections: keys are
the pixels inside the reference frames' boxes, queries the pixels inside the
key frame's boxes after a random jitter that stands in for the motion between
frames. Every level runs at training time.

Everything random here draws from torch's generator on the feature device, so
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
    """Scale ``(n, 4)`` boxes about their centres."""
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
    """``(n, 4)`` boxes in input-image pixels -> one ``(H, W)`` bool mask per
    level, true where the cell's centre lies inside a box.

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
    """SELSA-style attention of query pixels over key pixels, added residually.

    Each of ``num_heads`` heads scores the ``C / num_heads``-wide embeddings of
    every query against every key, softmaxes over the keys and sums their
    (separately projected) features with those weights.
    """

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
    """Per-level banks of pixel features: per-video inference state, never
    saved in a checkpoint.

    ``write`` keeps at most ``write_per_frame`` of the new pixels and, once a
    bank is full, replaces a random subset of the old ones, so the bank stays
    a sample of the whole video so far. ``sample`` returns at most ``num_keys``
    random rows.
    """

    def __init__(self, num_levels: int, capacity: int = 4096, num_keys: int = 1024,
                 write_per_frame: int = 512):
        if write_per_frame > capacity:
            raise ValueError("write_per_frame cannot exceed capacity")
        self.num_levels = num_levels
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
        feats = random_subset(feats.detach(), self.write_per_frame)
        bank = self.banks[level]
        if bank is None:
            self.banks[level] = feats
            return
        room = self.capacity - len(feats)
        if len(bank) > room:
            bank = random_subset(bank, room)
        self.banks[level] = torch.cat([bank, feats], dim=0)

    def sample(self, level: int) -> torch.Tensor | None:
        bank = self.banks[level]
        if bank is None or len(bank) == 0:
            return None
        return random_subset(bank, self.num_keys)


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
        location_prior: ``score_thr`` (a detection above it is *validated*,
            and feeds both priors and the memory), ``box_ratio`` (shrink
            factor of the prior boxes; the paper found 0.8 best for FCOS) and
            ``train_jitter`` (random shift and rescale of the ground-truth
            boxes standing in for the previous frame's detections, as a
            fraction of box size).
        size_prior: ``interval`` -- the head runs on every level once per
            this many frames; ``None`` disables the size prior.
        memory: ``capacity``, ``num_keys``, ``write_per_frame`` per level.
        aggregator: ``num_heads``; ``shared`` uses one aggregator for every
            level instead of one per level.
        ref_chunk_size: reference frames detected at once when seeding the
            memory at a video's first frame (bounds peak memory).
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
            dict(score_thr=0.5, box_ratio=0.8, train_jitter=0.1),
        )
        self.score_thr = float(location_prior["score_thr"])
        self.box_ratio = float(location_prior["box_ratio"])
        self.train_jitter = float(location_prior["train_jitter"])
        if size_prior is None:
            self.size_prior_interval = None
        else:
            size_prior = _with_defaults("size_prior", size_prior, dict(interval=7))
            self.size_prior_interval = int(size_prior["interval"])
            if self.size_prior_interval < 1:
                raise ValueError("size_prior.interval must be at least 1")
        memory = _with_defaults(
            "memory", memory, dict(capacity=4096, num_keys=1024, write_per_frame=512)
        )
        self.memory = PixelMemory(len(self.strides), **memory)
        aggregator = _with_defaults("aggregator", aggregator, dict(num_heads=16, shared=True))
        num_aggregators = 1 if aggregator["shared"] else len(self.strides)
        self.aggregators = nn.ModuleList(
            PixelAggregator(head.in_channels, aggregator["num_heads"])
            for _ in range(num_aggregators)
        )
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
        """Attend the masked cells of each ``(1, C, H, W)`` level over that
        level's keys; levels without a mask or keys pass through unchanged."""
        out = []
        for level, x in enumerate(feats):
            mask = None if masks is None else masks[level]
            key = keys[level]
            if mask is None or key is None or len(key) == 0 or not bool(mask.any()):
                out.append(x)
                continue
            queries = x[0][:, mask].t()  # (N, C)
            enhanced = self._aggregator(level)(queries, key)
            x = x.clone()
            x[0][:, mask] = enhanced.t()
            out.append(x)
        return out

    # ---- training --------------------------------------------------------------

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, ref_img, ref_img_metas,
                      ref_gt_bboxes=None, ref_gt_labels=None, gt_bboxes_ignore=None,
                      **kwargs) -> dict:
        """Losses for one key frame (``img``, batch size 1) and its reference
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
        """Pixels inside the reference frames' boxes at one level, ``(M, C)``;
        random pixels when no reference frame has a box, so the aggregator
        runs (and DDP finds its parameters used) on every step."""
        pixels = []
        for r in range(ref_feat.shape[0]):
            boxes = ref_boxes[ref_boxes[:, 0] == r, 1:]
            if boxes.numel():
                pixels.append(self._pixels_in_boxes(ref_feat[r], boxes, level))
        if pixels:
            return random_subset(torch.cat(pixels, dim=0), self.memory.num_keys)
        flat = ref_feat.permute(0, 2, 3, 1).reshape(-1, ref_feat.shape[1])
        return random_subset(flat, self.memory.num_keys)

    def _training_prior(self, gt_bboxes: torch.Tensor) -> torch.Tensor:
        """The key frame's boxes, jittered like a previous frame's detections
        would be, then shrunk by ``box_ratio``."""
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
        video must arrive in order; ``frame_id == 0`` resets the state and may
        come with reference frames (``ref_img`` ``[Tensor(1, R, C, H, W)]``,
        ``ref_img_metas`` ``[[[dict, ...]]]``) that seed the memory."""
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
                self._seed_from_references(ref_img[0][0], ref_img_metas[0][0])

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
        """Keep the validated boxes as the next frame's location prior, update
        the size prior on a full frame, and write the memory."""
        valid = det_bboxes[:, 4] > self.score_thr
        boxes = det_bboxes[valid, :4].detach()
        self._prev_boxes = boxes
        if self.size_prior_interval is not None and full:
            if bool(valid.any()):
                lowest = int(det_levels[valid].min())
                self._active_levels = list(range(lowest, self.num_levels))
                self._frames_until_full = self.size_prior_interval - 1
            else:
                # Nothing detected: no evidence about sizes, so keep every level.
                self._active_levels = None
        self._write_memory(feats, boxes)

    def _write_memory(self, feats, boxes: torch.Tensor) -> None:
        if boxes.numel() == 0:
            return
        for level, feat in enumerate(feats):
            self.memory.write(level, self._pixels_in_boxes(feat[0], boxes, level))

    def _seed_from_references(self, refs: torch.Tensor, ref_metas: list[dict]) -> None:
        """Plain detection on the reference frames ``(R, C, H, W)``: their
        validated pixels fill the memory, and the frame that is the current
        one (``frame_id == 0``) lends its detections as the location prior."""
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
                if metas[i].get("frame_id") == 0 and self._prev_boxes is None:
                    self._prev_boxes = boxes
