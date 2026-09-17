"""Geometric and photometric transforms. Port of the parts of
``mmdet.datasets.pipelines.transforms`` and mmtrack's ``Seq*`` wrappers that the
VID configs use.

Ported: keep-ratio resize at one scale or a random one of several
(``multiscale_mode="value"``), flips, random crops, normalise and pad, plus the
STPN-specific ``SeqResize2`` and ``SeqMaxSizePad``: everything MAMBA's and
STPN's pipelines use. Masks, segmentation maps and the unused options raise
``NotImplementedError``.

The ``Seq*`` variants run the single-frame transform on each frame of a clip;
with ``share_params`` every frame gets the key frame's random choices.
Random choices draw from numpy's global generator in the original order, so a
seeded pipeline reproduces the original samples exactly.
"""

from __future__ import annotations

import numpy as np

from vfe.datasets.builder import PIPELINES
from vfe.datasets.pipelines.image import imnormalize, impad, impad_to_multiple, imrescale

__all__ = ["Resize", "SeqResize", "SeqResize2", "RandomFlip", "SeqRandomFlip", "RandomCrop",
           "SeqRandomCrop", "Normalize", "SeqNormalize", "Pad", "SeqPad", "SeqMaxSizePad",
           "bbox_flip", "imflip"]


def _require_no_masks(results: dict, what: str) -> None:
    for key in ("mask_fields", "seg_fields"):
        if results.get(key):
            raise NotImplementedError(f"{what} of {key} is not ported")


@PIPELINES.register_module()
class Resize:
    """Resize to ``img_scale`` (a ``(long, short)`` edge limit) keeping aspect ratio.

    Adds ``img_shape``, ``pad_shape``, ``scale_factor`` (float32 ``[w, h, w, h]``)
    and ``keep_ratio``.
    """

    def __init__(self, img_scale=None, multiscale_mode="range", ratio_range=None,
                 keep_ratio=True, bbox_clip_border=True, backend="cv2", override=False):
        if img_scale is None:
            raise NotImplementedError("resizing without img_scale is not ported")
        if ratio_range is not None or override or not keep_ratio or backend != "cv2":
            raise NotImplementedError(
                "ratio_range / override / keep_ratio=False / non-cv2 backends are not ported"
            )
        self.img_scale = list(img_scale) if isinstance(img_scale, list) else [img_scale]
        if multiscale_mode not in ("value", "range"):
            raise ValueError(f"unknown multiscale_mode {multiscale_mode!r}")
        if len(self.img_scale) > 1 and multiscale_mode != "value":
            raise NotImplementedError("multiscale_mode='range' is not ported")
        self.keep_ratio = keep_ratio
        self.bbox_clip_border = bbox_clip_border

    def __call__(self, results: dict) -> dict:
        if "scale" not in results:
            if "scale_factor" in results:
                raise NotImplementedError("resizing by a preset scale_factor is not ported")
            # mmdet's random_select: one of the listed scales, uniformly.
            scale_idx = 0 if len(self.img_scale) == 1 else np.random.randint(len(self.img_scale))
            results["scale"] = self.img_scale[scale_idx]
            results["scale_idx"] = scale_idx
        elif "scale_factor" in results:
            raise ValueError("scale and scale_factor cannot both be set")

        for key in results.get("img_fields", ["img"]):
            h, w = results[key].shape[:2]
            img = imrescale(results[key], results["scale"])
            new_h, new_w = img.shape[:2]
            # From the rounded size, not the requested factor, as the original.
            w_scale, h_scale = new_w / w, new_h / h
            results[key] = img
            results["img_shape"] = img.shape
            results["pad_shape"] = img.shape
            results["scale_factor"] = np.array([w_scale, h_scale, w_scale, h_scale],
                                               dtype=np.float32)
            results["keep_ratio"] = self.keep_ratio

        for key in results.get("bbox_fields", []):
            bboxes = results[key] * results["scale_factor"]
            if self.bbox_clip_border:
                img_shape = results["img_shape"]
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes
        _require_no_masks(results, "Resize")
        return results


@PIPELINES.register_module()
class SeqResize(Resize):
    def __init__(self, share_params: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.share_params = share_params

    def __call__(self, results: list[dict]) -> list[dict]:
        outs, scale = [], None
        for i, frame in enumerate(results):
            if self.share_params and i > 0:
                frame["scale"] = scale
            frame = super().__call__(frame)
            if self.share_params and i == 0:
                scale = frame["scale"]
            outs.append(frame)
        return outs


@PIPELINES.register_module()
class SeqResize2(SeqResize):
    """A second ``SeqResize`` in the same pipeline (STPN's crop policy): the
    first resize's ``scale`` and ``scale_factor`` are dropped, so a new scale is
    drawn and ``scale_factor`` then describes this resize alone."""

    def __call__(self, results: list[dict]) -> list[dict]:
        for frame in results:
            if "scale" in frame:
                frame.pop("scale")
                frame.pop("scale_factor", None)
        return super().__call__(results)


def bbox_flip(bboxes: np.ndarray, img_shape: tuple, direction: str) -> np.ndarray:
    """Mirror ``(..., 4k)`` boxes within an image of ``img_shape``."""
    if bboxes.shape[-1] % 4:
        raise ValueError("box arrays must have 4k columns")
    flipped = bboxes.copy()
    if direction in ("horizontal", "diagonal"):
        w = img_shape[1]
        flipped[..., 0::4] = w - bboxes[..., 2::4]
        flipped[..., 2::4] = w - bboxes[..., 0::4]
    if direction in ("vertical", "diagonal"):
        h = img_shape[0]
        flipped[..., 1::4] = h - bboxes[..., 3::4]
        flipped[..., 3::4] = h - bboxes[..., 1::4]
    if direction not in ("horizontal", "vertical", "diagonal"):
        raise ValueError(f"invalid flip direction {direction!r}")
    return flipped


def imflip(img: np.ndarray, direction: str) -> np.ndarray:
    """A flipped *view* (negative strides), as ``mmcv.imflip``; the next
    transform that copies (``Normalize``) makes it contiguous."""
    axis = {"horizontal": 1, "vertical": 0, "diagonal": (0, 1)}[direction]
    return np.flip(img, axis=axis)


@PIPELINES.register_module()
class RandomFlip:
    """Flip image and boxes when ``results['flip']`` says so."""

    def __init__(self, flip_ratio=None, direction="horizontal"):
        if isinstance(flip_ratio, list) or isinstance(direction, list):
            raise NotImplementedError("per-direction flip ratios are not ported")
        if flip_ratio is not None and not 0 <= flip_ratio <= 1:
            raise ValueError(f"flip_ratio must be in [0, 1], got {flip_ratio}")
        if direction not in ("horizontal", "vertical", "diagonal"):
            raise ValueError(f"unknown flip direction {direction!r}")
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, results: dict) -> dict:
        if "flip" not in results:
            raise NotImplementedError("RandomFlip deciding by itself is not ported; "
                                      "use SeqRandomFlip(share_params=True)")
        if results["flip"]:
            direction = results["flip_direction"]
            for key in results.get("img_fields", ["img"]):
                results[key] = imflip(results[key], direction)
            for key in results.get("bbox_fields", []):
                results[key] = bbox_flip(results[key], results["img_shape"], direction)
            _require_no_masks(results, "RandomFlip")
        return results


@PIPELINES.register_module()
class SeqRandomFlip(RandomFlip):
    def __init__(self, share_params: bool, **kwargs):
        super().__init__(**kwargs)
        if not share_params:
            raise NotImplementedError("share_params=False is not ported")
        self.share_params = share_params

    def __call__(self, results: list[dict]) -> list[dict]:
        direction_list = [self.direction, None]
        flip_ratio_list = [self.flip_ratio, 1 - self.flip_ratio]
        # Always drawn, even at flip_ratio 0: the original consumed numpy's
        # global RNG here on every sample, and later random draws depend on it.
        cur_dir = np.random.choice(direction_list, p=flip_ratio_list)
        for frame in results:
            frame["flip"] = cur_dir is not None
            frame["flip_direction"] = cur_dir
        return [super(SeqRandomFlip, self).__call__(frame) for frame in results]


@PIPELINES.register_module()
class RandomCrop:
    """Crop a random region: its size is ``crop_size`` (``(h, w)``, ``absolute``)
    or drawn per side from ``[min(side, crop_size[0]), min(side, crop_size[1])]``
    (``absolute_range``), its position uniformly. Boxes are shifted, clipped
    (``bbox_clip_border``) and dropped with their labels when empty; with none
    left, the sample is rejected unless ``allow_negative_crop``.
    """

    BBOX2LABEL = {"gt_bboxes": "gt_labels", "gt_bboxes_ignore": "gt_labels_ignore"}

    def __init__(self, crop_size, crop_type: str = "absolute", allow_negative_crop: bool = False,
                 recompute_bbox: bool = False, bbox_clip_border: bool = True):
        if crop_type not in ("absolute", "absolute_range"):
            raise NotImplementedError(f"crop_type={crop_type!r} is not ported")
        if recompute_bbox:
            raise NotImplementedError("recompute_bbox needs masks, which are not ported")
        if not (crop_size[0] > 0 and crop_size[1] > 0):
            raise ValueError(f"invalid crop_size {crop_size}")
        if crop_type == "absolute_range" and crop_size[0] > crop_size[1]:
            raise ValueError("absolute_range needs crop_size[0] <= crop_size[1]")
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border

    def _get_crop_size(self, image_size: tuple[int, int]) -> tuple[int, int]:
        h, w = image_size
        if self.crop_type == "absolute":
            return min(self.crop_size[0], h), min(self.crop_size[1], w)
        crop_h = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
        crop_w = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
        return crop_h, crop_w

    def __call__(self, results: dict) -> dict | None:
        _require_no_masks(results, "RandomCrop")
        if "gt_instance_ids" in results:
            raise NotImplementedError("RandomCrop does not filter gt_instance_ids")
        crop_h, crop_w = self._get_crop_size(results["img"].shape[:2])
        for key in results.get("img_fields", ["img"]):
            img = results[key]
            offset_h = np.random.randint(0, max(img.shape[0] - crop_h, 0) + 1)
            offset_w = np.random.randint(0, max(img.shape[1] - crop_w, 0) + 1)
            results[key] = img[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w, ...]
            img_shape = results[key].shape
        results["img_shape"] = img_shape

        for key in results.get("bbox_fields", []):
            offset = np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
            bboxes = results[key] - offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
            if key == "gt_bboxes" and not valid.any() and not self.allow_negative_crop:
                return None
            results[key] = bboxes[valid, :]
            label_key = self.BBOX2LABEL.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid]
        return results


@PIPELINES.register_module()
class SeqRandomCrop(RandomCrop):
    """``RandomCrop`` on each frame independently (size and position), as the
    STPN configs' registered ``SeqRandomCrop`` did; ``SeqMaxSizePad`` then
    brings the frames back to one size."""

    def __init__(self, crop_size, crop_type: str = "absolute", allow_negative_crop: bool = False,
                 recompute_bbox: bool = False, bbox_clip_border: bool = True,
                 share_params: bool = False):
        if share_params:
            raise NotImplementedError("share_params=True is not ported")
        super().__init__(crop_size, crop_type, allow_negative_crop, recompute_bbox,
                         bbox_clip_border)

    def __call__(self, results: list[dict]) -> list[dict | None]:
        return [super(SeqRandomCrop, self).__call__(frame) for frame in results]


@PIPELINES.register_module()
class Normalize:
    """Adds ``img_norm_cfg``."""

    def __init__(self, mean, std, to_rgb: bool = True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results: dict) -> dict:
        for key in results.get("img_fields", ["img"]):
            results[key] = imnormalize(results[key], self.mean, self.std, self.to_rgb)
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@PIPELINES.register_module()
class SeqNormalize(Normalize):
    def __call__(self, results: list[dict]) -> list[dict]:
        return [super(SeqNormalize, self).__call__(frame) for frame in results]


@PIPELINES.register_module()
class Pad:
    """Pad bottom/right to ``size`` or to a multiple of ``size_divisor``.

    Adds ``pad_shape``, ``pad_fixed_size`` and ``pad_size_divisor``.
    """

    def __init__(self, size=None, size_divisor=None, pad_to_square=False,
                 pad_val=None):
        if pad_to_square:
            raise NotImplementedError("pad_to_square is not ported")
        if (size is None) == (size_divisor is None):
            raise ValueError("exactly one of size and size_divisor must be set")
        if isinstance(pad_val, (int, float)):
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        self.pad_val = pad_val if pad_val is not None else dict(img=0, masks=0, seg=255)
        self.size = size
        self.size_divisor = size_divisor

    def __call__(self, results: dict) -> dict:
        pad_val = self.pad_val.get("img", 0)
        for key in results.get("img_fields", ["img"]):
            if self.size is not None:
                padded = impad(results[key], self.size, pad_val=pad_val)
            else:
                padded = impad_to_multiple(results[key], self.size_divisor, pad_val=pad_val)
            results[key] = padded
        results["pad_shape"] = padded.shape
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor
        _require_no_masks(results, "Pad")
        return results


@PIPELINES.register_module()
class SeqPad(Pad):
    def __call__(self, results: list[dict]) -> list[dict]:
        return [super(SeqPad, self).__call__(frame) for frame in results]


@PIPELINES.register_module()
class SeqMaxSizePad:
    """Pad every frame bottom/right (with zeros) to the largest height and the
    largest width among the clip's frames."""

    def __call__(self, results: list[dict]) -> list[dict]:
        max_h = max(frame["img"].shape[0] for frame in results)
        max_w = max(frame["img"].shape[1] for frame in results)
        for frame in results:
            _require_no_masks(frame, "SeqMaxSizePad")
            for key in frame.get("img_fields", ["img"]):
                frame[key] = impad(frame[key], (max_h, max_w))
            frame["pad_shape"] = frame["img"].shape
            frame["pad_fixed_size"] = np.array([max_h, max_w])
        return results
