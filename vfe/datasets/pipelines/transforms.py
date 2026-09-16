"""Geometric and photometric transforms. Port of the parts of
``mmdet.datasets.pipelines.transforms`` and mmtrack's ``Seq*`` wrappers that the
VID configs use.

Status: the test-time paths are complete: single-scale keep-ratio resize, no
flip, normalise, pad. Training-only paths (box resizing, actual flipping,
multi-scale selection) raise ``NotImplementedError`` until Phase 5d ports them,
so a config needing them fails loudly instead of silently skipping a step.

The ``Seq*`` variants run the single-frame transform on each frame of a clip;
with ``share_params`` every frame gets the key frame's random choices.
"""

from __future__ import annotations

import numpy as np

from vfe.datasets.builder import PIPELINES
from vfe.datasets.pipelines.image import imnormalize, impad, impad_to_multiple, imrescale

__all__ = ["Resize", "SeqResize", "RandomFlip", "SeqRandomFlip", "Normalize", "SeqNormalize",
           "Pad", "SeqPad"]


def _require_no_annotations(results: dict, what: str) -> None:
    for key in ("bbox_fields", "mask_fields", "seg_fields"):
        if results.get(key):
            raise NotImplementedError(f"{what} of {key} is not ported yet (Phase 5d)")


@PIPELINES.register_module()
class Resize:
    """Resize to ``img_scale`` (a ``(long, short)`` edge limit) keeping aspect ratio.

    Adds ``img_shape``, ``pad_shape``, ``scale_factor`` (float32 ``[w, h, w, h]``)
    and ``keep_ratio``.
    """

    def __init__(self, img_scale=None, multiscale_mode="range", ratio_range=None,
                 keep_ratio=True, bbox_clip_border=True, backend="cv2", override=False):
        if img_scale is None or isinstance(img_scale, list) and len(img_scale) != 1:
            raise NotImplementedError("only a single img_scale is ported (Phase 5d)")
        if ratio_range is not None or override or not keep_ratio or backend != "cv2":
            raise NotImplementedError(
                "ratio_range / override / keep_ratio=False / non-cv2 backends are not ported"
            )
        self.img_scale = [img_scale] if isinstance(img_scale, tuple) else list(img_scale)
        self.keep_ratio = keep_ratio
        self.bbox_clip_border = bbox_clip_border

    def __call__(self, results: dict) -> dict:
        if "scale" not in results:
            if "scale_factor" in results:
                raise NotImplementedError("resizing by a preset scale_factor is not ported")
            results["scale"] = self.img_scale[0]
            results["scale_idx"] = 0
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
        _require_no_annotations(results, "Resize")
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
class RandomFlip:
    """Flip decided by ``results['flip']``; only the no-flip path is ported."""

    def __init__(self, flip_ratio=None, direction="horizontal"):
        if isinstance(flip_ratio, list) or isinstance(direction, list):
            raise NotImplementedError("per-direction flip ratios are not ported (Phase 5d)")
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
            raise NotImplementedError("flipping is not ported yet (Phase 5d)")
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
        _require_no_annotations(results, "Pad")
        return results


@PIPELINES.register_module()
class SeqPad(Pad):
    def __call__(self, results: list[dict]) -> list[dict]:
        return [super(SeqPad, self).__call__(frame) for frame in results]
