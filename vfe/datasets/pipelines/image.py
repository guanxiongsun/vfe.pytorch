"""Image operations of ``mmcv.image`` the pipelines use, with its exact semantics.

Each goes through the same OpenCV call mmcv made, not a numpy equivalent, so
the tensors match the original bit for bit (both envs ship OpenCV 4.14.0).
The rounding here is part of the published setup: a one-pixel difference in
the resized size changes every feature downstream.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["imfrombytes", "rescale_size", "imrescale", "imnormalize", "impad",
           "impad_to_multiple"]


def imfrombytes(content: bytes) -> np.ndarray:
    """Decode an encoded image to BGR uint8 (``mmcv.imfrombytes(flag='color')``)."""
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


def rescale_size(old_size: tuple[int, int], scale: tuple[int, int]) -> tuple[tuple[int, int], float]:
    """``(w, h)`` scaled to fit ``scale`` (long, short edge limits) with its
    aspect ratio kept, and the factor used. Rounds with ``int(x + 0.5)``."""
    w, h = old_size
    max_long_edge, max_short_edge = max(scale), min(scale)
    scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    new_size = (int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5))
    return new_size, scale_factor


def imrescale(img: np.ndarray, scale: tuple[int, int]) -> np.ndarray:
    """Resize with aspect ratio kept (bilinear, as mmcv's default)."""
    h, w = img.shape[:2]
    new_size, _ = rescale_size((w, h), scale)
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)


def imnormalize(img: np.ndarray, mean: np.ndarray, std: np.ndarray, to_rgb: bool = True
                ) -> np.ndarray:
    """``(img - mean) / std`` in float32, optionally BGR -> RGB first."""
    img = img.copy().astype(np.float32)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return img


def impad(img: np.ndarray, shape: tuple[int, int], pad_val: float = 0) -> np.ndarray:
    """Pad bottom and right to ``shape`` = ``(h, w)``."""
    return cv2.copyMakeBorder(img, 0, shape[0] - img.shape[0], 0, shape[1] - img.shape[1],
                              cv2.BORDER_CONSTANT, value=pad_val)


def impad_to_multiple(img: np.ndarray, divisor: int, pad_val: float = 0) -> np.ndarray:
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, (pad_h, pad_w), pad_val=pad_val)
