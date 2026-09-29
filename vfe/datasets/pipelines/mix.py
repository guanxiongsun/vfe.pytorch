"""YOLOX's training augmentations. Port of the ``Mosaic``, ``RandomAffine``,
``MixUp``, ``YOLOXHSVRandomAug`` and ``FilterAnnotations`` transforms of
mmdet 2.19.1.

``Mosaic`` and ``MixUp`` read other images: they implement ``get_indexes``,
and :class:`~vfe.datasets.dataset_wrappers.MultiImageMixDataset` hands them
those images' results as ``results['mix_results']``.

All randomness is numpy's global generator: mmdet's module imports
``from numpy import random``, so its ``random.randint`` / ``random.uniform``
are numpy's (``randint``'s upper bound exclusive). The same draws in the same
order reproduce mmdet's samples exactly
(``tools/checks/parity_yolox_pipeline.py``).
"""

from __future__ import annotations

import copy
import math

import cv2
import numpy as np
from numpy import random

from vfe.datasets.builder import PIPELINES
from vfe.datasets.pipelines.image import imresize

__all__ = ["find_inside_bboxes", "Mosaic", "RandomAffine", "MixUp", "YOLOXHSVRandomAug",
           "FilterAnnotations"]


def find_inside_bboxes(bboxes: np.ndarray, img_h: float, img_w: float) -> np.ndarray:
    """Boolean mask of the boxes with some part inside the image."""
    return ((bboxes[:, 0] < img_w) & (bboxes[:, 2] > 0)
            & (bboxes[:, 1] < img_h) & (bboxes[:, 3] > 0))


@PIPELINES.register_module()
class Mosaic:
    """Four images, each resized to fit ``img_scale``, around a random centre
    of a ``2 x img_scale`` canvas filled with ``pad_val``."""

    def __init__(self, img_scale=(640, 640), center_ratio_range=(0.5, 1.5), min_bbox_size=0,
                 bbox_clip_border=True, skip_filter=True, pad_val=114):
        self.img_scale = tuple(img_scale)
        self.center_ratio_range = center_ratio_range
        self.min_bbox_size = min_bbox_size
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter
        self.pad_val = pad_val

    def get_indexes(self, dataset) -> list[int]:
        return [random.randint(0, len(dataset)) for _ in range(3)]

    def __call__(self, results: dict) -> dict:
        if "mix_results" not in results:
            raise KeyError("Mosaic needs 'mix_results' (use MultiImageMixDataset)")
        mosaic_labels, mosaic_bboxes = [], []
        h2, w2 = int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)
        shape = (h2, w2, 3) if len(results["img"].shape) == 3 else (h2, w2)
        mosaic_img = np.full(shape, self.pad_val, dtype=results["img"].dtype)
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        for i, loc in enumerate(("top_left", "top_right", "bottom_left", "bottom_right")):
            patch = copy.deepcopy(results if i == 0 else results["mix_results"][i - 1])
            img_i = patch["img"]
            h_i, w_i = img_i.shape[:2]
            scale_ratio_i = min(self.img_scale[0] / h_i, self.img_scale[1] / w_i)
            img_i = imresize(img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
            paste, crop = self._mosaic_combine(loc, (center_x, center_y), img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste
            x1_c, y1_c, x2_c, y2_c = crop
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]
            gt_bboxes_i = patch["gt_bboxes"]
            if gt_bboxes_i.shape[0] > 0:
                gt_bboxes_i[:, 0::2] = scale_ratio_i * gt_bboxes_i[:, 0::2] + (x1_p - x1_c)
                gt_bboxes_i[:, 1::2] = scale_ratio_i * gt_bboxes_i[:, 1::2] + (y1_p - y1_c)
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(patch["gt_labels"])

        mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
        mosaic_labels = np.concatenate(mosaic_labels, 0)
        if self.bbox_clip_border:
            mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0, w2)
            mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0, h2)
        if not self.skip_filter:
            keep = ((mosaic_bboxes[:, 2] - mosaic_bboxes[:, 0] > self.min_bbox_size)
                    & (mosaic_bboxes[:, 3] - mosaic_bboxes[:, 1] > self.min_bbox_size))
            mosaic_bboxes, mosaic_labels = mosaic_bboxes[keep], mosaic_labels[keep]
        inside = find_inside_bboxes(mosaic_bboxes, h2, w2)
        results["img"] = mosaic_img
        results["img_shape"] = mosaic_img.shape
        results["gt_bboxes"] = mosaic_bboxes[inside]
        results["gt_labels"] = mosaic_labels[inside]
        return results

    def _mosaic_combine(self, loc, center, img_shape_wh):
        """``(paste, crop)`` boxes ``(x1, y1, x2, y2)`` on the canvas and on the image."""
        cx, cy = center
        w, h = img_shape_wh
        if loc == "top_left":
            x1, y1, x2, y2 = max(cx - w, 0), max(cy - h, 0), cx, cy
            crop = (w - (x2 - x1), h - (y2 - y1), w, h)
        elif loc == "top_right":
            x1, y1, x2, y2 = cx, max(cy - h, 0), min(cx + w, self.img_scale[1] * 2), cy
            crop = (0, h - (y2 - y1), min(w, x2 - x1), h)
        elif loc == "bottom_left":
            x1, y1, x2, y2 = max(cx - w, 0), cy, cx, min(self.img_scale[0] * 2, cy + h)
            crop = (w - (x2 - x1), 0, w, min(y2 - y1, h))
        else:
            x1, y1 = cx, cy
            x2, y2 = min(cx + w, self.img_scale[1] * 2), min(self.img_scale[0] * 2, cy + h)
            crop = (0, 0, min(w, x2 - x1), min(y2 - y1, h))
        return (x1, y1, x2, y2), crop


@PIPELINES.register_module()
class RandomAffine:
    """Random rotation, scaling, shear and translation, as one perspective
    warp; ``border`` grows (positive) or crops (negative) the output. Boxes
    become the bounds of their warped corners."""

    def __init__(self, max_rotate_degree=10.0, max_translate_ratio=0.1,
                 scaling_ratio_range=(0.5, 1.5), max_shear_degree=2.0, border=(0, 0),
                 border_val=(114, 114, 114), min_bbox_size=2, min_area_ratio=0.2,
                 max_aspect_ratio=20, bbox_clip_border=True, skip_filter=True):
        if not 0 <= max_translate_ratio <= 1:
            raise ValueError("max_translate_ratio must be in [0, 1]")
        if not 0 < scaling_ratio_range[0] <= scaling_ratio_range[1]:
            raise ValueError("scaling_ratio_range must be positive and increasing")
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter

    def __call__(self, results: dict) -> dict:
        img = results["img"]
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2
        rotation = self._rotation(random.uniform(-self.max_rotate_degree,
                                                 self.max_rotate_degree))
        scaling_ratio = random.uniform(*self.scaling_ratio_range)
        scaling = self._scaling(scaling_ratio)
        shear = self._shear(random.uniform(-self.max_shear_degree, self.max_shear_degree),
                            random.uniform(-self.max_shear_degree, self.max_shear_degree))
        translate = self._translation(
            random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width,
            random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height)
        warp_matrix = translate @ shear @ rotation @ scaling
        img = cv2.warpPerspective(img, warp_matrix, dsize=(width, height),
                                  borderValue=self.border_val)
        results["img"] = img
        results["img_shape"] = img.shape

        for key in results.get("bbox_fields", []):
            bboxes = results[key]
            n = len(bboxes)
            if not n:
                continue
            xs = bboxes[:, [0, 0, 2, 2]].reshape(n * 4)
            ys = bboxes[:, [1, 3, 3, 1]].reshape(n * 4)
            points = warp_matrix @ np.vstack([xs, ys, np.ones_like(xs)])
            points = points[:2] / points[2]
            xs, ys = points[0].reshape(n, 4), points[1].reshape(n, 4)
            warp_bboxes = np.vstack((xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T
            if self.bbox_clip_border:
                warp_bboxes[:, [0, 2]] = warp_bboxes[:, [0, 2]].clip(0, width)
                warp_bboxes[:, [1, 3]] = warp_bboxes[:, [1, 3]].clip(0, height)
            valid = find_inside_bboxes(warp_bboxes, height, width)
            if not self.skip_filter:
                valid = valid & self._filter(bboxes * scaling_ratio, warp_bboxes)
            results[key] = warp_bboxes[valid]
            if key == "gt_bboxes" and "gt_labels" in results:
                results["gt_labels"] = results["gt_labels"][valid]
        return results

    def _filter(self, origin, wrapped):
        ow, oh = origin[:, 2] - origin[:, 0], origin[:, 3] - origin[:, 1]
        ww, wh = wrapped[:, 2] - wrapped[:, 0], wrapped[:, 3] - wrapped[:, 1]
        aspect = np.maximum(ww / (wh + 1e-16), wh / (ww + 1e-16))
        return ((ww > self.min_bbox_size) & (wh > self.min_bbox_size)
                & (ww * wh / (ow * oh + 1e-16) > self.min_area_ratio)
                & (aspect < self.max_aspect_ratio))

    @staticmethod
    def _rotation(degrees):
        r = math.radians(degrees)
        return np.array([[np.cos(r), -np.sin(r), 0.0], [np.sin(r), np.cos(r), 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float32)

    @staticmethod
    def _scaling(ratio):
        return np.array([[ratio, 0.0, 0.0], [0.0, ratio, 0.0], [0.0, 0.0, 1.0]],
                        dtype=np.float32)

    @staticmethod
    def _shear(x_degrees, y_degrees):
        return np.array([[1, np.tan(math.radians(x_degrees)), 0.0],
                         [np.tan(math.radians(y_degrees)), 1, 0.0], [0.0, 0.0, 1.0]],
                        dtype=np.float32)

    @staticmethod
    def _translation(x, y):
        return np.array([[1, 0.0, x], [0.0, 1, y], [0.0, 0.0, 1.0]], dtype=np.float32)


@PIPELINES.register_module()
class MixUp:
    """Blend in (0.5 / 0.5) another image with boxes, resized to fit
    ``img_scale``, jittered in scale by ``ratio_range``, maybe flipped, and
    randomly cropped to this image's size."""

    def __init__(self, img_scale=(640, 640), ratio_range=(0.5, 1.5), flip_ratio=0.5,
                 pad_val=114, max_iters=15, min_bbox_size=5, min_area_ratio=0.2,
                 max_aspect_ratio=20, bbox_clip_border=True, skip_filter=True):
        self.dynamic_scale = tuple(img_scale)
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter

    def get_indexes(self, dataset) -> int:
        """An image with boxes, trying up to ``max_iters`` random ones."""
        for _ in range(self.max_iters):
            index = random.randint(0, len(dataset))
            if len(dataset.get_ann_info(index)["bboxes"]) != 0:
                break
        return index

    def __call__(self, results: dict) -> dict:
        if "mix_results" not in results or len(results["mix_results"]) != 1:
            raise KeyError("MixUp needs one image in 'mix_results' (use MultiImageMixDataset)")
        retrieve = results["mix_results"][0]
        if retrieve["gt_bboxes"].shape[0] == 0:
            return results
        retrieve_img = retrieve["img"]
        jit_factor = random.uniform(*self.ratio_range)
        is_flip = random.uniform(0, 1) > self.flip_ratio
        shape = self.dynamic_scale + ((3,) if len(retrieve_img.shape) == 3 else ())
        out_img = np.ones(shape, dtype=retrieve_img.dtype) * self.pad_val
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0],
                          self.dynamic_scale[1] / retrieve_img.shape[1])
        retrieve_img = imresize(retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                                               int(retrieve_img.shape[0] * scale_ratio)))
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img
        scale_ratio *= jit_factor
        out_img = imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                     int(out_img.shape[0] * jit_factor)))
        if is_flip:
            out_img = out_img[:, ::-1, :]

        ori_img = results["img"]
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w), 3)).astype(
            np.uint8)
        padded_img[:origin_h, :origin_w] = out_img
        x_offset = y_offset = 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        retrieve_gt_bboxes = retrieve["gt_bboxes"]
        retrieve_gt_bboxes[:, 0::2] = retrieve_gt_bboxes[:, 0::2] * scale_ratio
        retrieve_gt_bboxes[:, 1::2] = retrieve_gt_bboxes[:, 1::2] * scale_ratio
        if self.bbox_clip_border:
            retrieve_gt_bboxes[:, 0::2] = np.clip(retrieve_gt_bboxes[:, 0::2], 0, origin_w)
            retrieve_gt_bboxes[:, 1::2] = np.clip(retrieve_gt_bboxes[:, 1::2], 0, origin_h)
        if is_flip:
            retrieve_gt_bboxes[:, 0::2] = origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1]
        cp_bboxes = retrieve_gt_bboxes.copy()
        cp_bboxes[:, 0::2] = cp_bboxes[:, 0::2] - x_offset
        cp_bboxes[:, 1::2] = cp_bboxes[:, 1::2] - y_offset
        if self.bbox_clip_border:
            cp_bboxes[:, 0::2] = np.clip(cp_bboxes[:, 0::2], 0, target_w)
            cp_bboxes[:, 1::2] = np.clip(cp_bboxes[:, 1::2], 0, target_h)

        mixup_img = 0.5 * ori_img.astype(np.float32) + 0.5 * padded_cropped_img.astype(np.float32)
        retrieve_gt_labels = retrieve["gt_labels"]
        if not self.skip_filter:
            keep = self._filter(retrieve_gt_bboxes.T, cp_bboxes.T)
            if keep.sum() >= 1.0:
                retrieve_gt_labels = retrieve_gt_labels[keep]
                cp_bboxes = cp_bboxes[keep]
        gt_bboxes = np.concatenate((results["gt_bboxes"], cp_bboxes), axis=0)
        gt_labels = np.concatenate((results["gt_labels"], retrieve_gt_labels), axis=0)
        inside = find_inside_bboxes(gt_bboxes, target_h, target_w)
        results["img"] = mixup_img.astype(np.uint8)
        results["img_shape"] = mixup_img.shape
        results["gt_bboxes"] = gt_bboxes[inside]
        results["gt_labels"] = gt_labels[inside]
        return results

    def _filter(self, bbox1, bbox2):
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return ((w2 > self.min_bbox_size) & (h2 > self.min_bbox_size)
                & (w2 * h2 / (w1 * h1 + 1e-16) > self.min_area_ratio)
                & (ar < self.max_aspect_ratio))


@PIPELINES.register_module()
class YOLOXHSVRandomAug:
    """Random hue / saturation / value shifts, each applied with probability
    one half, on a BGR uint8 image (in place)."""

    def __init__(self, hue_delta=5, saturation_delta=30, value_delta=30):
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def __call__(self, results: dict) -> dict:
        img = results["img"]
        hsv_gains = np.random.uniform(-1, 1, 3) * [self.hue_delta, self.saturation_delta,
                                                    self.value_delta]
        hsv_gains *= np.random.randint(0, 2, 3)
        hsv_gains = hsv_gains.astype(np.int16)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)
        results["img"] = img
        return results


@PIPELINES.register_module()
class FilterAnnotations:
    """Drop boxes no wider / taller than ``min_gt_bbox_wh``; if none is left,
    drop the sample (``keep_empty``) or keep it unchanged."""

    def __init__(self, min_gt_bbox_wh, keep_empty=True):
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.keep_empty = keep_empty

    def __call__(self, results: dict):
        gt_bboxes = results["gt_bboxes"]
        if gt_bboxes.shape[0] == 0:
            return results
        keep = ((gt_bboxes[:, 2] - gt_bboxes[:, 0] > self.min_gt_bbox_wh[0])
                & (gt_bboxes[:, 3] - gt_bboxes[:, 1] > self.min_gt_bbox_wh[1]))
        if not keep.any():
            return None if self.keep_empty else results
        for key in ("gt_bboxes", "gt_labels"):
            if key in results:
                results[key] = results[key][keep]
        return results
