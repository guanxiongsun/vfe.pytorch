"""Packing a clip's per-frame results into model inputs. Port of mmtrack's
``VideoCollect``, ``ConcatVideoReferences``, ``MultiImagesToTensor`` and
``ToList``.

The originals wrapped image metas in ``mmcv.DataContainer`` so mmcv's collate
would leave them unstacked. Here metas stay plain Python objects and
:mod:`vfe.datasets.collate` handles them explicitly.

``SeqDefaultFormatBundle`` is the training counterpart of
``MultiImagesToTensor``: it also converts ground truth to tensors.
"""

from __future__ import annotations

import numpy as np
import torch

from vfe.datasets.builder import PIPELINES

__all__ = ["VideoCollect", "ConcatVideoReferences", "MultiImagesToTensor", "ToList",
           "SeqDefaultFormatBundle"]

DEFAULT_META_KEYS = ("filename", "ori_filename", "ori_shape", "img_shape", "pad_shape",
                     "scale_factor", "flip", "flip_direction", "img_norm_cfg", "frame_id",
                     "is_video_data")


@PIPELINES.register_module()
class VideoCollect:
    """Keep ``keys``, and gather meta keys into ``img_metas``.

    A meta key is looked up in the results, then in ``img_info``; one found in
    neither is left out rather than set to a default. That matters for MAMBA:
    adaptive-stride test samples have no ``frame_stride``, and its absence is
    what selects the adaptive path in ``MAMBA.extract_feats``.
    """

    def __init__(self, keys, meta_keys=None, default_meta_keys=DEFAULT_META_KEYS):
        self.keys = keys
        self.meta_keys = tuple(default_meta_keys)
        if meta_keys is not None:
            self.meta_keys += (meta_keys,) if isinstance(meta_keys, str) else tuple(meta_keys)

    def __call__(self, results):
        is_dict = isinstance(results, dict)
        outs = [self._collect(r) for r in ([results] if is_dict else results)]
        return outs[0] if is_dict else outs

    def _collect(self, results: dict) -> dict:
        img = results["img"]
        results.setdefault("pad_shape", img.shape)
        results.setdefault("scale_factor", 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results.setdefault("img_norm_cfg", dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32), to_rgb=False))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
            elif key in results["img_info"]:
                img_meta[key] = results["img_info"][key]
        data = {"img_metas": img_meta}
        for key in self.keys:
            data[key] = results[key]
        return data


@PIPELINES.register_module()
class ConcatVideoReferences:
    """``[key, ref_1, ..., ref_n]`` -> ``[key, refs]``: reference images stacked
    on a new last axis, metas gathered in a list, and ground-truth arrays
    concatenated with their reference index prepended as a column."""

    ARRAY_KEYS = ("proposals", "gt_bboxes", "gt_bboxes_ignore", "gt_labels", "gt_instance_ids")

    def __call__(self, results: list[dict]) -> list[dict]:
        if not isinstance(results, list):
            raise TypeError("results must be a list")
        outs = results[:1]
        for i, result in enumerate(results[1:], 1):
            if "img" in result:
                img = result["img"]
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if i == 1:
                    result["img"] = np.expand_dims(img, -1)
                else:
                    outs[1]["img"] = np.concatenate((outs[1]["img"], np.expand_dims(img, -1)),
                                                    axis=-1)
            for key in ("img_metas", "gt_masks"):
                if key in result:
                    if i == 1:
                        result[key] = [result[key]]
                    else:
                        outs[1][key].append(result[key])
            for key in self.ARRAY_KEYS:
                if key not in result:
                    continue
                value = result[key]
                if value.ndim == 1:
                    value = value[:, None]
                value = np.concatenate(
                    (np.full((value.shape[0], 1), i - 1, dtype=np.float32), value), axis=1)
                if i == 1:
                    result[key] = value
                else:
                    outs[1][key] = np.concatenate((outs[1][key], value), axis=0)
            if "gt_semantic_seg" in result:
                raise NotImplementedError("gt_semantic_seg is not ported")
            if i == 1:
                outs.append(result)
        return outs


@PIPELINES.register_module()
class MultiImagesToTensor:
    """Images to CHW (or NCHW for stacked references) tensors; the references'
    keys get ``ref_prefix``. Returns one dict."""

    def __init__(self, ref_prefix: str = "ref"):
        self.ref_prefix = ref_prefix

    def __call__(self, results: list[dict]) -> dict:
        outs = [self._to_tensor(r) for r in results]
        data = dict(outs[0])
        if len(outs) == 2:
            for key, value in outs[1].items():
                data[f"{self.ref_prefix}_{key}"] = value
        return data

    @staticmethod
    def _to_tensor(results: dict) -> dict:
        if "img" in results:
            img = results["img"]
            if len(img.shape) == 3:  # (H, W, 3) -> (3, H, W)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            else:  # (H, W, 3, N) -> (N, 3, H, W)
                img = np.ascontiguousarray(img.transpose(3, 2, 0, 1))
            results["img"] = torch.from_numpy(img)
        if "proposals" in results:
            results["proposals"] = torch.from_numpy(results["proposals"])
        return results


@PIPELINES.register_module()
class ToList:
    """Wrap every value in a one-item list (the test-time-augmentation axis)."""

    def __call__(self, results: dict) -> dict:
        return {key: [value] for key, value in results.items()}


@PIPELINES.register_module()
class SeqDefaultFormatBundle:
    """Images to CHW / NCHW tensors and ground-truth arrays to tensors; the
    references' keys get ``ref_prefix``. Returns one dict.

    The original wrapped values in ``DataContainer`` to steer mmcv's collate
    (stack images, keep boxes as per-sample lists, keep metas on the CPU);
    :func:`vfe.datasets.collate.collate_video_train` applies those rules by key.
    """

    GT_KEYS = ("proposals", "gt_bboxes", "gt_bboxes_ignore", "gt_labels", "gt_instance_ids",
               "gt_match_indices")

    def __init__(self, ref_prefix: str = "ref"):
        self.ref_prefix = ref_prefix

    def __call__(self, results: list[dict]) -> dict:
        outs = [self._format(r) for r in results]
        data = dict(outs[0])
        if len(outs) > 1:
            for key, value in outs[1].items():
                data[f"{self.ref_prefix}_{key}"] = value
        return data

    def _format(self, results: dict) -> dict:
        if "img" in results:
            img = results["img"]
            if len(img.shape) == 3:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            else:
                img = np.ascontiguousarray(img.transpose(3, 2, 0, 1))
            results["img"] = torch.from_numpy(img)
        for key in self.GT_KEYS:
            if key in results:
                results[key] = torch.from_numpy(results[key])
        if "gt_semantic_seg" in results:
            raise NotImplementedError("gt_semantic_seg is not ported")
        return results
