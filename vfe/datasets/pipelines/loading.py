"""Image loading. Port of ``mmdet.datasets.pipelines.loading.LoadImageFromFile``
and mmtrack's ``LoadMultiImagesFromFile``; only the local-disk backend and
colour decoding the configs use."""

from __future__ import annotations

import os.path as osp

import numpy as np

from vfe.datasets.builder import PIPELINES
from vfe.datasets.pipelines.image import imfrombytes

__all__ = ["LoadImageFromFile", "LoadMultiImagesFromFile", "LoadAnnotations",
           "SeqLoadAnnotations"]


@PIPELINES.register_module()
class LoadImageFromFile:
    """Adds ``img`` (BGR uint8, or float32 with ``to_float32``), ``filename``,
    ``ori_filename``, ``img_shape``, ``ori_shape`` and ``img_fields``."""

    def __init__(self, to_float32: bool = False, color_type: str = "color",
                 file_client_args: dict | None = None):
        if color_type != "color":
            raise NotImplementedError(f"color_type={color_type!r}; only 'color' is ported")
        if file_client_args not in (None, {"backend": "disk"}):
            raise NotImplementedError(f"file_client_args={file_client_args}; only local disk")
        self.to_float32 = to_float32

    def __call__(self, results: dict) -> dict:
        if results["img_prefix"] is not None:
            filename = osp.join(results["img_prefix"], results["img_info"]["filename"])
        else:
            filename = results["img_info"]["filename"]
        with open(filename, "rb") as f:
            img = imfrombytes(f.read())
        if self.to_float32:
            img = img.astype(np.float32)

        results["filename"] = filename
        results["ori_filename"] = results["img_info"]["filename"]
        results["img"] = img
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        results["img_fields"] = ["img"]
        return results


@PIPELINES.register_module()
class LoadMultiImagesFromFile(LoadImageFromFile):
    """``LoadImageFromFile`` over each frame's results dict."""

    def __call__(self, results: list[dict]) -> list[dict]:
        return [super(LoadMultiImagesFromFile, self).__call__(r) for r in results]


@PIPELINES.register_module()
class LoadAnnotations:
    """Copy ground truth from ``ann_info`` into the results: ``gt_bboxes``,
    ``gt_bboxes_ignore``, ``gt_labels``, and register the box fields so later
    geometric transforms move them with the image. Masks and segmentation
    are not ported (no config uses them)."""

    def __init__(self, with_bbox: bool = True, with_label: bool = True, with_mask: bool = False,
                 with_seg: bool = False, poly2mask: bool = True, file_client_args=None):
        if with_mask or with_seg:
            raise NotImplementedError("mask / segmentation annotations are not ported")
        self.with_bbox = with_bbox
        self.with_label = with_label

    def __call__(self, results: dict) -> dict:
        ann_info = results["ann_info"]
        if self.with_bbox:
            results["gt_bboxes"] = ann_info["bboxes"].copy()
            if ann_info.get("bboxes_ignore") is not None:
                results["gt_bboxes_ignore"] = ann_info["bboxes_ignore"].copy()
                results["bbox_fields"].append("gt_bboxes_ignore")
            results["bbox_fields"].append("gt_bboxes")
        if self.with_label:
            results["gt_labels"] = ann_info["labels"].copy()
        return results


@PIPELINES.register_module()
class SeqLoadAnnotations(LoadAnnotations):
    """Per frame; ``with_track`` also loads ``gt_instance_ids``."""

    def __init__(self, with_track: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.with_track = with_track

    def __call__(self, results: list[dict]) -> list[dict]:
        outs = []
        for frame in results:
            frame = super().__call__(frame)
            if self.with_track:
                frame["gt_instance_ids"] = frame["ann_info"]["instance_ids"].copy()
            outs.append(frame)
        return outs
