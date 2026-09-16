"""Image loading. Port of ``mmdet.datasets.pipelines.loading.LoadImageFromFile``
and mmtrack's ``LoadMultiImagesFromFile``; only the local-disk backend and
colour decoding the configs use."""

from __future__ import annotations

import os.path as osp

import numpy as np

from vfe.datasets.builder import PIPELINES
from vfe.datasets.pipelines.image import imfrombytes

__all__ = ["LoadImageFromFile", "LoadMultiImagesFromFile"]


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
