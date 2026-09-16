"""Index over a COCO-VID style annotation file (COCO plus ``videos``).

Replaces ``mmdet.datasets.parsers.CocoVID`` and the parts of pycocotools'
``COCO`` it inherited. Only the lookups the VID datasets use are provided, but
each keeps pycocotools' ordering semantics, because labels and frame order
derive from them:

* ``get_cat_ids(cat_names)`` returns ids in the *file's* category order, not in
  the order of ``cat_names``. ``cat2label`` is built from that order.
* ``get_ann_ids`` / ``get_img_ids`` / ``get_vid_ids`` return ids in file order.
* ``get_img_ids_from_vid`` returns a video's image ids indexed by ``frame_id``.
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

__all__ = ["CocoVID"]


def _as_list(value: Any) -> list:
    """pycocotools' ``_isArrayLike`` convention: a scalar means a one-item list."""
    if hasattr(value, "__iter__") and hasattr(value, "__len__"):
        return list(value)
    return [value]


class CocoVID:
    """Args:
        annotation_file: path to the JSON file.
        load_img_as_vid: treat each image as a one-frame video when the file
            has no ``videos`` (the DET annotations).
    """

    def __init__(self, annotation_file: str, load_img_as_vid: bool = False):
        with open(annotation_file) as f:
            self.dataset = json.load(f)
        if not isinstance(self.dataset, dict):
            raise TypeError(f"{annotation_file}: expected a JSON object at top level")
        self.load_img_as_vid = load_img_as_vid
        self._create_index()

    def _create_index(self) -> None:
        dataset = self.dataset
        if "videos" not in dataset and self.load_img_as_vid:
            dataset["videos"] = [
                dict(id=img["id"], name=img["file_name"]) for img in dataset.get("images", [])
            ]
            for img in dataset.get("images", []):
                img["video_id"] = img["id"]
                img["frame_id"] = 0
            for ann in dataset.get("annotations", []):
                ann["video_id"] = ann["image_id"]
                ann["instance_id"] = ann["id"]

        self.videos = {video["id"]: video for video in dataset.get("videos", [])}
        self.anns: dict[int, dict] = {}
        self.img_to_anns: dict[int, list[dict]] = defaultdict(list)
        self.vid_to_instances: dict[int, list[int]] = defaultdict(list)
        self.instances_to_imgs: dict[int, list[int]] = defaultdict(list)
        for ann in dataset.get("annotations", []):
            self.img_to_anns[ann["image_id"]].append(ann)
            self.anns[ann["id"]] = ann
            if "instance_id" in ann:
                self.instances_to_imgs[ann["instance_id"]].append(ann["image_id"])
                if ("video_id" in ann
                        and ann["instance_id"] not in self.vid_to_instances[ann["video_id"]]):
                    self.vid_to_instances[ann["video_id"]].append(ann["instance_id"])

        self.imgs: dict[int, dict] = {}
        self.vid_to_imgs: dict[int, list[dict]] = defaultdict(list)
        for img in dataset.get("images", []):
            if "video_id" in img:
                self.vid_to_imgs[img["video_id"]].append(img)
            self.imgs[img["id"]] = img

        self.cats = {cat["id"]: cat for cat in dataset.get("categories", [])}
        self.cat_to_imgs: dict[int, list[int]] = defaultdict(list)
        if "categories" in dataset:
            for ann in dataset.get("annotations", []):
                self.cat_to_imgs[ann["category_id"]].append(ann["image_id"])

    # ---- lookups ---------------------------------------------------------------

    def get_cat_ids(self, cat_names: Iterable[str] | str = ()) -> list[int]:
        names = _as_list(cat_names)
        cats = self.dataset.get("categories", [])
        if names:
            cats = [cat for cat in cats if cat["name"] in names]
        return [cat["id"] for cat in cats]

    def get_ann_ids(self, img_ids: Iterable[int] | int = (),
                    cat_ids: Iterable[int] | int = ()) -> list[int]:
        img_ids = _as_list(img_ids)
        cat_ids = _as_list(cat_ids)
        if img_ids:
            anns = list(itertools.chain.from_iterable(
                self.img_to_anns[img_id] for img_id in img_ids if img_id in self.img_to_anns
            ))
        else:
            anns = self.dataset.get("annotations", [])
        if cat_ids:
            anns = [ann for ann in anns if ann["category_id"] in cat_ids]
        return [ann["id"] for ann in anns]

    def get_img_ids(self) -> list[int]:
        return list(self.imgs)

    def get_vid_ids(self) -> list[int]:
        return list(self.videos)

    def get_img_ids_from_vid(self, vid_id: int) -> list[int]:
        img_infos = self.vid_to_imgs[vid_id]
        ids = [0] * len(img_infos)
        for img_info in img_infos:
            ids[img_info["frame_id"]] = img_info["id"]
        return ids

    def load_anns(self, ids: Iterable[int] | int) -> list[dict]:
        return [self.anns[i] for i in _as_list(ids)]

    def load_imgs(self, ids: Iterable[int] | int) -> list[dict]:
        return [self.imgs[i] for i in _as_list(ids)]

    def load_vids(self, ids: Iterable[int] | int) -> list[dict]:
        return [self.videos[i] for i in _as_list(ids)]
