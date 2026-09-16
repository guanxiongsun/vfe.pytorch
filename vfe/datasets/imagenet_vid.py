"""ImageNet VID / DET dataset: annotation loading, ground truth and evaluation.

Port of ``mmdet.datasets.{imagenet_vid_dataset,coco_video_dataset}`` (the
annotation half; the image pipeline and reference-frame sampling come with the
data path in Phase 5b/5d).

Frame order at test time is part of the published protocol, because MAMBA's
memory depends on it. With ``shuffle_video_frames``, each video's frames are
shuffled *except the first*. The original drew that shuffle from Python's
global ``random``, reseeded to 10 when its module was imported; here a private
``random.Random(shuffle_seed)`` with the same seed produces the same order
without depending on what else touched the global RNG.
"""

from __future__ import annotations

import os.path as osp
import random

import numpy as np

from vfe.datasets.cocovid import CocoVID
from vfe.evaluation import do_vid_evaluation

__all__ = ["ImagenetVIDDataset"]


class ImagenetVIDDataset:
    """Args:
        ann_file: COCO-VID JSON (VID) or COCO JSON (DET, with ``load_as_video=False``).
        img_prefix: directory the annotations' ``file_name`` entries are relative to.
        data_root: if given, relative ``ann_file`` / ``img_prefix`` are joined onto it.
        load_as_video: read ``videos`` and order images by video and frame.
        test_mode: keep every frame (training keeps only ``is_vid_train_frame``).
        shuffle_video_frames: shuffle frames within each video except the first.
        shuffle_seed: seed of that shuffle; 10 reproduces the original order.
    """

    CLASSES = (
        "airplane", "antelope", "bear", "bicycle", "bird", "bus", "car", "cattle", "dog",
        "domestic_cat", "elephant", "fox", "giant_panda", "hamster", "horse", "lion", "lizard",
        "monkey", "motorcycle", "rabbit", "red_panda", "sheep", "snake", "squirrel", "tiger",
        "train", "turtle", "watercraft", "whale", "zebra",
    )

    def __init__(
        self,
        ann_file: str,
        img_prefix: str = "",
        data_root: str | None = None,
        load_as_video: bool = True,
        test_mode: bool = False,
        shuffle_video_frames: bool = False,
        shuffle_seed: int = 10,
    ):
        if data_root is not None:
            if not osp.isabs(ann_file):
                ann_file = osp.join(data_root, ann_file)
            if img_prefix and not osp.isabs(img_prefix):
                img_prefix = osp.join(data_root, img_prefix)
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.load_as_video = load_as_video
        self.test_mode = test_mode
        self.shuffle_video_frames = shuffle_video_frames
        self.shuffle_seed = shuffle_seed

        self.data_infos = (
            self._load_video_anns(ann_file) if load_as_video else self._load_image_anns(ann_file)
        )

    def __len__(self) -> int:
        return len(self.data_infos)

    # ---- annotations ---------------------------------------------------------

    def _init_categories(self) -> None:
        # File category order, not CLASSES order; see CocoVID.get_cat_ids.
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

    def _load_video_anns(self, ann_file: str) -> list[dict]:
        self.coco = CocoVID(ann_file)
        self._init_categories()
        rng = random.Random(self.shuffle_seed)

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            if self.shuffle_video_frames:
                rest = img_ids[1:]
                rng.shuffle(rest)
                img_ids[1:] = rest
            for img_id in img_ids:
                info = dict(self.coco.load_imgs([img_id])[0])
                info["filename"] = info["file_name"]
                if self.test_mode:
                    if info["is_vid_train_frame"]:
                        raise ValueError(
                            f"image {img_id} is flagged is_vid_train_frame in test mode; "
                            "wrong annotation file?"
                        )
                    self.img_ids.append(img_id)
                    data_infos.append(info)
                elif info["is_vid_train_frame"]:
                    self.img_ids.append(img_id)
                    data_infos.append(info)
        return data_infos

    def _load_image_anns(self, ann_file: str) -> list[dict]:
        self.coco = CocoVID(ann_file)
        self._init_categories()
        self.img_ids = []
        data_infos = []
        for img_id in self.coco.get_img_ids():
            info = dict(self.coco.load_imgs([img_id])[0])
            info["filename"] = info["file_name"]
            if info["is_vid_train_frame"]:
                self.img_ids.append(img_id)
                data_infos.append(info)
        return data_infos

    def get_ann_info(self, img_info: dict | int) -> dict[str, np.ndarray]:
        """Ground truth for one image: ``bboxes`` (x1, y1, x2, y2, float32),
        0-based ``labels``, ``bboxes_ignore`` (crowd boxes), ``instance_ids``."""
        if isinstance(img_info, int):
            img_info = self.data_infos[img_info]
        ann_ids = self.coco.get_ann_ids(img_ids=[img_info["id"]], cat_ids=self.cat_ids)
        return self._parse_ann_info(img_info, self.coco.load_anns(ann_ids))

    def _parse_ann_info(self, img_info: dict, ann_info: list[dict]) -> dict[str, np.ndarray]:
        gt_bboxes, gt_labels, gt_bboxes_ignore, gt_instance_ids = [], [], [], []
        for ann in ann_info:
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            # x2 = x1 + w: mmdet v2's convention, no -1.
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann["category_id"]])
                if "instance_id" in ann:
                    gt_instance_ids.append(ann["instance_id"])

        if gt_bboxes:
            bboxes = np.array(gt_bboxes, dtype=np.float32)
            labels = np.array(gt_labels, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        if self.load_as_video:
            instance_ids = np.array(gt_instance_ids).astype(np.int64)
        else:
            instance_ids = np.arange(len(labels))
        return dict(bboxes=bboxes, labels=labels, bboxes_ignore=bboxes_ignore,
                    instance_ids=instance_ids)

    # ---- evaluation ----------------------------------------------------------

    def evaluate(self, results, vid_style: bool = True, **kwargs) -> dict[str, float]:
        """ImageNet VID AP50 (see :func:`vfe.evaluation.do_vid_evaluation`)."""
        if not vid_style:
            raise NotImplementedError(
                "COCO-style evaluation is not ported; the VID configs use vid_style=True"
            )
        return do_vid_evaluation(self, results)
