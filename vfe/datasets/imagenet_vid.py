"""ImageNet VID / DET dataset: annotation loading, ground truth and evaluation.

Port of ``mmdet.datasets.{imagenet_vid_dataset,coco_video_dataset}`` and the
parts of ``CustomDataset`` / ``CocoDataset`` they inherited.

In training mode (as in mmdet): images without an annotation of the 30
classes, or smaller than 32 px, are dropped; every image gets an aspect-ratio
``flag`` (1 if wider than tall) for the group sampler; and a sample the
pipeline rejects is replaced by a random one from the same group.

Reference frames are chosen per sample by ``ref_img_sampling``. At test time
with ``test_with_adaptive_stride`` (the released configs), the first frame of
each video gets ``num_ref_imgs`` frames spread evenly over the *whole* video,
and every later frame gets none, relying on MAMBA's memory instead.

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

from vfe.datasets.builder import DATASETS
from vfe.datasets.cocovid import CocoVID
from vfe.datasets.pipelines import Compose
from vfe.evaluation import do_vid_evaluation

__all__ = ["ImagenetVIDDataset"]


@DATASETS.register_module()
class ImagenetVIDDataset:
    """Args:
        ann_file: COCO-VID JSON (VID) or COCO JSON (DET, with ``load_as_video=False``).
        img_prefix: directory the annotations' ``file_name`` entries are relative to.
        data_root: if given, relative ``ann_file`` / ``img_prefix`` are joined onto it.
        load_as_video: read ``videos`` and order images by video and frame.
        test_mode: keep every frame (training keeps only ``is_vid_train_frame``).
        shuffle_video_frames: shuffle frames within each video except the first.
        shuffle_seed: seed of that shuffle; 10 reproduces the original order.
        pipeline: transform configs applied to ``[key frame, *reference frames]``.
        ref_img_sampler: keyword arguments of :meth:`ref_img_sampling`, or None
            to load the key frame alone.
        filter_empty_gt: in training, drop images with no annotation of
            ``CLASSES``.
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
        pipeline: list | None = None,
        ref_img_sampler: dict | None = None,
        filter_empty_gt: bool = True,
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

        self.ref_img_sampler = ref_img_sampler
        self.pipeline = Compose(pipeline or [])

        self.filter_empty_gt = filter_empty_gt

        self.data_infos = (
            self._load_video_anns(ann_file) if load_as_video else self._load_image_anns(ann_file)
        )
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            self._set_group_flag()

    def __len__(self) -> int:
        return len(self.data_infos)

    def __getitem__(self, idx: int):
        if self.test_mode:
            return self.prepare_data(idx)
        while True:
            data = self.prepare_data(idx)
            if data is not None:
                return data
            idx = self._rand_another(idx)

    # ---- training-mode bookkeeping ---------------------------------------------

    def _filter_imgs(self, min_size: int = 32) -> list[int]:
        """Indices of images to keep; also narrows ``img_ids`` to match."""
        ids_with_ann = {ann["image_id"] for ann in self.coco.anns.values()}
        ids_in_cat = set()
        for cat_id in self.cat_ids:
            ids_in_cat |= set(self.coco.cat_to_imgs[cat_id])
        ids_in_cat &= ids_with_ann

        valid_inds, valid_img_ids = [], []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info["width"], img_info["height"]) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _set_group_flag(self) -> None:
        """1 for landscape images, 0 otherwise: batches never mix the two."""
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i, img_info in enumerate(self.data_infos):
            if img_info["width"] / img_info["height"] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx: int) -> int:
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    # ---- samples -------------------------------------------------------------

    def prepare_results(self, img_info: dict) -> dict:
        results = dict(img_info=img_info, img_prefix=self.img_prefix, seg_prefix=None,
                       proposal_file=None, bbox_fields=[], mask_fields=[], seg_fields=[],
                       is_video_data=self.load_as_video)
        if not self.test_mode:
            results["ann_info"] = self.get_ann_info(img_info)
        return results

    def prepare_data(self, idx: int):
        img_info = self.data_infos[idx]
        if self.ref_img_sampler is not None:
            img_infos = self.ref_img_sampling(img_info, **self.ref_img_sampler)
            results = [self.prepare_results(info) for info in img_infos]
        else:
            results = self.prepare_results(img_info)
        return self.pipeline(results)

    def ref_img_sampling(self, img_info: dict, frame_range, stride: int = 1,
                         num_ref_imgs: int = 1, filter_key_img: bool = True,
                         method: str = "uniform", return_key_img: bool = True) -> list[dict]:
        """Reference frames for a key frame, sorted by ``frame_id``.

        Methods:
            uniform: ``num_ref_imgs`` random frames within ``frame_range``.
            bilateral_uniform: half from each side of the key frame.
            test_with_adaptive_stride: on a video's first frame only,
                ``num_ref_imgs`` frames evenly spread over the whole video
                (index ``round(i * stride)``, Python's round-half-to-even).
            test_with_fix_stride: a window of ``stride``-spaced frames on the
                first frame, then one new frame every ``stride`` frames.

        The random methods draw from Python's global ``random``, as the
        original did.
        """
        if isinstance(frame_range, int):
            if frame_range < 0:
                raise ValueError("frame_range can not be negative")
            frame_range = [-frame_range, frame_range]
        elif isinstance(frame_range, (list, tuple)):
            if len(frame_range) != 2 or frame_range[0] > 0 or frame_range[1] < 0:
                raise ValueError(f"frame_range must be [<=0, >=0], got {frame_range}")
            frame_range = list(frame_range)
        else:
            raise TypeError("frame_range must be an int or a list")
        if "test" in method and frame_range[1] - frame_range[0] != num_ref_imgs:
            # The original warned and rewrote its own config; refuse instead.
            raise ValueError(f"{method} needs num_ref_imgs == frame_range[1] - frame_range[0]")

        if (not self.load_as_video or img_info.get("frame_id", -1) < 0
                or frame_range == [0, 0]):
            ref_img_infos = [img_info.copy() for _ in range(num_ref_imgs)]
        else:
            vid_id, img_id, frame_id = img_info["video_id"], img_info["id"], img_info["frame_id"]
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            left = max(0, frame_id + frame_range[0])
            right = min(frame_id + frame_range[1], len(img_ids) - 1)

            ref_img_ids = []
            if method == "uniform":
                valid_ids = img_ids[left:right + 1]
                if filter_key_img and img_id in valid_ids:
                    valid_ids.remove(img_id)
                ref_img_ids.extend(random.sample(valid_ids, min(num_ref_imgs, len(valid_ids))))
            elif method == "bilateral_uniform":
                if num_ref_imgs % 2:
                    raise ValueError("bilateral_uniform needs an even num_ref_imgs")
                for valid_ids in (img_ids[left:frame_id + 1], img_ids[frame_id:right + 1]):
                    if filter_key_img and img_id in valid_ids:
                        valid_ids.remove(img_id)
                    num_samples = min(num_ref_imgs // 2, len(valid_ids))
                    ref_img_ids.extend(random.sample(valid_ids, num_samples))
            elif method == "test_with_adaptive_stride":
                if frame_id == 0:
                    adaptive = float(len(img_ids) - 1) / (num_ref_imgs - 1)
                    ref_img_ids.extend(img_ids[round(i * adaptive)] for i in range(num_ref_imgs))
            elif method == "test_with_fix_stride":
                if frame_id == 0:
                    ref_img_ids.extend(img_ids[0] for _ in range(frame_range[0], 1))
                    ref_img_ids.extend(
                        img_ids[min(round(i * stride), len(img_ids) - 1)]
                        for i in range(1, frame_range[1] + 1)
                    )
                elif frame_id % stride == 0:
                    ref_img_ids.append(
                        img_ids[min(round(frame_id + frame_range[1] * stride), len(img_ids) - 1)]
                    )
                # Only this method sets these; MAMBA reads their absence as
                # "adaptive stride".
                img_info["num_left_ref_imgs"] = abs(frame_range[0])
                img_info["frame_stride"] = stride
            else:
                raise NotImplementedError(f"unknown ref_img_sampling method {method!r}")

            if not self.test_mode:
                if not ref_img_ids:
                    ref_img_ids = [img_id] * num_ref_imgs
                if len(ref_img_ids) < num_ref_imgs:
                    ref_img_ids += [ref_img_ids[0]] * (num_ref_imgs - len(ref_img_ids))

            ref_img_infos = []
            for ref_img_id in ref_img_ids:
                info = dict(self.coco.load_imgs([ref_img_id])[0])
                info["filename"] = info["file_name"]
                ref_img_infos.append(info)
            ref_img_infos = sorted(ref_img_infos, key=lambda i: i["frame_id"])

        return [img_info, *ref_img_infos] if return_key_img else ref_img_infos

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
