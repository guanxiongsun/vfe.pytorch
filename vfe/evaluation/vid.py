"""ImageNet VID detection metric (AP50 overall and per motion speed).

Port of ``mmdet/datasets/mamba/vid_eval.py``, which follows the FGFA /
ImageNet VID devkit protocol. Its published numbers (MAMBA 83.8, STPN 85.2)
depend on several details that look like mistakes and must not be "fixed":

* **Box convention.** Boxes get +1 on x2/y2 *and* the IoU adds 1 to widths
  and heights again (maskrcnn-benchmark's ``boxlist_iou``). IoU is computed in
  float32, and ties are decided by exact float comparison.
* **Motion buckets.** A ground-truth box is "ignored" in a bucket when its
  motion IoU (``vid_motion_iou.npz``) lies outside the bucket's range.
  Detections matched only to ignored boxes, or to none, are weighted
  fractionally as false positives, using ``empty_weight``, the share of all
  motion entries inside the range.
* **Placeholder zeros.** A frame with no objects has one motion entry of 0 in
  the table. It never matches a box, but it *is* counted in ``empty_weight``.

Not ported: ``eval_proposals_vid`` (``box_only``; nothing calls it) and a
working ``motion_specific=False`` path (it raised ``NameError`` in the
original, so no published number used it).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

__all__ = ["do_vid_evaluation", "load_motion_ious", "MOTION_BUCKETS", "VID_CLASS_NAMES"]

logger = logging.getLogger(__name__)

MOTION_IOU_FILE = Path(__file__).with_name("vid_motion_iou.npz")

# Evaluated in this order; "all" comes first because per-class AP is reported
# from it.
MOTION_BUCKETS = (("all", (0.0, 1.0)), ("fast", (0.0, 0.7)), ("medium", (0.7, 0.9)),
                  ("slow", (0.9, 1.0)))

# Index 0 is background; labels in this metric are 1-based.
VID_CLASS_NAMES = (
    "__background__", "airplane", "antelope", "bear", "bicycle", "bird", "bus", "car", "cattle",
    "dog", "domestic_cat", "elephant", "fox", "giant_panda", "hamster", "horse", "lion", "lizard",
    "monkey", "motorcycle", "rabbit", "red_panda", "sheep", "snake", "squirrel", "tiger", "train",
    "turtle", "watercraft", "whale", "zebra",
)


def load_motion_ious(path: str | Path = MOTION_IOU_FILE) -> tuple[np.ndarray, np.ndarray]:
    """``(values, offsets)``: frame ``i`` (0-based, val image id ``i + 1``) has
    motion IoUs ``values[offsets[i]:offsets[i + 1]]``, one per ground-truth box."""
    with np.load(path, allow_pickle=False) as data:
        return data["values"], data["offsets"]


def _boxlist_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """maskrcnn-benchmark's ``boxlist_iou`` in float32 numpy: pairwise IoU with
    +1 added to widths and heights (``TO_REMOVE = 1``)."""
    boxes1 = boxes1.astype(np.float32, copy=False)
    boxes2 = boxes2.astype(np.float32, copy=False)
    area1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
    area2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = np.maximum(rb - lt + 1, 0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter)


def _empty_weight(values: np.ndarray, motion_range: tuple[float, float]):
    """Share of motion entries (placeholders included) inside ``motion_range``,
    or 0 when that share is 1."""
    inside = np.sum((values >= motion_range[0]) & (values <= motion_range[1]))
    weight = inside / float(len(values))
    return 0 if weight == 1 else weight


def _prec_rec(frames, values, offsets, motion_range, iou_thresh=0.5):
    """Precision / recall per class for one motion bucket. ``frames`` holds
    ``(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)`` per val
    frame, in image-id order."""
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    pred_ignore = defaultdict(list)
    empty_weight = _empty_weight(values, motion_range)

    for index, (pred_bbox, pred_label, pred_score, gt_bbox, gt_label) in enumerate(frames):
        gt_ignore = np.zeros(len(gt_bbox))
        motion_iou = values[offsets[index]:offsets[index + 1]]
        if len(motion_iou):
            for gt_index in range(len(gt_bbox)):
                outside = (motion_iou[gt_index] < motion_range[0]
                           or motion_iou[gt_index] > motion_range[1])
                gt_ignore[gt_index] = 1 if outside else 0

        for label in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask = pred_label == label
            pred_bbox_l = pred_bbox[pred_mask]
            pred_score_l = pred_score[pred_mask]
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask = gt_label == label
            gt_bbox_l = gt_bbox[gt_mask]
            gt_ignore_l = gt_ignore[gt_mask]

            # Python sum, as in the original: it makes n_pos a float.
            n_pos[label] += gt_bbox_l.shape[0] - sum(gt_ignore_l)
            score[label].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[label].extend((0,) * pred_bbox_l.shape[0])
                pred_ignore[label].extend((empty_weight,) * pred_bbox_l.shape[0])
                continue

            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = _boxlist_iou(pred_bbox_l, gt_bbox_l)

            num_obj, num_gt_obj = iou.shape
            selected = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for j in range(num_obj):
                iou_match = iou_thresh
                iou_match_ig = -1
                iou_match_nig = -1
                arg_match = -1
                for k in range(num_gt_obj):
                    if (gt_ignore_l[k] == 1) & (iou[j, k] > iou_match_ig):
                        iou_match_ig = iou[j, k]
                    if (gt_ignore_l[k] == 0) & (iou[j, k] > iou_match_nig):
                        iou_match_nig = iou[j, k]
                    if selected[k] or iou[j, k] < iou_match:
                        continue
                    if iou[j, k] == iou_match:
                        # An exact tie prefers a non-ignored box.
                        if arg_match < 0 or gt_ignore_l[arg_match]:
                            arg_match = k
                    else:
                        arg_match = k
                    iou_match = iou[j, k]

                if arg_match >= 0:
                    match[label].append(1)
                    pred_ignore[label].append(gt_ignore_l[arg_match])
                    selected[arg_match] = True
                else:
                    if iou_match_nig > iou_match_ig:
                        pred_ignore[label].append(0)
                    elif iou_match_ig > iou_match_nig:
                        pred_ignore[label].append(1)
                    else:
                        pred_ignore[label].append(sum(gt_ignore_l) / float(num_gt_obj))
                    match[label].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
    for label in n_pos:
        score_l = np.array(score[label])
        match_l = np.array(match[label], dtype=np.int8)
        pred_ignore_l = np.array(pred_ignore[label])

        order = score_l.argsort()[::-1]
        match_l = match_l[order]
        pred_ignore_l = pred_ignore_l[order]

        tps = np.logical_and(match_l == 1, np.logical_not(pred_ignore_l == 1))
        fps = np.logical_and(match_l == 0, np.logical_not(pred_ignore_l == 1))
        # Ignore weights of 0 mean "count fully"; fractional ones scale the FP.
        pred_ignore_l[pred_ignore_l == 0] = 1
        fps = fps * pred_ignore_l

        tp = np.cumsum(tps)
        fp = np.cumsum(fps)
        prec[label] = tp / (fp + tp + np.spacing(1))
        if n_pos[label] > 0:
            rec[label] = tp / n_pos[label]
    return prec, rec


def _average_precision(prec, rec) -> np.ndarray:
    """Area under the monotone precision envelope; NaN for classes with no
    detections or no positives."""
    ap = np.empty(len(prec))
    for label in range(len(prec)):
        if prec[label] is None or rec[label] is None:
            ap[label] = np.nan
            continue
        mpre = np.concatenate(([0], np.nan_to_num(prec[label]), [0]))
        mrec = np.concatenate(([0], rec[label], [1]))
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap[label] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def do_vid_evaluation(dataset, predictions, motion_iou_file: str | Path = MOTION_IOU_FILE
                      ) -> dict[str, float]:
    """AP50 of ``predictions`` on ``dataset`` (the ImageNet VID val set).

    Args:
        dataset: provides ``img_ids`` (in prediction order), ``coco.load_imgs``
            and ``get_ann_info``. Image ids must be 1..N, as in the val
            annotations, since they index the motion-IoU table.
        predictions: one entry per ``dataset.img_ids``, each a list of
            ``num_classes`` arrays of ``(n, 5)`` = ``[x1, y1, x2, y2, score]``.

    Returns:
        ``{"all", "fast", "medium", "slow", <class name>: AP50}``; per-class
        values come from the "all" bucket.
    """
    num_frames = len(dataset)
    if len(predictions) != num_frames:
        raise ValueError(f"{len(predictions)} predictions for {num_frames} frames")

    frames = [None] * num_frames
    for img_index, prediction in enumerate(predictions):
        img_id = dataset.img_ids[img_index]
        if not 1 <= img_id <= num_frames:
            raise ValueError(f"image id {img_id} outside 1..{num_frames}; the motion-IoU "
                             "table is indexed by val image id")
        pred_labels = []
        for cls_ind, bbox in enumerate(prediction):
            if len(bbox) > 0:
                pred_labels.extend([cls_ind + 1] * len(bbox))
        bboxes = np.vstack(prediction)
        gt = dataset.get_ann_info(dataset.coco.load_imgs(img_id)[0])
        frames[img_id - 1] = (
            bboxes[:, :4].astype(np.float32),
            np.asarray(pred_labels, dtype=np.int64),
            bboxes[:, -1],
            np.asarray(gt["bboxes"], dtype=np.float32),
            np.asarray(gt["labels"] + 1, dtype=np.int64),
        )

    values, offsets = load_motion_ious(motion_iou_file)
    if len(offsets) - 1 != num_frames:
        raise ValueError(f"motion-IoU table has {len(offsets) - 1} frames, dataset {num_frames}")

    results: dict[str, float] = {}
    per_class_ap = None
    for name, motion_range in MOTION_BUCKETS:
        logger.info("evaluating motion IoU range %s - %s", *motion_range)
        prec, rec = _prec_rec(frames, values, offsets, motion_range)
        ap = _average_precision(prec, rec)
        results[name] = np.nanmean(ap)
        if per_class_ap is None:
            per_class_ap = ap
    for label, ap in enumerate(per_class_ap):
        if label > 0:
            results[VID_CLASS_NAMES[label]] = ap
    return results
