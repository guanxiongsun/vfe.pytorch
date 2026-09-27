"""COCO-style box evaluation of a VID dataset, on pycocotools.

The ImageNet VID protocol (:mod:`vfe.evaluation.vid`) reports AP50 with a
motion-speed breakdown. One-stage VID papers, EOVOD among them, instead report
COCO's AP averaged over IoU 0.5:0.95, with AP50 / AP75 and the small / medium
/ large split -- what ``CocoDataset.evaluate('bbox')`` gave in mmdet. The two
are not comparable, so a dataset can report both.

Keys follow mmdet: ``bbox_mAP``, ``bbox_mAP_50``, ``bbox_mAP_75``,
``bbox_mAP_s``, ``bbox_mAP_m``, ``bbox_mAP_l``.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

__all__ = ["do_coco_bbox_evaluation", "results_to_coco"]

METRIC_NAMES = ("mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l")


def results_to_coco(dataset, results) -> list[dict]:
    """``bbox2result`` outputs, one per ``dataset.img_ids`` -> COCO result dicts
    (``xywh`` boxes, category ids from ``dataset.cat_ids``)."""
    if len(results) != len(dataset.img_ids):
        raise ValueError(f"{len(results)} results for {len(dataset.img_ids)} images")
    out = []
    for img_id, result in zip(dataset.img_ids, results, strict=True):
        for label, bboxes in enumerate(result):
            bboxes = np.asarray(bboxes)
            for x1, y1, x2, y2, score in bboxes.tolist():
                out.append(dict(
                    image_id=int(img_id),
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    score=float(score),
                    category_id=int(dataset.cat_ids[label]),
                ))
    return out


def do_coco_bbox_evaluation(dataset, results, classwise: bool = False,
                            quiet: bool = False) -> dict[str, float]:
    """COCO box metrics of ``results`` on ``dataset`` (an
    :class:`~vfe.datasets.ImagenetVIDDataset`, whose annotation file is COCO
    JSON with an extra ``videos`` table pycocotools ignores)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(dataset.ann_file)
    # COCOeval reads these two fields unconditionally; the VID converter
    # writes them, but a hand-made file may not.
    for ann in coco_gt.dataset["annotations"]:
        ann.setdefault("iscrowd", 0)
        if "area" not in ann:
            ann["area"] = float(ann["bbox"][2] * ann["bbox"][3])
    coco_gt.createIndex()

    coco_results = results_to_coco(dataset, results)
    metrics = {}
    if not coco_results:
        for name in METRIC_NAMES:
            metrics[f"bbox_{name}"] = 0.0
        return metrics

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = list(dataset.img_ids)
    coco_eval.params.catIds = list(dataset.cat_ids)
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    if not quiet:
        print(captured.getvalue(), end="")

    for i, name in enumerate(METRIC_NAMES):
        metrics[f"bbox_{name}"] = float(f"{coco_eval.stats[i]:.4f}")

    if classwise:
        # precision: (iou, recall, cls, area, max_dets); per-class AP over IoUs.
        precision = coco_eval.eval["precision"]
        for idx in range(len(dataset.cat_ids)):
            p = precision[:, :, idx, 0, -1]
            p = p[p > -1]
            metrics[f"bbox_AP_{dataset.CLASSES[idx]}"] = float(p.mean()) if p.size else float("nan")
    return metrics
