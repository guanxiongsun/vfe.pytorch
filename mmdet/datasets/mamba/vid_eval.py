import numpy as np
import os
import scipy.io as sio
import torch
from collections import defaultdict

from .bounding_box import BoxList
from .boxlist_ops import boxlist_iou


def do_vid_evaluation(
    dataset,
    predictions,
    output_folder=None,
    box_only=False,
    motion_specific=True,
):
    # create list to hold results
    pred_boxlists = [None] * len(dataset)
    gt_boxlists = [None] * len(dataset)
    for img_index, prediction in enumerate(predictions):
        # img_info = dataset.get_img_info(image_id)
        # image_width = img_info["width"]
        # image_height = img_info["height"]
        # prediction = prediction.resize((image_width, image_height))
        # pred_boxlists.append(prediction)
        img_id = dataset.img_ids[img_index]
        img_info = dataset.coco.load_imgs(img_id)
        width = img_info[0]["width"]
        height = img_info[0]["height"]
        # convert prediction to boxlist
        # [n, 5]
        labels = []
        for cls_ind, bbox in enumerate(prediction):
            if len(bbox) > 0:
                cat_id = cls_ind + 1
                labels.extend([cat_id] * len(bbox))
        labels = torch.as_tensor(labels, dtype=torch.int64)
        bboxes = np.vstack(prediction)
        scores = torch.as_tensor(bboxes[:, -1])
        bboxes = torch.as_tensor(bboxes[:, :4])
        prediction_boxlist = BoxList(bboxes, (width, height), mode="xyxy")
        prediction_boxlist.add_field("labels", labels)
        prediction_boxlist.add_field("scores", scores)
        pred_boxlists[img_id - 1] = prediction_boxlist

        # get annotations
        gt_coco = dataset.get_ann_info(img_info[0])
        # convert ground truth to boxlist
        bboxes = gt_coco['bboxes']
        labels = gt_coco['labels'] + 1
        labels = torch.as_tensor(labels, dtype=torch.int64)
        gt_boxlist = BoxList(bboxes, (width, height), mode="xyxy")
        gt_boxlist.add_field("labels", labels)

        gt_boxlists[img_id - 1] = gt_boxlist

    if box_only:
        result = eval_proposals_vid(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=0.5,
        )
        result_str = "Recall: {:.4f}".format(result["recall"])
        print(result_str)
        if output_folder:
            with open(os.path.join(output_folder, "proposal_motion_eval_metrics.txt"), "w") as fid:
                fid.write(result_str)
        return result

    if motion_specific:
        motion_ranges = [[0.0, 1.0], [0.0, 0.7], [0.7, 0.9], [0.9, 1.0]]
        motion_name = ["all", "fast", "medium", "slow"]
    else:
        motion_ranges = [[0.0, 1.0]]
        motion_name = ["all"]
    result = eval_detection_vid(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5,
        motion_ranges=motion_ranges,
        motion_specific=motion_specific,
        use_07_metric=False,
    )
    result_dict = {}
    result_str = ""
    template_str = "AP50 | motion={:>6s} = {:0.4f}\n"
    for motion_index in range(len(motion_name)):
        result_str += template_str.format(
            motion_name[motion_index], result[motion_index]["map"]
        )
        result_dict.update(
            {motion_name[motion_index] : result[motion_index]["map"]})

    result_str += "Category AP:\n"
    map_class_id_to_class_name = [
        "__background__",  # always index 0
        "airplane",
        "antelope",
        "bear",
        "bicycle",
        "bird",
        "bus",
        "car",
        "cattle",
        "dog",
        "domestic_cat",
        "elephant",
        "fox",
        "giant_panda",
        "hamster",
        "horse",
        "lion",
        "lizard",
        "monkey",
        "motorcycle",
        "rabbit",
        "red_panda",
        "sheep",
        "snake",
        "squirrel",
        "tiger",
        "train",
        "turtle",
        "watercraft",
        "whale",
        "zebra",
    ]

    for i, ap in enumerate(result[0]["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(map_class_id_to_class_name[i], ap)

        result_dict.update({map_class_id_to_class_name[i] : ap})

    # print("\n" + result_str)
    if output_folder:
        with open(os.path.join(output_folder, "motion_eval_metrics.txt"), "w") as fid:
            fid.write(result_str)

    return result_dict


def eval_proposals_vid(pred_boxlists, gt_boxlists, iou_thresh=0.5, limit=300):
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."

    gt_overlaps = []
    num_pos = 0
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        inds = pred_boxlist.get_field("objectness").sort(descending=True)[1]
        pred_boxlist = pred_boxlist[inds]

        if len(pred_boxlist) > limit:
            pred_boxlist = pred_boxlist[:limit]

        num_pos += len(gt_boxlist)

        if len(gt_boxlist) == 0:
            continue

        if len(pred_boxlist) == 0:
            continue

        overlaps = boxlist_iou(pred_boxlist, gt_boxlist)

        _gt_overlaps = torch.zeros(len(gt_boxlist))
        for j in range(min(len(pred_boxlist), len(gt_boxlist))):
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0

            box_ind = argmax_overlaps[gt_ind]

            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr

            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    recall = (gt_overlaps >= iou_thresh).float().sum() / float(num_pos)

    return {"recall": recall}


def eval_detection_vid(
    pred_boxlists,
    gt_boxlists,
    iou_thresh=0.5,
    motion_ranges=[[0.0, 0.7], [0.7, 0.9], [0.9, 1.0]],
    motion_specific=False,
    use_07_metric=False,
):
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."

    if motion_specific:
        motion_iou_file = "mmdet/datasets/mamba/vid_groundtruth_motion_iou.mat"
        motion_ious = sio.loadmat(motion_iou_file)
        motion_ious = np.array(
            [
                [
                    motion_ious["motion_iou"][i][0][j][0]
                    if len(motion_ious["motion_iou"][i][0][j]) != 0
                    else 0
                    for j in range(len(motion_ious["motion_iou"][i][0]))
                ]
                for i in range(len(motion_ious["motion_iou"]))
            ]
        )
    else:
        motion_ious = None

    motion_ap = defaultdict(dict)
    for motion_index, motion_range in enumerate(motion_ranges):
        print(
            "Evaluating motion iou range {} - {}".format(
                motion_range[0], motion_range[1]
            )
        )
        prec, rec = calc_detection_vid_prec_rec(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            motion_ious=motion_ious,
            iou_thresh=iou_thresh,
            motion_range=motion_range,
        )
        ap = calc_detection_vid_ap(prec, rec, use_07_metric=use_07_metric)
        motion_ap[motion_index] = {"ap": ap, "map": np.nanmean(ap)}
    return motion_ap


def args_check(motion_ious, gt_boxlists, motion_range):
    if motion_ious is None:
        motion_ious = [None] * len(gt_boxlists)
        empty_weight = 0
    else:
        all_motion_iou = np.concatenate(motion_ious, axis=0)
        empty_weight = sum(
            [
                (all_motion_iou[i] >= motion_range[0])
                & (all_motion_iou[i] <= motion_range[1])
                for i in range(len(all_motion_iou))
            ]
        ) / float(len(all_motion_iou))
        if empty_weight == 1:
            empty_weight = 0
    return motion_ious, empty_weight, all_motion_iou


def get_gt_ignore(motion_iou, gt_bbox, motion_range, gt_ignore):
    for gt_index, gt in enumerate(gt_bbox):
        if motion_iou:
            if (
                motion_iou[gt_index] < motion_range[0]
                or motion_iou[gt_index] > motion_range[1]
            ):
                gt_ignore[gt_index] = 1
            else:
                gt_ignore[gt_index] = 0

    return gt_ignore


def calc_detection_vid_prec_rec(
    gt_boxlists, pred_boxlists, motion_ious, iou_thresh=0.5, motion_range=[0.0, 1.0]
):
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    pred_ignore = defaultdict(list)

    motion_ious, empty_weight, _ = args_check(motion_ious, gt_boxlists, motion_range)

    for gt_boxlist, pred_boxlist, motion_iou in zip(
        gt_boxlists, pred_boxlists, motion_ious
    ):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_ignore = np.zeros(len(gt_bbox))

        gt_ignore = get_gt_ignore(motion_iou, gt_bbox, motion_range, gt_ignore)

        for _l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == _l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]

            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == _l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_ignore_l = gt_ignore[gt_mask_l]

            n_pos[_l] += gt_bbox_l.shape[0] - sum(gt_ignore_l)
            score[_l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[_l].extend((0,) * pred_bbox_l.shape[0])
                pred_ignore[_l].extend((empty_weight,) * pred_bbox_l.shape[0])
                continue

            # VID evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()

            num_obj, num_gt_obj = iou.shape

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for j in range(0, num_obj):
                iou_match = iou_thresh
                iou_match_ig = -1
                iou_match_nig = -1
                arg_match = -1
                for k in range(0, num_gt_obj):
                    if (gt_ignore_l[k] == 1) & (iou[j, k] > iou_match_ig):
                        iou_match_ig = iou[j, k]
                    if (gt_ignore_l[k] == 0) & (iou[j, k] > iou_match_nig):
                        iou_match_nig = iou[j, k]
                    if selec[k] or iou[j, k] < iou_match:
                        continue
                    if iou[j, k] == iou_match:
                        if arg_match < 0 or gt_ignore_l[arg_match]:
                            arg_match = k
                    else:
                        arg_match = k
                    iou_match = iou[j, k]

                if arg_match >= 0:
                    match[_l].append(1)
                    pred_ignore[_l].append(gt_ignore_l[arg_match])
                    selec[arg_match] = True
                else:
                    if iou_match_nig > iou_match_ig:
                        pred_ignore[_l].append(0)
                    elif iou_match_ig > iou_match_nig:
                        pred_ignore[_l].append(1)
                    else:
                        pred_ignore[_l].append(sum(gt_ignore_l) / float(num_gt_obj))
                    match[_l].append(0)
                    # pred_ignore[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    print(n_pos)
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for _l in n_pos.keys():
        score_l = np.array(score[_l])
        match_l = np.array(match[_l], dtype=np.int8)
        pred_ignore_l = np.array(pred_ignore[_l])

        order = score_l.argsort()[::-1]
        match_l = match_l[order]
        pred_ignore_l = pred_ignore_l[order]

        tps = np.logical_and(match_l == 1, np.logical_not(pred_ignore_l == 1))
        fps = np.logical_and(match_l == 0, np.logical_not(pred_ignore_l == 1))
        pred_ignore_l[pred_ignore_l == 0] = 1
        fps = fps * pred_ignore_l

        tp = np.cumsum(tps)
        fp = np.cumsum(fps)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[_l] = tp / (fp + tp + np.spacing(1))
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[_l] > 0:
            rec[_l] = tp / n_pos[_l]

    return prec, rec


def calc_detection_vid_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of VID.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in VID.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for _l in range(n_fg_class):
        if prec[_l] is None or rec[_l] is None:
            ap[_l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[_l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[_l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[_l])[rec[_l] >= t])
                ap[_l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[_l]), [0]))
            mrec = np.concatenate(([0], rec[_l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[_l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
