"""COCO-style box evaluation on a synthetic COCO-VID annotation file."""

import json

import numpy as np
import pytest

from vfe.datasets.imagenet_vid import ImagenetVIDDataset
from vfe.evaluation import results_to_coco

CLASSES = ImagenetVIDDataset.CLASSES


@pytest.fixture
def dataset(tmp_path):
    ann = dict(
        categories=[dict(id=i + 1, name=name) for i, name in enumerate(CLASSES)],
        videos=[dict(id=1, name="v")],
        images=[
            dict(id=1, file_name="v/000000.JPEG", width=100, height=80, video_id=1, frame_id=0,
                 is_vid_train_frame=False),
            dict(id=2, file_name="v/000001.JPEG", width=100, height=80, video_id=1, frame_id=1,
                 is_vid_train_frame=False),
        ],
        annotations=[
            dict(id=1, image_id=1, category_id=3, bbox=[10, 10, 30, 20], area=600,
                 iscrowd=0, instance_id=1),
            dict(id=2, image_id=1, category_id=7, bbox=[50, 20, 40, 40], area=1600,
                 iscrowd=0, instance_id=2),
            dict(id=3, image_id=2, category_id=3, bbox=[12, 11, 30, 20], area=600,
                 iscrowd=0, instance_id=1),
        ],
    )
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(ann))
    return ImagenetVIDDataset(str(path), test_mode=True, pipeline=[], ref_img_sampler=None)


def perfect_results(dataset):
    results = []
    for img_info in dataset.data_infos:
        ann = dataset.get_ann_info(img_info)
        per_class = [np.zeros((0, 5), dtype=np.float32) for _ in CLASSES]
        for box, label in zip(ann["bboxes"], ann["labels"], strict=True):
            det = np.concatenate([box, [1.0]]).astype(np.float32)[None]
            per_class[label] = np.concatenate([per_class[label], det])
        results.append(per_class)
    return results


def test_results_to_coco_uses_dataset_ids_and_xywh(dataset):
    coco = results_to_coco(dataset, perfect_results(dataset))
    assert len(coco) == 3
    first = coco[0]
    assert first["image_id"] == 1 and first["category_id"] == 3
    assert first["bbox"] == pytest.approx([10, 10, 30, 20])


def test_perfect_detections_score_full_ap_and_none_score_zero(dataset):
    metrics = dataset.evaluate(perfect_results(dataset), vid_style=False, coco_style=True)
    assert metrics["bbox_mAP"] == 1.0 and metrics["bbox_mAP_50"] == 1.0
    empty = [[np.zeros((0, 5), dtype=np.float32) for _ in CLASSES] for _ in dataset.data_infos]
    assert dataset.evaluate(empty, vid_style=False, coco_style=True)["bbox_mAP"] == 0.0
    with pytest.raises(ValueError):
        dataset.evaluate(empty, vid_style=False, coco_style=False)


def test_wrong_classes_score_zero(dataset):
    results = perfect_results(dataset)
    shifted = [[*per_class[1:], per_class[0]] for per_class in results]  # every label off by one
    assert dataset.evaluate(shifted, vid_style=False, coco_style=True)["bbox_mAP"] == 0.0
