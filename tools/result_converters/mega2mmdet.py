import torch
from tqdm import tqdm
import argparse
import mmcv
import numpy as np
from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert images to coco format without annotations')
    parser.add_argument('mega_results', help='The root path of images')
    parser.add_argument(
        'out',
        type=str,
        help='The output annotation json file name, The save dir is in the '
        'same directory as img_path')
    args = parser.parse_args()
    return args


def cvt_mega_results_to_coco_format(mega_results, dataset):
    all_results = []
    for img_id, result in enumerate(tqdm(mega_results)):
        img_info = dataset.data_infos[img_id]
        all_results.append(
            cvt_mega_results_to_coco_format_single(
                result, img_info['width'], img_info['height']
            )
        )
    return all_results


def cvt_mega_results_to_coco_format_single(prediction, img_width, img_height):
    prediction = prediction.resize((img_width, img_height))
    coco_prediction_list = [[] for _ in range(30)]
    if len(prediction) > 0:
        for box, label, score in zip(prediction.bbox,
                                     prediction.get_field('labels'),
                                     prediction.get_field('scores')
                                     ):
            label = label.int().tolist()
            label_ind = label - 1
            bbox = box.tolist()
            bbox.append(score.tolist())
            coco_prediction_list[label_ind].append(bbox)
    coco_prediction_np = []
    for list_of_a_class in coco_prediction_list:
        coco_prediction_np.append(
            np.asarray(list_of_a_class, dtype=np.float32).reshape(-1, 5)
        )

    return coco_prediction_np


def main():
    args = parse_args()
    assert args.out.endswith(
        'pkl'), 'The output file name must be json suffix'

    # dataset settings
    dataset_type = "ImagenetVIDDataset"
    data_root = "data/ILSVRC/"

    test_pipeline = [
        dict(type="LoadImageFromFile"),
    ]
    dataset_cfg = dict(
            type=dataset_type,
            ann_file=data_root + "annotations/imagenet_vid_val.json",
            img_prefix=data_root + "Data/VID",
            ref_img_sampler=None,
            pipeline=test_pipeline,
            test_mode=True,
    )

    dataset = build_dataset(dataset_cfg)

    # 1 load mega results
    mega_results = torch.load(args.mega_results)

    # 2 convert mega results to coco results
    mega_results_coco_format = cvt_mega_results_to_coco_format(mega_results, dataset)

    # 3 dump
    # save_dir = os.path.join(args.img_path, '..', 'annotations')
    # mmcv.mkdir_or_exist(save_dir)
    # save_path = os.path.join(save_dir, args.out)
    mmcv.dump(mega_results_coco_format, args.out)
    print(f'save pkl file: {args.out}')


if __name__ == '__main__':
    main()
