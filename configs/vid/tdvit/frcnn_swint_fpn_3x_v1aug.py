# The Swin-T baseline with v1's single-frame pipeline -- a test of where the
# paper's Swin-T row (47.1 AP, 77.2 AP50) comes from. v1's Swin-T VID config
# (configs/vid/single_frame/faster_rcnn/faster_rcnn_swint_fpn_3x.py on the v1
# branch) resized to 600 on the shorter side and flipped, nothing more; TDViT
# trained with Swin's multi-scale and crop augmentation, as the configs here
# do for both models. Everything else is frcnn_swint_fpn_3x.py's.
_base_ = ["./frcnn_swint_fpn_3x.py"]

dataset_type = "ImagenetVIDDataset"
data_root = "data/ILSVRC/"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type="LoadMultiImagesFromFile"),
    dict(type="SeqLoadAnnotations", with_bbox=True, with_mask=False),
    dict(type="SeqResize", img_scale=(1000, 600), keep_ratio=True),
    dict(type="SeqRandomFlip", share_params=True, flip_ratio=0.5),
    dict(type="SeqNormalize", **img_norm_cfg),
    dict(type="SeqPad", size_divisor=16),
    dict(type="VideoCollect", keys=["img", "gt_bboxes", "gt_labels"]),
    dict(type="ConcatVideoReferences"),
    dict(type="SeqDefaultFormatBundle", ref_prefix="ref"),
]
data = dict(
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root + "annotations/imagenet_vid_train.json",
            img_prefix=data_root + "Data/VID",
            ref_img_sampler=dict(num_ref_imgs=0, frame_range=0),  # the key frame alone
            pipeline=train_pipeline,
        ),
        dict(
            type=dataset_type,
            load_as_video=False,
            ann_file=data_root + "annotations/imagenet_det_30plus1cls.json",
            img_prefix=data_root + "Data/DET",
            ref_img_sampler=dict(num_ref_imgs=0, frame_range=0),
            pipeline=train_pipeline,
        ),
    ],
)
