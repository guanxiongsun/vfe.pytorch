# TDViT-T with joint attention, trained with v1's plain pipeline (shorter side
# 600, flips) instead of Swin's multi-scale and crop augmentation -- the
# recipe on which Swin-T scores higher (51.1 against 50.2 AP,
# frcnn_swint_fpn_3x_v1aug.py). The references, one per stage, as before.
_base_ = ["./tdvit_t_joint_frcnn_fpn_3x.py"]

temporal_dilations = (4, 8, 16, 32)
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
ref_img_sampler = dict(
    num_ref_imgs=len(temporal_dilations),
    frame_range=list(temporal_dilations),
    filter_key_img=True,
    method="stagewise_uniform",
)
data = dict(
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root + "annotations/imagenet_vid_train.json",
            img_prefix=data_root + "Data/VID",
            ref_img_sampler=ref_img_sampler,
            pipeline=train_pipeline,
        ),
        dict(
            type=dataset_type,
            load_as_video=False,
            ann_file=data_root + "annotations/imagenet_det_30plus1cls.json",
            img_prefix=data_root + "Data/DET",
            ref_img_sampler=dict(ref_img_sampler, filter_key_img=False),
            pipeline=train_pipeline,
        ),
    ],
)
