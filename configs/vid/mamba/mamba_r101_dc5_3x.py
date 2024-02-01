_base_ = [
    '../../_base_/models/vid/faster_rcnn_r50_dc5.py',
    '../../_base_/datasets/vid/imagenet_vid_multi_frame.py',
    '../../_base_/default_runtime.py',
    "../../_base_/schedules/schedule_1x.py",
]


is_video_model = True

model = dict(
    type='MAMBA',
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', 
                checkpoint='torchvision://resnet101')),
        roi_head=dict(
            type='MambaRoIHead',
            bbox_head=dict(
                type='MambaBBoxHead',
                num_shared_fcs=2,
                topk=75,
                aggregator=dict(
                    type='MambaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16,
                    )))))

# dataset settings
dataset_type = "ImagenetVIDDataset"
data_root = "data/ILSVRC/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadMultiImagesFromFile"),
    dict(type="SeqLoadAnnotations", with_bbox=True, with_track=True),
    dict(type="SeqResize", img_scale=(1000, 600), keep_ratio=True),
    dict(type="SeqRandomFlip", share_params=True, flip_ratio=0.5),
    dict(type="SeqNormalize", **img_norm_cfg),
    dict(type="SeqPad", size_divisor=16),
    dict(
        type="VideoCollect", keys=["img", "gt_bboxes", "gt_labels", "gt_instance_ids"]
    ),
    dict(type="ConcatVideoReferences"),
    dict(type="SeqDefaultFormatBundle", ref_prefix="ref"),
]
test_pipeline = [
    dict(type="LoadMultiImagesFromFile"),
    dict(type="SeqResize", img_scale=(1000, 600), keep_ratio=True),
    dict(type="SeqRandomFlip", share_params=True, flip_ratio=0.0),
    dict(type="SeqNormalize", **img_norm_cfg),
    dict(type="SeqPad", size_divisor=16),
    dict(
        type="VideoCollect",
        keys=["img"],
        meta_keys=("num_left_ref_imgs", "frame_stride"),
    ),
    dict(type="ConcatVideoReferences"),
    dict(type="MultiImagesToTensor", ref_prefix="ref"),
    dict(type="ToList"),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root + "annotations/imagenet_vid_train.json",
            img_prefix=data_root + "Data/VID",
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=1000,
                filter_key_img=True,
                method="bilateral_uniform",
            ),
            pipeline=train_pipeline,
        ),
        dict(
            type=dataset_type,
            load_as_video=False,
            ann_file=data_root + "annotations/imagenet_det_30plus1cls.json",
            img_prefix=data_root + "Data/DET",
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method="bilateral_uniform",
            ),
            pipeline=train_pipeline,
        ),
    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/imagenet_vid_val.json",
        img_prefix=data_root + "Data/VID",
        ref_img_sampler=dict(
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride',
        ),
        pipeline=test_pipeline,
        test_mode=True,
        shuffle_video_frames=True,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/imagenet_vid_val.json",
        img_prefix=data_root + "Data/VID",
        ref_img_sampler=dict(
            num_ref_imgs=14,
            frame_range=[-7, 7],
            method='test_with_adaptive_stride',
        ),
        pipeline=test_pipeline,
        test_mode=True,
        shuffle_video_frames=True,
    ),
)

# optimizer
optimizer = dict(
    type="SGD",
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=1.0 / 3, step=[2]
)
# runtime settings
total_epochs = 3
checkpoint_config = dict(interval=3)
evaluation = dict(metric=["bbox"], vid_style=True, interval=total_epochs)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
