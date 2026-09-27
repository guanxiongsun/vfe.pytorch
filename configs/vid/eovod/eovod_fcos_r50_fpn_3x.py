# EOVOD on FCOS / ResNet-50-FPN on the paper's FCOS recipe: 3 epochs at batch
# 4, lr 1e-3 for two epochs then 1e-4, shorter side 600. The paper's reported
# model is the ResNet-101 3x config next to this one.
_base_ = [
    '../../_base_/models/vid/fcos_r50_fpn.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_1x.py',
]

is_video_model = True

model = dict(
    type='EOVOD',
    # A detection above score_thr is *validated*: it feeds both priors and
    # the memory. box_ratio shrinks the prior boxes (0.8 per the paper);
    # train_jitter perturbs the ground truth that stands in for them.
    location_prior=dict(score_thr=0.5, box_ratio=0.8, train_jitter=0.1),
    # Full detection on every 7th frame; in between only the levels from the
    # lowest one that produced a validated box upward. None disables it.
    size_prior=dict(interval=7),
    memory=dict(capacity=4096, num_keys=1024, write_per_frame=512),
    aggregator=dict(num_heads=16, shared=True),
    ref_chunk_size=4,
)

# dataset settings: caffe-style normalisation (BGR, mean subtraction only), as
# the detectron backbone expects; padding to 32 for P7.
dataset_type = 'ImagenetVIDDataset'
data_root = 'data/ILSVRC/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(
        type='VideoCollect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList'),
]
# Test frames stay in order (no shuffle_video_frames): the location prior
# comes from the previous frame, and the size prior counts frames.
test_data = dict(
    type=dataset_type,
    ann_file=data_root + 'annotations/imagenet_vid_val.json',
    img_prefix=data_root + 'Data/VID',
    ref_img_sampler=dict(
        num_ref_imgs=14,
        frame_range=[-7, 7],
        method='test_with_adaptive_stride'),
    pipeline=test_pipeline,
    test_mode=True,
)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=[
        dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/imagenet_vid_train.json',
            img_prefix=data_root + 'Data/VID',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=9,
                filter_key_img=True,
                method='bilateral_uniform'),
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            load_as_video=False,
            ann_file=data_root + 'annotations/imagenet_det_30plus1cls.json',
            img_prefix=data_root + 'Data/DET',
            ref_img_sampler=dict(
                num_ref_imgs=2,
                frame_range=0,
                filter_key_img=False,
                method='bilateral_uniform'),
            pipeline=train_pipeline),
    ],
    val=test_data,
    test=test_data,
)

# optimizer: the paper's (one image per process; batch 4 = 4 processes x
# --accumulate 1). The warmup and clipping come from the released code.
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step', warmup='linear', warmup_iters=500, warmup_ratio=1.0 / 3, step=[2])
# runtime settings
total_epochs = 3
checkpoint_config = dict(interval=3)
# COCO-style AP is what EOVOD's paper reports; the VID metric is what MAMBA
# and STPN report. Both are computed.
evaluation = dict(metric=['bbox'], vid_style=True, coco_style=True, interval=total_epochs)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
