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
    # The recipe below is the one chosen on 2026-09-28 (v4 in
    # docs/eovod-plan.md). Each departure from the paper's text is marked.
    #
    # A detection above score_thr is *validated*: it feeds both priors and the
    # key set. Departure: 0.3 on the detection score (class score x
    # centerness), the released code's threshold. The paper's 0.5 on that
    # score almost never lets the prior engage with FCOS; 0.5 on the class
    # score alone (validate_on='cls_score', the paper's wording) comes second.
    # box_ratio is the paper's adjustment ratio r.
    # Departure: the train_* options. With the ground-truth boxes alone as the
    # training mask, the head learns to detect the aggregated region instead
    # of the object (E-M3), so the training prior is made to look like a
    # previous frame's detections: a quarter of the steps have none (as every
    # video's first frame; 0.5 halves the prior's gain, 0.1 costs AP75), and
    # otherwise objects are missed, moved, and joined by up to two false
    # positives.
    location_prior=dict(
        score_thr=0.3,
        box_ratio=0.8,
        train_plain_prob=0.25,
        train_drop_prob=0.3,
        train_jitter=0.1,
        train_distractors=2),
    # The paper's T: after a full detection, 7 frames run only the levels the
    # validated boxes came from, so a full detection happens every 8th frame.
    # Departure: margin=1 also runs each recorded level's neighbours; objects
    # near a range boundary otherwise fall on skipped levels (T = 7 alone cost
    # 1.1 AP, with the margin 0.1). size_prior=None gives the LPN-only model.
    size_prior=dict(interval=7, margin=1),
    # The paper's key set: pixels inside the reference frames' detections,
    # fixed for the video. Departure: num_keys reads a random 4,096 of them
    # per frame -- no AP change on the full val set, and at 3 epochs the
    # uncapped set averages ~30k keys (peak memory 52 GB, aggregation 11 ms).
    memory=dict(update=False, num_keys=4096),
    # Departure: branches='cls' feeds the aggregated maps to the
    # classification tower only; the regression tower reads the original
    # ones (aggregating both raised scores but degraded the boxes, AP75).
    aggregator=dict(num_heads=16, shared=True, query_chunk=1024, branches='cls'),
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
