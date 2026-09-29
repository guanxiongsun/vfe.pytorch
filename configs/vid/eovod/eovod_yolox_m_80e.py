# EOVOD on YOLOX-M. This config trains the detector itself, alone, with the
# paper's YOLOX recipe: 640x640, Mosaic / MixUp (and YOLOX's random affine),
# batch 32, SGD at lr 1e-3 with a cosine schedule, 80 epochs, on ImageNet VID
# (15 frames per video) and the DET subset -- still images, so EOVOD's
# aggregators sit out (EOVOD.forward_train without reference frames). The
# location prior's own training follows as a second stage, from this model.
#
# What the paper leaves open, and the choice here (each is an option):
# - Initialisation: the official COCO-trained YOLOX-M (46.9 AP), converted by
#   tools/convert_yolox_megvii.py with the 80-class classifier dropped.
# - The schedule's shape: mmdet's YOLOX policy -- 5 warmup epochs, cosine to
#   5% of the rate, then a constant rate and no Mosaic / MixUp / affine for
#   the last 15 epochs (with the L1 loss on) -- and its EMA of the weights,
#   which the checkpoints and evaluation use.
# - Test: the top 100 detections per frame (the paper's FCOS setting) above
#   YOLOX's test threshold of 0.001.
_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_1x.py',
]

is_video_model = True
img_scale = (640, 640)

model = dict(
    type='EOVOD',
    detector=dict(
        type='YOLOX',
        input_size=img_scale,
        random_size_range=(15, 25),
        random_size_interval=10,
        backbone=dict(type='CSPDarknet', deepen_factor=0.67, widen_factor=0.75),
        neck=dict(
            type='YOLOXPAFPN',
            in_channels=[192, 384, 768],
            out_channels=192,
            num_csp_blocks=2),
        bbox_head=dict(
            type='YOLOXHead', num_classes=30, in_channels=192, feat_channels=192),
        train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(
            score_thr=0.001, nms=dict(type='nms', iou_threshold=0.65), max_per_img=100)),
    # The FCOS recipe's settings (eovod_fcos_r50_fpn_3x.py explains each); the
    # location prior is trained in the second stage. With three levels the
    # size prior's neighbour margin would skip almost nothing, so this uses
    # the paper's rule (margin 0).
    location_prior=dict(
        score_thr=0.3,
        box_ratio=0.8,
        train_plain_prob=0.25,
        train_drop_prob=0.3,
        train_jitter=0.1,
        train_distractors=2),
    size_prior=dict(interval=7, margin=0),
    memory=dict(update=False, num_keys=4096),
    aggregator=dict(num_heads=16, shared=True, query_chunk=1024, branches='cls'),
    ref_chunk_size=4,
)

# dataset settings: YOLOX takes raw BGR pixels, letterboxed and padded with 114
# (a per-channel tuple: OpenCV pads a scalar into the first channel only).
dataset_type = 'ImagenetVIDDataset'
data_root = 'data/ILSVRC/'
pad_val = dict(img=(114.0, 114.0, 114.0))
load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
]
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='MixUp', img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=pad_val),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqResize', img_scale=img_scale, keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(type='SeqPad', size=img_scale, pad_val=pad_val),
    dict(type='VideoCollect', keys=['img'], meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList'),
]
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
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='MultiImageMixDataset',
        dataset=[
            dict(
                type=dataset_type,
                ann_file=data_root + 'annotations/imagenet_vid_train.json',
                img_prefix=data_root + 'Data/VID',
                pipeline=load_pipeline),
            dict(
                type=dataset_type,
                load_as_video=False,
                ann_file=data_root + 'annotations/imagenet_det_30plus1cls.json',
                img_prefix=data_root + 'Data/DET',
                pipeline=load_pipeline),
        ],
        pipeline=train_pipeline),
    val=test_data,
    test=test_data,
)

# optimizer: batch 32 = 4 processes x 8 images; YOLOX's SGD otherwise.
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(_delete_=True, grad_clip=None)
max_epochs = 80
num_last_epochs = 15
lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # epochs
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=num_last_epochs, priority=48),
    dict(type='SyncNormHook', num_last_epochs=num_last_epochs, interval=10, priority=48),
    dict(type='ExpMomentumEMAHook', resume_from=None, momentum=0.0001, priority=49),
]
total_epochs = max_epochs
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=10)
evaluation = dict(metric=['bbox'], vid_style=True, coco_style=True, interval=max_epochs)
load_from = '/projects/b5cs/vfe/checkpoints/coco/yolox_m_coco_eovod.pth'
