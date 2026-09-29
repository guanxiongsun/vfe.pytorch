# Stage B of EOVOD on YOLOX-M: the location prior's training, from the stage-A
# model (pass it with --load-from: a YOLOX-M trained alone by
# eovod_yolox_m_10e.py or _80e.py). Video clips of a key frame and two support
# frames, letterboxed to 640, with the FCOS work's training prior and schedule
# (eovod_fcos_r50_fpn_3x.py): 3 epochs at batch 8, lr 1e-3 then 1e-4 after the
# second. EOVOD trains one key frame per GPU, too few for YOLOX's batch
# statistics, so every BatchNorm keeps the stage-A statistics (freeze_norm);
# that also allows gradient accumulation (4 GPUs x 2 = batch 8).
_base_ = ['./eovod_yolox_m_80e.py']

model = dict(freeze_norm=True)

pad_val = dict(img=(114.0, 114.0, 114.0))
img_scale = (640, 640)
dataset_type = 'ImagenetVIDDataset'
data_root = 'data/ILSVRC/'
clip_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=img_scale, keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqPad', size=img_scale, pad_val=pad_val),
    dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]
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
            pipeline=clip_pipeline),
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
            pipeline=clip_pipeline),
    ],
)

optimizer = dict(
    _delete_=True,
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    _delete_=True,
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[2])
custom_hooks = [dict(type='NumClassCheckHook')]
max_epochs = 3
total_epochs = max_epochs
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=max_epochs)
load_from = None
