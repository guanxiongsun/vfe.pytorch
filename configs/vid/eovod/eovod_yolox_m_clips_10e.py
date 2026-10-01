# EOVOD on YOLOX-M with YOLOX's full recipe and the location prior together.
# Stage A's recipe (eovod_yolox_m_10e.py: COCO weights, Mosaic, RandomAffine,
# MixUp, HSV, multi-scale, 10 epochs at batch 32, YOLOX's schedule, EMA) on
# clips: every transform draws once for the whole clip (SeqShared), so a key
# frame and its two support frames share the mosaic layout, warp, mixed-in
# clip, colour and flip, and every pixel keeps its previous frames. Each GPU
# trains 8 clips at once, so BatchNorm trains as in stage A; clip_multiscale
# resizes whole clips. The prior is the before-PAFPN design, which trained
# jointly gained 1.7 AP on clips without Mosaic. The control, the same with
# every step plain, is eovod_yolox_m_clips_10e_plain.py.
_base_ = ['./eovod_yolox_m_10e.py']

model = dict(
    clip_multiscale=True,
    location_prior=dict(queries='all', train_keys='random', train_random_keys=2000,
                        train_plain_prob=0.0),
    memory=dict(update=True, capacity=20000, num_keys=2000, write_per_frame=1000),
    aggregator=dict(position='backbone', levels=[1, 2], shared=False, branches='cls',
                    backbone_strides=(8, 16, 32)),
)

img_scale = (640, 640)
pad_val = dict(img=(114.0, 114.0, 114.0))
dataset_type = 'ImagenetVIDDataset'
data_root = 'data/ILSVRC/'
clip_load_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True),
]
clip_train_pipeline = [
    dict(type='SeqShared', transform=dict(type='Mosaic', img_scale=img_scale, pad_val=114.0)),
    dict(type='SeqShared', transform=dict(
        type='RandomAffine', scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2))),
    dict(type='SeqShared', transform=dict(
        type='MixUp', img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=114.0)),
    dict(type='SeqShared', transform=dict(type='YOLOXHSVRandomAug')),
    dict(type='SeqShared', transform=dict(type='RandomFlip', flip_ratio=0.5)),
    dict(type='SeqShared', transform=dict(type='Resize', img_scale=img_scale, keep_ratio=True)),
    dict(type='SeqShared', transform=dict(type='Pad', pad_to_square=True, pad_val=pad_val)),
    dict(type='SeqShared', transform=dict(
        type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False)),
    dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]
data = dict(
    workers_per_gpu=12,
    train=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=[
            dict(
                type=dataset_type,
                ann_file=data_root + 'annotations/imagenet_vid_train.json',
                img_prefix=data_root + 'Data/VID',
                ref_img_sampler=dict(
                    num_ref_imgs=2,
                    frame_range=9,
                    filter_key_img=True,
                    method='bilateral_uniform'),
                pipeline=clip_load_pipeline),
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
                pipeline=clip_load_pipeline),
        ],
        pipeline=clip_train_pipeline),
)
