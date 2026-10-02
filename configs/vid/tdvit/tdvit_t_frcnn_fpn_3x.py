# TDViT-T with Faster R-CNN on ImageNet VID: the paper's Table 2 row (49.1 AP,
# 78.5 AP50, COCO-style). docs/tdvit-plan.md has the design and the runs.
#
# Where the recipe comes from: the paper (3 epochs, the learning rate down
# 10x after the 2nd, AdamW, weight decay 0.05, Swin's augmentation and
# regularisation, one reference per stage within +-D_t), the authors' answers
# where it is silent or wrong (ImageNet-1K Swin-T weights; AdamW at 2.5e-5 for
# batch 8 -- the paper's 1e-3 is a typo; Swin's multi-scale and crop
# augmentation, every frame of a clip cut alike), and v1's Swin-T VID baseline
# (configs/vid/single_frame/faster_rcnn/faster_rcnn_swint_fpn_3x.py: the
# detector, drop path 0.2, warmup and gradient clipping).
_base_ = [
    "../../_base_/models/vid/faster_rcnn_r50_fpn.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_1x.py",
]
pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"  # noqa

is_video_model = True

# D_t of the four stages (Table 7's best): the TDTBs' memory length and
# reference reuse at test time, and the training references' ranges below.
temporal_dilations = (4, 8, 16, 32)

model = dict(
    type="TDViTDetector",
    detector=dict(
        backbone=dict(
            _delete_=True,
            type="TDViT",
            # The split scheme (Sec. 3.3, Table S1): Swin blocks first, TDTBs last.
            layout=("st", "st", "sssttt", "st"),
            temporal_dilations=temporal_dilations,
            memory_sampling="earliest",  # Table 8's default
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            convert_weights=True,
            init_cfg=dict(type="Pretrained", checkpoint=pretrained),
        ),
        neck=dict(in_channels=[96, 192, 384, 768]),
    ),
)

# dataset settings
dataset_type = "ImagenetVIDDataset"
data_root = "data/ILSVRC/"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
multi_scale = [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333),
               (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)]
# Swin's detection augmentation (STPN's pipeline), the same draws for the key
# frame and its references: TDTBs attend between frames window by window.
train_pipeline = [
    dict(type="LoadMultiImagesFromFile"),
    dict(type="SeqLoadAnnotations", with_bbox=True, with_mask=False),
    dict(type="SeqRandomFlip", share_params=True, flip_ratio=0.5),
    dict(
        type="AutoAugment",
        policies=[
            [dict(type="SeqResize", img_scale=multi_scale, multiscale_mode="value",
                  keep_ratio=True)],
            [
                dict(type="SeqResize", img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                     multiscale_mode="value", keep_ratio=True),
                dict(type="SeqRandomCrop", crop_type="absolute_range", crop_size=(384, 600),
                     allow_negative_crop=True, share_params=True),
                dict(type="SeqResize2", img_scale=multi_scale, multiscale_mode="value",
                     keep_ratio=True),
            ],
        ],
    ),
    dict(type="SeqNormalize", **img_norm_cfg),
    dict(type="SeqPad", size_divisor=16),
    dict(type="VideoCollect", keys=["img", "gt_bboxes", "gt_labels"]),
    dict(type="ConcatVideoReferences"),
    dict(type="SeqDefaultFormatBundle", ref_prefix="ref"),
]
# Frames in video order, one at a time; no references: the TDTBs' memories
# fill as each video goes on.
test_pipeline = [
    dict(type="LoadMultiImagesFromFile"),
    dict(type="SeqResize", img_scale=(1000, 600), keep_ratio=True),
    dict(type="SeqRandomFlip", share_params=True, flip_ratio=0.0),
    dict(type="SeqNormalize", **img_norm_cfg),
    dict(type="SeqPad", size_divisor=16),
    dict(type="VideoCollect", keys=["img"]),
    dict(type="ConcatVideoReferences"),
    dict(type="MultiImagesToTensor", ref_prefix="ref"),
    dict(type="ToList"),
]
# One reference per stage, from within +-D_t of the key frame (Sec. 4), in
# stage order. Still images (DET) are their own references.
ref_img_sampler = dict(
    num_ref_imgs=len(temporal_dilations),
    frame_range=list(temporal_dilations),
    filter_key_img=True,
    method="stagewise_uniform",
)
test_data = dict(
    type=dataset_type,
    ann_file=data_root + "annotations/imagenet_vid_val.json",
    img_prefix=data_root + "Data/VID",
    ref_img_sampler=dict(num_ref_imgs=0, frame_range=0),  # the key frame alone
    pipeline=test_pipeline,
    test_mode=True,
)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
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
    val=test_data,
    test=test_data,
)

# optimizer: batch 8 (one clip per process; 4 GPUs with --accumulate 2).
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=0.000025,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    _delete_=True, policy="step", warmup="linear", warmup_iters=500, warmup_ratio=1.0 / 3,
    step=[2],
)
# runtime settings
total_epochs = 3
checkpoint_config = dict(interval=1)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
# The paper's Table 2 is COCO-style; its Table 3 is the VID metric.
evaluation = dict(metric=["bbox"], vid_style=True, coco_style=True, interval=total_epochs)
