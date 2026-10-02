# SELSA on TDViT-T: Table 3's "SELSA+Ours" (83.9 VID AP50; fast 67.7,
# medium 83.8, slow 88.6). SELSA with RDN's distillation, as the paper's
# reimplementation ("SELSA*"): each reference frame offers its top 75 of 300
# proposals, aggregated after each shared FC of the box head (MAMBA's
# instance level, given its references on every frame, so its memory is
# never read). TDViT-T's training and recipe are unchanged; the references of
# SELSA are two more frames within +-9 of the key frame (SELSA's range in
# mmtracking) and, at test time, 14 frames spread over each video
# (test_with_adaptive_stride), all seen frame by frame.
_base_ = ["./tdvit_t_frcnn_fpn_3x.py"]

temporal_dilations = (4, 8, 16, 32)

model = dict(
    detector=dict(
        roi_head=dict(
            type="MambaRoIHead",
            bbox_head=dict(
                type="MambaBBoxHead",
                num_shared_fcs=2,
                topk=75,
                aggregator=dict(type="MambaAggregator", in_channels=1024,
                                num_attention_blocks=16),
            ),
        ),
    ),
)

# dataset settings: TDViT-T's pipelines (a test checks they stay identical),
# with SELSA's references added.
dataset_type = "ImagenetVIDDataset"
data_root = "data/ILSVRC/"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
multi_scale = [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333),
               (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)]
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
# TDViT's references (one per stage, within +-D_t), then SELSA's two.
selsa_range = 9
ref_img_sampler = dict(
    num_ref_imgs=len(temporal_dilations) + 2,
    frame_range=[*temporal_dilations, selsa_range, selsa_range],
    filter_key_img=True,
    method="stagewise_uniform",
)
test_data = dict(
    type=dataset_type,
    ann_file=data_root + "annotations/imagenet_vid_val.json",
    img_prefix=data_root + "Data/VID",
    ref_img_sampler=dict(num_ref_imgs=14, frame_range=[-7, 7],
                         method="test_with_adaptive_stride"),
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
