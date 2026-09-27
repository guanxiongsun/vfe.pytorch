# FCOS with a caffe-style ResNet-50 and FPN (P3-P7): mmdet's
# fcos_r50_caffe_fpn_gn-head config, with ImageNet VID's 30 classes and the VID
# configs' test thresholds. The detector EOVOD wraps.

model = dict(
    detector=dict(
        type='FCOS',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='caffe',
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://detectron/resnet50_caffe')),
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',  # P6 and P7 from P5
            num_outs=5,
            relu_before_extra_convs=True),
        bbox_head=dict(
            type='FCOSHead',
            num_classes=30,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            strides=[8, 16, 32, 64, 128],
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='IoULoss', loss_weight=1.0),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
        # FCOS assigns targets by size range, not by an assigner; these are
        # carried for config compatibility.
        train_cfg=dict(allowed_border=-1, pos_weight=-1, debug=False),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
