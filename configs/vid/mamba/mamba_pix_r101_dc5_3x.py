# MAMBA's pixel level alone (paper Table 3, "Ours_pix": 81.8, against 75.4
# for Faster R-CNN): the full model with a plain box head instead of the
# instance level's.
_base_ = ['./mamba_full_r101_dc5_3x.py']

model = dict(
    detector=dict(
        roi_head=dict(
            type='StandardRoIHead',
            bbox_head=dict(
                _delete_=True,
                type='Shared2FCBBoxHead',
                in_channels=512,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=30,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.2, 0.2, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))))
