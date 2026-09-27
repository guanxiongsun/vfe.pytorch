# EOVOD on FCOS / ResNet-101-FPN, 9 epochs: the paper's reported setting
# (FCOS+LPN+SPN, ResNet-101). The schedule follows the released code's
# 9x recipe; the paper itself states batch size 32.
_base_ = ['./eovod_fcos_r50_fpn_3x.py']

model = dict(
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://detectron/resnet101_caffe'))))

# learning policy
lr_config = dict(
    policy='step', warmup='linear', warmup_iters=500, warmup_ratio=1.0 / 3, step=[6])
# runtime settings
total_epochs = 9
checkpoint_config = dict(interval=3)
evaluation = dict(metric=['bbox'], vid_style=True, coco_style=True, interval=total_epochs)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
