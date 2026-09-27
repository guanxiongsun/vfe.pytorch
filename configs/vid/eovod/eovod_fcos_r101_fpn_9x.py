# EOVOD on FCOS / ResNet-101-FPN, 9 epochs: the recipe of the released
# checkpoint (fcos_att_r101_fpn_9x in the released code: batch 8, x0.1 after
# epoch 6). The paper's own FCOS recipe is the 3x config beside this one.
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
