# The best one-epoch design (eovod_fcos_r101_fpn_3x_backbone_cls_ctrreg.py:
# C4 and C5 aggregated before the FPN, every pixel a query, the
# classification tower only, centerness from the regression tower) on the
# released code's schedule, as eovod_fcos_r101_fpn_9x.py does for the default
# recipe: 9 epochs at batch 8 (4 GPUs x 2 micro-steps), lr x0.1 after epoch 6.
# At one epoch: 36.9 / 65.4 / 38.3 against the default recipe's 36.0 / 63.2 /
# 37.8; the default recipe reached 51.7 at 9 epochs.
_base_ = ['./eovod_fcos_r101_fpn_3x_backbone_cls_ctrreg.py']

# learning policy
lr_config = dict(
    policy='step', warmup='linear', warmup_iters=500, warmup_ratio=1.0 / 3, step=[6])
# runtime settings
total_epochs = 9
checkpoint_config = dict(interval=3)
evaluation = dict(metric=['bbox'], vid_style=True, coco_style=True, interval=total_epochs)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
