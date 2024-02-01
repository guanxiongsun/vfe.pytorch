_base_ = [
    './mamba_r101_dc5_3x.py'
]

# learning policy
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=1.0 / 3, step=[4]
)
# runtime settings
total_epochs = 6
checkpoint_config = dict(interval=3)
evaluation = dict(metric=["bbox"], vid_style=True, interval=total_epochs)
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)