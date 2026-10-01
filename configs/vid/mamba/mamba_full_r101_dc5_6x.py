# The full model (pixel + instance level, random training keys) on the released
# model's 6-epoch schedule (mamba_r101_dc5_6x.py: lr 1e-3, x0.1 after epoch 4).
# At one epoch it scores 75.6 VID AP50 against the instance level's 72.1; the
# released instance-only model scores 83.80 (84.06 retrained), the paper's full
# model 84.6.
_base_ = ['./mamba_full_r101_dc5_3x.py']

lr_config = dict(step=[4])
total_epochs = 6
checkpoint_config = dict(interval=3)
evaluation = dict(interval=total_epochs)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
