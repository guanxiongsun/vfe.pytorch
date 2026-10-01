# The YOLOX-M recipe at 10 epochs, to check the setup before the paper's 80:
# one warmup epoch, the last two without Mosaic / MixUp.
_base_ = ['./eovod_yolox_m_80e.py']

max_epochs = 10
num_last_epochs = 2
lr_config = dict(warmup_iters=1, num_last_epochs=num_last_epochs)
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=num_last_epochs, priority=48),
    dict(type='SyncNormHook', num_last_epochs=num_last_epochs, interval=5, priority=48),
    dict(type='ExpMomentumEMAHook', resume_from=None, momentum=0.0001, priority=49),
]
total_epochs = max_epochs
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=5)
evaluation = dict(interval=max_epochs)
