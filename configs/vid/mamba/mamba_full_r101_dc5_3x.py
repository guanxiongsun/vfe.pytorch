# MAMBA's full model (paper Table 3, "Ours": 84.6 against the instance-only
# 83.7): the released model's instance level plus the pixel level, which
# enhances the backbone's DC5 map before the ChannelMapper, RPN and RoI head.
# The paper fixes K = 100 pixels per detected box and 2,000 keys (the memory's
# default key_length); the score threshold and the per-frame cap follow the
# released EOVOD code's MPN, the same design on FCOS.
_base_ = ['./mamba_r101_dc5_3x.py']

model = dict(
    pixel=dict(
        in_channels=2048,
        num_attention_blocks=16,
        position='backbone',
        stride=16,
        train_keys='gt',
        score_thr=0.3,
        pixels_per_box=100,
        pixels_per_frame=1000,
    ))
