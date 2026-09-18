# STPN with a Swin-S backbone.
#
# Swin-S differs from Swin-T in exactly two settings: stage 3 is 18 blocks deep
# instead of 6 (a 50M-parameter backbone instead of 28M), and the original Swin
# detection configs raise stochastic depth from 0.2 to 0.3 to match. Everything
# else -- embedding width, head counts, window size, the prompt tokens STPN
# prepends, the detector, the data pipeline and the AdamW schedule -- is the
# Swin-T config's, so this file inherits it and overrides only those two plus
# the pretrained weights.
#
# No released STPN Swin-S checkpoint exists: this config is provided to train
# from, and its accuracy has not been measured. See configs/README.md.
_base_ = ["./stpn_swint_adam_9x.py"]

pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.2/mask_rcnn_swin_small_patch4_window7.pth"  # noqa

model = dict(
    detector=dict(
        backbone=dict(
            depths=[2, 2, 18, 2],
            drop_path_rate=0.3,
            init_cfg=dict(type="Pretrained", checkpoint=pretrained),
        ),
    ),
)
