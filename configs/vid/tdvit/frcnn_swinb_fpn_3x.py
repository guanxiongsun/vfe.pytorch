# The Swin-B baseline of TDViT's Table 2 (53.2 AP, 82.7 AP50): Swin-S's with
# Swin-B's width (128, heads 4 / 8 / 16 / 32) and its ImageNet-1K weights.
_base_ = ["./frcnn_swins_fpn_3x.py"]
pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth"  # noqa

model = dict(
    detector=dict(
        backbone=dict(
            embed_dims=128,
            num_heads=[4, 8, 16, 32],
            init_cfg=dict(type="Pretrained", checkpoint=pretrained),
        ),
        neck=dict(in_channels=[128, 256, 512, 1024]),
    ),
)
