# The Swin-S baseline of TDViT's Table 2 (52.6 AP, 82.4 AP50): Swin-T's
# baseline (frcnn_swint_fpn_3x.py) with stage 3 at 18 blocks, the ImageNet-1K
# Swin-S weights and Swin-S's drop path (0.3).
_base_ = ["./frcnn_swint_fpn_3x.py"]
pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth"  # noqa

model = dict(
    detector=dict(
        backbone=dict(
            depths=[2, 2, 18, 2],
            drop_path_rate=0.3,
            init_cfg=dict(type="Pretrained", checkpoint=pretrained),
        ),
    ),
)
