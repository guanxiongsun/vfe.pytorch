# TDViT-S with joint attention (Table 2: 55.4 AP, 84.1 AP50): TDViT-T joint
# with Swin-S's depth -- stage 3 at nine Swin blocks then nine TDTBs (Table
# S1) -- its ImageNet-1K weights and drop path 0.3.
_base_ = ["./tdvit_t_joint_frcnn_fpn_3x.py"]
pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth"  # noqa

model = dict(
    detector=dict(
        backbone=dict(
            layout=("st", "st", "s" * 9 + "t" * 9, "st"),
            depths=[2, 2, 18, 2],
            drop_path_rate=0.3,
            init_cfg=dict(type="Pretrained", checkpoint=pretrained),
        ),
    ),
)
