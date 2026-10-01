# The Swin-T baseline of TDViT's Table 2 (47.1 AP, 77.2 AP50): Faster R-CNN on
# Swin-T, trained and evaluated exactly as TDViT-T -- the same key frames,
# augmentation and schedule -- with every block spatial. The reference frames
# are loaded and ignored, so one seed draws the same samples for both models.
_base_ = ["./tdvit_t_frcnn_fpn_3x.py"]
pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"  # noqa

model = dict(
    detector=dict(
        backbone=dict(
            _delete_=True,
            type="SwinTransformer",
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            convert_weights=True,
            init_cfg=dict(type="Pretrained", checkpoint=pretrained),
        ),
    ),
)
