# SELSA on Swin-T, the single-frame-backbone counterpart of SELSA + TDViT-T,
# trained and tested exactly as it: the same key frames, references, head and
# recipe. TDViT's per-stage references are loaded and ignored.
_base_ = ["./selsa_tdvit_t_fpn_3x.py"]
pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"  # noqa

model = dict(
    backbone_refs=4,  # TDViT's four per-stage references come first: skip them
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
