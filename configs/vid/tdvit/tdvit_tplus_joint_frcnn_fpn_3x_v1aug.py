# TDViT-T+ with joint attention and its two new blocks copied from pretrained
# ones (tdvit_tplus_joint_frcnn_fpn_3x.py), trained with v1's plain pipeline
# (shorter side 600, flips) as tdvit_t_joint_frcnn_fpn_3x_v1aug.py is.
_base_ = ["./tdvit_t_joint_frcnn_fpn_3x_v1aug.py"]

model = dict(detector=dict(backbone=dict(extra_tdtbs=(0, 0, 2, 0), extra_init="copy")))
