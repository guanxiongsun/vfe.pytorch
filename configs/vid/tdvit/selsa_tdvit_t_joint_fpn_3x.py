# SELSA on TDViT-T with joint attention (tdvit_t_joint_frcnn_fpn_3x.py's
# backbone, selsa_tdvit_t_fpn_3x.py's head and references).
_base_ = ["./selsa_tdvit_t_fpn_3x.py"]

model = dict(detector=dict(backbone=dict(attention="joint")))
