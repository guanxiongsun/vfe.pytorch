# TDViT-T+ with joint attention: TDViT-T joint plus two TDTBs at the end of
# stage 3 (s*3 t*5). The two new blocks start as the identity -- their
# residual branches' output projections at zero -- so the network starts as
# the pretrained TDViT-T; from torch's default initialisation (the authors'
# code) they cost 1.1 AP after one epoch (docs/tdvit-plan.md).
_base_ = ["./tdvit_tplus_frcnn_fpn_3x.py"]

model = dict(detector=dict(backbone=dict(attention="joint", extra_init="zero")))
