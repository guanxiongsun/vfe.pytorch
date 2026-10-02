# TDViT-T+ with joint attention: TDViT-T joint plus two TDTBs at the end of
# stage 3 (s*3 t*5). Table 2: 50.9 AP, 79.9 AP50; here 51.4 / 80.8. The two new
# blocks have no ImageNet weights. Started from torch's initialisation (the
# authors' code) or as the identity they learn nothing at this learning rate and
# TDViT-T+ equals TDViT-T (50.7 AP); started as copies of the two pretrained
# blocks before them they train at once (docs/tdvit-plan.md).
_base_ = ["./tdvit_tplus_frcnn_fpn_3x.py"]

model = dict(detector=dict(backbone=dict(attention="joint", extra_init="copy")))
