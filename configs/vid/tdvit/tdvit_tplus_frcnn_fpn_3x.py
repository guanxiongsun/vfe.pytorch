# TDViT-T+ with Faster R-CNN: TDViT-T plus two TDTBs at the end of stage 3,
# s*3 t*5 (Table 6 and Table S1; 51.3M parameters in the paper's count).
# Table 2: 50.9 AP, 79.9 AP50. The two extra blocks have no ImageNet weights:
# they start from PyTorch's default initialisation, as the authors' code left
# them, and take no part in stochastic depth.
_base_ = ["./tdvit_t_frcnn_fpn_3x.py"]

model = dict(detector=dict(backbone=dict(extra_tdtbs=(0, 0, 2, 0))))
