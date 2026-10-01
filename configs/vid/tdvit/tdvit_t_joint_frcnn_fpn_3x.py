# TDViT-T with joint attention: each TDTB attends over its window's own tokens
# and the reference's together (docs/tdvit-plan.md, "the block, not only the
# protocol"). The paper's block attends to the reference alone, giving up the
# frame's spatial attention in half of TDViT-T's blocks: on the same samples it
# trains worse than Swin-T, and fast objects suffer at test time. Joint
# attention keeps everything else -- the parameters (TDViT-T is still Swin-T's
# size), the memory, the dilations, the training references -- and is Swin
# exactly when a frame is its own reference. A learnable per-head bias on the
# reference's keys (temporal_bias=True) made no difference at one epoch.
_base_ = ["./tdvit_t_frcnn_fpn_3x.py"]

model = dict(detector=dict(backbone=dict(attention="joint")))
