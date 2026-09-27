# EOVOD on FCOS / ResNet-101-FPN on the paper's FCOS recipe: 3 epochs at batch
# 4, SGD lr 1e-3 for two epochs then 1e-4 (its implementation details), which
# produced the reported 54.1 / 53.8 AP. Batch 4 is four processes with
# --accumulate 1, or two with --accumulate 2. The 9x config beside this one is
# the released checkpoint's longer recipe.
_base_ = ['./eovod_fcos_r50_fpn_3x.py']

model = dict(
    detector=dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='open-mmlab://detectron/resnet101_caffe'))))
