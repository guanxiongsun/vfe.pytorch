# The default recipe (eovod_fcos_r101_fpn_3x.py) with FCOS's centerness
# computed from the regression tower: the control for how much
# centerness_on_reg gives by itself, against the combined design that uses it
# (eovod_fcos_r101_fpn_3x_backbone_cls_ctrreg.py). With --cfg-options
# model.location_prior.train_plain_prob=1.0 it trains FCOS alone.
_base_ = ['./eovod_fcos_r101_fpn_3x.py']

model = dict(detector=dict(bbox_head=dict(centerness_on_reg=True)))
