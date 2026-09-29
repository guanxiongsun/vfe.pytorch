# eovod_fcos_r101_fpn_3x_backbone_cls.py with FCOS's centerness computed from
# the regression tower (centerness_on_reg), which reads the original maps.
# Centerness ranks overlapping boxes; computed from the classification tower,
# which this model aggregates on every step, it learns on aggregated maps.
# AP75 fell wherever that tower was aggregated on (nearly) every step: the
# released-style and combined variants, and v5 (docs/eovod-plan.md).
_base_ = ['./eovod_fcos_r101_fpn_3x_backbone_cls.py']

model = dict(detector=dict(bbox_head=dict(centerness_on_reg=True)))
