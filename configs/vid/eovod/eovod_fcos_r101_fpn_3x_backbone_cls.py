# The released code's aggregation (eovod_fcos_r101_fpn_3x_backbone.py: C4 and
# C5 before the FPN, every pixel a query, a memory bank) with the regression
# tower kept on the original features: the aggregated maps go through the FPN
# to the classification tower only, and the FPN runs a second time on the
# original backbone maps for the regression tower. At one epoch the released
# design gained 3.3 AP50 and lost 3.5 AP75 against the paper-text model; the
# AP75 loss is the same one classification-only aggregation removed there.
_base_ = ['./eovod_fcos_r101_fpn_3x_backbone.py']

model = dict(aggregator=dict(branches='cls'))
