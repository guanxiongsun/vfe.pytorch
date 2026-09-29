# EOVOD with the released code's aggregation (guanxiongsun/EOVOD at 84576bb:
# MPN with before_fpn=True, start_level=2 in its R-101 config). The backbone's
# C4 and C5 maps are aggregated before the FPN, every pixel a query, so the
# FPN and both head towers read the aggregated maps. Training keys are 2,000
# random pixels of each support frame; at test time the keys come from a
# memory bank (at most 20,000 pixels per level, 2,000 read per frame) written
# with the pixels inside each frame's validated detections, starting from the
# reference frames'. There is no location-prior mask, so no training prior;
# the size prior still applies at test time. Everything else is the default
# recipe (eovod_fcos_r101_fpn_3x.py), which follows the paper's text instead.
_base_ = ['./eovod_fcos_r101_fpn_3x.py']

model = dict(
    location_prior=dict(queries='all', train_keys='random', train_random_keys=2000,
                        train_plain_prob=0.0),
    memory=dict(update=True, capacity=20000, num_keys=2000, write_per_frame=1000),
    aggregator=dict(position='backbone', levels=[2, 3], shared=False, branches='all'),
)
