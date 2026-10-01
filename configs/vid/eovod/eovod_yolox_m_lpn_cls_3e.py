# Stage B with the detector frozen except its classification branch, the one
# the aggregated maps feed (branches='cls'): the backbone, the PAFPN and the
# box / objectness branch keep stage A's weights exactly. The first stage-B
# runs trained the whole detector and lost 4-5 AP to stage A; only the
# aggregators and the classification branch (1e-4) learn here. The
# aggregators' zero-initialised output projections gave the other aggregator
# layers no gradient until they grew, and at 1e-3 they grew to norms of 0.06-0.2
# in an epoch (typical: 7-15), so the aggregators learn at 1e-2. Start from the
# stage-A detector without its (untrained) aggregators, or load_from replaces
# the zero initialisation: epoch_10_detector.pth.
_base_ = ['./eovod_yolox_m_lpn_3e.py']

model = dict(frozen_modules=[
    'detector.backbone',
    'detector.neck',
    'detector.bbox_head.multi_level_reg_convs',
    'detector.bbox_head.multi_level_conv_reg',
    'detector.bbox_head.multi_level_conv_obj',
])
optimizer = dict(paramwise_cfg=dict(custom_keys={'aggregators': dict(lr_mult=100.)}))
