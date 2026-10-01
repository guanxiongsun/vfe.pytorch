# The before-PAFPN design (eovod_yolox_m_lpn_3e_backbone.py) with the detector
# frozen except its classification branch, as eovod_yolox_m_lpn_cls_3e.py: the
# aggregated backbone maps reach it through the frozen PAFPN.
_base_ = ['./eovod_yolox_m_lpn_3e_backbone.py']

model = dict(frozen_modules=[
    'detector.backbone',
    'detector.neck',
    'detector.bbox_head.multi_level_reg_convs',
    'detector.bbox_head.multi_level_conv_reg',
    'detector.bbox_head.multi_level_conv_obj',
])
optimizer = dict(paramwise_cfg=dict(custom_keys={'aggregators': dict(lr_mult=100.)}))
