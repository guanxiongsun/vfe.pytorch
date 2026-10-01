# EOVOD on YOLOX-M trained with the location prior from the start, as EOVOD was
# on FCOS: from the COCO weights, on video clips (eovod_yolox_m_lpn_3e.py's
# data), the whole detector learning together with the before-PAFPN
# aggregation (eovod_yolox_m_lpn_3e_backbone.py's design) at lr 1e-3, the
# aggregators at their default initialisation. Adding the prior to a finished
# YOLOX (stage B) gained nothing; on FCOS the prior's gain came from training
# with it. No Mosaic or MixUp: a mosaic has no previous frame. BatchNorm keeps
# COCO's statistics (one key frame per GPU). The control is
# eovod_yolox_m_joint_3e_plain.py.
_base_ = ['./eovod_yolox_m_lpn_3e_backbone.py']

model = dict(aggregator=dict(zero_init=False))
optimizer = dict(
    lr=0.001,
    paramwise_cfg=dict(_delete_=True, norm_decay_mult=0., bias_decay_mult=0.))
load_from = '/projects/b5cs/vfe/checkpoints/coco/yolox_m_coco_eovod.pth'
