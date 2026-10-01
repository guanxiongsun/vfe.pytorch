# EOVOD on YOLOX-M trained with the location prior from the start, as EOVOD was
# on FCOS: from the COCO weights, on video clips (eovod_yolox_m_lpn_3e.py's
# data), the whole detector learning together with the before-PAFPN
# aggregation (eovod_yolox_m_lpn_3e_backbone.py's design), the aggregators at
# their default initialisation. Adding the prior to a finished
# YOLOX (stage B) gained nothing; on FCOS the prior's gain came from training
# with it. No Mosaic or MixUp: a mosaic has no previous frame. BatchNorm keeps
# COCO's statistics (one key frame per GPU). The rates are stage B's: the
# detector at 1e-4, the aggregators at 1e-3. At 1e-3 for everything (FCOS's
# rate) YOLOX did not learn: the objectness loss stayed near 4, the box loss
# rose from COCO's 1.55 to 2.5, and the control scored 2.8 AP. The control is
# eovod_yolox_m_joint_3e_plain.py.
_base_ = ['./eovod_yolox_m_lpn_3e_backbone.py']

model = dict(aggregator=dict(zero_init=False))
load_from = '/projects/b5cs/vfe/checkpoints/coco/yolox_m_coco_eovod.pth'
