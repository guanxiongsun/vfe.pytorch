# The control for eovod_yolox_m_joint_3e_backbone.py: the same training with
# every step plain -- YOLOX-M fine-tuned from COCO on the same clips, the
# aggregators unused.
_base_ = ['./eovod_yolox_m_joint_3e_backbone.py']

model = dict(location_prior=dict(train_plain_prob=1.0))
