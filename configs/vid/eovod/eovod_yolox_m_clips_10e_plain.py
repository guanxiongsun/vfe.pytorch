# The control for eovod_yolox_m_clips_10e.py: the same clips, recipe and
# schedule with every step plain -- YOLOX-M alone, its aggregators unused.
_base_ = ['./eovod_yolox_m_clips_10e.py']

model = dict(location_prior=dict(train_plain_prob=1.0))
