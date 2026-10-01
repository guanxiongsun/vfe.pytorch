# Stage B on YOLOX-M with the design that reproduced the paper on FCOS
# (eovod_fcos_r101_fpn_9x_backbone_cls_ctrreg.py: 54.0 AP): the backbone's
# stride-16 and -32 maps aggregated before the PAFPN, every pixel a query, keys
# from a memory bank (random pixels of the support frames in training), and
# the aggregated maps feeding the classification tower only -- the PAFPN runs a
# second time on the original maps for the box and objectness branch, where
# YOLOX already computes objectness (FCOS needed centerness_on_reg for that).
_base_ = ['./eovod_yolox_m_lpn_3e.py']

model = dict(
    location_prior=dict(queries='all', train_keys='random', train_random_keys=2000,
                        train_plain_prob=0.0),
    memory=dict(update=True, capacity=20000, num_keys=2000, write_per_frame=1000),
    aggregator=dict(position='backbone', levels=[1, 2], shared=False, branches='cls',
                    backbone_strides=(8, 16, 32)),
)
