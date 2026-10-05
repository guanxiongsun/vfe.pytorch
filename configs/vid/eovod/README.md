# EOVOD

**[Efficient One-stage Video Object Detection by Exploiting Temporal Consistency](https://arxiv.org/abs/2402.09241)**
(ECCV 2022), implemented from its paper with the location and size priors
(LPN and SPN) that the original release left out, on FCOS and on YOLOX
(ported from MMDetection and checked against it).

## Results

ImageNet VID validation, COCO-style AP as the paper reports it:

| Detector | | AP | AP50 | AP75 | paper |
| :-- | :-- | :--: | :--: | :--: | :--: |
| FCOS, ResNet-101 | alone | 49.8 | 73.6 | 54.6 | 49.8 / 73.3 / 54.6 |
| | + LPN | 54.0 | 79.2 | 59.3 | 54.1 / 79.8 / 59.5 |
| | + LPN + SPN | 53.8 | 78.9 | 59.2 | 53.8 / 76.9 / 58.9 |
| YOLOX-M | alone | 55.5 | 75.1 | 61.6 | 49.4 / 69.4 / 55.4 |
| | + LPN | 56.1 | 75.8 | 62.3 | 53.3 / 75.1 / 58.1 |
| | + LPN + SPN | 55.4 | 74.7 | 61.6 | 52.7 / 74.5 / 56.7 |

With the VID metric, FCOS + LPN scores 79.7 AP50 and YOLOX-M + LPN 76.3.

FCOS trains for 9 epochs with
[`eovod_fcos_r101_fpn_9x_backbone_cls_ctrreg.py`](eovod_fcos_r101_fpn_9x_backbone_cls_ctrreg.py),
which aggregates before the FPN as the original code does; FCOS alone trains on
the same schedule. YOLOX-M trains for 10 epochs from the COCO weights, with
YOLOX's own recipe applied to video clips
([`eovod_yolox_m_clips_10e.py`](eovod_yolox_m_clips_10e.py); alone,
[`eovod_yolox_m_clips_10e_plain.py`](eovod_yolox_m_clips_10e_plain.py)).
Started from COCO, YOLOX-M is already stronger than the paper's, and the
location prior adds 0.6 AP to it, against 4.2 on FCOS; trained alone on still
images, as YOLOX usually is, it scores 56.1. The size prior is a test-time
setting of the same model. The checkpoint released with the original EOVOD code
scores 54.0 / 79.7 / 59.3 here.

## Configs

FCOS, ResNet-101-FPN unless the name says otherwise:

| Config | Model |
| :-- | :-- |
| [`eovod_fcos_r101_fpn_9x_backbone_cls_ctrreg.py`](eovod_fcos_r101_fpn_9x_backbone_cls_ctrreg.py) | **the reported model**: C4 and C5 aggregated before the FPN, the classification tower only, centerness from the regression tower; 9 epochs at batch 8 |
| [`eovod_fcos_r101_fpn_3x.py`](eovod_fcos_r101_fpn_3x.py) | the paper-text design on the paper's FCOS recipe (3 epochs at batch 4) |
| [`eovod_fcos_r101_fpn_9x.py`](eovod_fcos_r101_fpn_9x.py) | the same on the released checkpoint's recipe (9 epochs at batch 8) |
| [`eovod_fcos_r101_fpn_3x_backbone.py`](eovod_fcos_r101_fpn_3x_backbone.py) | the released code's aggregation: before the FPN, both towers |
| [`eovod_fcos_r101_fpn_3x_backbone_cls.py`](eovod_fcos_r101_fpn_3x_backbone_cls.py) | the same with the regression tower on the original maps |
| [`eovod_fcos_r101_fpn_3x_backbone_cls_ctrreg.py`](eovod_fcos_r101_fpn_3x_backbone_cls_ctrreg.py) | the reported model's design, 3 epochs |
| [`eovod_fcos_r101_fpn_3x_ctrreg.py`](eovod_fcos_r101_fpn_3x_ctrreg.py) | the paper-text design with centerness from the regression tower (a control) |
| [`eovod_fcos_r50_fpn_3x.py`](eovod_fcos_r50_fpn_3x.py) | the paper-text design on ResNet-50 |

YOLOX-M:

| Config | Model |
| :-- | :-- |
| [`eovod_yolox_m_clips_10e.py`](eovod_yolox_m_clips_10e.py) | **the reported model**: YOLOX's recipe on video clips, the location prior trained with it, 10 epochs from COCO |
| [`eovod_yolox_m_clips_10e_plain.py`](eovod_yolox_m_clips_10e_plain.py) | its control: YOLOX-M alone on the same clips |
| [`eovod_yolox_m_10e.py`](eovod_yolox_m_10e.py), [`eovod_yolox_m_80e.py`](eovod_yolox_m_80e.py) | YOLOX-M alone on still images (stage A), 10 and 80 epochs |
| [`eovod_yolox_m_lpn_3e.py`](eovod_yolox_m_lpn_3e.py), [`eovod_yolox_m_lpn_3e_backbone.py`](eovod_yolox_m_lpn_3e_backbone.py) | stage B: the prior trained on a finished stage-A model (the paper-text and the before-PAFPN design) |
| [`eovod_yolox_m_lpn_cls_3e.py`](eovod_yolox_m_lpn_cls_3e.py), [`eovod_yolox_m_lpn_cls_3e_backbone.py`](eovod_yolox_m_lpn_cls_3e_backbone.py) | stage B with only the classification branch training |
| [`eovod_yolox_m_joint_3e_backbone.py`](eovod_yolox_m_joint_3e_backbone.py), [`eovod_yolox_m_joint_3e_plain.py`](eovod_yolox_m_joint_3e_plain.py) | the prior trained from the start without Mosaic, and its control |

## Notes

EOVOD's test set keeps frames in order (no `shuffle_video_frames`): the
location prior reads the previous frame, and the size prior counts frames. The
size prior is inference-only, so one trained model gives both of the paper's
rows: as configured (`size_prior.interval=7`) and LPN-only
(`--cfg-options model.size_prior=None`). It evaluates with both the VID metric
and COCO-style AP, which is what the paper reports.

EOVOD on YOLOX starts from Megvii's COCO-trained
[YOLOX-M](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth),
converted once and passed with `--load-from` (the configs name the path used on
Isambard-AI):

```bash
python tools/convert_yolox_megvii.py yolox_m.pth yolox_m_coco_eovod.pth --drop-classifier --prefix detector.
```

FCOS and YOLOX have their own two-environment parity checks:
`tools/checks/parity_fcos.py`, `parity_yolox.py` and `parity_yolox_pipeline.py`.
