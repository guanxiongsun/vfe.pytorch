# Configs

The trainable configs, plus the `_base_` files they inherit. The syntax is
MMDetection's, and `vfe.config.Config` resolves it identically — that
equivalence is itself one of the parity checks (`run_parity.py check config`).

Each method's folder has its results against the paper, its configs and what
they do not say:

| Folder | Method |
| :-- | :-- |
| [`vid/mamba/`](vid/mamba) | MAMBA (AAAI 2021), ported from the released code; checkpoint released |
| [`vid/stpn/`](vid/stpn) | STPN (ICCV 2023), ported from the released code; checkpoint released |
| [`vid/eovod/`](vid/eovod) | EOVOD (ECCV 2022), implemented from its paper, on FCOS and YOLOX |
| [`vid/tdvit/`](vid/tdvit) | TDViT (ECCV 2022), implemented from its paper |

## `_base_`

| File | Holds |
| :-- | :-- |
| [`_base_/default_runtime.py`](_base_/default_runtime.py) | logging, checkpoint and evaluation intervals |
| [`_base_/schedules/schedule_1x.py`](_base_/schedules/schedule_1x.py) | SGD, the step schedule and warmup |
| [`_base_/datasets/vid/imagenet_vid_multi_frame.py`](_base_/datasets/vid/imagenet_vid_multi_frame.py) | ImageNet VID + DET, reference-frame sampling, the train and test pipelines |
| [`_base_/models/vid/faster_rcnn_r50_dc5.py`](_base_/models/vid/faster_rcnn_r50_dc5.py) | the Faster R-CNN DC5 detector MAMBA wraps |
| [`_base_/models/vid/fcos_r50_fpn.py`](_base_/models/vid/fcos_r50_fpn.py) | the FCOS detector EOVOD wraps (caffe-style ResNet, FPN P3–P7, GroupNorm head) |
| [`_base_/models/vid/faster_rcnn_r50_fpn.py`](_base_/models/vid/faster_rcnn_r50_fpn.py) | the Faster R-CNN FPN detector TDViT and its Swin baselines wrap (from v1) |

Configs for models this release does not implement — SELSA on ResNet, the
ResNet single-frame baselines, and MMDetection's 700-odd COCO configs — are at
the `v1.0.0` tag.
