# STPN

**[Spatio-temporal Prompting Network for Robust Video Feature Extraction](https://arxiv.org/abs/2402.02574)**
(ICCV 2023), video object detection, ported from the released code.

## Results

ImageNet VID validation, AP50, measured on 4× GH200:

| | released checkpoint, evaluated here | trained here, from scratch | originally published |
| :-- | :--: | :--: | :--: |
| STPN, Swin-T | 85.15 | 84.54 | 85.15 |

The released checkpoint scores 64.1 / 84.1 / 91.4 AP50 on fast / medium /
slow objects; it loads into this code unchanged:
[`stpn_swint_adam_9x`](https://huggingface.co/guanxiongsun/vfe.pytorch/tree/main/work_dirs/stpn_swint_adam_9x).
Training here lands 0.61 low, with per-epoch losses within 0.6% of the
original run: run-to-run variance, most likely, though that was not confirmed
with a second seed.

## Configs

| Config | Model |
| :-- | :-- |
| [`stpn_swint_adam_9x.py`](stpn_swint_adam_9x.py) | STPN, Swin-T, 9 epochs: the released model |
| [`stpn_swins_adam_9x.py`](stpn_swins_adam_9x.py) | STPN, Swin-S, 9 epochs: never trained (below) |

**The Swin-S config has never been trained.** It is the Swin-T config with
stage 3 at 18 blocks, `drop_path_rate` 0.3 and the Swin-S pretrained weights.
Both stacks build it identically (66,324,528 parameters) and both config
loaders resolve it identically, but no accuracy has been measured and no
checkpoint is published. In 1.x this file existed but was empty.
