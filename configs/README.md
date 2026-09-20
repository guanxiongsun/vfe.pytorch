# Configs

Four trainable configs, plus the `_base_` files they inherit. The syntax is
MMDetection's, and `vfe.config.Config` resolves it identically — that
equivalence is itself one of the parity checks (`run_parity.py check config`).

| Config | Model | Released checkpoint |
| :-- | :-- | :-- |
| [`vid/mamba/mamba_r101_dc5_6x.py`](vid/mamba/mamba_r101_dc5_6x.py) | MAMBA, ResNet-101-DC5, 6 epochs | [`mamba_r101_dc5_6x`](https://huggingface.co/guanxiongsun/vfe.pytorch/tree/main/work_dirs/mamba_r101_dc5_6x) — AP50 83.8 |
| [`vid/mamba/mamba_r101_dc5_3x.py`](vid/mamba/mamba_r101_dc5_3x.py) | the same, 3 epochs | — |
| [`vid/stpn/stpn_swint_adam_9x.py`](vid/stpn/stpn_swint_adam_9x.py) | STPN, Swin-T, 9 epochs | [`stpn_swint_adam_9x`](https://huggingface.co/guanxiongsun/vfe.pytorch/tree/main/work_dirs/stpn_swint_adam_9x) — AP50 85.2 |
| [`vid/stpn/stpn_swins_adam_9x.py`](vid/stpn/stpn_swins_adam_9x.py) | STPN, Swin-S, 9 epochs | none — see below |

Both released checkpoints load into this code unchanged and reproduce their
published scores to within 0.02 AP50.

## Two things the configs do not say

**MAMBA's published model did not train on the schedule its config describes.**
Its checkpoint records a 4-GPU run resumed on 8, and mmcv rescaled the
iteration count on resume, so epochs 1–3 ran at batch 4 and epochs 4–6 at batch
8. Read literally — batch 8 throughout — the config gives half the steps in
epochs 1–3 and scores 83.16 instead of 83.8. To reproduce the published model,
train epochs 1–3 with half the batch and resume:

```bash
torchrun --standalone --nproc_per_node=4 -m vfe.cli.train \
    configs/vid/mamba/mamba_r101_dc5_6x.py --launcher pytorch \
    --accumulate 1 --max-epochs 3 --work-dir WORK_DIR         # batch 4
torchrun --standalone --nproc_per_node=4 -m vfe.cli.train \
    configs/vid/mamba/mamba_r101_dc5_6x.py --launcher pytorch \
    --accumulate 2 --resume-from auto --work-dir WORK_DIR     # batch 8
```

**The Swin-S config has never been trained.** It is the Swin-T config with
stage 3 at 18 blocks, `drop_path_rate` 0.3 and the Swin-S pretrained weights.
Both stacks build it identically (66,324,528 parameters) and both config
loaders resolve it identically, but no accuracy has been measured and no
checkpoint is published. In 1.x this file existed but was empty.

## `_base_`

| File | Holds |
| :-- | :-- |
| [`_base_/default_runtime.py`](_base_/default_runtime.py) | logging, checkpoint and evaluation intervals |
| [`_base_/schedules/schedule_1x.py`](_base_/schedules/schedule_1x.py) | SGD, the step schedule and warmup |
| [`_base_/datasets/vid/imagenet_vid_multi_frame.py`](_base_/datasets/vid/imagenet_vid_multi_frame.py) | ImageNet VID + DET, reference-frame sampling, the train and test pipelines |
| [`_base_/models/vid/faster_rcnn_r50_dc5.py`](_base_/models/vid/faster_rcnn_r50_dc5.py) | the Faster R-CNN DC5 detector MAMBA wraps |

Configs for models this release does not implement — SELSA, the single-frame
baselines, and MMDetection's 700-odd COCO configs — are at the `v1.0.0` tag.
