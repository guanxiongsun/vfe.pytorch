# Configs

The trainable configs, plus the `_base_` files they inherit. The syntax is
MMDetection's, and `vfe.config.Config` resolves it identically — that
equivalence is itself one of the parity checks (`run_parity.py check config`).

| Config | Model | Released checkpoint |
| :-- | :-- | :-- |
| [`vid/mamba/mamba_r101_dc5_6x.py`](vid/mamba/mamba_r101_dc5_6x.py) | MAMBA, ResNet-101-DC5, 6 epochs | [`mamba_r101_dc5_6x`](https://huggingface.co/guanxiongsun/vfe.pytorch/tree/main/work_dirs/mamba_r101_dc5_6x) — AP50 83.8 |
| [`vid/mamba/mamba_r101_dc5_3x.py`](vid/mamba/mamba_r101_dc5_3x.py) | the same, 3 epochs | — |
| [`vid/stpn/stpn_swint_adam_9x.py`](vid/stpn/stpn_swint_adam_9x.py) | STPN, Swin-T, 9 epochs | [`stpn_swint_adam_9x`](https://huggingface.co/guanxiongsun/vfe.pytorch/tree/main/work_dirs/stpn_swint_adam_9x) — AP50 85.2 |
| [`vid/stpn/stpn_swins_adam_9x.py`](vid/stpn/stpn_swins_adam_9x.py) | STPN, Swin-S, 9 epochs | none — see below |
| [`vid/eovod/eovod_fcos_r101_fpn_3x.py`](vid/eovod/eovod_fcos_r101_fpn_3x.py) | EOVOD on FCOS, ResNet-101-FPN, the paper's recipe (3 epochs at batch 4) | none — untrained, see below |
| [`vid/eovod/eovod_fcos_r101_fpn_9x.py`](vid/eovod/eovod_fcos_r101_fpn_9x.py) | the same, the released checkpoint's recipe (9 epochs at batch 8) | — |
| [`vid/eovod/eovod_fcos_r50_fpn_3x.py`](vid/eovod/eovod_fcos_r50_fpn_3x.py) | the same on ResNet-50, 3 epochs | — |

Both released checkpoints load into this code unchanged and reproduce their
published scores to within 0.02 AP50.

**EOVOD is implemented from its paper, not ported**, and has not been trained
here yet; [`docs/eovod-plan.md`](../docs/eovod-plan.md) has the design, what
is verified, and the run plan. Its test set keeps frames in order (no
`shuffle_video_frames`): the location prior reads the previous frame, and the
size prior counts frames. The size prior is inference-only, so one trained
model gives both of the paper's rows: as configured (`size_prior.interval=7`)
and LPN-only (`--cfg-options model.size_prior=None`). It evaluates with both
the VID metric and COCO-style AP, which is what the paper reports.

## TDViT

[`vid/tdvit/`](vid/tdvit) holds TDViT (ECCV 2022) and its baselines: Faster
R-CNN with an FPN on ImageNet VID, 3 epochs at batch 8, AdamW at 2.5e-5 from
ImageNet-1K Swin weights, with Swin's augmentation unless the name says
otherwise. No checkpoint is published yet; the results are in the
[README](../README.md#tdvit) and [`docs/tdvit-plan.md`](../docs/tdvit-plan.md).

| Config | Model |
| :-- | :-- |
| [`frcnn_swint_fpn_3x.py`](vid/tdvit/frcnn_swint_fpn_3x.py) | Swin-T, the single-frame baseline (Table 2) |
| [`tdvit_t_frcnn_fpn_3x.py`](vid/tdvit/tdvit_t_frcnn_fpn_3x.py) | TDViT-T as published: a temporal block attends to its reference alone |
| [`tdvit_t_joint_frcnn_fpn_3x.py`](vid/tdvit/tdvit_t_joint_frcnn_fpn_3x.py) | **TDViT-T with joint attention** -- the one to use |
| [`tdvit_tplus_frcnn_fpn_3x.py`](vid/tdvit/tdvit_tplus_frcnn_fpn_3x.py) | TDViT-T+ as published: two more temporal blocks in stage 3, from torch's initialisation |
| [`tdvit_tplus_joint_frcnn_fpn_3x.py`](vid/tdvit/tdvit_tplus_joint_frcnn_fpn_3x.py) | TDViT-T+ with joint attention, the two new blocks starting as the identity |
| [`frcnn_swins_fpn_3x.py`](vid/tdvit/frcnn_swins_fpn_3x.py), [`tdvit_s_joint_frcnn_fpn_3x.py`](vid/tdvit/tdvit_s_joint_frcnn_fpn_3x.py) | Swin-S, and TDViT-S with joint attention |
| [`frcnn_swinb_fpn_3x.py`](vid/tdvit/frcnn_swinb_fpn_3x.py), [`tdvit_b_joint_frcnn_fpn_3x.py`](vid/tdvit/tdvit_b_joint_frcnn_fpn_3x.py) | Swin-B and TDViT-B: they build, but were never trained |
| [`frcnn_swint_fpn_3x_v1aug.py`](vid/tdvit/frcnn_swint_fpn_3x_v1aug.py), [`tdvit_t_joint_frcnn_fpn_3x_v1aug.py`](vid/tdvit/tdvit_t_joint_frcnn_fpn_3x_v1aug.py) | the two tiny models on v1's plain pipeline (resize to 600, flip), which scores higher |
| [`selsa_swint_fpn_3x.py`](vid/tdvit/selsa_swint_fpn_3x.py), [`selsa_tdvit_t_joint_fpn_3x.py`](vid/tdvit/selsa_tdvit_t_joint_fpn_3x.py) | SELSA\* (Table 3: SELSA with RDN's top-75 reference proposals) on Swin-T and on TDViT-T with joint attention |
| [`selsa_tdvit_t_fpn_3x.py`](vid/tdvit/selsa_tdvit_t_fpn_3x.py) | SELSA\* on TDViT-T as published; never trained |

TDViT's test set keeps each video's frames in order, first frame first: the
detector resets the backbone's memories at `frame_id == 0` and fills them as
the video goes on. The memory is a test-time setting of one trained model, so
one checkpoint gives the paper's Table 8 and more through `--cfg-options`:
`model.detector.backbone.memory_sampling=nms` (or `patch_shuffle`,
`channel_shuffle`), `model.detector.backbone.memory_reuse=1` (a new reference
every frame), `model.detector.backbone.temporal_dilations=(1,2,4,8)`, and
`model.online=False` (the memory off: the weights as a still-image detector).
`model.detector.backbone.fused_attention=True` runs the attention through
PyTorch's fused kernel, the same detections faster; it is off by default so
Swin stays bit-identical to its reference.

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
| [`_base_/models/vid/fcos_r50_fpn.py`](_base_/models/vid/fcos_r50_fpn.py) | the FCOS detector EOVOD wraps (caffe-style ResNet, FPN P3–P7, GroupNorm head) |
| [`_base_/models/vid/faster_rcnn_r50_fpn.py`](_base_/models/vid/faster_rcnn_r50_fpn.py) | the Faster R-CNN FPN detector TDViT and its Swin baselines wrap (from v1) |

Configs for models this release does not implement — SELSA on ResNet, the
ResNet single-frame baselines, and MMDetection's 700-odd COCO configs — are at
the `v1.0.0` tag.
