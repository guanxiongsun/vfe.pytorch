# EOVOD on YOLOX

> Working document for EOVOD's second one-stage detector. The FCOS work, and
> why EOVOD is built the way it is, is in [eovod-plan.md](eovod-plan.md).
> Update the checkboxes and the Progress log as work proceeds.

- **Goal:** the paper's YOLOX-M rows on ImageNet VID with `vfe`'s EOVOD:
  YOLOX-M alone, + LPN, + LPN + SPN.
- **Method:** port YOLOX from mmdet 2.19.1 and check it against it the way
  FCOS was checked; train YOLOX-M with the paper's recipe; then train the
  location prior on top, reusing the FCOS work's recipe where it applies.

## The paper

Table 3 and Table 5 (COCO-style AP on VID val, V100 FPS; Table 5 labels the
rows YOLOX-S, but its LPN row repeats Table 3's YOLOX-M numbers):

| | AP | AP50 | AP75 | APs | APm | APl | FPS |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| YOLOX-M | 49.4 | 69.4 | 55.4 | 11.1 | 25.0 | 55.4 | 39.7 |
| + LPN | 53.3 | 75.1 | 58.1 | 11.6 | 30.2 | 58.9 | 35.8 |
| + LPN + SPN | 52.7 | 74.5 | 56.7 | 11.2 | 28.9 | 57.7 | 50.5 |

Recipe: "images are resized to 640 x 640 and use additional data
augmentations, including MixUp, Mosaic, RandomCrop"; "batch size 32 using
the SGD optimizer. The initial learning rate is set to 10^-3 with a cosine
learning rate schedule for 80 epochs". The released code has no YOLOX (its
`tools/speed_test.py` names a `YOLOAtt` defined nowhere).

**Left open**, and the choice here:

- *Initialisation:* the official COCO-trained YOLOX-M (Megvii release
  0.1.1rc0, 46.9 AP), converted by `tools/convert_yolox_megvii.py`, with the
  80-class classifier dropped. A 1e-3 rate is a fine-tuning rate for YOLOX,
  whose from-scratch rate is 1e-2 at batch 64.
- *Schedule details:* mmdet's YOLOX recipe -- 5 warmup epochs, cosine to 5%,
  a constant rate and no Mosaic / MixUp / affine for the last 15 epochs (L1
  loss on), BatchNorm statistics averaged across processes, and an EMA of
  the weights that checkpoints and evaluation use. "RandomCrop" is taken to
  be what `RandomAffine`'s translation and scale do.
- *How the location prior trains with Mosaic:* it cannot (a mosaic has no
  previous frame). Two stages: YOLOX-M alone on still images with the full
  recipe (stage A), then EOVOD's training on video clips from that model
  (stage B), with frozen BatchNorm (one key frame per GPU).
- *Test:* the top 100 detections per frame above YOLOX's 0.001, as the
  paper's FCOS setting.

## The port

| mmdet 2.19.1 | `vfe/` |
| :-- | :-- |
| `CSPDarknet`, `Focus`, `SPPBottleneck` | `vfe/models/backbones/csp_darknet.py` |
| `CSPLayer`, `DarknetBottleneck` | `vfe/layers/csp_layer.py` |
| `YOLOXPAFPN` | `vfe/models/necks/yolox_pafpn.py` |
| `YOLOXHead` | `vfe/models/dense_heads/yolox_head.py`, plus EOVOD's `level_ids`, `with_levels`, `with_cls_scores`, `reg_feats` (the box and objectness branch) |
| `SimOTAAssigner` | `vfe/core/bbox/assigners.py` |
| `YOLOX` (multi-scale steps) | `vfe/models/detectors/single_stage.py` |
| `Mosaic`, `RandomAffine`, `MixUp`, `YOLOXHSVRandomAug`, `FilterAnnotations` | `vfe/datasets/pipelines/mix.py` |
| `MultiImageMixDataset` | `vfe/datasets/dataset_wrappers.py` |
| `DefaultFormatBundle`, `Collect`, `Pad(pad_to_square)`, `RandomFlip(flip_ratio)` | `vfe/datasets/pipelines/{formatting,transforms}.py` |
| `YOLOXLrUpdaterHook` | `vfe/engine/lr_scheduler.py` (`policy='YOLOX'`) |
| `ExpMomentumEMAHook`, `YOLOXModeSwitchHook`, `SyncNormHook` | `vfe/engine/hooks.py`, called by the trainer at mmcv's points |

EOVOD itself needed one addition: without reference frames,
`EOVOD.forward_train` trains the wrapped detector on the still-image batch as
it would alone (YOLOX's own multi-scale step included), so one model and one
checkpoint layout serve both stages and the evaluation.

**Checked against mmdet 2.19.1** (the legacy stack built on this laptop, CPU):

- `tools/checks/parity_yolox.py` (282 artifacts, a YOLOX-S-sized model with
  30 classes): float64 -- forward passes, decoding, SimOTA's targets, the
  multi-scale resize and every parameter gradient within 2.3e-13 relative;
  `loss_obj` 1.35e-7, because mmdet's `binary_cross_entropy` casts targets
  with `label.float()` and so computes that loss in float32 in both stacks.
  float32 -- forward 8.6e-7, losses 4.2e-6, targets 3.3e-5, post-NMS
  detections 1.8e-7 (compared as sets: random weights tie thousands of
  scores). float32 gradients differ by 1.9e-2, all of it on the legacy side:
  against float64, torch 1.10's CPU backward through the whole network is off
  by 1.9e-2, `vfe`'s by 9.3e-5 -- the same finding as the rewrite plan's.
- `tools/checks/parity_yolox_pipeline.py`: 24 mixed-image samples and 4 after
  the switch, **128 / 128 artifacts bit-exact** (images, boxes, labels, scale
  factors, flips). Getting there showed that mmdet's transforms module does
  `from numpy import random`: every `random.*` call in Mosaic, RandomAffine
  and MixUp is numpy's (exclusive upper bounds), which the port now follows.
- The official YOLOX-M weights load with every one of 606 tensors mapped
  (25,326,495 parameters, the published 25.3M) and find the demo image's
  bench (0.94) and cars (0.93-0.78) where they are.
- `tests/test_yolox.py`: the head's EOVOD extensions, the multi-scale step,
  the LR policy, the EMA, the mode switch, the transforms, the converter,
  EOVOD training and inference on YOLOX, and the recipe through the trainer.

## Milestones

- [x] **Y1 — the model**, parity with mmdet (above).
- [x] **Y2 — the training recipe**: transforms (bit-exact), schedule, hooks,
  the still-image path through EOVOD.
- [x] **Y3 — the COCO weights**, converted and checked.
- [x] **Y4 — stage A at 10 epochs**: YOLOX-M alone, **56.1 / 75.7 / 62.5**
  on the full val set -- above the paper's YOLOX-M at 80 epochs (49.4) and
  its + LPN (53.3). The 80-epoch run awaits the user's decision.
- [ ] **Y5 — stage B**: the location prior on YOLOX (frozen BatchNorm, video
  clips at 640, zero-initialised aggregation), two designs at one epoch, then
  LPN and LPN + SPN evaluations and speed.

## Running it

```bash
# weights (once): the official release, converted for EOVOD
python tools/convert_yolox_megvii.py yolox_m.pth yolox_m_coco_eovod.pth --drop-classifier --prefix detector.
# stage A on 4 GPUs (batch 32 = 4 x 8; BatchNorm trains, so no accumulation)
CONFIG=configs/vid/eovod/eovod_yolox_m_10e.py NAME=eovod_yolox_m_10e ACCUMULATE=1 EXTRA="--no-validate" \
    sbatch tools/isambard/train.sbatch
# YOLOX alone on VID val: a location prior that never validates, no size prior
... -m vfe.cli.test CONFIG CKPT --cfg-options model.location_prior.score_thr=2.0 model.size_prior=None
```

## Progress log

- **2026-09-30 (stage A at 10 epochs; stage B submitted)** — YOLOX-M alone,
  10 epochs from COCO (job 6955929, 1 h 20 min on 4 GH200s, 0.14 s per step
  of 32 images): full val **56.1 / 75.7 / 62.5** (APs / APm / APl 13.1 /
  28.7 / 62.5; VID AP50 76.2, fast 51.3). That is 6.7 AP above the paper's
  YOLOX-M, whose initialisation the paper does not state; COCO is the likely
  difference, so the paper's absolute numbers are not the bar here -- LPN's
  gain over this model is. Stage B at one epoch from it (`--load-from`),
  each evaluated with and without the size prior: the paper-text design
  (`eovod_yolox_m_lpn_3e.py`, job 6959337) and the before-PAFPN design
  (`_backbone.py`, 6959340); stage A's speed, 6959343. Stage B starts from a
  trained detector, so the aggregators' output projections start at zero
  (`aggregator.zero_init`: aggregation is the identity until it learns) and
  learn at 1e-3 while the detector continues at 1e-4.
- **2026-09-29 (the smoke run, and an EMA bug)** — The 1-GPU smoke (6955928)
  passed the tests on the GH200 (including CUDA graphs of YOLOX's head),
  trained 60 steps on real data, and ran 3 videos plain (58 frames/s) and
  with both priors (62 frames/s). It also exposed a bug: the trainer
  registered the EMA's buffers before applying `load_from`, so the first
  epoch's swap put the initial weights in place of the COCO ones (649
  detections on 1,392 frames at a 0.001 threshold). Fixed before stage A
  started: the average now restarts from the loaded weights, as in mmdet,
  with a test that fails without the fix.
- **2026-09-29 (Y1-Y3)** — The port, its parity checks and the weights, as
  above. Isambard: a separate worktree `~/code/vfe-yolox`; the converted
  weights at `/projects/b5cs/vfe/checkpoints/coco/yolox_m_coco_eovod.pth`
  (sha256 `ac266a86...`). Submitted: a 1-GPU smoke (tests, 60 training steps
  on real data, 3-video inference plain and with the priors) 6955928 ->
  stage A at 10 epochs 6955929 -> its plain full-val evaluation 6955930.
