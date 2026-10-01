# EOVOD on YOLOX

> Working document for EOVOD's second one-stage detector. The FCOS work, and
> why EOVOD is built the way it is, is in [eovod-plan.md](eovod-plan.md).
> Update the checkboxes and the Progress log as work proceeds.

- **Goal:** the paper's YOLOX-M rows on ImageNet VID with `vfe`'s EOVOD:
  YOLOX-M alone, + LPN, + LPN + SPN.
- **Method:** port YOLOX from mmdet 2.19.1 and check it against it the way
  FCOS was checked; train YOLOX-M with the paper's recipe; then train the
  location prior on top, reusing the FCOS work's recipe where it applies.

## Status (2026-10-01): concluded

- **The port is exact.** YOLOX matches mmdet 2.19.1 to 2.3e-13 in float64
  (forward passes, decoding, SimOTA's targets, every gradient), its training
  pipeline matches bit for bit (128 / 128), and the official COCO YOLOX-M loads
  with all 606 tensors (*The port*, below).
- **YOLOX-M alone is far stronger than the paper's.** From the COCO weights,
  with the paper's recipe cut to 10 epochs: **56.1 / 75.7 / 62.5**, against
  the paper's YOLOX-M at 49.4 (and its YOLOX-M + LPN at 53.3). The paper does
  not say how its YOLOX was initialised; COCO is the likely difference.
- **The location prior on YOLOX** (full val, AP):

  | How the prior was trained | without it | with LPN | gain | LPN + SPN |
  | :-- | --: | --: | --: | --: |
  | added to the finished model (detector frozen but for classification) | 56.0 | 56.0 | 0.0 | 55.1 |
  | with the detector, on clips without Mosaic, 1 epoch | 41.5 | 43.2 | +1.7 | 42.3 |
  | with the detector, YOLOX's full recipe on clips, 10 epochs | 55.5 | **56.1** | **+0.6** | 55.4 |

  The prior helps when it trains with the detector, and helps less the
  stronger the detector: +4.2 AP on FCOS (from 49.8), the paper's +3.9 on its
  YOLOX-M (from 49.4), +1.7 on a 41.5 YOLOX, +0.6 on a 55.5 one (APs +1.7,
  APl +0.8, nothing on fast motion). The size prior costs 0.7-0.9 AP on
  YOLOX; with the full recipe that is more than the location prior adds.
  Single runs: the +0.6 is within what a second seed might move.
- **Speed** on one GH200 at batch 1 (`tools/eovod_speed.py`, the final model,
  8 videos):

  | | eager FPS | fast engine FPS | AP on the 8 videos |
  | :-- | --: | --: | --: |
  | no aggregation | 46-49 | 181-184 | 52.1 |
  | LPN | 38.8 | 120 | 52.7 |
  | LPN + SPN | 44.1 | 130 | 47.3 |

  The paper (V100): 39.7 / 35.8 / 50.5 FPS, so its LPN + SPN is 27% faster
  than YOLOX-M alone. Here it is 5-10% slower eager and 28-29% slower with the
  fast engine: the backbone and PAFPN take 15.9-17.6 ms of an eager frame and
  the head only 3.7-4.1 ms, so running 1.38 of the 3 head levels saves 1.5
  ms while the aggregation and its bookkeeping add 2.9. On these videos the
  size prior also costs 5.3 AP (the paper: 0.6 on val).
- **Not done:** the paper's 80 epochs (at 10, YOLOX-M alone is already above
  the paper's YOLOX-M + LPN) and a second seed of the last comparison.
- **What carries over:** `SeqShared` with clip-aware `MultiImageMixDataset`
  (a still-image recipe with Mosaic and MixUp, applied to clips with the same
  random draws per clip) and batched clip training in EOVOD (BatchNorm can
  train; one key frame per GPU is unchanged, bit for bit).

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
  its + LPN (53.3). The 80-epoch run was not needed and not run.
- [x] **Y5 — the prior on YOLOX**: added to the finished stage-A model it
  gains nothing; trained jointly on clips without Mosaic +1.7 AP (41.5 ->
  43.2); with YOLOX's full recipe on clips (`SeqShared`) +0.6 (55.5 -> 56.1),
  and LPN + SPN 55.4.

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

- **2026-10-01 (speed of the final model; the YOLOX work stops here)** — The
  clip-recipe model with the prior (job 6990037,
  `/projects/b5cs/vfe/speed/speed_yolox_m_clips_10e.json`), eager / fast
  engine, ms per frame after the first: plain 19.9-21.1 / 5.2-5.3, LPN 25.3 /
  8.1, LPN + SPN 22.2 / 7.4 (1.38 levels, aggregation 1.65 ms). Eager
  breakdown, plain against LPN + SPN: backbone and PAFPN 16.5 in both, head
  3.9 -> 2.4, aggregation and the rest 0.5 -> 3.4. The findings are in
  *Status* above.
- **2026-10-01 (the full recipe with the prior: +0.6 AP)** — 10 epochs from
  COCO at batch 32, YOLOX's recipe on clips (jobs 6985198 / 6985201), full val:

  | | AP | AP50 | AP75 | APs / APm / APl | VID AP50 | fast |
  | :-- | --: | --: | --: | :-: | --: | --: |
  | control (every step plain) | 55.5 | 75.1 | 61.6 | 13.2 / 29.1 / 61.6 | 75.6 | 53.3 |
  | with the prior, LPN | **56.1** | 75.8 | 62.3 | 14.9 / 28.8 / 62.4 | 76.3 | 53.2 |
  | with the prior, LPN + SPN | 55.4 | 74.7 | 61.6 | 15.0 / 28.5 / 61.7 | 75.1 | 52.2 |
  | stage A (still images) | 56.1 | 75.7 | 62.5 | 13.1 / 28.7 / 62.5 | 76.2 | 51.3 |

  The clip recipe trains a full-strength YOLOX (the control is 0.6 AP below
  stage A), and the prior adds 0.6 AP to it -- the most on small objects
  (+1.7 APs), none on fast motion -- where training without Mosaic it added
  1.7 to a 41.5 model. The size prior gives that back (55.4). On FCOS the prior
  added 4.2 AP to a 49.8 model, and the paper's YOLOX-M gains 3.9 from 49.4:
  on a YOLOX-M this strong, the prior helps little. Single runs; the 0.6 is
  within what a second seed might move.
- **2026-10-01 (YOLOX's recipe on clips, with the prior)** — To have stage A's
  detector and the prior's gain at once, the stage-A recipe now runs on clips.
  `SeqShared` applies each single-image transform to all three frames of a
  clip with the same random draws (the same mosaic layout, warp, mixed-in clip,
  colour, flip and size), so every pixel keeps its previous frames;
  `MultiImageMixDataset` mixes whole clips. EOVOD trains on batches of clips,
  so BatchNorm trains as in stage A (one key frame per GPU trains exactly as
  before, checked on losses, gradients and the random stream), and
  `clip_multiscale` applies YOLOX's multi-scale step to whole clips.
  `eovod_yolox_m_clips_10e.py` (the before-PAFPN prior) against
  `_plain.py` (every step plain), 10 epochs from COCO at batch 32: a 1-GPU
  smoke 6985197, then trainings 6985198 / 6985201 and their evaluations.
- **2026-10-01 (joint training: the prior helps YOLOX)** — YOLOX-M trained with
  the before-PAFPN prior from the COCO weights on video clips, against the same
  training with every step plain; one epoch each, the detector at 1e-4 and the
  aggregators at 1e-3 (jobs 6982189, 6982192). Full val:

  | | AP | AP50 | AP75 | VID AP50 | fast |
  | :-- | --: | --: | --: | --: | --: |
  | control (plain) | 41.5 | 59.2 | 46.3 | 59.5 | 39.8 |
  | joint, LPN | **43.2** | 61.7 | 47.7 | 62.0 | 40.7 |
  | joint, LPN + SPN | 42.3 | 60.2 | 46.6 | 60.6 | 39.0 |

  +1.7 AP / +2.5 AP50 for LPN, FCOS's one-epoch gain (v4: +1.7). So the prior
  works on YOLOX when trained with it, and not when added to a finished model.
  The size prior again costs about 1 AP. A first attempt at 1e-3, FCOS's rate,
  did not learn (control 2.8 AP: objectness loss flat near 4, box loss rising
  from COCO's 1.55 to 2.5); YOLOX with frozen BatchNorm at batch 8 needs 1e-4.
  Both are far below stage A (56.1): clip training has no Mosaic, MixUp,
  multi-scale or EMA, and one epoch is short.
- **2026-09-30 (stage B, second attempt: no drift, and no gain)** — The
  detector frozen except its classification branch, zero-initialised
  aggregators learning at 1e-2, one epoch from stage A's detector (jobs
  6961022, 6961026). Full val, AP / AP50 / AP75:

  | Design | plain | LPN | LPN + SPN |
  | :-- | --: | --: | --: |
  | paper-text | 55.9 / 75.8 / 62.4 | 56.0 / 75.8 / 62.5 | 55.0 / 74.4 / 61.4 |
  | before the PAFPN | 56.0 / 75.7 / 62.5 | 56.0 / 75.8 / 62.6 | 55.1 / 74.5 / 61.5 |

  Freezing removed the drift (plain 55.9-56.0 against stage A's 56.1), but
  the location prior adds at most 0.1 AP, and the size prior costs 1 AP (the
  paper: -0.6). The aggregators learned more than before -- output
  projections at norms 0.6-1.8, against 0.06-0.2 -- but still little. On
  this YOLOX-M, which starts 6.7 AP above the paper's and already above its
  YOLOX-M + LPN (53.3), a prior trained onto the finished detector does not
  help; on FCOS the prior's gain (+4.2) came from training with it from the
  start.
- **2026-09-30 (stage B, first attempt: the detector drifted, the prior never
  learned)** — One epoch from stage A (56.1), full val:

  | Design | plain | LPN | LPN + SPN |
  | :-- | --: | --: | --: |
  | paper-text (the prior's cells on the PAFPN levels) | 51.1 | 51.1 / 70.7 / 57.5 | 49.8 / 68.7 / 56.1 |
  | before the PAFPN | 52.0 | 52.0 / 71.5 / 58.1 | 50.8 / 69.8 / 56.8 |

  Plain equals LPN to the last digit in both. Training the whole detector
  for an epoch (batch 8, lr 1e-4, no EMA) cost it 4-5 AP, and the prior did
  not learn to help. The before-PAFPN aggregators' zero-initialised output
  projections reached norms of 0.06 and 0.22 (the other layers 10-15), so
  they stayed nearly the identity; the paper-text design's zero
  initialisation never applied, because `load_from` restored stage A's
  untrained aggregator of the same name, and its prior engages on 94% of
  frames yet moves AP by 0.3 on the timed videos. Second attempt (jobs
  6961022, 6961026, each evaluated plain / LPN / LPN + SPN): the detector
  frozen except the classification branch (`frozen_modules`), started from
  `epoch_10_detector.pth` (stage A without aggregators or EMA buffers), the
  aggregators learning at 1e-2.

  Speed on one GH200, batch 1 (`tools/eovod_speed.py`, 8 videos): YOLOX-M
  alone 41.8 FPS eager (23.3 ms: backbone and PAFPN 18.6, head 5.9) and 199
  FPS with the fast engine (4.8 ms). The before-PAFPN stage-B model: LPN
  35.4 FPS eager / 122 fast (aggregation 1.8 ms); with SPN 36.6 / 128.5,
  running 1.37 of 3 levels on average -- and dropping 3.5 AP on those videos,
  where the paper's YOLOX SPN costs 0.6.
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
