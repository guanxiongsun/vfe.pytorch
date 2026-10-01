# EOVOD — Phase 9

> Working document for EOVOD in `vfe/`, the phase after 2.0.
> [rewrite-plan.md](rewrite-plan.md) records how MAMBA and STPN were ported and
> why the method is what it is. This phase is different in kind: EOVOD is
> **implemented from its paper**, not ported from its released code. Update the
> checkboxes and the Progress log as work proceeds.
> EOVOD on YOLOX, the paper's second detector: [eovod-yolox-plan.md](eovod-yolox-plan.md).

- **Goal:** EOVOD (*Efficient One-stage Video Object Detection by Exploiting
  Temporal Consistency*, Sun, Hua, Hu, Robertson; ECCV 2022,
  [arXiv 2402.09241](https://arxiv.org/abs/2402.09241)) on plain PyTorch in
  `vfe/`, trained and evaluated on ImageNet VID with this repository's tooling.
  The paper's abstract names this repository as the code's home.
- **Method:** read the paper, build its two ideas — the location prior and the
  size prior over a one-stage detector with pixel-level attention — on a
  faithfully ported FCOS, check the FCOS part against mmdet 2.19.1 the way
  every other layer of `vfe/` was checked, and judge the whole by task metrics
  once it is trained.
- **Not the goal:** reproducing the released code, which the audit (below)
  found to be a different model from the paper's: every pixel a query, no
  level ever skipped. Its FCOS is the same mmdet FCOS this implementation is
  checked against, and its checkpoint's score matches the paper's LPN-only row.

## Status (2026-09-30)

- **EOVOD's paper reproduced at 9 epochs** (full val, COCO-style) by
  `eovod_fcos_r101_fpn_9x_backbone_cls_ctrreg.py`: LPN **54.0 / 79.2 / 59.3**
  (the paper 54.1 / 79.8 / 59.5); LPN + SPN **53.8 / 78.9 / 59.2** (the
  paper 53.8 / 76.9 / 58.9); VID AP50 79.7 / 79.4. FCOS alone on the same
  schedule is 49.8 / 73.6 / 54.6 (the paper's 49.8 / 73.3 / 54.6), so the
  gain is +4.2 AP (the paper +4.3). The design: C4 and C5 aggregated before
  the FPN, every pixel a query, a memory bank of validated pixels (the
  released code's), the aggregated maps feeding the classification tower only,
  and FCOS's centerness computed from the regression tower. The released
  checkpoint scores 54.0 / 79.7 / 59.3.
- **The paper-text design** (the prior's cells on the FPN levels; the v4
  recipe below) reaches 51.7 / 77.1 / 56.5 at 9 epochs (52.2 with r 1.2 and
  validation at 0.2): +1.9 AP over FCOS.
- **Speed follows the paper's pattern** when measured eager, as the paper
  would have: LPN −18% FPS against plain (the paper −19%), LPN + SPN with the
  paper's rule (margin 0) +7% (the paper +7%). With neighbour levels
  (margin 1, no AP cost) it is −1%. A fast engine (CUDA graphs, one sync per
  frame; same detections) runs plain at 105 FPS and EOVOD + SPN at 78–88 FPS.
- **The recipe** (the configs' default since 2026-09-28; each departure from
  the paper is commented in `eovod_fcos_r50_fpn_3x.py`): decorrelated
  training prior (`train_plain_prob` 0.25, drop 0.3, jitter 0.1, up to 2
  distractors), classification-only aggregation, validation at detection
  score > 0.3, the key set capped at 4,096, the size prior with neighbour
  levels. Why each is there: the Progress log (E-M3's leak, v2 → v5, the
  size-prior study).
- **The FCOS underneath is exact.** `tools/checks/parity_fcos.py` compares the
  head against mmdet 2.19.1 on CPU: in float64 all 82 artifacts agree to
  1e-15 relative; in float32, forward passes to 8.5e-7, decoding to 2.4e-6
  and gradients to 7.8e-4, the torch 1.10 → 2.10 accumulation floor
  documented for the training-step checks in the rewrite plan.
- **Tools:** `tools/eovod_inspector/` (a web page of per-frame state, served
  from Isambard) and `tools/eovod_speed.py` (timing); see *Running it*.

## The paper

Read in full from the PDF on 2026-09-27 (the first pass, through search
excerpts, had four details wrong; the implementation now follows the text).

**Analysis (Section 3).** The accurate VID methods (SELSA, RDN, MEGA, LRTR)
aggregate features with attention, Eq. 1:
`A(q_i, K) = q_i + Σ_j w_ij · (W · k_j)`, with `w_ij` from the similarity of
every query–key pair; complexity `O(N_q² · C)`. In two-stage detectors the
queries are ~300 proposals. The naive adaptation — every pixel of the feature
maps a query, pixels of randomly sampled reference frames the keys — costs
(Table 1, two reference frames, a 32 GB V100): SELSA 300 queries, 1.8 GB,
18.5 FPS; FCOS 12,958 queries at 600×1000, 21.9 GB, 4.6 FPS; CenterNet 16,384
at 512², 31.2 GB, 4.3 FPS; YOLOX 8,400 at 640², 16.7 GB, 8.9 FPS. More than
two reference frames ran out of memory. Second bottleneck (Table 2, FCOS
R-50): backbone 7.7 ms (18%), FPN 0.5 ms, heads 34.5 ms of which the stride-8
level takes 27.7 ms (64.9%); strides 16 / 32 / 64 / 128 take 3.9 / 1.4 /
0.8 / 0.7 ms. Both bottlenecks are attacked with *temporal consistency*:
objects change gradually in location and size between consecutive frames.

**Location prior network (Section 4.1).** Two steps. *Foreground region
selection*: the previous frame's detections with classification score above
0.5 are *validated*; if there is none, aggregation is skipped. Validated boxes
are resized by an adjustment ratio *r*, projected to each feature level by
dividing by its stride, and a binary mask `M` per level is 1 where a location
falls into any box. *Partial feature aggregation*: where `M` is 1 the pixel is
enhanced with the reference features through Eq. 1 and replaces the original
(Eq. 2); elsewhere the map is unchanged; the new maps go to the heads. The
reference frames follow SELSA's open-source implementation (mmtracking): 14
frames. *Training*: FGFA's temporal dropout picks two support frames from the
same video; the ground-truth boxes generate the mask and select the query and
key pixels; the current frame's detection losses train everything end to end.
*Inference*: the mask of frame *t* is propagated from the detections of
*t − 1*; the key set is the pixels within the detected boxes on the reference
frames.

**Size prior network (Section 4.2).** After a full detection at time *t*, the
validated boxes' levels are recorded; for the next *T* frames the heads run
only on those levels; then a full detection follows. With *T* = 7 a full
detection happens every eighth frame (Section 5.3). Boxes only from the top
level mean no small objects, so the low levels are skipped.

**Setup (Section 5.1).** VID, 15 frames per training video, plus at most
2,000 DET images per class. FCOS: ResNet-101, FPN, heads of two branches with
four 3×3 convs of 256 channels and a 3×3 predictor. Four V100s. Images
resized to a shorter side of 600 and a longer side at most 1000; **batch 4;
3 epochs; SGD with momentum 0.9 and weight decay 1e-4; lr 1e-3 for two
epochs, 1e-4 for the last.** Inference keeps the top 100 detections per
frame. (CenterNet: 512², batch 32, lr 1e-4 for 50 epochs then 1e-5 for 30.
YOLOX: 640², MixUp / Mosaic, batch 32, lr 1e-3 cosine, 80 epochs.)

**Results (COCO-style AP on VID val; all V100 FPS).**

| | AP | AP50 | AP75 | APs | APm | APl | FPS |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| FCOS R-101 (Table 3) | 49.8 | 73.3 | 54.6 | 10.1 | 22.4 | 56.1 | 25.1 |
| + LPN, *r* = 0.8 | **54.1** | **79.8** | 59.5 | 10.5 | 28.3 | 60.1 | 20.4 |
| + LPN + SPN, *T* = 7 (Table 5) | **53.8** | **76.9** | 58.9 | 9.8 | 27.3 | 59.5 | 26.9 |
| RDN\* (Table 7) | 53.4 | 81.2 | 60.1 | 8.5 | 27.4 | 59.6 | 7.1 |
| SELSA\* | 52.6 | 81.6 | 57.9 | 9.3 | 28.6 | 58.4 | 6.4 |
| MEGA\* | 53.2 | 82.4 | 59.2 | 9.1 | 29.4 | 59.1 | 5.3 |
| Faster R-CNN R-101\* | 49.7 | 75.6 | 55.9 | 7.4 | 23.7 | 56.0 | 22.5 |

Table 4, *r* (AP / FPS): 0.5 → 53.6 / 21.4; 0.8 → 54.1 / 20.4; 1.0 → 54.2 /
19.1; 1.2 → 54.2 / 17.5; 1.5 → 54.2 / 14.7. Table 6, *T* (AP / AP50 / FPS):
0 → 54.1 / 79.8 / 20.4; 7 → 53.8 / 76.9 / 26.9; 14 → 53.0 / 75.4 / 28.4;
21 → 51.9 / 73.8 / 29.0; 28 → 48.5 / 73.3 / 29.5. CenterNet + LPN 53.4 /
79.8 at 35.5 FPS; YOLOX-M + LPN 53.3 / 75.1 at 35.8; + SPN 52.7 / 74.5 at
50.5. The released checkpoint's 54.0 / 79.7 / 59.3 matches the LPN-only row,
consistent with its code never skipping a level.

**What the paper leaves open**, and the choice made here (each is an option):

- The attention's heads and projections beyond Eq. 1: SELSA's implementation
  is cited, so 16 heads with SELSA's four linear layers.
- Whether *r* also resizes the training mask: assumed yes (the mask is
  "generated" the same way).
- Which detections on the reference frames supply keys: assumed the validated
  ones (score above 0.5), the paper's one threshold.
- How many keys: all of them (no cap).
- Warmup and gradient clipping: the released code's 500 iterations from 1/3
  and clipping at 35.

## Design: paper → `vfe/`

| Paper | Here | Notes |
| :-- | :-- | :-- |
| FCOS (Fig. 1) | `FCOS` / `FCOSHead` (`vfe/models/detectors/single_stage.py`, `vfe/models/dense_heads/fcos_head.py`) | ported from mmdet 2.19.1, plus `MlvlPointGenerator`, `DistancePointBBoxCoder`, `FocalLoss`, `IoULoss`, `Scale`, `open-mmlab://` URIs |
| validated detections, score > 0.5 | `location_prior.score_thr` | one threshold for both priors and the key set |
| adjustment ratio *r* = 0.8 | `location_prior.box_ratio`, `scale_boxes` | about the box centre |
| mask `M` per level | `boxes_to_level_masks` | a cell is foreground if its centre lies in a box (the box divided by the stride); a box smaller than a cell still marks the cell holding its centre |
| Eq. 1 | `PixelAggregator` | SELSA's multi-head form, residual |
| Eq. 2 | `EOVOD._enhance` | enhanced pixels written back into a copy of the map; `query_chunk` bounds the heads × queries × keys tensor without changing the result |
| key set: pixels in detected boxes on the 14 reference frames | `PixelMemory` (default `update=False`, no caps), `_gather_reference_keys` | gathered once per video from the frames `test_with_adaptive_stride` supplies; detection on them in chunks of `ref_chunk_size` |
| skip without a validated box | `_enhance` passes levels through | the first frame of every video, and any frame after one with no validated box |
| size prior, *T* | `size_prior.interval`; `FCOSHead(..., level_ids=...)` | the head runs a subset of levels; `with_levels=True` reports each detection's level; `interval` frames run only the recorded levels, then a full frame |
| training: GT boxes make the mask and pick queries and keys; two support frames | `forward_train` | keys from the support frames' boxes (random pixels when a support frame has none, so DDP sees every parameter used); queries from the key frame's boxes at ratio *r*; every level runs |

Everything EOVOD-specific is `vfe/models/vid/eovod.py` (about 480 lines).

**Options beyond the paper, all off by default:**

1. `memory.update=True` with `capacity` / `num_keys` / `write_per_frame`:
   a MAMBA-style bank that also takes every frame's validated pixels, with
   random replacement once full and a random subset read per frame.
   `num_keys` alone caps the paper's key set, for small GPUs.
2. `size_prior.keep_higher_levels=True`: run every level above the lowest
   validated one, not only the levels boxes came from. The high levels cost
   almost nothing (Table 2), and an object growing into the next level is then
   not lost until the next full frame.
3. `location_prior.bootstrap_first_frame=True`: the reference frames include
   the video's first frame, so its plain detections can be its own prior — one
   extra head pass per video instead of a plain first frame.
4. `location_prior.train_jitter=0.1`: a random shift and rescale of the
   ground-truth boxes standing in for the previous frame's detections, as a
   fraction of box size, to mimic the motion between frames.

Two implementation points: only torch's generator is used, on the feature
device, so exact gradient accumulation (`RngStreams`) covers all of it (the
released code drew from numpy, which the per-virtual-rank swap does not
cover); and both metrics are reported, `evaluation = dict(vid_style=True,
coco_style=True)` — the VID AP50 with the motion breakdown for the MAMBA /
STPN table, and COCO AP / AP50 / AP75 / small / medium / large for the paper's
(`vfe/evaluation/coco.py`, on pycocotools).

## Verification so far

- **FCOS vs mmdet 2.19.1** (`tools/checks/parity_fcos.py`, both sides on CPU
  in this container; the mmdet side from the `v1.0.0` tree): forward outputs
  on five levels, grid points and valid flags, pre-NMS candidates, post-NMS
  detections (sorted by label, score, x1), size-range targets, the three
  losses and every parameter gradient on a two-image batch, and the focal and
  IoU losses alone. float64: 82/82 within 1e-15 relative (40 bit-exact).
  float32: forward 8.5e-7, decode 2.4e-6, gradients 7.8e-4 (the
  classification tower, whose gradient sums 12k × 30 focal terms). The harness
  is standalone for now; it joins `run_parity.py`'s matrix when frozen on the
  machine with the legacy environment.
- **Unit tests** (`tests/test_one_stage.py`, `tests/test_eovod.py`,
  `tests/test_coco_eval.py`, 26 tests, CPU, seconds): point coordinates and
  valid flags against hand-computed values; the distance coder's round trip
  and clipping; focal loss against its formula with index and one-hot
  targets; IoU loss modes and the all-zero-weight case; FCOS target assignment
  by size range on a 32×32 image; loss, decoding, level subsets and the level
  report; box scaling; mask projection including the tiny-box rule; the fixed
  key set and the optional bank; the aggregator's residual form; query
  chunking leaving the result and the background unchanged; the size-prior
  rule (the paper's level set, the superset option, *T* = 0, no validated
  box); key gathering and the first-frame prior with and without the
  bootstrap; EOVOD rejecting unknown options and two-stage detectors; a
  training step with gradients reaching every aggregator, with and without
  support-frame boxes, and the same through the trainer's `train_step` with
  two micro-batches; four frames of stateful inference (fixed key set,
  restricted then full levels, reset on a new video, missing `frame_id`); the
  updating memory; plain detection when nothing validates; all three configs
  building at 32,443,240 and 51,435,368 parameters; and COCO-style
  evaluation on a synthetic annotation file (perfect detections 1.0, none
  0.0, shifted labels 0.0).
- **CI-equivalent checks** pass locally: ruff, all 61 tests, no mm\* module
  at runtime, every config building, the CLI entry points, Python 3.8 syntax
  in `tools/checks/`, the manifest, and the sdist's contents.

## Running it

Configs: `configs/vid/eovod/eovod_fcos_r101_fpn_3x.py` (the paper's
setting), `eovod_fcos_r101_fpn_9x.py` (the released checkpoint's recipe) and
`eovod_fcos_r50_fpn_3x.py` (quick). Test frames stay in order — do not set
`shuffle_video_frames`: the location prior comes from the previous frame and
the size prior counts frames.

```bash
python -m pytest                                    # 61 tests, CPU

# smoke: 300 iterations on the RTX 4060 (batch 1, no accumulation, no evaluation)
python -m vfe.cli.train configs/vid/eovod/eovod_fcos_r50_fpn_3x.py \
    --work-dir work_dirs/eovod_smoke --max-epochs 1 --max-iters-per-epoch 300 --no-validate
# inference smoke on two videos (cap the keys on an 8 GB GPU)
python -m vfe.cli.test configs/vid/eovod/eovod_fcos_r50_fpn_3x.py work_dirs/eovod_smoke/latest.pth \
    --work-dir work_dirs/eovod_smoke --max-videos 2 --cfg-options model.memory.num_keys=4096

# Isambard: four GH200s x 1 micro-step = the paper's batch of 4
torchrun --standalone --nproc_per_node=4 -m vfe.cli.train \
    configs/vid/eovod/eovod_fcos_r101_fpn_3x.py --launcher pytorch \
    --accumulate 1 --work-dir work_dirs/eovod_r101_3x --seed 0

# the size prior is inference-only, so one trained model gives both rows:
torchrun --standalone --nproc_per_node=4 -m vfe.cli.test \
    configs/vid/eovod/eovod_fcos_r101_fpn_3x.py work_dirs/eovod_r101_3x/epoch_3.pth \
    --launcher pytorch --work-dir work_dirs/eovod_r101_3x                    # T = 7: 53.8 / 76.9
torchrun --standalone --nproc_per_node=4 -m vfe.cli.test \
    configs/vid/eovod/eovod_fcos_r101_fpn_3x.py work_dirs/eovod_r101_3x/epoch_3.pth \
    --launcher pytorch --work-dir work_dirs/eovod_r101_3x_lpn \
    --cfg-options model.size_prior=None                                      # LPN only: 54.1 / 79.8
# the single-frame FCOS baseline (49.8 / 73.3) from the same weights: never validate, never skip
#   --cfg-options model.location_prior.score_thr=1.1 model.size_prior=None
```

What to look at first: the log's `loss_cls` / `loss_bbox` / `loss_centerness`
falling from about 1.1 / 0.7 / 0.65 (a fresh FCOS on 30 classes) — the first
`loss_cls` is dominated by the 1% prior init and should drop fast; peak
memory per GPU (three 1000×600 frames through R-101-FPN plus the attention;
expect well under 20 GB); and seconds per iteration. At batch 4 an epoch is
27,422 iterations; MAMBA's batch-4 epochs ran at about 0.09 s each on four
GH200s (three epochs in 1 h 58 min), and FCOS-FPN with three frames per
sample should be in the same range, so the 3-epoch run is likely 3–4 hours,
≈15 GPU-hours.

Inference cost: the attention tensor per level is heads × queries × keys ×
4 bytes; with 14 reference frames of large objects the stride-8 level can hold
tens of thousands of keys. `query_chunk` (1024) bounds the tensor; on an 8 GB
GPU also set `model.memory.num_keys=4096`, which subsamples the key set.

**The frame inspector** (`tools/eovod_inspector/`). `dump.py` runs a
checkpoint's own `simple_test` over the first frames of a few val videos and
records, per frame, the detections against ground truth, the prior and its
masks, score maps with and without aggregation, the levels run, the key set,
and where two queries (top detection, first ground-truth box) attend on the
reference frames; it builds the model from the checkpoint's saved training
config. `server.py` (standard library only) serves those dumps as a web page
and runs new dumps on its GPU on request. On Isambard:

```bash
sbatch tools/isambard/inspect.sbatch            # 1 GPU, 4 h; cancel when done
grep -A4 "On your laptop" eovod_inspect_<jobid>.out
# on the laptop:  ssh -N -L 8765:<node>:8765 b5cs.aip2.isambard
# then open the printed http://localhost:8765/?token=... link
```

Dumps live in `/projects/b5cs/vfe/viz` (`runs/*.json`, `img/`); the page's
*Diagnostics* section reads `runs/diagnostics.json`, kept by hand.

**Optional: the released checkpoint's FCOS.** Its `detector.*` keys match
this implementation's names (the head's `cls_convs.N.{conv,gn}`, `conv_cls`,
`conv_reg`, `conv_centerness`, `scales.N.scale`), so
`load_checkpoint(model, ckpt)` loads the backbone, FPN and head, logs
`memory.*` as unexpected and `aggregators.*` as missing, and evaluating gives
a plain-FCOS score from weights trained with the released aggregation. It is
not a check of this implementation, since the aggregator differs by design.

## Milestones

- [x] **9a — one-stage machinery**, checked against mmdet (above).
- [x] **9b — EOVOD's modules**, from the full paper, unit-tested on a tiny FCOS.
- [x] **9c — COCO-style evaluator**, on a synthetic annotation file.
- [ ] **9d — smoke on your machine:** the unit tests, a few hundred training
  iterations of the R-50 config on real data, `python -m vfe.cli.test ...
  --max-videos 2` on the resulting checkpoint (stateful inference on real
  videos), and `parity_fcos.py` against your `vfe` conda env; then freeze it
  into `run_parity.py`'s matrix.
- [x] **E-M2 — a short Isambard run** (500 iterations of the R-101 3x config,
  batch 4): speed, peak memory, the LR schedule, the loss curve.
- [x] **E-M3 — the full R-101 3x run** (done as the v4 recipe at 3 and at 9 epochs; see the Progress log), evaluated twice from the same
  weights: LPN only → COCO AP within ±0.5 of 54.1 (AP50 79.8); with *T* = 7 →
  53.8 (AP50 76.9). Plus the VID AP50 for the MAMBA / STPN table. The 9x
  recipe is the fallback if 3x lands low: the released checkpoint trained on
  it, at three times the cost.
- [x] **Speed** on a fixed GPU (one GH200; `tools/eovod_speed.py`): FPS with and without the size prior (the
  paper: 20.4 → 26.9 on a V100), since that is the paper's second claim.
- [ ] **Ablations**, if the budget allows: *r* ∈ {0.5, 0.8, 1.0} and
  *T* ∈ {0, 7, 14} are inference-only, so cheap; this implementation's own
  options (`keep_higher_levels`, `bootstrap_first_frame`, `memory.update`)
  likewise; `train_jitter` and `aggregator.shared=False` need retraining.

## Decisions needed

1. **Memory on small GPUs.** The paper caps nothing; the key set from 14
   frames of large objects can be tens of thousands of pixels at stride 8.
   Keep the default uncapped for E-M3 on the GH200s, and cap only for the
   RTX 4060 smoke (`model.memory.num_keys=4096`)?
2. **Isambard budget:** E-M2 ≈ 0.2 GPU-hours; E-M3 ≈ 15 GPU-hours if the
   iteration time matches MAMBA's; the two evaluations ≈ 1.5 GPU-hours each.
3. **Acceptance:** ±0.5 COCO AP against 54.1 (LPN) and 53.8 (LPN + SPN), as
   for the other models' ±0.5 AP50?
4. **The training leak (E-M3, 2026-09-27)** — *chosen: (a), 2026-09-27; see the Progress log.* Training on the paper's text —
   ground-truth boxes as the mask — lets the head read aggregation as the
   label. Options, each needing a retrain (≈ 11 GPU-hours for 3 epochs):
   (a) keep the paper's masked training but decorrelate the mask from the
   objects: a share of samples with no prior (plain FCOS, which is also what
   the first frame of every test video sees), and on the rest ground-truth
   boxes dropped at random, jittered (`train_jitter`), plus random distractor
   boxes; (b) the released code's way: every key-frame pixel a query in
   training, the location prior used only at test time for speed; (c) also a
   plain FCOS R-101 3x run, the paper's 49.8 baseline, which checks this
   repository's FCOS training on VID independently of EOVOD.

## The released code (audit, 2026-09-26)

Source: [guanxiongsun/EOVOD](https://github.com/guanxiongsun/EOVOD) at
`84576bb`. Its `mmdet/` is `v1.0.0`'s MMDetection 2.19.1 fork plus two files:
`mmdet/models/vid/fcos_att.py` (`FCOSAtt`, 292 lines) and
`mmdet/models/memory/mpn.py` (`MPN`, 346 lines). Of the 33 other files that
differ, 20 differ only in whitespace; the rest give `MemoryBank` a SELSA
aggregator, make evaluation COCO-only, drop the frame shuffle, and change
registrations.

- **It does not import as published:** `mmdet/models/__init__.py` names a
  `CenterNetAtt` defined nowhere (and `tools/speed_test.py` a `YOLOAtt`).
  With those two references removed it runs on the `vfe` legacy stack
  (Python 3.8, torch 1.10.1, mmcv-full 1.3.17) rather than its pinned torch
  1.8: both configs build, train a step and run stateful inference on CPU.
- **Its model:** stock FCOS with `MPN` between backbone and FPN
  (`before_fpn=True`): one `MemoryBank` per backbone level from
  `start_level` (C4–C5 for the released R-101 config), every pixel attending
  over up to 2,000 random pixels of the memory through a SELSA aggregator,
  residually. Training keys: 2,000 random pixels per reference frame per level
  (`np.random`). Test writes: pixels inside detections scoring above 0.3, at
  most 300 per box and 1,000 per frame, else the 50 highest-norm pixels.
  Nothing is called LPN or SPN; `filter_with_mask` is always called without a
  mask, so every pixel is a query, and no level is ever skipped.
- **How its 79.7 was measured:** COCO-style AP50 (54.0 / 79.7 / 59.3; small /
  medium / large 9.8 / 26.6 / 60.4), frames in order, 14 references over ±7
  frames at the first frame, and **the memory reset only when the video's
  number is a multiple of 1000**: VID val numbers its 555 snippets in 178
  blocks, so 377 snippets start with memory left by earlier snippets.
- **Recipe (its config):** SGD lr 0.001, momentum 0.9, weight decay 1e-4,
  clipping at 35, warmup 500 from 1/3, ×0.1 after epoch 6 of 9, one image per
  GPU on eight GPUs; caffe-style ResNet-101 from
  `open-mmlab://detectron/resnet101_caffe`. The checkpoint and log are on
  Google Drive, linked from its README.

## Where each part can run

- **Cloud sessions (this container):** both stacks build on CPU from PyPI
  alone — the legacy one in 3 min 20 s with the recipe below — so the parity
  harnesses and the unit tests run here. Until the environment was given full
  network access, Hugging Face, Google Drive, the OpenMMLab and PyTorch
  download hosts and the paper's hosts were blocked; checkpoints and data
  still live on your machine.
- **Your machine:** the data and checkpoints, CUDA comparisons, freezing
  goldens, the `vfe` conda env.
- **Isambard:** evaluation and training.

```bash
uv python install 3.8 && uv venv --python 3.8 legacy38
PY=legacy38/bin/python
uv pip install --python $PY torch==1.10.1 torchvision==0.11.2
uv pip install --python $PY numpy==1.23.5 "opencv-python-headless<5" "matplotlib<3.8" pycocotools \
    "scipy<1.11" yapf==0.32.0 addict terminaltables pyyaml packaging Pillow six "setuptools<60" wheel ninja
MMCV_WITH_OPS=1 FORCE_CUDA=0 MAX_JOBS=4 uv pip install --python $PY --no-build-isolation mmcv-full==1.3.17
# mmdet comes from a checkout on PYTHONPATH: the v1.0.0 tree, or EOVOD with the import patch
PYTHONPATH=/path/to/v1.0.0 $PY tools/checks/parity_fcos.py --impl mmdet --out fcos_mmdet.pt
python tools/checks/parity_fcos.py --impl vfe --out fcos_vfe.pt
python tools/checks/parity_fcos.py --compare fcos_mmdet.pt fcos_vfe.pt        # add --dtype float64 to both runs for the exact check
```

## Progress log

- **2026-09-30 (the paper reproduced at 9 epochs; the centerness controls)**
  — The best one-epoch design at 9 epochs (job 6954576, 7 h on 4 GH200s):
  **54.0 / 79.2 / 59.3** (APs / APm / APl 10.8 / 26.8 / 60.4; VID AP50
  79.7, fast 59.3); with SPN (margin 1) 53.8 / 78.9 / 59.2 (VID 79.4). The
  paper: LPN 54.1 / 79.8 / 59.5, + SPN 53.8 / 76.9 / 58.9. Against FCOS
  alone on the same schedule (49.8), +4.2 AP, the paper's +4.3. The two
  one-epoch controls separate the centerness change: FCOS alone with
  `centerness_on_reg` 35.2 / 62.1 / 36.5 (without it 33.7 / 59.8 / 35.0,
  so +1.5 AP on its own), but the paper-text design with it 35.3 / 63.1 /
  35.4 (LPN; plain 33.1) against 36.0 / 63.2 / 37.8 without. The change
  helps where the classification tower is aggregated on every step (the
  before-FPN design: 35.0 -> 36.9) and not where the prior aggregates a
  quarter of the steps' cells. Speed of the 9-epoch model: job 6959332.
- **2026-09-29 (centerness on the regression tower: the best one-epoch
  model)** — `eovod_fcos_r101_fpn_3x_backbone_cls_ctrreg.py` (the combined
  variant with `centerness_on_reg=True`). Full val, one epoch: **36.9 /
  65.4 / 38.3** (APs / APm / APl 8.0 / 17.9 / 42.0; VID AP50 65.8, fast
  45.8); with SPN (margin 1) 36.7 / 65.1 / 38.1. AP75 recovers from 35.1 to
  38.3 — above v4's 37.8 — which confirms that centerness learned on
  always-aggregated classification features was the AP75 cost. Against v4:
  +0.9 AP, +2.2 AP50, +2.2 VID AP50, +5.6 on fast objects. Not yet
  separated: how much `centerness_on_reg` gives on its own (mmdet reports
  it helping plain FCOS too).

- **2026-09-29 (before the FPN, classification only, at one epoch)** —
  `eovod_fcos_r101_fpn_3x_backbone_cls.py`: the released code's aggregation
  with the aggregated maps feeding the classification tower only (the FPN
  runs a second time on the original backbone maps for the regression
  tower). Full val, one epoch: **35.0 / 64.5 / 35.1** (APs / APm / APl 8.6 /
  18.5 / 39.4; VID AP50 64.9, fast 42.6); with SPN 34.8 / 64.2 / 34.9.
  Against v4 (36.0 / 63.2 / 37.8) and the released-style variant (35.6 /
  66.5 / 34.3), it lands between them on AP50 and AP75 and below both on
  AP. Clean regression inputs recovered only 0.8 of the released-style
  model's 3.5-point AP75 loss. What the three share with v5 (whose AP75
  also fell when the prior was on 90% of steps): FCOS's centerness here
  comes from the classification tower, so when that tower is aggregated on
  (nearly) every step, the centerness that ranks overlapping boxes learns on
  aggregated maps. The released-style and combined variants never train a
  plain step. Candidate next: `centerness_on_reg=True`, which moves
  centerness to the regression tower.

- **2026-09-29 (the released code's aggregation, at one epoch)** — The
  variant `eovod_fcos_r101_fpn_3x_backbone.py` (C4 and C5 aggregated before
  the FPN, every pixel a query, 2,000 random training keys per support
  frame, a memory bank at test time: the released R-101 model), trained one
  epoch on the same schedule and seed as v4. Full val:

  | 1 epoch | AP | AP50 | AP75 | APs / APm / APl | VID AP50 (fast) |
  | :-- | --: | --: | --: | :-- | --: |
  | v4 (paper's text), plain | 34.3 | 59.8 | 36.3 | 5.4 / 15.3 / 39.4 | 60.1 (36.6) |
  | v4 (paper's text), LPN | **36.0** | 63.2 | **37.8** | 5.4 / 15.1 / 41.7 | 63.6 (40.2) |
  | released-style | 35.6 | **66.5** | 34.3 | 6.3 / 19.9 / 39.9 | **66.9** (41.7) |
  | released-style + SPN (margin 1) | 35.5 | 66.2 | 34.2 | 6.5 / 19.7 / 39.9 | 66.6 (41.2) |
  | released-style, no memory | 22.1 | 44.4 | 19.3 | | 44.6 (26.0) |

  Same AP, different trade: aggregating every pixel before the FPN gains
  3.3 AP50 (and small / medium objects) and loses 3.5 AP75, the same box
  degradation as v2's aggregating both towers — before the FPN, the
  regression tower reads aggregated maps too. It never trains without
  aggregation, so it cannot detect without its memory (22.1 AP).

- **2026-09-29 (larger prior boxes on the 9-epoch model)** — Inference
  only. On the 10-video subset (LPN, key cap) AP rose monotonically with the
  box ratio r and with a lower validation threshold: r 0.8 / 1.0 / 1.2 /
  1.5 at detection score > 0.3: 46.3 / 46.5 / 46.8 / 46.9, at > 0.2: 46.6 /
  46.9 / 47.2 / 47.4 (plain 45.1). Full val: LPN at r 1.2, > 0.2 **52.2** /
  77.7 / 57.0 (VID AP50 78.2); r 1.5, > 0.2 52.2 / 77.5 / 57.2; LPN + SPN
  (margin 1) at r 1.2, > 0.2 **52.2** / 77.6 / 57.0 — +0.5 AP over the
  default (r 0.8, > 0.3). The cost is speed: about 3,500 queries per frame
  instead of 1,500, aggregation 3.2 → 4.5 ms, FPS −7% (fast engine: LPN +
  SPN 73.2 → 68.3). The paper's Table 4 shows the same trade (r 0.8 → 1.2:
  +0.1 AP, −14% FPS) and keeps r = 0.8; so does the default here. Best
  result so far: 52.2 AP, 1.9 short of the paper's 54.1.

- **2026-09-29 (9 epochs)** — The v4 recipe on the released code's
  schedule (`eovod_fcos_r101_fpn_9x.py`: 9 epochs, batch 8 as 4 GH200s × 2
  micro-steps, ×0.1 after epoch 6), 7 h 58 min. Full val:

  | 9 epochs | AP | AP50 | AP75 | APs / APm / APl | VID AP50 | slow / med / fast |
  | :-- | --: | --: | --: | :-- | --: | :-- |
  | plain | 49.8 | 73.6 | 54.6 | 8.8 / 23.6 / 56.2 | 74.0 | 81.5 / 71.8 / 49.4 |
  | LPN (key cap) | **51.7** | **77.1** | 56.5 | 8.6 / 25.4 / 58.4 | **77.6** | 83.8 / 75.8 / 54.8 |
  | LPN + SPN, margin 1 (default) | **51.7** | 76.9 | 56.4 | 8.6 / 25.3 / 58.3 | 77.4 | 83.8 / 75.6 / 54.2 |
  | LPN + SPN, margin 0 (paper's rule) | 50.9 | 75.5 | 55.7 | 8.5 / 24.5 / 57.6 | 76.0 | 83.0 / 74.0 / 51.7 |
  | paper: FCOS / LPN / LPN + SPN | 49.8 / 54.1 / 53.8 | 73.3 / 79.8 / 76.9 | 54.6 / 59.5 / 58.9 | | | |

  The plain path equals the paper's FCOS; EOVOD misses 54.1 by 2.4 AP,
  all of it in the prior's gain (+1.9 against +4.3). Speed (one GH200,
  eager / fast engine, FPS over a mean-length video): plain 42.0 / 105.3;
  LPN with the key cap 34.5 / 73.0; LPN + SPN margin 1 41.4 / 78.5; margin
  0 45.1 / 88.4; LPN uncapped 28.2 / 44.4 (35.7k keys, aggregation 11.7
  ms). Eager, the paper's ratios reproduce: −18% / +7% for LPN / LPN + SPN
  (margin 0) against −19% / +7%.

- **2026-09-28 (fewer syncs in EOVOD's inference)** — `EOVOD._enhance`
  now gathers every level's masked cells with one `nonzero` (and one small
  transfer to split them) instead of up to three syncs per level, and
  `_after_frame` uses one `nonzero`; a test pins `_enhance` to the old
  boolean-mask version, outputs and gradients. `tools/eovod_speed.py` now
  takes FPS from a clean pass (no synchronisation of its own) and the
  breakdown from a second, instrumented pass. v4 (3 epochs), fast engine,
  ms per frame: plain 9.3–9.5 (102–104 FPS); LPN with the key cap 14.7 →
  13.3 (72.9 FPS); LPN + SPN margin 1 13.3 → 12.3 (77.6 FPS); margin 0
  12.4 → 11.1 (87.4 FPS). Aggregation 3.2 → 2.7 ms; bookkeeping about
  1.8 ms over plain. With the whole head at 4.8 ms of a 9.4 ms frame, SPN
  saves 8% (margin 1) to 17% (margin 0) of LPN's time and cannot bring
  EOVOD to plain FCOS's speed on this GPU; the paper's "faster than FCOS"
  rests on a head that dominates the frame (V100, eager).

- **2026-09-28 (the v4 recipe as default; the 9-epoch run; a fast
  inference engine)** — The user accepted SPN as validated by mechanism and
  chose v4's recipe for the 9-epoch run, with speed engineering in parallel.
  The v4 recipe is now the configs' default (each departure commented in
  `eovod_fcos_r50_fpn_3x.py`). The first 9-epoch attempt got its 4 GPUs on
  two nodes and `torchrun --standalone` saw one; `--nodes=1` is now in the
  4-GPU job scripts. The 9-epoch run (`eovod_r101_9x_v4`, 4 GH200s × 2
  micro-steps = batch 8, 13,711 iterations per epoch) is training.
  **Fast engine** (inference only, bitwise the same detections): CUDA
  graphs for backbone + FPN and the head's per-level forward
  (`vfe/engine/cuda_graphs.py`, one private graph per input shape), a
  post-processing path with one GPU sync per image instead of one per level
  (`FCOSHead.one_sync_postprocess`; a stable sort over all scores with the
  sub-threshold ones set to −inf, so the candidates and their order equal
  mmdet's), and cuDNN autotuning. On v4 (3 epochs), per frame: plain 23.6 →
  **9.3 ms** (41.8 → 103.5 FPS; backbone + FPN 12.8 → 4.2, head 10.4 →
  4.7); LPN with the key cap 30.4 → 14.7 ms; LPN + SPN margin 1 25.9 → 13.3
  ms; margin 0 23.3 → 12.4 ms. With the engine, SPN saves 10% (margin 1) to
  16% (margin 0) of LPN's time, but EOVOD's eager aggregation (3.2 ms) and
  prior bookkeeping (2.4 ms, many small GPU→CPU syncs) now exceed what SPN
  can save from a 4.7 ms head, so EOVOD + SPN runs at 71–76% of plain
  FCOS's frame rate.

- **2026-09-28 (why the size prior buys little speed on a GH200)** — v4 at
  3 epochs, LPN + SPN (T = 7) + key cap, full val: margin 0 48.3 (−1.1
  against LPN), lower neighbour only (`margin=1, margin_up=0`) 48.7 (−0.7),
  margin 1 49.3 (−0.1); on the dumped videos the objects on skipped levels
  are 5.8% / 0.9% / 0%. Timed: 24.9 / 25.7 / 27.2 ms against plain
  24.4–25.6 ms and LPN 31.4 ms. A per-level profile of the head (one GH200,
  batch 1, a 608×1024 input) explains the ceiling: every level costs
  2.4–2.6 ms alone — P3 (76×128) and P7 (5×8) alike, forward about 1.2 ms
  and post-processing about 1.2 ms — so the head is launch- and
  synchronisation-bound, and skipping a level saves about 2 ms whichever it
  is; backbone + FPN take 13.8 ms. On the paper's V100, P3 was 65% of the
  head and the head 80% of the frame. The size prior works as a mechanism
  (with margin 1 it keeps the accuracy and cuts LPN's time by 13%); its
  speed benefit over plain FCOS depends on the head being compute-bound,
  which it is not in eager PyTorch on this GPU.

- **2026-09-28 (v4 at 3 epochs: LPN works; the size prior's trade-off on a
  GH200)** — v4 resumed to 3 epochs. Full val (all validated at detection
  score 0.3):

  | 3 epochs | AP | AP50 | AP75 | VID AP50 | VID fast |
  | :-- | --: | --: | --: | --: | --: |
  | FCOS alone | 47.5 | 72.5 | 52.0 | 72.9 | 49.4 |
  | v4, plain | 47.7 | 72.1 | 52.2 | 72.6 | 48.5 |
  | v4, LPN | **49.4** | **76.0** | 53.3 | **76.5** | **54.5** |
  | v4, LPN + SPN (T = 7, margin 1) + key cap | 49.3 | 75.7 | 53.3 | 76.2 | 53.8 |
  | v4, same, no P3 added (`margin_min_level=1`) | 49.2 | 75.5 | 53.2 | 76.0 | 53.1 |

  LPN over FCOS: +1.9 AP, +3.5 AP50, +5.1 VID AP50 on fast objects (the
  paper: +4.3 / +6.5). Speed (v4; FPS against plain's 39.3): LPN −33%
  uncapped, −15% with the key cap; LPN + SPN with the key cap +8% at margin
  0, −9% at margin 1 (the paper: −19%, then +7%). Forbidding the margin to
  add P3 saved no head time: on a GH200 the head's cost follows the number
  of levels run (each carries launch-bound per-level work) more than P3's
  size, unlike the paper's V100. Since every size-prior miss was an object
  recorded one level above where it belongs, `size_prior.margin_up` now
  lets the margin add only the lower neighbour; that is being evaluated.

- **2026-09-28 (the size prior fixed; the key cap is free)** — The user's
  gate before any 9-epoch run: at 3 epochs, LPN must give a clear gain over
  FCOS and SPN must restore the speed for a small AP loss. On v3 at 3
  epochs (all validated at detection score 0.3), full val:

  | v3, 3 epochs | AP | AP50 | AP75 | APs / APm / APl | VID AP50 |
  | :-- | --: | --: | --: | :-- | --: |
  | LPN | 48.9 | 74.7 | 53.3 | 8.1 / 22.6 / 55.6 | 75.2 |
  | LPN, `memory.num_keys=4096` | 48.9 | 74.7 | 53.3 | 8.1 / 22.6 / 55.6 | 75.2 |
  | LPN + SPN, T = 7 | 47.7 | 72.7 | 52.2 | 7.6 / 20.9 / 54.7 | 73.1 |
  | LPN + SPN, T = 7, `keep_higher_levels` | 48.1 | 73.3 | 52.6 | 7.6 / 21.1 / 55.1 | 73.7 |
  | LPN + SPN, T = 3 | 48.1 | 73.2 | 52.6 | 7.6 / 21.3 / 55.0 | 73.6 |
  | LPN + SPN, T = 7, `size_prior.margin=1` | **48.7** | **74.4** | **53.2** | 7.8 / 22.3 / 55.6 | **74.9** |

  The key cap costs nothing (peak memory 52.5 → 3.8 GB). The size prior's
  loss was objects near a level boundary: detected on P6 on the full frame
  but assigned to P5 by size, so on partial frames their level was not run
  — 7.3% of the dumped objects on partial frames, detected 57% of the time.
  The new `size_prior.margin` (run the neighbours of each recorded level;
  inference only) cuts that to 0.9% and the size prior's cost to 0.2 AP
  (the paper: 0.3). Speed (one GH200, warmed up, plain repeated as a
  check), v3: plain 24.4 ms / 40.1 FPS; LPN 38.3 ms / 25.8 FPS (33k keys,
  aggregation 11.7 ms); LPN with the key cap 30.5 ms / 32.3 FPS; LPN + SPN
  (T = 7) with the key cap 24.1 ms / 40.6 FPS — the paper's pattern (−19%,
  then +7% against FCOS). Margin 1 runs more levels and is being timed; v4
  at 3 epochs is training for a larger LPN gain.

- **2026-09-28 (v5, and the first speed measurements)** — **v5**
  (`train_plain_prob=0.1`), one epoch: plain 31.8 / 58.8 / 31.6, LPN
  (detection 0.3) 34.1 / 63.2 / 33.7. LPN AP50 equals v4's but AP75 falls
  4–5 points in both modes, so `train_plain_prob=0.25` (v4) is the best of
  0.5 / 0.25 / 0.1. The regression tower reads clean features in both; the
  likely cause is centerness, which here comes from the classification
  tower and so from aggregated features on nearly every step
  (`centerness_on_reg=True` would move it to the clean tower).
  **Speed** (`tools/eovod_speed.py`: one GH200, the first 200 frames of 8
  val videos preloaded on the GPU, synchronised timing, ms per frame after
  each video's first; FPS over a mean-length val video). v4: plain 24.4 ms
  (40.3 FPS); LPN (detection 0.3) 32.3 ms (30.5 FPS; aggregation 6.4 ms,
  engaged on 64% of frames, 22.8k keys on average, p90 52 ms); LPN with the
  key set capped at 4,096 29.3 ms (aggregation 2.6 ms, p90 32.5 ms, AP on
  the timed frames unchanged); LPN + SPN (T = 7) 26.6 ms (36.9 FPS; head
  10.7 → 5.4 ms). v5 engages less and gathers fewer keys (58%, 13.7k), so
  its aggregation costs 3.7 ms. About 2 ms per frame is the priors'
  bookkeeping. TF32 matmuls save about 1 ms of aggregation. The first frame
  of a video (plain detection on 14 references) costs 155–180 ms. The v3
  rows of that first run were measured before the GPU reached its steady
  clock (v3 plain 39.5 ms where v4/v5 plain took 24.4–24.9 ms); the tool
  now warms up for two minutes first and repeats plain last as a check.

- **2026-09-28 (FCOS at 3 epochs, v4, and the released recipes)** — FCOS
  alone (`train_plain_prob=1`), resumed from its epoch-1 checkpoint to 3
  epochs: full val **47.5** / 72.5 / 52.0 (APs/m/l 8.7 / 22.8 / 53.6; VID
  AP50 72.9), 2.3 AP below the paper's FCOS and 0.3 below v3's plain path, so
  sharing the detector with EOVOD costs nothing and the base gap is the FCOS
  recipe. **v4** (v3 with `train_plain_prob=0.25`), one epoch: plain 34.3 /
  59.8 / 36.3, LPN (detection 0.3) **36.0 / 63.2 / 37.8** (VID AP50 63.6,
  fast 40.2) — the prior's gain doubles (+1.7 AP, +3.4 AP50 against v3's +0.8
  / +1.9) with plain detection intact. The released repository
  (`84576bb`) explains much of the remaining gap to the paper: its EOVOD
  configs are this recipe at **9 epochs, batch 8, ×0.1 after epoch 6**
  (ImageNet caffe R-101, lr 1e-3, linear warmup, no paramwise settings), which
  produced the 54.0 checkpoint; its R-101 FCOS VID baseline
  (`configs/vid/fcos/fcos_r101_fpn_vid.py`) starts from a **COCO-trained
  FCOS backbone** (`fcos_r101_caffe_fpn_gn-head_mstrain_coco_backbone.pth`)
  with `bias_lr_mult=2, bias_decay_mult=0`, constant warmup and 9 epochs.
  The paper's 49.8 FCOS row is most likely that run, not the 3-epoch
  ImageNet recipe the text describes. **v5** (`train_plain_prob=0.1`, one
  epoch) is running.

- **2026-09-27 (v3, 3 epochs: working, 5 AP short of the paper)** — v3
  resumed from epoch 1 to the 3-epoch recipe (1 h 39 min more; lr 1e-4 in
  epoch 3). Full val (COCO-style; VID AP50 with slow / medium / fast):

  | 3 epochs | AP | AP50 | AP75 | APs / APm / APl | VID AP50 | slow / med / fast |
  | :-- | --: | --: | --: | :-- | --: | :-- |
  | v3, plain | 47.8 | 72.4 | 52.4 | 9.1 / 22.4 / 54.1 | 72.9 | 80.8 / 70.4 / 47.5 |
  | v3, LPN, detection score > 0.3 | **48.9** | **74.7** | **53.3** | 8.1 / 22.5 / 55.6 | **75.2** | 82.5 / 73.1 / 51.3 |
  | v3, LPN, class score > 0.5 | 48.6 | 74.2 | 53.1 | 8.4 / 22.3 / 55.3 | 74.7 | 82.4 / 72.4 / 50.5 |
  | v3, LPN + SPN (T = 7), detection > 0.3 | 47.7 | 72.7 | 52.2 | 7.6 / 20.9 / 54.7 | 73.1 | 81.7 / 70.6 / 47.6 |
  | paper: FCOS / + LPN / + LPN + SPN | 49.8 / 54.1 / 53.8 | 73.3 / 79.8 / 76.9 | 54.6 / 59.5 / 58.9 | | | |

  The prior helps (+1.1 AP, +2.3 AP50, +3.8 VID AP50 on fast objects) but
  the result misses E-M3's acceptance (±0.5 of 54.1) by 5.2 AP: about 2 AP
  in the base detector (v3 plain 47.8 against the paper's FCOS 49.8; at one
  epoch v3 plain matched FCOS trained alone, but FCOS alone was not run to
  3 epochs) and 3.2 AP in the prior's gain (+1.1 against +4.3). The size
  prior costs 1.2 AP (the paper: 0.3), most of it on small objects.

- **2026-09-27 (one-epoch iterations: v2, the validation criterion, v3)** —
  Options (a) from *Decisions needed* item 4, implemented as
  `location_prior.train_plain_prob / train_drop_prob / train_distractors`
  (with `train_jitter`), on in the configs at 0.5 / 0.3 / 2 / 0.1. A first
  4-GPU attempt died in DDP on its first steps: a prior whose boxes were all
  dropped aggregates nothing, so the aggregators joined no graph; a
  zero-weighted term now keeps them in it on every step (test added).
  **v2, one epoch** (R-101, 4 GH200s × 1, seed 0, 50 min):
  - The leak is gone: with random boxes as the prior the best score on them
    is p50 0.034 / p90 0.088 (E-M3: 0.354 / 0.495).
  - Full val (176,126 frames), COCO AP / AP50 / AP75, VID AP50:

    | 1 epoch | AP | AP50 | AP75 | VID AP50 |
    | :-- | --: | --: | --: | --: |
    | FCOS alone (`train_plain_prob=1`), plain | **33.7** | **59.8** | **35.0** | **60.1** |
    | v2, plain | 32.2 | 58.6 | 33.0 | 59.0 |
    | v2, LPN, detection score > 0.5 | 31.9 | 58.6 | 32.4 | 58.9 |
    | v2, LPN, class score > 0.5 | 32.0 | 59.4 | 31.9 | 59.8 |

  - The validation criterion decides whether the prior runs. FCOS's
    detection score is class × centerness; at 0.5 on it the prior engaged on
    15% of the subset's frames. Subset AP gain over plain (10 videos, v2):
    detection score 0.5 / 0.4 / 0.3 / 0.2 → +0.3 / +1.8 / +3.8 / +5.1;
    class score 0.6 / 0.5 / 0.4 → +1.2 / +2.7 / +4.1. The paper says
    "classification score above 0.5", so `location_prior.validate_on`
    (`'score'` default, `'cls_score'`) now exists; FCOSHead can return each
    detection's class score (`with_cls_scores`).
  - Aggregation helps classification and hurts the boxes: the prior raises
    AP50 and lowers AP75 on full val, and on the dumped frames where it
    engaged (class score 0.5), the top same-class detection's IoU with its
    object fell from 0.818 to 0.804 and the share at IoU ≥ 0.75 from 0.81 to
    0.76 (worse on 200 objects, better on 127), while its score rose from
    0.41 to 0.46. v2's plain path trails FCOS mostly on AP75 (−2.0) too.
  - A turtle video showed the prior carrying a class confusion (the turtle
    detected as "lizard" at 0.45 plain, 0.55 aggregated), not a background
    false positive.

  **v3**: `aggregator.branches='cls'` feeds the aggregated maps to the
  classification tower only; the regression tower reads the original maps
  (FCOSHead takes `reg_feats`), so localisation trains cleanly on every step,
  as in FCOS. One epoch, full val:

  | 1 epoch | AP | AP50 | AP75 | VID AP50 | VID fast |
  | :-- | --: | --: | --: | --: | --: |
  | FCOS alone, plain | 33.7 | 59.8 | 35.0 | 60.1 | 33.3 |
  | v3, plain | 34.2 | 59.6 | 35.8 | 60.0 | 33.9 |
  | v3, LPN, class score > 0.5 | 34.7 | 60.6 | 36.4 | 60.9 | 34.5 |
  | v3, LPN, detection score > 0.3 | **35.0** | **61.5** | **36.6** | **61.8** | **36.2** |

  v3's plain path now matches FCOS and the prior improves every metric,
  AP75 included; on the dumped frames where it engaged, box IoU is unchanged
  (0.832 → 0.832, 28 objects worse, 27 better) while the matched score rises
  (0.43 → 0.49). The gain over FCOS (+1.3 AP) is below the paper's +4.3 at
  full training. v3 was resumed from epoch 1 to the 3-epoch recipe, with
  full-val evaluations of plain, LPN (detection 0.3, class 0.5) and LPN +
  SPN (T = 7) chained on it. The frame inspector (`tools/eovod_inspector/`,
  *Running it*) was built to review all of this.

- **2026-09-27 (E-M3: trained, failed on a training leak)** — The R-101 3x
  run (4 GH200s × 1, seed 0, `--no-validate`) took 2 h 38 min at 0.115
  s/iter and 14.3 GB per GPU; lr stepped to 1e-4 at epoch 3 as configured.
  Mean loss over each epoch's last 1,000 iterations: 0.943 / 0.862 / 0.773
  (`loss_cls` 0.177 / 0.129 / 0.086, `loss_bbox` 0.198 / 0.168 / 0.121,
  `loss_centerness` 0.569 / 0.566 / 0.566). The three evaluations of
  `epoch_3.pth` (T = 7, LPN only, plain) gave **identical** numbers: COCO AP
  1.3, AP50 5.1, AP75 0.4; VID AP50 5.0 (slow 5.4, medium 5.3, fast 3.5).
  Inference ran at 34–36 frames/s per GPU in all three (data-bound, so no
  FPS comparison). Diagnosis, all on this checkpoint:
  - The checkpoint loads completely (683/683 keys); the caffe backbone
    loaded with only its 104 zero conv biases unexpected, as in mmdet.
  - Scores are tiny on plain frames: the per-frame top score has median
    0.017, 99th percentile 0.127, maximum 0.351 over all 176,126 val
    frames. Nothing reaches 0.5, so neither prior nor the key set ever
    started — hence the identical results. Boxes are sometimes right
    (IoU 0.78 with the correct class) at scores of about 0.01.
  - Lowering the threshold on the first 10 val videos (LPN only, subset
    COCO AP / AP50): 0.5 → 0.5 / 2.2; 0.1 → 0.2 / 1.2 (scores rise to a
    0.65 maximum, AP falls); 0.05 → 0.6 / 2.1; 0 (every frame aggregated
    over its top-100 boxes) → 13.7 / 30.1.
  - Shortcut test on the first 5 videos: two random boxes per frame as the
    prior, keys from the references' top-100 boxes. The best detection
    matching a random box (IoU > 0.5) scores p50 0.354 / p90 0.495 with
    aggregation and 0.000 / 0.000 without; the best elsewhere 0.090 / 0.232
    and 0.015 / 0.065. The head boxes whatever region was aggregated.
  So the paper's training as implemented — the key frame's ground-truth
  boxes are exactly the aggregated region — teaches the head to detect the
  aggregation, not the object. The low training `loss_cls` is the leak, not
  a good fit. Options under *Decisions needed*, item 4.

- **2026-09-27 (Isambard: E-M2 and inference smoke)** — The Isambard checkout
  moved to `cb0dfa4`; the caffe-style R-50 / R-101 backbones went into
  `/projects/b5cs/vfe/torch_home/hub/checkpoints` (SHA-256 prefixes match
  their names). The 61 tests pass on a GH200 node. **E-M2** (500 iterations,
  4 GH200s × 1 micro-step, seed 0): 0.13–0.15 s/iter after start-up, 13.0 GB
  peak per GPU; lr warmed from 3.3e-4 to 1e-3; loss 4.09 → 1.66 (`loss_cls`
  1.03 → 0.61, `loss_bbox` 2.39 → 0.46, `loss_centerness` 0.68 → 0.59),
  grad norm 41 → 16 (clipping at 35 acted only in the first ~50 iterations).
  `loss_bbox` starts well above the 0.7 guessed below: the IoU loss in log
  mode on an untrained regression branch, at a warmup lr ten times COCO's.
  **Inference smoke** on the first two val videos (928 frames, one GH200)
  from E-M2's checkpoint: after 500 iterations no score passes 0.5 (max
  0.088), so the paper's thresholds ran plain FCOS; at `score_thr=0.05` the
  size prior's partial frames dropped 596 of 92,800 detections and LPN-only
  kept all of them; at `score_thr=0` (every box validated, the key set's worst
  case) the peak was 6.0 GB. So the uncapped key set fits a GH200 by far
  (decision 1). `vfe.cli.test` had no `--cfg-options`, which the commands
  below use; it now has, as in training. The full run is at 3 × 27,422
  iterations × ~0.14 s ≈ 3.2 h.

- **2026-09-27 (the paper in full)** — With the PDF in hand, four details of
  the first reading were corrected: the key set is the pixels inside the 14
  reference frames' detections, fixed for the video (not a growing memory);
  the first frame is detected plainly (no bootstrap); training uses the
  ground-truth boxes as they are (no jitter); and the size prior runs exactly
  the levels the validated boxes came from, with *T* = 7 meaning seven partial
  frames after each full one. Each earlier choice remains an option, off by
  default. The paper's tables are recorded above as the targets; the
  released checkpoint's 79.7 matches the LPN-only row. 61 tests pass.
- **2026-09-27 (recipe)** — Further excerpts gave the paper's FCOS recipe:
  3 epochs at batch 4, lr 1e-3 → 1e-4 after two, shorter side 600; the
  "batch 32" was CenterNet's. The R-101 3x config follows it and becomes the
  E-M3 target; the 9x config stays as the released checkpoint's recipe.
- **2026-09-27 (implemented)** — EOVOD built from the paper: FCOS ported from
  mmdet 2.19.1 and checked exact against it in float64 (82 artifacts,
  1e-15) and to the float32 accumulation floor; `EOVOD` with the location
  prior, the size prior and SELSA-style residual attention over the FPN
  outputs; COCO-style evaluation beside the VID metric; configs; unit tests.
  The paper's hosts were blocked at the time, so it was read through search
  excerpts. Nothing trained yet.
- **2026-09-26 (audit)** — Phase 9 opened. EOVOD's public code is `v1.0.0`'s
  MMDetection fork plus `FCOSAtt` and `MPN`. It fails to import
  (`CenterNetAtt` is imported but never defined); with that fixed it runs on
  the existing legacy stack, so one oracle environment serves both projects.
  Its 79.7 is COCO-style AP50, measured with memory carried across 377 of
  VID val's 555 snippets. Both stacks were built from PyPI in a cloud
  container, so CPU parity work can happen there.
