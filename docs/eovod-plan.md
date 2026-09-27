# EOVOD — Phase 9

> Working document for EOVOD in `vfe/`, the phase after 2.0.
> [rewrite-plan.md](rewrite-plan.md) records how MAMBA and STPN were ported and
> why the method is what it is. This phase is different in kind: EOVOD is
> **implemented from its paper**, not ported from its released code. Update the
> checkboxes and the Progress log as work proceeds.

- **Goal:** EOVOD (*Efficient One-stage Video Object Detection by Exploiting
  Temporal Consistency*, Sun, Hua, Hu, Robertson; ECCV 2022,
  [arXiv 2402.09241](https://arxiv.org/abs/2402.09241)) on plain PyTorch in
  `vfe/`, trained and evaluated on ImageNet VID with this repository's tooling.
- **Method:** read the paper, build its two ideas — the location prior and the
  size prior over a one-stage detector with pixel-level attention — on a
  faithfully ported FCOS, check the FCOS part against mmdet 2.19.1 the way
  every other layer of `vfe/` was checked, and judge the whole by task metrics
  once it is trained.
- **Not the goal:** reproducing the released code line by line. Its audit is
  kept below because it explains the published 79.7, and because its FCOS is
  the same mmdet FCOS this implementation is checked against.

## Status (2026-09-27)

- **Implemented and unit-tested; not yet trained.** Everything runs on CPU in
  this container on random weights; the data, GPUs and checkpoints are on
  your machine and Isambard.
- **The FCOS underneath is exact.** `tools/checks/parity_fcos.py` compares the
  head against mmdet 2.19.1 on CPU: in float64 all 82 artifacts agree to
  1e-15 relative; in float32, forward passes to 8.5e-7, decoding to 2.4e-6
  and gradients to 7.8e-4, the torch 1.10 → 2.10 accumulation floor
  documented for the training-step checks in the rewrite plan.
- **Next: a smoke run on your machine, then E-M3 on Isambard.** Commands and
  what to look at are under *Running it*.

## The paper, as read here

The paper's hosts (arXiv, ECVA, Springer, QUB's repository) are denied by this
environment's network policy, so it was read through search-engine excerpts of
the PDF. What follows is what those excerpts support; where an implementation
detail is not in them it is marked *chosen*, and the choice is under *Design*.

**Analysis.** (1) The accurate VID methods aggregate features with attention.
(2) In two-stage detectors that is cheap: the attention runs over ~300
proposals. (3) A one-stage detector has no proposals; FCOS's pyramid holds
~13k pixels, and attention over all of them is unaffordable. (4) About 80% of
a one-stage detector's head time goes on the low-level feature maps (~65% on
the lowest), which exist for small objects. *Temporal consistency*: objects
change gradually in location and size between consecutive frames.

**Location prior network (LPN).** Given the previous frame's detections,
those scoring above 0.5 are *validated*; their boxes are adjusted by a ratio
*r* (0.8 for FCOS: smaller *r*, fewer pixels, faster), projected to each
feature level by dividing by the stride, and turned into a binary foreground
mask. Attention-based aggregation runs on those pixels only ("partial feature
aggregation"). Without a validated box the aggregation is skipped. Queries
are pixels of the current frame; keys are pixels from other frames of the
video (at training time, randomly sampled reference frames).

**Size prior network (SPN).** One-stage detectors assign objects to levels by
size. After detecting at time *t*, the levels the validated boxes came from
are recorded; for the next *T* frames the heads run only on those levels
("if validated boxes are generated from the top level, there might not be
small objects in the following frames"). *T* = 7 for the reported trade-off.
SPN alone takes FCOS+LPN from 20.4 to 26.9 FPS and YOLOX+LPN from 35.8 to
50.5, at 53.8 / 52.7 AP.

**Training.** VID (15 frames per video) plus at most 2,000 DET images per
class. FCOS: images resized to a shorter side of 600 (longer side at most
1000); batch size 4; SGD with momentum 0.9 and weight decay 1e-4; 3 epochs,
lr 1e-3 for the first two and 1e-4 for the last; ResNet-101. CenterNet:
batch 32, lr 1e-4 for 50 epochs then 1e-5 for 30. YOLOX: DarkNet-53 at
640×640 with MixUp / Mosaic, lr 1e-3 on a cosine schedule for 80 epochs.
Inference uses **2 reference frames** — more ran a 32 GB V100 out of memory
for the naive one-stage adaptation, so the comparison kept 2 throughout.

**Results (ImageNet VID, COCO-style AP).** FCOS R-101: 49.8 → 54.1 AP with
LPN (+4.3), 53.8 with LPN+SPN at 26.9 FPS on a V100; RDN, the best
competitor, is 0.7 lower and ~3× slower. With *T* = 7 the size prior takes
YOLOX from 25.1 to 40 FPS in one table and YOLOX-M+LPN from 35.8 to 50.5 in
another. STPN's paper later quotes EOVOD at 54.1 AP / 79.8 AP50. The
released checkpoint (a 9-epoch, batch-8 run per its config) scores
54.0 / 79.7 / 59.3.

**Not in the excerpts:** the attention formulation, the memory's size and
update rule, how many keys a query sees, how the first frame is handled
beyond "skip", and whether aggregation runs before or after the FPN.

## Design: paper → `vfe/`

| Paper | Here | Notes |
| :-- | :-- | :-- |
| one-stage detector | `FCOS` / `FCOSHead` (`vfe/models/detectors/single_stage.py`, `vfe/models/dense_heads/fcos_head.py`) | ported from mmdet 2.19.1; plus `MlvlPointGenerator`, `DistancePointBBoxCoder`, `FocalLoss`, `IoULoss`, `Scale`, `open-mmlab://` URIs |
| validated detections | `location_prior.score_thr` (0.5) | one threshold feeds both priors and the memory |
| box ratio *r* | `location_prior.box_ratio` (0.8) | `scale_boxes` about the centre |
| projection to a level's grid | `boxes_to_level_masks` | a cell is foreground if its centre lies in a box; a box smaller than a cell still marks the cell holding its centre |
| partial feature aggregation | `EOVOD._enhance` + `PixelAggregator` | SELSA-style multi-head attention (16 heads), added residually to the query pixel |
| keys from other frames | `PixelMemory` | per level: a random-replacement bank of pixel features (capacity 4096), 1024 sampled per frame |
| size prior, interval *T* | `size_prior.interval` (7); `FCOSHead(..., level_ids=...)` | the head runs a subset of levels; `with_levels=True` reports each detection's level |
| skip without a prior | `_enhance` passes levels through | first frame, or a frame after one with no validated box |

Everything EOVOD-specific is `vfe/models/vid/eovod.py` (about 450 lines).

**Choices the paper left open, and the reasons:**

1. **Aggregation runs on the FPN outputs (P3–P7).** The location masks and
   the size prior are both per pyramid level, so the FPN outputs are the
   natural place; all levels share a channel width there, which lets one
   aggregator serve every level (`aggregator.shared=True`, 263k parameters).
   The released code instead attends over the backbone's C3–C5 before the
   FPN, with a 22M-parameter module.
2. **The size prior keeps every level from the lowest validated one upward**,
   not exactly the set of levels boxes came from. The paper's motivation is
   skipping low levels when there are no small objects; the high levels cost
   almost nothing, and keeping them means an object that grows into the next
   level is not lost until the next full frame.
3. **A frame with no validated detection at a full frame runs every level**
   afterwards: there is no evidence about sizes to skip on.
4. **The first frame's prior is its own plain detection.** The test sampler
   gives the first frame reference frames spread over the video, one of which
   is the frame itself. They are detected plainly to seed the memory, and the
   frame's own detections become its location prior — one extra head pass on
   one frame per video. The memory resets at every video, not on the released
   code's `video_id % 1000` quirk.
5. **Memory holds the enhanced features**, as MAMBA's memory holds enhanced
   RoI features, written from inside the validated boxes at ratio 1.0, at
   most 512 pixels per level per frame.
6. **Training mirrors inference with ground truth in place of detections.**
   Keys: pixels inside the reference frames' boxes (falling back to random
   pixels when a reference has none, so the aggregator — and DDP — see every
   parameter used on every step). Queries: pixels inside the key frame's
   boxes after a random jitter of ±10% in position and size (standing in for
   the motion between frames), then shrunk by *r*. Every level runs.
7. **Only torch's generator is used**, on the feature device, so exact
   gradient accumulation (`RngStreams`) covers all of it. The released code
   drew from numpy, which the per-virtual-rank swap does not cover.
8. **The recipe is the paper's FCOS recipe** — batch 4, SGD lr 1e-3 for two
   epochs then 1e-4, 3 epochs, shorter side 600 — in the 3x configs, with
   the released code's 500 warmup iterations from 1/3 and gradient clipping
   at 35, VID references 2 within ±9 frames plus DET, and 14 references over
   the video for a first frame's memory. The 9x R-101 config keeps the
   released checkpoint's longer recipe (9 epochs at batch 8, ×0.1 after the
   sixth) for comparison with it.
9. **Both metrics are reported.** `evaluation = dict(vid_style=True,
   coco_style=True)` gives the VID AP50 with the motion breakdown (comparable
   to MAMBA and STPN) and COCO AP / AP50 / AP75 / small / medium / large
   (comparable to the paper). `vfe/evaluation/coco.py`, on pycocotools.

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
  `tests/test_coco_eval.py`, 20 tests, CPU, seconds): point coordinates and
  valid flags against hand-computed values; the distance coder's round trip
  and clipping; focal loss against its formula with index and one-hot
  targets; IoU loss modes and the all-zero-weight case; FCOS target assignment
  by size range on a 32×32 image; loss, decoding, level subsets and the
  level report; box scaling; mask projection including the tiny-box rule;
  memory capacity and replacement; the aggregator's residual form; EOVOD
  rejecting unknown options and two-stage detectors; a training step with
  gradients reaching the aggregator, with and without reference boxes, and
  the same through the trainer's `train_step` with two micro-batches; four
  frames of stateful inference (seeding, prior, restricted then full levels,
  reset on a new video, missing `frame_id`); plain detection when nothing
  validates; both configs building at 32,443,240 and 51,435,368 parameters;
  and COCO-style evaluation on a synthetic annotation file (perfect
  detections 1.0, none 0.0, shifted labels 0.0).
- **CI-equivalent checks** pass locally: ruff, all 55 tests, no mm\* module
  at runtime, every config building, the CLI entry points, Python 3.8 syntax
  in `tools/checks/`, the manifest, and the sdist's contents.

## Running it

Configs: `configs/vid/eovod/eovod_fcos_r101_fpn_3x.py` (the paper's
setting), `eovod_fcos_r101_fpn_9x.py` (the released checkpoint's recipe) and
`eovod_fcos_r50_fpn_3x.py` (quick). Test frames stay in order — do not set
`shuffle_video_frames`: the location prior comes from the previous frame and
the size prior counts frames.

```bash
python -m pytest                                    # 55 tests, CPU

# smoke: 300 iterations on the RTX 4060 (batch 1, no accumulation, no evaluation)
python -m vfe.cli.train configs/vid/eovod/eovod_fcos_r50_fpn_3x.py \
    --work-dir work_dirs/eovod_smoke --max-epochs 1 --max-iters-per-epoch 300 --no-validate

# Isambard: four GH200s x 1 micro-step = the paper's batch of 4
torchrun --standalone --nproc_per_node=4 -m vfe.cli.train \
    configs/vid/eovod/eovod_fcos_r101_fpn_3x.py --launcher pytorch \
    --accumulate 1 --work-dir work_dirs/eovod_r101_3x --seed 0

torchrun --standalone --nproc_per_node=4 -m vfe.cli.test \
    configs/vid/eovod/eovod_fcos_r101_fpn_3x.py work_dirs/eovod_r101_3x/epoch_3.pth \
    --launcher pytorch --work-dir work_dirs/eovod_r101_3x
```

What to look at first: the log's `loss_cls` / `loss_bbox` / `loss_centerness`
falling from about 1.1 / 0.7 / 0.65 (a fresh FCOS on 30 classes) — the first
`loss_cls` is dominated by the 1% prior init and should drop fast; peak
memory per GPU (three 1000×600 frames through R-101-FPN plus the attention;
expect well under 20 GB); and seconds per iteration, from which the cost
follows. At batch 4 an epoch is 27,422 iterations; MAMBA's batch-4 epochs
ran at about 0.09 s each on four GH200s (three epochs in 1 h 58 min), and
FCOS-FPN with three frames per sample should be in the same range, so the
3-epoch run is likely 3–4 hours, ≈15 GPU-hours.

Evaluation prints both metric sets. The paper's numbers to compare with are
COCO AP 53.8 / AP50 ≈ 79.7 (FCOS+LPN+SPN, R-101); `size_prior=None` in the
config gives the LPN-only model (54.1 in the paper) at lower speed.

**Optional: the released checkpoint's FCOS.** Its `detector.*` keys match
this implementation's names (the head's `cls_convs.N.{conv,gn}`, `conv_cls`,
`conv_reg`, `conv_centerness`, `scales.N.scale`), so
`load_checkpoint(model, ckpt)` loads the backbone, FPN and head, logs
`memory.*` as unexpected and `aggregators.*` as missing, and evaluating gives
a plain-FCOS score from weights trained with the released aggregation. It is
not a check of this implementation, since the aggregator differs by design.

## Milestones

- [x] **9a — one-stage machinery**, checked against mmdet (above).
- [x] **9b — EOVOD's modules**, unit-tested on a tiny FCOS.
- [x] **9c — COCO-style evaluator**, on a synthetic annotation file.
- [ ] **9d — smoke on your machine:** the unit tests, a few hundred training
  iterations of the R-50 config on real data, `python -m vfe.cli.test ...
  --max-videos 2` on the resulting checkpoint (stateful inference on real
  videos), and `parity_fcos.py` against your `vfe` conda env; then freeze it
  into `run_parity.py`'s matrix.
- [ ] **E-M2 — a short Isambard run** (500 iterations of the R-101 3x config,
  batch 4): speed, peak memory, the LR schedule, the loss curve.
- [ ] **E-M3 — the full R-101 3x run → COCO AP within ±0.5 of 53.8 and AP50
  of 79.7**, plus the VID AP50 for the MAMBA/STPN table. The 9x recipe is
  the fallback if 3x lands low: it is what the released checkpoint trained
  on, at three times the cost.
- [ ] **Ablations the paper reports**, if the budget allows: `size_prior=None`
  (54.1 AP), *r* ∈ {0.6, 0.8, 1.0}, *T* ∈ {3, 7, 15}; and this
  implementation's own: `aggregator.shared=False`, memory sizes.
- [ ] **Speed measurement** on a fixed GPU: FPS with and without the size
  prior, since that is the paper's second claim.

## Decisions needed

1. **Which recipe for E-M3.** The paper's 3 epochs at batch 4 (its 54.1 /
   53.8 AP) or the released checkpoint's 9 epochs at batch 8 (its 54.0 AP)?
   Configs exist for both; the 3x costs a third. The plan assumes 3x first.
2. **Memory defaults** (4096 / 1024 / 512 per level): the paper gives none,
   and says its inference saw 2 reference frames. Keep, or run E-M2 at two
   sizes and pick by loss?
3. **Isambard budget:** E-M2 ≈ 0.2 GPU-hours; E-M3 ≈ 15 GPU-hours if the
   iteration time matches MAMBA's; each ablation the same again.
4. **Acceptance:** ±0.5 COCO AP against the paper's 53.8, as for the other
   models' ±0.5 AP50?

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
  harnesses and the unit tests run here. The network policy blocks Hugging
  Face, Google Drive, `download.openmmlab.com`, `download.pytorch.org` and the
  paper's hosts: no checkpoints, pretrained weights, data or PDF.
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

- **2026-09-27 (recipe)** — Further excerpts gave the paper's FCOS recipe:
  3 epochs at batch 4, lr 1e-3 → 1e-4 after two, shorter side 600, 2
  reference frames at inference; the "batch 32" was CenterNet's. The R-101
  3x config follows it and becomes the E-M3 target; the 9x config stays as
  the released checkpoint's recipe. arXiv, ECVA, QUB (including the PhD
  thesis, which likely holds the chapter in full) remain blocked here.
- **2026-09-27 (implemented)** — EOVOD built from the paper: FCOS ported from
  mmdet 2.19.1 and checked exact against it in float64 (82 artifacts,
  1e-15) and to the float32 accumulation floor; `EOVOD` with the location
  prior, the size prior, per-level pixel memory and SELSA-style residual
  attention over the FPN outputs; COCO-style evaluation beside the VID
  metric; two configs; 19 unit tests. The paper's hosts are blocked here, so
  it was read through search excerpts; the open details and the choices made
  for them are listed under *Design*. Nothing trained yet: next is a smoke
  run on the local machine and E-M2/E-M3 on Isambard.
- **2026-09-26 (audit)** — Phase 9 opened. EOVOD's public code is `v1.0.0`'s
  MMDetection fork plus `FCOSAtt` and `MPN`. It fails to import
  (`CenterNetAtt` is imported but never defined); with that fixed it runs on
  the existing legacy stack, so one oracle environment serves both projects.
  Its 79.7 is COCO-style AP50, measured with memory carried across 377 of
  VID val's 555 snippets. Both stacks were built from PyPI in a cloud
  container, so CPU parity work can happen there.
