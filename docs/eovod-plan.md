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

## Status (2026-09-27)

- **Implemented from the full paper text and unit-tested; not yet trained.**
  Everything runs on CPU in this container on random weights; the data, GPUs
  and checkpoints are on your machine and Isambard.
- **The FCOS underneath is exact.** `tools/checks/parity_fcos.py` compares the
  head against mmdet 2.19.1 on CPU: in float64 all 82 artifacts agree to
  1e-15 relative; in float32, forward passes to 8.5e-7, decoding to 2.4e-6
  and gradients to 7.8e-4, the torch 1.10 → 2.10 accumulation floor
  documented for the training-step checks in the rewrite plan.
- **Next: a smoke run on your machine, then E-M2 and E-M3 on Isambard.**
  Commands and what to look at are under *Running it*.

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
- [ ] **E-M2 — a short Isambard run** (500 iterations of the R-101 3x config,
  batch 4): speed, peak memory, the LR schedule, the loss curve.
- [ ] **E-M3 — the full R-101 3x run**, evaluated twice from the same
  weights: LPN only → COCO AP within ±0.5 of 54.1 (AP50 79.8); with *T* = 7 →
  53.8 (AP50 76.9). Plus the VID AP50 for the MAMBA / STPN table. The 9x
  recipe is the fallback if 3x lands low: the released checkpoint trained on
  it, at three times the cost.
- [ ] **Speed** on a fixed GPU: FPS with and without the size prior (the
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
