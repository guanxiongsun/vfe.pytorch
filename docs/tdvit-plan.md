# TDViT

> Working document for TDViT in `vfe/`. Like EOVOD ([eovod-plan.md](eovod-plan.md)),
> TDViT was never released: its repository
> ([guanxiongsun/TDViT](https://github.com/guanxiongsun/TDViT)) holds only a
> README, and none of its code was ever in this one (issue #3 asks for it).
> Update the checkboxes and the Progress log as work proceeds.

- **Goal:** TDViT (*TDViT: Temporal Dilated Video Transformer for Dense Video
  Tasks*, Sun, Hua, Hu, Robertson; ECCV 2022,
  [arXiv 2402.09257](https://arxiv.org/abs/2402.09257)) in `vfe/`, trained and
  evaluated on ImageNet VID, its Table 2 reproduced first (Faster R-CNN on
  Swin-T and TDViT-T), then the rest.
- **Method:** built from the paper, with the authors' code where the paper is
  silent: the backbone of the CVPR 2022 submission's supplementary code
  (mmdet 2.x; the TDTB, the split scheme, how the training references are
  formed) survives in the authors' archive, and the authors answered what
  neither records (below).

## Status (2026-10-02)

- Implemented and unit-tested (57 tests): the backbone with both attention
  forms (`cross`, the paper's; `joint`) and both reference passes
  (`spatial`, the authors' code; `paired`), the detector (Faster R-CNN and
  SELSA), the per-stage reference sampling, crops shared by a clip, the
  configs for T, T+, S, B and SELSA, the fused attention, and the speed
  and attention-share tools.
- The paper's block, implemented as the authors' code has it, trails a
  Swin-T trained the same way (3 epochs: 46.0 against 50.2 AP): it gives up
  the frame's own spatial attention in half of TDViT-T's blocks. With one
  change -- *joint* attention over the frame's window and the reference's --
  TDViT-T reaches **50.6 AP / 80.3 AP50** (VID 80.8) at 3 epochs, above the
  paper's TDViT-T (49.1 / 78.5) and the Swin-T trained alongside it (50.2 /
  79.0), at Swin-T's size and, with the fused kernel, its speed; on v1's
  plain pipeline both score higher (51.1 and 51.8). Table 8's order is
  reproduced; the reference's distance at test is worth half a point.
- TDViT-T+ reaches **51.4 / 80.8** (VID 81.4) once its two new blocks start
  as copies of pretrained blocks (`extra_init='copy'`); from torch's
  initialisation or the identity they learn nothing at this learning rate
  and TDViT-T+ equals TDViT-T. On the plain pipeline it equals TDViT-T even
  so (51.7 against 51.8), so the extra blocks are worth between nothing and
  a point here, not the paper's 1.8. A second seed of Swin-T and TDViT-T puts the
  run-to-run spread at 0.4 AP / 0.6 AP50: TDViT-T's AP50 gain holds in both
  seeds, its AP gain is within the spread. The paired reference pass
  changes nothing for the tiny models and stays an option for the small.
  SELSA on TDViT-T reproduces the paper's 83.9 (83.8) but SELSA on Swin-T
  scores 84.5; no further SELSA work. S had one round (Swin-S 56.7, TDViT-S
  55.2, its diagnosis in the log) and waits until T is final; B is dropped.
- The tiny models' settings are final (Progress log, evening of
  2026-10-02); what is left is the README and the PR.

## The paper

**Idea.** Video backbones (3D CNNs, video transformers) suit *sparse* tasks,
one prediction per video; *dense* tasks (one per frame: VID detection, VIS)
still use 2D backbones plus temporal modules. TDViT is a backbone that is
itself temporal and costs one frame per frame.

**The temporal dilated transformer block (TDTB, Sec. 3.2, Eq. 1).**
`f^R = Sampling(M, D_t)`; `f' = MCA(LN(f), LN(f^R)) + f`;
`out = MLP(LN(f')) + f'`. Queries come from the current frame, keys and values
from a reference map `f^R` sampled from the block's memory `M` of earlier
frames; `D_t`, the temporal dilation, controls the sampling. The memory is
updated every frame, the oldest map dropped. A sampled `f^R` is reused for
`D_t` frames, so only the queries are computed per frame -- "slightly faster"
than Swin (23.9 vs 22.8 FPS).

**Sampling strategies (Table 8, TDViT-T):** temporal earliest (the oldest
map; 78.5 AP50, 23.9 FPS, the default), temporal NMS (largest L2 norm; 78.5),
channel shuffle (each channel from a random frame; 77.7), patch shuffle (four
quarters from four groups of frames; 78.8, 19.7 FPS).

**Local attention (Table 9):** window (Swin's, 7x7: 78.5 / 23.9 FPS, the
default) vs correlation (a 7x7 neighbourhood: 78.8 / 18.4 FPS).

**Schemes (Sec. 3.3, Table 6, stage 3 of TDViT-T):** Swin `[s]*6` 77.2;
factorised `[s, t]*3` 78.0; split `s*3, t*3` **78.5** (default); TDViT-T+
`s*3, t*5` 79.9. Stages 1, 2 and 4 are `[s, t]`.

**Dilations (Table 7, AP50):** (1,2,4,8) 77.2; (2,4,8,16) 77.9; (3,6,12,24)
78.1; **(4,8,16,32) 78.5** (default); (5,10,20,40) 78.4. (Fig. 2 shows
3/6/12/24, an older setting a reviewer noticed.)

**Variants (Sec. 3.5, Table S1):** T: C 96, L (2,2,6,2); S: 96, (2,2,18,2);
B: 128, (2,2,18,2); "+" adds two TDTBs at the end of stage 3. Split layouts:
T `st, st, s3 t3, st`; S/B `st, st, s9 t9, st`; T+ `... s3 t5 ...`; S+/B+
`... s9 t11 ...`.

**Setup (Sec. 4).** VID train (15 frames per video) plus DET (at most 2,000
images per class). "AdamW for 3 epochs, initial learning rate 10^-3 and 10^-4
after the 2nd epoch, weight decay 0.05; the augmentation and regularisation
of Swin. Given the key frame I_k, we randomly sample four frames from
{I_tau}, k - D_t .. k + D_t, to approximately form the memory of TDTBs in
four stages, respectively. Losses only for I_k." V100 GPUs.

**Table 2 (single-frame Faster R-CNN, COCO-style AP on VID val):**

| Backbone | AP | AP50 | AP75 | APs | APm | APl | #Param | FPS |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| R-50 | 44.3 | 72.5 | 47.2 | 6.3 | 19.7 | 49.3 | 44.4M | 23.0 |
| R-101 | 48.5 | 75.5 | 53.1 | 7.6 | 23.1 | 53.7 | 63.4M | 19.5 |
| Swin-T | 47.1 | 77.2 | 51.5 | 9.5 | 23.2 | 52.4 | 47.8M | 22.8 |
| Swin-S | 52.6 | 82.4 | 59.3 | 10.0 | 26.5 | 58.6 | 69.1M | 16.7 |
| Swin-B | 53.2 | 82.7 | 60.4 | 9.1 | 27.7 | 59.0 | 107.1M | 12.8 |
| **TDViT-T** | **49.1** | **78.5** | 52.7 | 8.0 | 25.6 | 53.1 | 47.8M | 23.9 |
| TDViT-T+ | 50.9 | 79.9 | 55.7 | 9.1 | 26.9 | 57.2 | 51.3M | 21.9 |
| TDViT-S | 55.4 | 84.1 | 63.4 | 10.3 | 29.3 | 61.2 | 69.1M | 16.8 |
| TDViT-S+ | 55.7 | 84.1 | 63.2 | 10.5 | 30.4 | 61.4 | 72.7M | 16.2 |
| TDViT-B | 56.0 | 84.4 | 64.1 | 10.7 | 28.0 | 62.3 | 107.1M | 12.9 |
| TDViT-B+ | 56.1 | 84.7 | 64.2 | 10.0 | 29.9 | 61.8 | 113.5M | 12.2 |

The #Param column is MMDetection's COCO Mask R-CNN sizes (Swin-T 47.8M);
the VID Faster R-CNN here has 44.9M, TDViT-T+ 48.4M -- the same +3.55M.

**Table 3 (VID AP50 with the motion breakdown):** SELSA\* R-101 81.5;
SELSA + TDViT-T **83.9** (slow 88.6, medium 83.8, fast 67.7, 16.2 FPS);
SELSA + TDViT-T+ 84.5. SELSA\* is SELSA with RDN's top-75 reference
proposals, i.e. MAMBA's instance level without its memory bank.

**Tables 4-5 (YouTube VIS, MaskTrack R-CNN):** TDViT-T 35.4 mask AP vs Swin-T
33.7. Out of scope here (no VIS data or MaskTrack R-CNN in this repository).

## What the paper leaves open, and the choices

From the authors' code (CVPR 2022 supplementary):

- **A TDTB is a Swin block** with the same parameters: `norm1` normalises both
  maps, and the one `qkv` projection gives the queries of the current frame
  and the keys and values of the reference; the relative position bias and
  the shifted windows apply as in Swin (a query attends to the reference
  tokens of its own window). TDViT-T therefore has Swin-T's parameter count
  and loads Swin-T's weights unchanged.
- **The memory keeps the block's input**, not its output as the text says
  (`memory_feature='output'` gives the text's version). Training agrees:
  each TDTB keeps the map it receives for its stage's reference.
- **Training references:** reference *s* passes, without gradients, through
  stages 1..*s*, every TDTB attending to the frame itself; each TDTB of stage
  *s* keeps the map it receives; the key frame then attends to those maps.
- **The shift** of a block's windows alternates with its index in the stage,
  for TDTBs as for Swin blocks; the advanced variants' extra TDTBs get drop
  path 0 and no pretrained weights.

From the authors (2026-10-01):

- **Weights:** ImageNet-1K Swin-T (`swin_tiny_patch4_window7_224.pth`), as in
  v1's Swin-T VID baseline (`configs/vid/single_frame/faster_rcnn/faster_rcnn_swint_fpn_3x.py`
  on the `v1` branch).
- **Learning rate:** AdamW at 2.5e-5 for batch 8 (one clip per GPU on eight
  GPUs), 2.5e-6 after the second epoch; the paper's 10^-3 is a typo. Weight
  decay 0.05, none on norms and position tables.
- **Augmentation:** Swin's: random flip, then either a multi-scale resize
  (shorter side 480-800, longer at most 1333) or a resize to 400-600, a
  random 384-600 crop and the multi-scale resize (STPN's pipeline) -- with
  every frame of a clip drawn alike, crops included (`SeqRandomCrop`
  `share_params=True`, new here; STPN's crops are per frame).
- **Training references:** one per stage, uniformly within `±D_t` of the key
  frame for that stage's `D_t`, the key frame excluded
  (`ref_img_sampling(method='stagewise_uniform')`).

From v1's Swin-T VID baseline config: Faster R-CNN with an FPN and VID's
settings (600 training proposals, 512 RoI samples), Swin's drop path 0.2,
500 warmup iterations from 1/3, gradient clipping at 35. Testing at 600 on
the shorter side (1000 on the longer), as all VID configs here.

Here, where nobody records it:

- **The sampling schedule.** A frame with an empty memory (a video's first)
  attends to itself, which makes TDViT exactly Swin on it. Draws happen at
  frames 1, 1 + D_t, 1 + 2 D_t, ...; with *temporal earliest* a stage looks
  between `D_t` and `2 D_t - 1` frames back. A drawn reference's keys and
  values are computed once and reused (the speed claim).
- **Patch shuffle:** the memory split into four groups of consecutive frames;
  quarter *q* (top-left, top-right, bottom-left, bottom-right) from a random
  frame of group *q*. **Temporal NMS:** the map with the largest L2 norm
  over all its tokens. Randomness from torch's CPU generator (`--seed`).

## Design: paper -> `vfe/`

| Paper | `vfe/` |
| :-- | :-- |
| TDTB (Eq. 1, Fig. 2b) | `TDTB` (a `SwinBlock` whose attention is `TemporalShiftWindowMSA`: queries from the first third of `qkv`, keys and values from the rest) in `vfe/models/backbones/tdvit.py` |
| memory `M` and `Sampling(M, D_t)` (Fig. 3) | `MemoryQueue`: `earliest`, `nms`, `patch_shuffle`, `channel_shuffle`; reuse for `D_t` frames |
| split / factorised schemes (Sec. 3.3) | `TDViT(layout=...)`: per stage, `'s'` and `'t'` in order; `extra_tdtbs` for the "+" variants |
| training: four references, one per stage | `TDViT.reference_maps` (backbone); `ref_img_sampling(method='stagewise_uniform')` (dataset) |
| inference, one frame at a time | `TDTB.forward_online`; `TDViTDetector` resets the memories at each video's first frame |
| Faster R-CNN on TDViT (Table 2) | `TDViTDetector` around `FasterRCNN`; `configs/vid/tdvit/{frcnn_swint,tdvit_t,tdvit_tplus}_*_3x.py` |
| SELSA\* on TDViT (Table 3) | `TDViTDetector` with a `MambaRoIHead` given reference RoIs on every frame (top 75 per frame, RDN's distillation); SELSA's references through `TDViT.forward_spatial`; `configs/vid/tdvit/selsa_{tdvit_t,swint}_fpn_3x.py` |

The changes to existing code are additive: `swin.py` gains three hooks
(`WindowMSA.attend`, `ShiftWindowMSA.shift_attn_mask`, `make_stage` with
per-stage arguments) and stays bit-identical (outputs and every gradient of
Swin, and STPN's prompted Swin, checked against a snapshot); `RandomCrop`'s
box handling moved into `_crop_boxes` unchanged.

**Verified** (`tests/test_tdvit.py`, 31 tests): TDViT-T has Swin-T's
parameters and Table S1's layout; a TDTB attending to its own frame equals a
Swin block (both shifts), and a query sees only its window's reference
tokens; a video's first frame is exactly Swin; every memory policy and the
reuse schedule; online frames match a recomputation with the expected
reference, keys and values cached; training maps equal each reference run
alone through its stages; every parameter learns, references get no
gradient; per-stage sampling ranges and order; shared crops. An end-to-end
run of the real config's pipelines on synthetic JPEGs feeds the detector.

## Milestones

- [x] **T1 -- the implementation**, unit-tested (above).
- [x] **T2 -- smoke on Isambard** (job 6995011): the 160 tests pass on a
  GH200; 60 iterations each of TDViT-T (0.160 s, 3.5 GB) and Swin-T (0.123
  s); online inference over 3 val videos at 43 frames/s.
- [x] **T3 -- Table 2, tiny:** the Swin-T baseline and TDViT-T, 3 epochs at
  batch 8 (4 GH200s, `--accumulate 2`), evaluated COCO-style and VID-style.
  Target: Swin-T 47.1 / 77.2, TDViT-T 49.1 / 78.5 -- the gain (+2.0 AP, +1.3
  AP50) matters more than the absolute numbers. Done: Swin-T 50.2 / 79.0,
  TDViT-T (joint attention) 50.6 / 80.3; the paper's block 46.0 / 75.7.
- [x] **T4 -- TDViT-T+** (50.9 / 79.9): 51.4 / 80.8 with its new blocks
  copied from pretrained ones; 50.7, equal to TDViT-T, from torch's
  initialisation or the identity (Progress log, 2026-10-02).
- [x] **T5 -- ablations:** sampling strategies (inference only: one trained
  model, `--cfg-options model.detector.backbone.memory_sampling=...`),
  dilations and schemes (retraining). Done: Table 8's order reproduced;
  dilations 1 / 2 / 4 / 8 equal to 4 / 8 / 16 / 32 at one epoch.
- [x] **T6 -- SELSA + TDViT** (Table 3, 83.9 VID AP50): implemented (MAMBA's
  instance level given its references every frame is SELSA\*); smoke
  6995312 passed (SELSA on TDViT-T 0.281 s an iteration on one GPU, 8.6 GB;
  on Swin-T 0.221 s; inference with 14 references a video at 33 frames/s).
  Done: 83.8, the paper's number, but SELSA on Swin-T scores 84.5.
- [x] **T7 -- speed** against Swin, eager, as the paper measured
  (`tools/tdvit_speed.py`). First run (6995305, untrained weights, five
  videos x 100 frames, GH200): Swin-T 25.6 ms a frame (39.0 FPS, backbone
  11.7 ms), TDViT-T 26.0 ms (38.5 FPS, 11.9 ms), TDViT-T+ 27.1 ms (36.9 FPS,
  13.7 ms). The paper's V100: 22.8 / 23.9 / 21.9 FPS. Reusing keys and
  values saves FLOPs that a GH200 running one frame at a time, bound by
  kernel launches, does not turn into time; T+'s cost (-5%) is the paper's
  (-4%). Repeated with trained weights, and with the fused kernel: joint
  TDViT-T at Swin-T's speed (Progress log, 2026-10-02).
- [ ] **T8 -- S and B.** One round of S (Swin-S 56.7 / 83.5, TDViT-S 55.2 /
  83.3 with the spatial reference pass; Progress log) and then deferred, B
  dropped (2026-10-02): the tiny models' settings are finalised first, and
  S follows by config. The S and B configs build; only the two S ones above
  were trained.

## Progress log

- **2026-10-03 (TDViT-T+ on the plain recipe)** — Full val, 3 epochs, v1's
  plain pipeline (resize to 600, flip), the paper's test protocol:

  | | AP | AP50 | AP75 | VID AP50 | fast | medium | slow |
  | :-- | --: | --: | --: | --: | --: | --: | --: |
  | Swin-T | 51.1 | 79.4 | 57.0 | 79.9 | 56.9 | 78.7 | 86.5 |
  | TDViT-T | **51.8** | **81.2** | **57.6** | **81.7** | **59.6** | **80.8** | 88.4 |
  | TDViT-T+, new blocks copied | 51.7 | 81.0 | 57.5 | 81.6 | 58.8 | 80.4 | 88.4 |

  On this recipe TDViT-T+ equals TDViT-T (-0.1 AP), where on Swin's
  augmentation it led by 0.8 (51.4 against 50.6). With a run-to-run spread
  of 0.4 AP, the two extra blocks are worth between nothing and a point: a
  smaller and less certain gain than the paper's +1.8, and one that does not
  show on the recipe where both models score higher. TDViT-T's lead over
  Swin-T, by contrast, holds on both recipes (+0.4 and +0.7 AP, +1.3 and
  +1.8 AP50). The tiny models' numbers are complete
  (`tdvit_tplus_joint_frcnn_fpn_3x_v1aug.py`).

- **2026-10-02 (evening: the tiny models' settings are final)** — Full val,
  3 epochs, Swin's augmentation, the paper's test protocol:

  | | AP | AP50 | AP75 | VID AP50 | fast | medium | slow |
  | :-- | --: | --: | --: | --: | --: | --: | --: |
  | Swin-T, seed 1 | 50.2 | 79.0 | 55.4 | 79.5 | 57.3 | 78.8 | 85.7 |
  | Swin-T, seed 2 | 50.6 | 79.0 | 56.3 | 79.6 | 57.9 | 78.8 | 85.1 |
  | TDViT-T, seed 1 | 50.6 | 80.3 | 56.0 | 80.8 | **59.5** | **80.5** | 87.3 |
  | TDViT-T, seed 2 | 50.6 | 79.7 | 56.5 | 80.3 | 58.0 | 79.6 | 86.7 |
  | TDViT-T, paired reference pass | 50.6 | 79.8 | 56.2 | 80.4 | 58.3 | 80.2 | 86.2 |
  | TDViT-T+, new blocks copied | 51.4 | **80.8** | 57.3 | **81.4** | 57.7 | 80.0 | **89.0** |
  | TDViT-T+, copied, paired reference pass | **51.5** | **80.8** | **57.5** | **81.4** | 58.2 | 80.1 | 88.7 |
  | *the paper: Swin-T* | *47.1* | *77.2* | *51.5* | | | | |
  | *the paper: TDViT-T* | *49.1* | *78.5* | *52.7* | | | | |
  | *the paper: TDViT-T+* | *50.9* | *79.9* | *55.7* | | | | |

  **Noise.** Two seeds of one model differ by up to 0.4 AP, 0.6 AP50 and 0.5
  VID AP50 at 3 epochs. TDViT-T's lead over Swin-T is +0.4 and 0.0 AP, +1.3
  and +0.7 AP50, +1.3 and +0.7 VID AP50 in the two seeds: the AP50 gain
  holds, the AP gain is within the spread. (On the plain pipeline, one
  seed: +0.7 AP, +1.8 AP50.)

  **The paired reference pass changes nothing for the tiny models:**
  TDViT-T 50.6 / 79.8 against 50.6 / 80.3 and 79.7, TDViT-T+ 51.5 against
  51.4. Their references pass too few TDTBs for the mismatch to tell;
  `reference_mode` stays `'spatial'`, the authors' way, and the option
  remains for the small models, where the mismatch was diagnosed.

  **TDViT-T+ works once its new blocks start as blocks.** Copied from the
  two pretrained blocks before them (`extra_init='copy'`), they train at
  once (the loss over iterations 1,000-3,000: 0.2514 against TDViT-T's
  0.2603, where from torch's initialisation or the identity it was 0.256 /
  0.255 and the end result equal to TDViT-T), and TDViT-T+ reaches 51.4 /
  80.8 / 57.3, VID 81.4: +0.8 AP over TDViT-T, slow objects +1.7, and above
  the paper's TDViT-T+ by 0.5 AP, 0.9 AP50 and 1.6 AP75.
  `tdvit_tplus_joint_frcnn_fpn_3x.py` now copies.

  **The settings, final for T and T+:** joint attention; the paper's
  dilations 4 / 8 / 16 / 32 and split layouts; the spatial reference pass;
  T+'s new blocks copied; Swin's augmentation as the main recipe, with v1's
  plain pipeline as the stronger variant; at test the paper's protocol (a
  reference held `D_t` frames), near references as an optional row (+0.6
  AP, -0.2 VID AP50); `fused_attention` for speed. S follows by config when
  wanted, with `reference_mode='paired'` the first thing to try there.

- **2026-10-02 (TDViT-S: where it loses, and a paired reference pass)** —
  TDViT-S trails its Swin-S by 1.5 AP where TDViT-T leads its Swin-T by
  0.4. The training logs (one seed, the same samples) place the loss in
  training: TDViT-S's loss is above Swin-S's at every epoch (0.1819 /
  0.1535 / 0.1310 against 0.1795 / 0.1504 / 0.1284, 1.3-2.0% more), where
  joint TDViT-T's is *below* Swin-T's (0.1444 against 0.1484 at epoch 3) --
  the signature of the paper's block on the tiny models, back at the small
  scale. The attention shares do not discriminate: at 3 epochs every TDTB of
  either model gives 0.47-0.50 of its attention to the reference, S's deep
  stack included (`tools/tdvit_attention.py`, 20 videos).

  What differs between the scales is how many TDTBs a reference passes. The
  training references' maps are formed as the authors' code forms them: a
  reference passes through the TDTBs *attending to itself*, so the map a
  TDTB receives is a Swin map. At test time the memory holds maps computed
  online, every TDTB attending to its reference, so the keys and values a
  TDTB sees come from another kind of map than in training. In TDViT-T a
  stage-3 reference passes two TDTBs before the stage's last one uses it
  (plus one in each of stages 1 and 2); in TDViT-S, ten. The fix keeps the
  model and makes training form both sides alike: `reference_mode='paired'`
  advances the key frame and its references block by block; at every TDTB
  the key frame attends over its stage's reference map and every reference,
  without gradients, over the key frame's map (the key frame standing in
  for the reference's own earlier frames), so the maps the TDTBs receive are
  joint-attention maps on both sides. With the frame as its own reference
  this is still exactly Swin, and inference is unchanged (5 tests: the
  paired pass equals the spatial one wherever a reference has met no TDTB,
  and differs after one). The S runs that would test it (TDViT-S paired,
  and TDViT-S with T's three TDTBs in stage 3, `s15 t3`) were submitted and
  then cancelled with every other S experiment: the small models wait until
  the tiny ones' settings are final, when extending them is a config change.
  The paired pass is tested on TDViT-T (7017703) and on TDViT-T+ with its
  new blocks copied (7018305), against their spatial counterparts (50.6 and
  7017010); for T, a new reference every frame at the paper's dilations
  (7017539) splits the reference's distance from the hold.

- **2026-10-02 (three epochs: TDViT-T+, Swin-S, SELSA)** — Full val,
  Faster R-CNN, 3 epochs, Swin's augmentation:

  | | AP | AP50 | AP75 | VID AP50 | fast | medium | slow |
  | :-- | --: | --: | --: | --: | --: | --: | --: |
  | Swin-T | 50.2 | 79.0 | 55.4 | 79.5 | 57.3 | 78.8 | 85.7 |
  | TDViT-T | 50.6 | 80.3 | 56.0 | 80.8 | 59.5 | 80.5 | 87.3 |
  | TDViT-T+, new blocks from torch's initialisation | 50.7 | 80.4 | 56.4 | 81.0 | 58.4 | 80.0 | 88.0 |
  | TDViT-T+, new blocks from the identity | 50.7 | 80.3 | 56.0 | 80.8 | 58.5 | 79.9 | 87.8 |
  | Swin-S | 56.7 | 83.5 | 64.3 | 84.0 | 63.5 | 83.3 | 89.9 |
  | TDViT-S | 55.2 | 83.3 | 62.7 | 83.8 | 62.1 | 83.8 | 90.4 |
  | *the paper: TDViT-T+* | *50.9* | *79.9* | *55.7* | | | | |
  | *the paper: Swin-S* | *52.6* | *82.4* | *59.3* | | | | |
  | *the paper: TDViT-S* | *55.4* | *84.1* | *63.4* | | | | |

  **TDViT-T+ equals TDViT-T** (+0.1 AP) however its two new blocks start,
  so the paper's +1.8 AP from them is not reproduced. The weight norms say
  why: at 2.5e-5 the new blocks never leave their initialisation (from
  torch's, 11.2 / 11.3 at the start and at the end of training; from zero,
  1.3 / 2.2 against a pretrained block's 17.5 / 50.5), so two blocks that
  cannot be trained from scratch on this recipe are carried along. One
  initialisation left to try, the natural one: each new block a copy of the
  pretrained block two before it (`extra_init='copy'`, 7017010), so both
  start as working blocks with their shifts in place.

  **Swin-S** trained here scores 56.7 AP, 4.1 above the paper's Swin-S and
  above the paper's TDViT-S (55.4) -- the pattern of the tiny models, whose
  Swin-T was 3.1 above the paper's. **TDViT-S** lands on the paper's TDViT-S
  (55.2) but *below* its own Swin-S: -1.5 AP, -0.2 AP50, fast objects -1.4,
  slow +0.5. At the small scale the gain of the tiny models is gone. TDViT-S
  has 12 TDTBs of 24, nine of them in a row in stage 3 (Table S1's `s9 t9`)
  against TDViT-T's 6 of 12, so nine blocks in sequence lean on one
  reference 16-31 frames back. Its diagnosis follows in the next entry; the
  S experiments themselves stop there (the user's call, 2026-10-02): the
  tiny models' settings come first, and the small ones follow by config.

  **SELSA** (Table 3; SELSA\*, RDN's top-75 reference proposals; 3 epochs):

  | SELSA on | AP | AP50 | AP75 | VID AP50 | fast | medium | slow |
  | :-- | --: | --: | --: | --: | --: | --: | --: |
  | Swin-T | **53.4** | **83.9** | **59.2** | **84.5** | **68.6** | **84.5** | **89.6** |
  | TDViT-T, joint attention | 52.1 | 83.2 | 57.6 | 83.8 | 67.7 | 84.3 | 88.4 |
  | the same, memory off at test | 52.2 | 82.6 | 58.0 | 83.2 | 67.8 | 83.5 | 87.6 |
  | the same, references 1 / 2 / 4 / 8 back | 52.9 | 83.5 | 58.9 | 84.0 | **69.2** | 84.5 | 88.1 |
  | *the paper: TDViT-T* | | | | *83.9* | *67.7* | *83.8* | *88.6* |

  SELSA on TDViT-T lands on the paper's row (83.8 against 83.9, the same
  67.7 on fast objects). But SELSA on Swin-T, the comparison the paper does
  not make, scores 84.5: on top of proposal aggregation across the video
  the backbone's memory costs 0.7 VID AP50 and 1.3 AP, at epoch 1 as at
  epoch 3 (43.3 / 78.1 against 42.4 / 77.8 there). Two inference-only runs
  of the checkpoint place the loss. With the memory off -- TDViT's weights
  as a still backbone under SELSA -- it scores 52.2 / VID 83.2: the memory
  does add at test time (+0.6 VID AP50, nothing in AP), and the weights
  themselves are 1.2 AP behind SELSA on Swin-T's. The backbone trained under
  SELSA serves two modes at once -- the key frame with its references, and
  SELSA's reference frames on their own (`forward_spatial`, the references
  being spread over the video rather than in order) -- and comes out weaker
  than Swin-T trained for one. With near references (1 / 2 / 4 / 8 back, a
  new one every frame) it reaches 52.9 / VID 84.0 and 69.2 on fast objects,
  above SELSA on Swin-T's 68.6; the 0.5 left is all on slow objects (88.1
  against 89.6), where SELSA's aggregation over the whole video is at its
  best. As with Faster R-CNN, the paper's test protocol -- a reference held
  `D_t` frames, so 16 to 63 frames back in the later stages -- costs what
  nearer references would give.

  Swin-B and TDViT-B were dropped (the user's call); their configs stay,
  untrained.

- **2026-10-02 (the plain recipe; how noisy one epoch is)** — TDViT-T
  (joint) trained 3 epochs with v1's plain pipeline (resize to 600, flip),
  next to the Swin-T trained the same way:

  | 3 epochs | AP | AP50 | AP75 | VID AP50 | fast | medium | slow |
  | :-- | --: | --: | --: | --: | --: | --: | --: |
  | Swin-T, Swin's augmentation | 50.2 | 79.0 | 55.4 | 79.5 | 57.3 | 78.8 | 85.7 |
  | TDViT-T, Swin's augmentation | 50.6 | 80.3 | 56.0 | 80.8 | 59.5 | 80.5 | 87.3 |
  | Swin-T, resize and flip | 51.1 | 79.4 | 57.0 | 79.9 | 56.9 | 78.7 | 86.5 |
  | **TDViT-T, resize and flip** | **51.8** | **81.2** | **57.6** | **81.7** | **59.6** | **80.8** | **88.4** |

  On the plain recipe too TDViT-T leads Swin-T (+0.7 AP, +1.8 AP50, every
  motion class 1.9-2.7 VID AP50 up), and both are better than with Swin's
  augmentation, which the paper used (`*_v1aug.py`).

  Seed 2 measures one epoch's noise (`--seed 2`: other initial heads, other
  samples):

  | Epoch 1 | AP | AP50 | AP75 | VID AP50 |
  | :-- | --: | --: | --: | --: |
  | TDViT-T, seed 1 | 42.0 | 74.6 | 43.9 | 75.1 |
  | TDViT-T, seed 2 | 41.4 | 73.7 | 43.1 | 74.2 |
  | TDViT-T+ from the identity, seed 1 | 40.7 | 72.8 | 42.0 | 73.3 |
  | TDViT-T+ from the identity, seed 2 | 41.3 | 73.7 | 42.7 | 74.2 |

  Two seeds of one model differ by 0.6 AP and 0.9 AP50 after one epoch, and
  with seed 2 TDViT-T+ equals TDViT-T: TDViT-T+'s gap at one epoch was noise.
  The one-epoch comparisons above that found no difference (the temporal
  bias, past references, small dilations) stand; differences under about a
  point at one epoch are not evidence. To make the 3-epoch comparison firm,
  Swin-T and TDViT-T are trained again with seed 2 (7011182, 7011184).

- **2026-10-02 (TDViT-T+, the dilations, the speed)** — One epoch each,
  joint attention, the default test protocol unless marked:

  | Epoch 1 | AP | AP50 | AP75 | VID AP50 | fast | medium | slow |
  | :-- | --: | --: | --: | --: | --: | --: | --: |
  | TDViT-T | **42.0** | **74.6** | **43.9** | 75.1 | 51.4 | 75.1 | **82.7** |
  | the same, references 1 / 2 / 4 / 8 back | 42.6 | 74.4 | 45.3 | 74.9 | 52.4 | 74.8 | 81.8 |
  | TDViT-T trained with dilations 1 / 2 / 4 / 8 | **42.0** | **74.6** | 43.5 | **75.2** | **53.0** | **75.3** | 82.1 |
  | TDViT-T+, new blocks from torch's initialisation | 40.9 | 72.8 | 42.0 | 73.3 | 49.3 | 72.6 | 80.6 |
  | the same, references 1 / 2 / 4 / 8 back | 41.7 | 72.8 | 43.9 | 73.3 | 50.5 | 72.8 | 79.7 |
  | TDViT-T+, new blocks starting as the identity | 40.7 | 72.8 | 42.0 | 73.3 | 50.3 | 72.8 | 80.6 |

  The dilations trained small do no better than the paper's (Table 7 has
  1 / 2 / 4 / 8 1.3 AP50 worse); 4 / 8 / 16 / 32 stay. TDViT-T+ trails
  TDViT-T by 1.8 AP50 whether its two new TDTBs start from torch's
  initialisation (as the authors' code leaves blocks missing from the
  checkpoint) or as the identity (`extra_init='zero'`: the attention's and
  the MLP's output projections zero), and with near references as well, so
  neither the initialisation nor the distance explains it. Nor do the new
  blocks learn much in an epoch: their weights' norms barely move from
  torch's initialisation (attention projection 11.2, MLP output 11.3), and
  from zero they reach only 1.3 and 2.2 against a pretrained block's 17.5
  and 50.5 -- about what AdamW's steps add up to with no steady direction
  (`sqrt(13,711) x 2.5e-5` = 0.003 a weight). Started from the identity,
  TDViT-T+ is TDViT-T plus two near-identity blocks, yet 1.3 AP lower: the
  gap may be the run's own noise (with a different architecture the heads
  start from different random numbers). Submitted: TDViT-T again with seed 2
  (7008142), and both TDViT-T+ runs resumed to 3 epochs (7008182, 7008184),
  where the paper's +1.8 AP is measured.

  **Speed** with the trained 3-epoch weights (GH200, eager, five val videos
  x 100 frames, one frame at a time): Swin-T 19.8 ms a frame (50.5 FPS),
  the paper's block 20.1 ms (49.7), joint attention 20.9 ms (47.8): twice
  the keys cost 5% (7007335). Attention through PyTorch's fused kernel
  (`fused_attention=True`, `F.scaled_dot_product_attention` with the
  position bias as its mask; off by default, which keeps Swin bit-identical)
  takes it back; in one run (7007390), Swin-T goes from 20.6 to 19.3 ms a
  frame (51.8 FPS) and joint TDViT-T from 21.5 to 19.2 ms (52.0).
  The paper's V100 had TDViT-T 5% faster than Swin-T (23.9 against 22.8
  FPS) from reusing the reference's keys and values; on a GH200 running one
  frame at a time, launch-bound, the saving does not show.

- **2026-10-02 (three epochs: joint TDViT-T beats Swin-T and the paper)** —
  Full val, Faster R-CNN, 3 epochs:

  | | AP | AP50 | AP75 | VID AP50 | fast | medium | slow |
  | :-- | --: | --: | --: | --: | --: | --: | --: |
  | Swin-T | 50.2 | 79.0 | 55.4 | 79.5 | 57.3 | 78.8 | 85.7 |
  | TDViT-T, the paper's block | 46.0 | 75.7 | 49.4 | 76.2 | 51.1 | 75.2 | 83.8 |
  | **TDViT-T, joint attention** | **50.6** | **80.3** | 56.0 | **80.8** | 59.5 | 80.5 | **87.3** |
  | the same, memory off at test | 49.9 | 78.5 | 55.5 | 79.0 | 57.6 | 78.3 | 85.2 |
  | the same, references 1 / 2 / 4 / 8 back | **51.2** | 80.0 | **57.2** | 80.6 | **60.1** | 80.4 | 86.4 |
  | the same, a new reference every frame, 4 / 8 / 16 / 32 back | 50.8 | 80.3 | 56.5 | 80.9 | 59.7 | 80.8 | 87.3 |
  | the same, temporal NMS | 50.8 | 80.3 | 56.5 | 80.9 | 59.9 | 80.7 | 87.2 |
  | the same, patch shuffle | 50.9 | 80.4 | 56.6 | 80.9 | 59.9 | 80.8 | 87.2 |
  | *the paper: Swin-T* | *47.1* | *77.2* | *51.5* | | | | |
  | *the paper: TDViT-T* | *49.1* | *78.5* | *52.7* | | | | |

  With joint attention TDViT-T passes both the paper's TDViT-T (+1.5 AP,
  +1.8 AP50) and the Swin-T trained alongside it (+0.4 AP, +1.3 AP50 -- the
  paper's AP50 gain -- +2.2 on fast objects); at test the memory adds 0.7 AP
  and 1.8 AP50. The sampling strategies keep the paper's order (Table 8:
  patch shuffle first, NMS level with earliest). The reference's distance is
  worth half a point at most, nearly all of it AP75: a new reference every
  frame at the paper's distances (no hold, so no reuse of keys and values)
  gives 50.8, near references 51.2, and the VID AP50 hardly moves (80.6 to
  80.9). One epoch more of joint attention's diagnosis: every stage's TDTBs
  give 47-50% of their attention to the reference
  (`tools/tdvit_attention.py`, 20 videos).

  The baseline hypothesis is refuted: Swin-T trained with v1's plain pipeline
  scores *more*, 51.1 / 79.4 / 57.0 (VID 79.9), than with Swin's
  augmentation; the paper's 47.1 stays unexplained. Plain augmentation is
  the stronger recipe here, so TDViT-T joint is being trained with it too.

  At one epoch, two more problems: TDViT-T+ (joint) scores 40.9 AP against
  TDViT-T's 42.0 -- its two new blocks start from torch's initialisation and
  disturb the pretrained network -- and SELSA on TDViT-T (joint) 42.4 / VID
  77.8 against SELSA on Swin-T's 43.3 / 78.1 (which reaches 53.4 / 83.9 / VID
  **84.5** at 3 epochs, already above the paper's SELSA + TDViT-T, 83.9).
  Submitted: TDViT-T+ with its new blocks starting as the identity
  (`extra_init='zero'`, 7007313), TDViT-T joint with the plain pipeline at 3
  epochs (7007316), SELSA + TDViT-T joint resumed to 3 epochs (7007318), and
  TDViT-T joint trained with dilations 1 / 2 / 4 / 8 (7007320).

- **2026-10-01 (joint attention works)** — One epoch each, evaluated with the
  paper's test protocol (temporal earliest, references held `D_t` frames):

  | Epoch 1 | AP | AP50 | AP75 | VID AP50 | fast | medium | slow |
  | :-- | --: | --: | --: | --: | --: | --: | --: |
  | Swin-T | 41.0 | 73.1 | 42.4 | 73.6 | 51.4 | 73.1 | 81.1 |
  | TDViT-T, the paper's block | 37.5 | 69.9 | 36.6 | 70.4 | 42.0 | 69.1 | 79.8 |
  | TDViT-T, joint attention | **42.0** | 74.6 | **43.9** | 75.1 | 51.4 | 75.1 | **82.7** |
  | the same, + temporal bias | 41.5 | **74.7** | 42.4 | **75.3** | 51.5 | 75.2 | 82.6 |
  | the same, + bias, past references | 41.9 | 74.5 | 43.2 | 75.1 | **52.0** | **75.3** | 82.5 |

  Joint attention turns TDViT-T from 3.5 AP behind Swin-T to 1.0 ahead
  (+1.5 AP50, +1.5 VID AP50): fast objects are no longer hurt, medium and
  slow ones gain about two points. The temporal bias and the past references
  make no difference worth their complexity, so the design kept is the
  paper's with one change -- a TDTB attends over its own window and the
  reference's -- and TDViT-T stays exactly Swin-T's size
  (`tdvit_t_joint_frcnn_fpn_3x.py`). Its speed is Swin-T's (25.8 ms a frame
  on a GH200, untrained weights). Submitted: the joint run resumed to 3
  epochs (7003775), TDViT-T+ and SELSA + TDViT-T with joint attention at one
  epoch (7003777, 7003779), SELSA + Swin-T's epoch 1 evaluated (7003781), and
  the joint checkpoint with its memory off and with near references
  (7003782, 7003783).

- **2026-10-01 (three epochs; a stronger baseline than the paper's)** — The
  3-epoch checkpoints, evaluated apart (full val):

  | | AP | AP50 | AP75 | VID AP50 | fast | medium | slow |
  | :-- | --: | --: | --: | --: | --: | --: | --: |
  | Swin-T | **50.2** | **79.0** | **55.4** | **79.5** | 57.3 | 78.8 | 85.7 |
  | TDViT-T, the paper's protocol | 46.0 | 75.7 | 49.4 | 76.2 | 51.1 | 75.2 | 83.8 |
  | TDViT-T, memory off | 48.2 | 76.8 | 52.8 | 77.2 | 55.3 | 76.6 | 83.2 |
  | TDViT-T, references 1 / 2 / 4 / 8 back | 49.0 | 78.0 | 53.8 | 78.5 | 56.4 | 78.3 | 84.1 |
  | *the paper: Swin-T* | *47.1* | *77.2* | *51.5* | | | | |
  | *the paper: TDViT-T* | *49.1* | *78.5* | *52.7* | | | | |

  TDViT-T with near references lands on the paper's TDViT-T; the Swin-T
  baseline trained the same way is 3.1 AP above the paper's Swin-T. So the
  paper's Swin-T was probably trained more weakly -- v1's Swin-T VID config,
  for one, resized to 600 and flipped, nothing more, where TDViT followed
  Swin's multi-scale and crop augmentation. Under test: Swin-T with v1's
  pipeline, 3 epochs (`frcnn_swint_fpn_3x_v1aug.py`, 7002832). Meanwhile the
  running losses of the joint-attention runs, on the same samples, drop
  *below* Swin-T's (iterations 3,000-6,000: joint 0.2204, Swin-T 0.2231, the
  paper's block 0.2264): with joint attention the other frame helps the
  network fit.

- **2026-10-01 (the block, not only the protocol)** — Two more findings.
  The test-matched references (past, at the test's distances) lift TDViT-T
  at epoch 1 from 37.5 to 38.8 AP (fast 42.0 to 45.6), still under memory
  off (40.1) and Swin-T (41.0). And on the same samples (one seed), TDViT-T
  *trains* worse than Swin-T at every epoch: a loss of 0.2021 against 0.1983
  at epoch 1 and 0.1522 against 0.1484 at epoch 3, where SELSA on Swin-T,
  aggregating proposals, reaches 0.1415. Training references are near and
  informative, so this is the block itself: a TDTB *replaces* the frame's
  spatial attention with attention to the reference, so half of TDViT-T's
  blocks no longer mix the frame's own tokens. Where the reference matches,
  little is lost; where content moved, the block brings in another frame's
  tokens and has nothing to fall back on -- the fast-object loss at test.

  The fix tried first keeps the paper's parameters, memory and dilations:
  `attention='joint'` lets a query attend over its window's own tokens *and*
  the reference's (a two-frame space-time window, Video Swin's 3D window with
  a dilated time step), so it can keep to its own frame where the reference
  does not match. With the frame as its own reference this is exactly Swin
  (first frames, DET images). `temporal_bias` adds a per-head logit on the
  reference's keys, from 0. One epoch each: joint (7002429), joint + bias
  (7002431), joint + bias + past references (7002434), each evaluated as
  default. The 3-epoch checkpoints of Swin-T and TDViT-T are being evaluated
  apart (7002376-9): their runs trained to the end but died in the final
  evaluation, where rank 0 scores for about 17 minutes while the others
  wait at a barrier and PyTorch 2's NCCL watchdog aborts after 10;
  `init_dist` now waits two hours.

- **2026-10-01 (distance is the problem)** — The epoch-1 checkpoint of
  TDViT-T under other test protocols (inference only):

  | Test protocol | AP | AP50 | AP75 | VID AP50 | fast | medium | slow |
  | :-- | --: | --: | --: | --: | --: | --: | --: |
  | *temporal earliest* held `D_t` frames (the default) | 37.5 | 69.9 | 36.6 | 70.4 | 42.0 | 69.1 | 79.8 |
  | a new reference every frame, exactly 4 / 8 / 16 / 32 back | 38.4 | 70.8 | 37.8 | 71.3 | 43.0 | 70.6 | 79.8 |
  | the same, exactly 1 / 2 / 4 / 8 back | **40.7** | **72.8** | 41.5 | **73.3** | 50.1 | 73.1 | 79.9 |
  | the previous frame for every stage | 40.6 | 72.3 | 41.8 | 72.8 | 50.4 | 72.3 | 79.2 |
  | memory off | 40.1 | 71.6 | 41.0 | 72.1 | 49.8 | 71.4 | 78.7 |
  | *Swin-T* | *41.0* | *73.1* | *42.4* | *73.6* | *51.4* | *73.1* | *81.1* |

  Near references help (+0.6 AP, +1.2 VID AP50 over memory off); far ones
  cost fast objects most of 8 points. Trained on references within `±D_t`,
  the model cannot use the farther, one-sided references the memory gives
  it. The likely fix is training on the test's distances: submitted, one
  epoch of TDViT-T with references 4-7, 8-15, 16-31 and 32-63 frames back
  (`[lo, hi]` windows, new in `stagewise_uniform`), evaluated as default
  (train 6998657, `work_dirs/tdvit_t_pastrefs_e1`; evaluation 6998658).

- **2026-10-01 (epoch 1: the memory hurts fast objects)** — Full val, after
  one epoch (before the learning-rate step):

  | | AP | AP50 | AP75 | VID AP50 | fast | medium | slow |
  | :-- | --: | --: | --: | --: | --: | --: | --: |
  | Swin-T | 41.0 | 73.1 | 42.4 | 73.6 | 51.4 | 73.1 | 81.1 |
  | TDViT-T | 37.5 | 69.9 | 36.6 | 70.4 | 42.0 | 69.1 | 79.8 |
  | TDViT-T, memory off at test | 40.1 | 71.6 | 41.0 | 72.1 | 49.8 | 71.4 | 78.7 |

  TDViT's weights make a sound still-image detector (memory off, -0.9 AP
  against Swin-T), and its memory helps slow objects (+1.1 VID AP50) but
  costs fast ones 7.8. The likely cause is the reference's distance: at test
  time a reference held for `D_t` frames lies `D_t` to `2 D_t - 1` frames
  back (stage 4: 32 to 63), where training drew references uniformly within
  `±D_t`, mostly much closer; a fast object leaves its 7x7 window. The draft
  of November 2021 reads *temporal earliest* as `f_{t - D_t}`, "similar to
  the manner of spatial dilated convolutions", and Fig. 5 joins each frame to
  the one exactly `D_t` back, so the paper's reuse may not mean holding a
  reference. TDViT-T+ and SELSA + TDViT-T, which inherit the protocol, are
  cancelled (6995580, 6995581); SELSA + Swin-T goes on. Inference-only
  diagnostics on the epoch-1 checkpoint: a new reference every frame (exactly
  `D_t` back, 6998077), dilations 1/2/4/8 the same way (6998078), the previous
  frame for every stage (6998079).

- **2026-10-01 (the next tier, submitted early)** — To have results by the
  next login, submitted before epoch 1's evaluation, to be cancelled if
  TDViT-T does not lead there: TDViT-T+ 6995580 (`work_dirs/tdvit_tplus_3x`,
  0.281 s an iteration), SELSA + TDViT-T 6995581 (`tdvit_selsa_t_3x`, 0.469
  s) and SELSA + Swin-T 6995582 (`tdvit_selsa_swint_3x`, 0.454 s). Chained on
  TDViT-T's run: inference-only ablations of its last checkpoint (temporal
  NMS 6995396, patch shuffle 6995397, channel shuffle 6995398, a new
  reference every frame 6995399) and the speed with trained weights
  (6995400). Epoch-1 evaluations: Swin-T 6996696, TDViT-T 6996825, and
  TDViT-T with its memory off at test time (`model.online=False`, the
  weights without the temporal attention) 6996827.

- **2026-10-01 (T2, T3 submitted)** — Smoke 6995011 passed. Submitted from
  the worktree `~/code/vfe-tdvit` (the same seed for both, so both draw the
  same samples): Swin-T baseline 6995240 (`work_dirs/tdvit_swint_base_3x`),
  TDViT-T 6995241 (`work_dirs/tdvit_t_3x`), 4 GH200s with `--accumulate 2`.
  Both run at 0.24 s an iteration (13,711 an epoch): TDViT's four reference
  passes add under 3% here. SELSA added to the detector; its smoke is
  6995312; the speed tool's first run (untrained weights) 6995305.

- **2026-10-01 (T1)** — Read the paper (PDF and arXiv HTML), the CVPR 2022
  supplementary (code, Table S1, rebuttal), the ECCV supplementary and
  reviews, and the November 2021 draft. Asked the authors the four things no
  source records (weights, learning rate, augmentation, references).
  Implemented and tested as above.
