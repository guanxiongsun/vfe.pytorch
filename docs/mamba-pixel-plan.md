# MAMBA's pixel level

> Working document for adding the pixel level to `vfe`'s MAMBA, the part of
> the paper's full model that was never released (issue #4). Update the
> checkboxes and the Progress log as work proceeds.

- **Goal:** the paper's full MAMBA (pixel + instance level) in `vfe/`, trained
  and evaluated on ImageNet VID; first the paper's ablation at a short
  schedule, then the full schedule if the pixel level earns its place.
- **Why:** the released model is the instance-only row of the paper's
  Table 3 (`Ours_ins`, 83.7; the released checkpoint scores 83.80 through
  `vfe`). The headline 84.6 needs the pixel level, which the repository has
  never contained (not in `v1.0.0`, not since).

## The paper

*MAMBA: Multi-level Aggregation via Memory Bank for Video Object Detection*
(Sun, Hua, Hu, Robertson; AAAI 2021; arXiv 2401.09923).

- **Pipeline:** backbone -> the feature maps enhanced by the pixel-level
  memory bank -> RPN on the enhanced maps -> proposals enhanced by the
  instance-level memory bank -> detection head.
- **Both levels:** the generalized enhancement operation (GEO), SELSA's
  attention with a residual; `N_pix = 1` GEO at the pixel level, `N_ins = 2`
  at the instance level (after each shared FC); `N_k = 2000` keys read from
  a memory bank (light-weight key-set construction: a random subset), and
  feature-wise random replacement when writing.
- **Memory writes:** K = 100 random pixels within each detected box (pixel
  level); the top K = 75 proposals by objectness (instance level).
- **Training:** two phases -- the pixel level alone for 60K iterations (lr
  1e-3 for 40K, then 1e-4), then both end to end for 120K (1e-3 for 80K,
  then 1e-4); 4 Titan RTX GPUs, SGD, momentum 0.9, weight decay 1e-4.
- **Table 3 (Faster R-CNN, ResNet-101):**

  | Variant | Pixel | Instance | mAP | Runtime |
  | :-- | :-: | :-: | --: | --: |
  | Faster R-CNN | | | 75.4 | 51.8 ms |
  | Ours_pix | ✓ | | 81.8 | 81.6 ms |
  | Ours_ins | | ✓ | 83.7 | 79.6 ms |
  | Ours | ✓ | ✓ | 84.6 | 110.3 ms |

**Left open**, and the choice here (each is an option of `MambaPixelLevel`):

- *Which map:* the backbone's DC5 output (2048 channels, stride 16, before
  the ChannelMapper; `position='backbone'`) -- the released EOVOD code's
  `MPN`, the same design on FCOS, enhances backbone maps (`before_fpn`).
  `position='neck'` enhances the ChannelMapper's 512-channel output.
- *Training keys:* 2,000 random pixels per reference frame, capped at 2,000
  like a test-time read (`train_keys='random'`, `MPN`'s released configs).
  The alternative, K random pixels inside each reference frame's ground-truth
  boxes (`'gt'`), leaks and costs 4.1 AP50 in the full model at one epoch.
- *Which detections are written:* score above 0.3, at most 1,000 pixels per
  frame, the 50 highest-norm pixels when nothing qualifies (all `MPN`'s).
- *The first frame:* detection on the 14 reference frames and the key frame
  (the instance level initialising its memory from their top-75 RoIs, as the
  released model does), then their pixels inside those detections fill the
  pixel memory. `MPN` did the same with the plain detector.

## Design: paper -> `vfe/`

| Paper | `vfe/` |
| :-- | :-- |
| pixel-level GEO over the memory | `MambaPixelLevel.forward`: every pixel a query, `x + MambaAggregator(x, keys)` -- the instance level's SELSA aggregator and memory bank (20,000 / 2,000, random), on pixels |
| K = 100 pixels per detected box | `MambaPixelLevel.write` / `pixels` (cells whose centres lie inside the box, EOVOD's rule) |
| RPN on the enhanced map | `MAMBA._stage_one` / `_stage_two` around the pixel level; the RPN and the RoI head read the enhanced map |
| the instance level | unchanged: `MambaBBoxHead`; a plain `StandardRoIHead` leaves it out |
| Table 3's rows | `frcnn_r101_dc5_3x.py` (baseline), `mamba_pix_r101_dc5_3x.py` (Ours_pix), `mamba_r101_dc5_3x.py` (Ours_ins, the released model), `mamba_full_r101_dc5_3x.py` (Ours); `mamba_pix_neck_r101_dc5_3x.py` for the map ablation |

Without `pixel` the model is the released one, bit for bit: training losses,
every gradient, and five frames of stateful inference on a tiny model match
the code before the change exactly (checked 2026-09-29). Pixel-level
inference supports the adaptive-stride protocol only (what every released
config uses).

## Milestones

- [x] **P1 — the pixel level and the variants**, unit-tested
  (`tests/test_mamba_pixel.py`: key and write rules, every parameter trained
  in every variant, stateful inference, the rescale path, the released
  checkpoint's keys unchanged), and a full-size CPU smoke of the R-101
  configs at 600x1000 with 14 references.
- [x] **P2 — Table 3 at one epoch** (batch 8, lr 1e-3, ImageNet init): with
  random training keys, pixel-only 72.6 (+6.6 AP50 over Faster R-CNN's 66.0)
  and full **75.6** (+3.5 over instance-only's 72.1). Ground-truth keys, the
  first choice, leaked (below).
- [ ] **P3 — the full schedule**, with the user's approval: the paper's two
  phases or a single 6x run of the chosen variant.

## Progress log

- **2026-09-30 (random training keys fix the full model)** — With
  `train_keys='random'`, one epoch, VID AP50: pixel-only **72.6** (fast 53.0,
  medium 71.6, slow 78.7), up from 70.5; full **75.6** (56.4 / 74.6 / 82.1),
  up from 71.5 -- now 3.5 above instance-only (72.1) and 9.6 above Faster
  R-CNN, where the paper's full schedule has +0.9 and +9.2. Ground-truth keys
  had leaked, as the training losses suggested. Random keys are now the
  default. Next, with the user's approval: the full model on a full schedule.
- **2026-09-30 (P2: the ablation at one epoch)** — Full val, VID AP50:

  | Variant | all | fast | medium | slow | vs baseline | paper, full schedule |
  | :-- | --: | --: | --: | --: | --: | --: |
  | Faster R-CNN | 66.0 | 42.2 | 64.9 | 74.0 | | (75.4) |
  | pixel (backbone map) | 70.5 | 50.0 | 69.4 | 77.9 | +4.5 | +6.4 |
  | pixel (neck map) | 70.6 | 53.1 | 70.0 | 75.7 | +4.6 | |
  | instance (released) | 72.1 | 52.3 | 71.6 | 78.3 | +6.1 | +8.3 |
  | full (pixel + instance) | 71.5 | 48.1 | 70.0 | 79.7 | +5.5 | +9.2 |

  Each level alone follows the paper's pattern; together they do not add,
  and fast motion drops (48.1 against 52.3). The training losses point at the
  keys: every model with the pixel level ends the epoch far lower (0.23)
  than instance-only (0.29) or the baseline (0.31), yet does worse on val.
  Ground-truth training keys hand the key frame its objects' pixels -- on
  DET images the "references" are the key image itself -- while test keys
  come from imperfect detections: EOVOD's E-M3 leak in another form. `MPN`'s
  released configs, and the FCOS design that reproduced EOVOD's paper, used
  random keys. Submitted: pixel and full with `train_keys='random'`, one
  epoch each (jobs 6959333 / 6959335, evaluations 6959334 / 6959336).
- **2026-09-29 (P2 submitted)** — Five 1-epoch runs, each with a chained
  full-val evaluation, from a separate Isambard worktree
  (`~/code/vfe-mamba`, so the queued EOVOD jobs keep their checkout): ins
  6955515 / 6955516, full 6955517 / 6955518, frcnn 6955519 / 6955520, pix
  6955521 / 6955522, pix_neck 6955523 / 6955524; work dirs
  `/projects/b5cs/vfe/work_dirs/mamba_{ins,full,frcnn,pix,pix_neck}_e1`.
- **2026-09-29 (P1)** — Issue #4 checked: no pixel-level code anywhere in
  the repository's history. The `v1.0.0` detector imported `MemoryBank` and
  accepted an unused `memory_cfg`, possibly what was left of it. Built from
  the paper with `MPN`'s choices where the paper is silent.
