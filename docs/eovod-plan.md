# EOVOD port — Phase 9

> Working document for bringing EOVOD into `vfe/`, the phase after 2.0.
> [rewrite-plan.md](rewrite-plan.md) records how MAMBA and STPN were ported and
> why the method is what it is; this document applies the same method to
> EOVOD. Update the checkboxes and the Progress log as work proceeds.

- **Goal:** EOVOD (*Efficient One-stage Video Object Detection by Exploiting
  Temporal Consistency*, ECCV 2022) running on plain PyTorch in `vfe/`, checked
  against its original code, with its released checkpoint reproducing its
  published score.
- **Source:** [guanxiongsun/EOVOD](https://github.com/guanxiongsun/EOVOD) at
  `84576bb` (2023-04-03), the only public code. The checkpoint and training log
  are on Google Drive, linked from its README. Issues #1 and #2 on this
  repository ask about it.
- **Method:** as for MAMBA and STPN. The original code is the oracle; port
  bottom-up with a parity harness per layer; evaluate the released checkpoint
  (E-M1) before spending any training compute; freeze the oracle's side into
  `run_parity.py` before retiring it.

## Status (2026-09-26)

- **Audited, nothing ported.** A static read of the code, plus a CPU run of it
  on the legacy stack. Findings below.
- **Next: E-M0, on the local machine.** Download the checkpoint and log, check
  that the checkpoint loads into the public code with no unexpected keys, and
  confirm it scores 79.7 through the original code. Everything after depends on
  the answer.

## Audit findings (2026-09-26)

**EOVOD is `v1.0.0` plus two files.** Its `mmdet/` is the same MMDetection
2.19.1 fork. It adds `mmdet/models/vid/fcos_att.py` (`FCOSAtt`, 292 lines) and
`mmdet/models/memory/mpn.py` (`MPN`, 346 lines). Of the 33 other files that
differ, 20 differ only in whitespace; the other 13 carry about 130 changed
lines:

- `MemoryBank` takes `in_channels` and owns a `SelsaAggregator`;
  `forward(x, ref)` returns `x + aggregator(x, ref)`. Sampling and updating are
  v1's (`torch.randperm` on the CPU generator).
- `CocoVideoDataset.evaluate` is COCO-style only, and `ImagenetVIDDataset`
  predates `shuffle_video_frames`, so EOVOD evaluates frames in order.
  (`mamba/vid_eval.py` differs too, but EOVOD never calls it.)
- Registrations: `FCOSAtt` and `MPN` in place of MAMBA and STPN.

Everything else EOVOD runs is stock 2.19.1 code that `v1.0.0` already holds:
FCOS, FocalLoss, IoULoss, the point generator, the data pipeline, the samplers.

**The public repository does not import.** `mmdet/models/__init__.py` imports
`CenterNetAtt` from `mmdet.models.vid`, which does not define it, so
`import mmdet.models` raises `ImportError`. (`tools/speed_test.py` also names a
`YOLOAtt` that does not exist.) Removing `CenterNetAtt` from the two lines that
name it is enough to run everything else.

**It runs on the existing legacy stack.** EOVOD's README pins Python 3.7 and
PyTorch 1.8.0; with that two-line patch it runs as-is on the `vfe` legacy stack
(Python 3.8, PyTorch 1.10.1, mmcv-full 1.3.17). On CPU with random weights, both
configs build, a training step runs forward and backward, and stateful inference
carries its memory from frame to frame. One legacy environment can therefore be
the oracle for both projects, with `mmdet` resolving to whichever checkout a
harness needs.

| config | parameters | of which MPN | memory on | training pixels |
| :-- | --: | --: | :-- | :-- |
| `fcos_att_r50_fpn_3x_vid_caffe_random.py` | 54,214,504 | 22,034,432 | C3–C5 | random |
| `fcos_att_r101_fpn_9x_vid_caffe_random_level2_imagenet.py` | 72,156,008 | 20,983,808 | C4–C5 | random |

The R-101 config is the one the README trains; its released checkpoint is
labelled "FCOS+LPN".

**The model.** `FCOSAtt` wraps a stock FCOS detector (caffe-style ResNet, FPN
with `on_output` extra convs, `FCOSHead` with GroupNorm, focal / IoU /
centerness losses) and inserts `MPN` between the backbone and the FPN
(`before_fpn=True`). MPN keeps one `MemoryBank` per backbone level from
`start_level` on; every pixel of that level's feature map attends over pixels
drawn from the memory, through a SELSA aggregator added residually.

- *Training:* the key frame's pixels attend over up to 2,000 randomly chosen
  pixels from each of its two reference frames (`np.random.shuffle`). A
  box-based sampling mode exists, but no config uses it.
- *Testing:* after each frame, pixels inside detections scoring above 0.3 are
  written to memory: up to 300 per box (`np.random.choice`) and 1,000 per frame,
  or the 50 highest-norm pixels when nothing qualifies. Each frame reads 2,000
  of up to 20,000 stored pixels at random.
- *The paper's terms:* nothing in the code is called LPN or SPN.
  `MPN.filter_with_mask`, the natural place for a location prior, is always
  called without a mask, so every pixel queries the memory; `start_level`
  statically skips the low levels. Whether this is the model behind the README's
  "FCOS+LPN" row is for the author to say. A checkpoint with parameters the
  public code lacks will show up as unexpected keys at E-M0.

**How 79.7 was measured.**

- *COCO-style*, not the ImageNet VID (FGFA) metric MAMBA and STPN report:
  mmdet's `CocoDataset.evaluate('bbox')`, which is pycocotools' COCOeval over
  VID val (maxDets 100/300/1000). The published AP 54.0 / AP50 79.7 / AP75 59.3
  (small / medium / large 9.8 / 26.6 / 60.4) are therefore not comparable with
  MAMBA's 83.8 or STPN's 85.2. Once `vfe` has the COCO path, it can report both.
- *Frames in order.* At each video's first frame, its 14 references over ±7
  frames (`test_with_adaptive_stride`, which `vfe` already has) are detected and
  written into memory.
- **Memory is reset only when a video's number is a multiple of 1000**
  (`int(name.split('_')[-1]) % 1000 == 0`). VID val numbers its 555 snippets in
  178 blocks (`…_00000000` to `…_00000005`, then `…_00001000`, …; 1 to 46
  snippets per block), so 377 of the 555 snippets start with the memory the
  earlier snippets in their block left behind. Reproducing 79.7 requires
  reproducing this, and a comparison with methods that treat every snippet
  independently should say so.
- *Stochastic*, like MAMBA's: both the pixel choice (numpy) and the memory read
  (torch) draw from global generators.
- *A parity trap to expect:* boxes reach feature-map cells as
  `int(box * scale_factor / stride)` after a rescale round trip, so float noise
  between stacks can move a cell: the pixel-level counterpart of 4f's NMS flips.

**Training recipe (from the config).** SGD, lr 1e-3, momentum 0.9, weight decay
1e-4, gradient clipping at 35; 500-iteration linear warmup from 1/3; ×0.1 after
epoch 6 of 9; one image per GPU, eight GPUs in the README's example. Data: VID
(2 references within ±9 frames, bilateral uniform) plus the DET 30-class subset,
as for MAMBA. Backbone: caffe-style ResNet-101 (BGR, mean subtraction only) from
`open-mmlab://detectron/resnet101_caffe`. The batch actually used, the seed, the
speed and the per-epoch scores come from the log at E-M0.

## What `vfe` has, and what it needs

**Already ported:** ResNet in caffe style with frozen stages and BN, FPN
(including `on_output` and `relu_before_extra_convs`), GroupNorm,
`multiclass_nms` with `score_factors`, `bbox2result`, v1's `MemoryBank`, the VID
dataset with in-order frames, every transform in EOVOD's pipelines, the four
reference samplers, the group and video samplers, the SGD constructor, the LR
schedule and the training loop. Caution: caffe style and FPN's extra convs were
ported with the rest, but no parity check has exercised them. MAMBA uses
pytorch style and a `ChannelMapper`; STPN uses an FPN whose extra level is a max
pool.

**To port:**

| piece | source (`v1.0.0` unless noted) | lines | note |
| :-- | :-- | --: | :-- |
| FCOS head | `dense_heads/{fcos_head,anchor_free_head,base_dense_head,dense_test_mixins}.py` | 453 / 350 / 526 / 206 | only the paths FCOS takes: GroupNorm, `Scale`, centerness, regress ranges, `get_bboxes` with score factors |
| point generator | `core/anchor/point_generator.py` | 263 | `MlvlPointGenerator` |
| `FocalLoss` | `losses/focal_loss.py` | 182 | the original ran mmcv's compiled kernel on CUDA and the Python formula on CPU; the port is the formula, so CUDA agreement is to float noise, not exact |
| `IoULoss` | `losses/iou_loss.py` | part of 474 | `mode='log'`, `eps=1e-6` |
| single-stage detector | `detectors/{single_stage,fcos}.py` | 171 / 19 | plus `distance2bbox` |
| `SelsaAggregator` | `aggregators/selsa_aggregator.py` | 77 | dropped from 2.0 with SELSA |
| EOVOD's `MemoryBank` | EOVOD `memory/memory_bank.py` | 81 | v1's, plus `in_channels` and the aggregator |
| `MPN`, `FCOSAtt` | EOVOD | 346 / 292 | stateful test, numpy draws |
| COCO-style evaluator | `datasets/coco.py` (`evaluate`, `results2json`) | — | on pycocotools, already a dependency |
| `open-mmlab://` | mmcv's `open_mmlab.json` | 2 entries | `detectron/resnet{50,101}_caffe`; dropped from 2.0 because nothing used it |

**One change to existing code.** Exact gradient accumulation swaps *torch*
generator states per virtual rank (`RngStreams` in `vfe/engine/trainer.py`);
numpy's and Python's are per process. MAMBA and STPN draw only from torch
inside the model, but MPN's training step calls `np.random.shuffle`, so
`--accumulate` would give EOVOD's micro-steps one shared numpy stream where the
original ranks each had their own. Add numpy's state to the per-rank swap, and
extend the 1-process × 2-micro-step vs 2-process bit-identity check to an
EOVOD model.

## Where each part can run

- **Cloud sessions (verified 2026-09-26).** Both stacks build on CPU from PyPI
  alone, so both sides of every CPU parity harness on synthetic inputs and
  random weights can run there. The legacy side took 3 min 20 s (recipe below).
  On the `vfe` side, PyPI's x86_64 `torch==2.10.0` is the same `+cu128` build
  `pyproject.toml` pins, and the 35 unit tests pass. The egress policy blocks
  Hugging Face, Google Drive, `download.openmmlab.com` and
  `download.pytorch.org`, so there are no checkpoints, pretrained weights or
  data there.
- **The local machine:** the real data and checkpoints, CUDA comparisons,
  freezing goldens.
- **Isambard:** full evaluation and training.

The legacy stack without conda (Python 3.8 from uv; PyPI's torch 1.10.1 is the
cu102 build and runs on CPU):

```bash
uv python install 3.8 && uv venv --python 3.8 legacy38
PY=legacy38/bin/python
uv pip install --python $PY torch==1.10.1 torchvision==0.11.2
uv pip install --python $PY numpy==1.23.5 "opencv-python-headless<5" "matplotlib<3.8" pycocotools \
    "scipy<1.11" yapf==0.32.0 addict terminaltables pyyaml packaging Pillow six "setuptools<60" wheel ninja
MMCV_WITH_OPS=1 FORCE_CUDA=0 MAX_JOBS=4 uv pip install --python $PY --no-build-isolation mmcv-full==1.3.17
# mmdet comes from a checkout on PYTHONPATH: the v1.0.0 worktree, or EOVOD with the import patch
```

## Milestones

- [ ] **E-M0 — the original, on the local machine.**
  1. Download the R-101 checkpoint and its log from the Drive folder in EOVOD's
     README, and mirror both to the Hugging Face model repository beside
     MAMBA's and STPN's, where Isambard can fetch them and they outlive the
     Drive link.
  2. Check out EOVOD at `84576bb` beside this repository (for example
     `~/code/eovod.legacy`) and drop `CenterNetAtt` from
     `mmdet/models/__init__.py`. Use the `vfe` conda environment with that
     checkout first on `PYTHONPATH`.
  3. Load the checkpoint into the R-101 config's `FCOSAtt`: expect 0 missing
     and 0 unexpected keys. An unexpected key is a module the public code
     lacks, and stops the plan until it is found.
  4. Evaluate on VID val through the original code
     (`python tools/test.py CONFIG --checkpoint CKPT --eval bbox`): expect AP50
     79.7 and AP 54.0, ±0.2. Fill in *Reproduction facts* from the log.
- [ ] **9a — one-stage machinery:** point generator, `Scale`, `distance2bbox`,
  FocalLoss, IoULoss, the FCOS head (forward, targets and loss, `get_bboxes`),
  the single-stage detector, `open-mmlab://`. One harness per layer, as in
  Phase 4: bit-exact on CPU wherever the arithmetic is elementwise, forward
  passes only on CUDA.
- [ ] **9b — EOVOD's modules:** `SelsaAggregator`, EOVOD's `MemoryBank`, `MPN`,
  `FCOSAtt`. Random weights with both generators aligned: memory contents
  exact, the training step's losses and CPU gradients, and multi-frame test
  detections compared as sets, across a %1000 reset boundary.
- [ ] **9c — COCO-style evaluator**, checked on synthetic detections over the
  full val set, as 5a was.
- [ ] **E-M1 — the released checkpoint through `vfe` on Isambard → AP50 79.7
  ± 0.2.** MAMBA's M1 took ≈1.35 GPU-hours.
- [ ] **9d — training:** numpy streams per virtual rank; training-step parity
  from the released checkpoint on real batches, as 6c.
- [ ] **E-M2, E-M3 — a short run, then the full 9x schedule**, overlaid on the
  original log. Budget once E-M2 has measured the speed; for scale, MAMBA's 6x
  took ≈17 GPU-hours and STPN's 9x ≈40.
- [ ] **Freeze** the EOVOD variants into `run_parity.py` before the EOVOD
  checkout is retired.

## Decisions needed

1. **Scope:** only the R-101 model, the one with a checkpoint? The R-50 config
   has no released weights, and the paper's CenterNet and YOLOX variants have
   no code (`CenterNetAtt` and `YOLOAtt` are named but never defined).
2. **Checkpoint hosting:** mirror the Drive files to Hugging Face?
3. **Cross-snippet memory:** reproduce it for E-M1 (needed to reach 79.7), and
   also report the score with the memory reset at every snippet?
4. **Isambard budget:** E-M1 ≈ 2 GPU-hours now; training after E-M2.
5. **Acceptance:** E-M1 within ±0.2 AP50, training within ±0.5, as for MAMBA
   and STPN.

## Reproduction facts (from the original log)

To be filled in at E-M0: hardware and batch, iterations per epoch, speed, seed,
per-epoch AP, final AP.

## Progress log

- **2026-09-26 (audit)** — Phase 9 opened. EOVOD's public code is `v1.0.0`'s
  MMDetection fork plus `FCOSAtt` and `MPN`. It fails to import (`CenterNetAtt`
  is imported but never defined); with that fixed it runs on the existing
  legacy stack, so one oracle environment serves both projects. Its 79.7 is
  COCO-style AP50, measured with memory carried across 377 of VID val's 555
  snippets. Both stacks were built from PyPI in a cloud container, so CPU parity
  work can happen there. Nothing ported yet; next is E-M0 on the local machine.
