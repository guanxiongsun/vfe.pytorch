# vfe.pytorch — Video Feature Enhancement in PyTorch

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Video object detection on ImageNet VID, in plain PyTorch. This repository holds
reference implementations of

- **[MAMBA](https://arxiv.org/abs/2401.09923)** — Multi-level Aggregation via Memory Bank (AAAI 2021)
- **[STPN](https://arxiv.org/abs/2402.02574)** — Spatio-temporal Prompting Network (ICCV 2023)
- **[EOVOD](https://arxiv.org/abs/2402.09241)** — Efficient One-stage Video Object Detection by
  Exploiting Temporal Consistency (ECCV 2022), on FCOS and YOLOX

together with the ImageNet VID data and annotations needed to train and
evaluate them, since the official dataset links are no longer reachable.

## News

- **2026-10-01** — EOVOD reproduced on FCOS: 54.0 AP with LPN and 53.8 with
  LPN + SPN, against the paper's 54.1 and 53.8. EOVOD also runs on YOLOX,
  ported from MMDetection and checked against it. MAMBA's pixel level, which
  the original release left out, is implemented
  ([#7](https://github.com/guanxiongsun/vfe.pytorch/pull/7)).
- **2026-09-27** — EOVOD implemented from its paper, on a ported FCOS
  ([#6](https://github.com/guanxiongsun/vfe.pytorch/pull/6)).
- **2026-09-20** — **v2.0**, a rewrite in plain PyTorch. MAMBA and STPN
  trained with it score 84.06 and 84.54 AP50 (originally 83.82 and 85.15).
- **2024-02** — v1.0: MAMBA and STPN code and models, with a mirror of
  ImageNet VID and COCO-style annotations.

## Highlights

- **Plain PyTorch.** Up to 1.x this was a fork of MMDetection 2.19.1 that
  needed `mmcv-full` and a Python 3.8 environment pinned to PyTorch 1.10. The
  `vfe/` package now implements everything it uses — models, data pipeline,
  evaluator, training loop, samplers, config system — on `torch` and
  `torchvision.ops`; nothing from mmcv, mmdet or mmengine is imported at
  runtime. It runs on Python 3.12 and PyTorch 2.10, on x86 and Arm. See
  [what changed in 2.0](#what-changed-in-20).
- **Checked against the original.** Every layer of the port is compared,
  tensor by tensor, with frozen outputs of the original implementation
  ([docs/parity.md](docs/parity.md)). The YOLOX port matches MMDetection to
  2.3e-13 in float64, and its training pipeline bit for bit.
- **Retrained, not only ported.** MAMBA's and STPN's released checkpoints score
  within 0.02 AP50 of their published results here. Trained from scratch here,
  MAMBA reaches 84.06 AP50 (published: 83.82), STPN 84.54 (85.15) and EOVOD
  54.0 AP (the paper: 54.1).
- **What the original releases left out:** MAMBA's pixel level, EOVOD's
  location and size priors (LPN and SPN), and EOVOD on YOLOX.
- **Exact multi-GPU training.** With BatchNorm frozen, `--accumulate k` makes
  `n` GPUs train exactly as `n * k` did, and `--resume-from auto` continues a
  run exactly.
- **Fast inference**, optional and with the same detections: CUDA graphs and
  fewer GPU syncs run FCOS at 105 FPS and YOLOX-M at over 180 FPS on one GH200.

## Results

ImageNet VID validation, AP50 and the standard motion-speed breakdown:

| Model | Backbone | AP50 | AP (fast) | AP (med) | AP (slow) | |
| :-- | :-- | :--: | :--: | :--: | :--: | :-- |
| Faster R-CNN | ResNet-101 | 76.7 | 52.3 | 74.1 | 84.9 | [reference](https://github.com/Scalsol/mega.pytorch#main-results) |
| SELSA | ResNet-101 | 81.5 | — | — | — | [reference](https://github.com/open-mmlab/mmtracking/tree/master/configs/vid/selsa) |
| MEGA | ResNet-101 | 82.9 | 62.7 | 81.6 | 89.4 | [reference](https://github.com/Scalsol/mega.pytorch) |
| **MAMBA** | ResNet-101 | **83.8** | 65.3 | 83.8 | 89.5 | [config](configs/vid/mamba) · [model](https://huggingface.co/guanxiongsun/vfe.pytorch/tree/main/work_dirs/mamba_r101_dc5_6x) |
| **STPN** | Swin-T | **85.2** | 64.1 | 84.1 | 91.4 | [config](configs/vid/stpn) · [model](https://huggingface.co/guanxiongsun/vfe.pytorch/tree/main/work_dirs/stpn_swint_adam_9x) |

### Reproduced with this code

Measured on 4× GH200, AP50 on the same validation set:

| | released checkpoint, evaluated here | trained here, from scratch | originally published |
| :-- | :--: | :--: | :--: |
| MAMBA | 83.80 | 84.06 | 83.82 |
| STPN | 85.15 | 84.54 | 85.15 |

Evaluation reproduces the released checkpoints to within 0.02 AP50. Training
reproduces MAMBA and lands 0.61 low on STPN, with per-epoch losses within 0.6%
of the original run — run-to-run variance, most likely, though that was not
confirmed with a second seed.

> **MAMBA's schedule.** The published MAMBA model trained epochs 1–3 at batch 4
> and epochs 4–6 at batch 8 (its checkpoint records a 4-GPU run resumed on 8,
> and mmcv rescaled the iteration count). Reading the config literally — batch 8
> throughout — halves the steps in epochs 1–3 and scores 83.16. The 84.06 above
> follows the published model's own schedule.

> **MAMBA's pixel level.** The released model is the paper's instance-level
> variant (Table 3, "Ours_ins": 83.7), which the numbers above reproduce. The
> paper's full model also enhances the feature map before the RPN, and that
> pixel level was never released
> ([#4](https://github.com/guanxiongsun/vfe.pytorch/issues/4)). It is now
> implemented: [`mamba_full_r101_dc5_3x.py`](configs/vid/mamba/mamba_full_r101_dc5_3x.py).
> After one epoch it scores 75.6 AP50 against 72.1 for the instance level
> alone ([docs/mamba-pixel-plan.md](docs/mamba-pixel-plan.md)). There is no
> full-schedule checkpoint of it yet.

### EOVOD

ImageNet VID validation, COCO-style AP as the paper reports it:

| Detector | | AP | AP50 | AP75 | paper |
| :-- | :-- | :--: | :--: | :--: | :--: |
| FCOS, ResNet-101 | alone | 49.8 | 73.6 | 54.6 | 49.8 / 73.3 / 54.6 |
| | + LPN | 54.0 | 79.2 | 59.3 | 54.1 / 79.8 / 59.5 |
| | + LPN + SPN | 53.8 | 78.9 | 59.2 | 53.8 / 76.9 / 58.9 |
| YOLOX-M | alone | 55.5 | 75.1 | 61.6 | 49.4 / 69.4 / 55.4 |
| | + LPN | 56.1 | 75.8 | 62.3 | 53.3 / 75.1 / 58.1 |
| | + LPN + SPN | 55.4 | 74.7 | 61.6 | 52.7 / 74.5 / 56.7 |

FCOS trains for 9 epochs with
[`eovod_fcos_r101_fpn_9x_backbone_cls_ctrreg.py`](configs/vid/eovod/eovod_fcos_r101_fpn_9x_backbone_cls_ctrreg.py),
which aggregates before the FPN as the original code does; FCOS alone trains on
the same schedule. YOLOX-M trains for 10 epochs from the COCO weights, with
YOLOX's own recipe applied to video clips
([`eovod_yolox_m_clips_10e.py`](configs/vid/eovod/eovod_yolox_m_clips_10e.py);
alone, [`eovod_yolox_m_clips_10e_plain.py`](configs/vid/eovod/eovod_yolox_m_clips_10e_plain.py)).
Started from COCO, YOLOX-M is already stronger than the paper's, and the
location prior adds 0.6 AP to it, against 4.2 on FCOS; trained alone on still
images, as YOLOX usually is, it scores 56.1. The size prior is a test-time
setting of the same model. The checkpoint released with the original EOVOD code
scores 54.0 / 79.7 / 59.3 here. How each number was reached:
[docs/eovod-plan.md](docs/eovod-plan.md) and
[docs/eovod-yolox-plan.md](docs/eovod-yolox-plan.md).

## Install

```bash
uv venv --python 3.12 && uv sync          # uv.lock pins torch 2.10.0+cu128
# or
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
```

PyPI's default aarch64 `torch` wheel is CPU-only, so the CUDA index matters on
Arm machines (Isambard-AI, Grace Hopper) as well as on x86.

## Data

Download ILSVRC2015 DET and VID from
[this mirror](https://huggingface.co/datasets/guanxiongsun/imagenetvid/tree/main)
and the [COCO-style annotations](https://huggingface.co/datasets/guanxiongsun/imagenetvid/blob/main/annotations.tar.gz),
then arrange (or symlink) them as:

```
data/ILSVRC/
├── Annotations/{DET,VID}
├── Data/{DET,VID}
├── ImageSets
└── annotations/          # imagenet_vid_{train,val}.json, imagenet_det_30plus1cls.json
```

The `ImageSets` lists come from
[FGFA](https://github.com/msracver/Flow-Guided-Feature-Aggregation/tree/master/data/ILSVRC2015/ImageSets).

## Evaluate

```bash
# one GPU
python -m vfe.cli.test configs/vid/mamba/mamba_r101_dc5_6x.py CHECKPOINT --work-dir WORK_DIR

# one node, several GPUs
torchrun --standalone --nproc_per_node=4 -m vfe.cli.test CONFIG CHECKPOINT \
    --launcher pytorch --work-dir WORK_DIR
```

Video detectors are evaluated one video per process: frames must arrive in
order on the same process, which the sampler guarantees.

## Train

```bash
torchrun --standalone --nproc_per_node=4 -m vfe.cli.train CONFIG \
    --launcher pytorch --accumulate 2 --work-dir WORK_DIR --seed 1466607766
```

`--accumulate k` runs `k` micro-steps per process, so `n` GPUs train exactly as
`n * k` did: each micro-step gets its own data loader and its own random
streams, mirroring one original rank. Four GPUs with `--accumulate 2` therefore
reproduce the original eight-GPU batch, at the same speed per iteration. This is
exact rather than approximate — halving a loss halves every gradient — provided
no layer uses batch statistics, and the loop refuses to accumulate unless every
BatchNorm is frozen.

Runs write checkpoints and `*.log.json` logs in the original format, so
`tools/analyze_train_log.py --compare` can overlay a run on the original one,
and `--resume-from auto` continues a run exactly, generator states included.
Slurm scripts for Isambard-AI are in [tools/isambard/](tools/isambard/).

EOVOD on YOLOX starts from Megvii's COCO-trained
[YOLOX-M](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth),
converted once and passed with `--load-from` (the configs name the path used on
Isambard-AI):

```bash
python tools/convert_yolox_megvii.py yolox_m.pth yolox_m_coco_eovod.pth --drop-classifier --prefix detector.
```

## Tests and parity

```bash
python -m pytest                       # fast unit tests, no data needed; GPU-only ones skip on CPU
python tools/checks/run_parity.py check --all
```

The second command re-runs every layer of the port against frozen outputs of
the original implementation and compares them tensor by tensor. See
[docs/parity.md](docs/parity.md) for how that works, what it does and does not
claim, and how to rebuild the oracle. FCOS and YOLOX, ported for EOVOD, have
their own two-environment checks: `tools/checks/parity_fcos.py`,
`parity_yolox.py` and `parity_yolox_pipeline.py`.

## What changed in 2.0

Version 1.x is preserved: `git checkout v1.0.0`, or browse the
[`v1` branch](https://github.com/guanxiongsun/vfe.pytorch/tree/v1). Its
environment is still the reference this rewrite is checked against.

| 1.x | 2.0 |
| :-- | :-- |
| `python tools/train.py CONFIG` | `python -m vfe.cli.train CONFIG` (or `vfe-train`) |
| `./tools/dist_train.sh CONFIG 8` | `torchrun --nproc_per_node=8 -m vfe.cli.train CONFIG --launcher pytorch` |
| `python tools/test.py CONFIG --checkpoint CKPT --eval bbox` | `python -m vfe.cli.test CONFIG CKPT` |
| `mmcv.Config` | `vfe.config.Config` (same syntax, `_base_` included) |
| `mmdet.apis.train_detector` | `vfe.engine.train_detector` |
| mmcv registry + `build_detector` | `vfe.models.build_model` |
| python 3.8, torch 1.10, mmcv-full 1.3.17 | python 3.12, torch 2.10, no mm\* |

Configs are unchanged: the same files load in both stacks and resolve
identically, which is one of the parity checks.

Not carried over: SELSA and the single-frame baselines, fp16 training, and the
MMDetection model zoo this was forked from — all still in `v1.0.0`. A
single-frame Faster R-CNN baseline is back, as
[`frcnn_r101_dc5_3x.py`](configs/vid/mamba/frcnn_r101_dc5_3x.py).

## Citation

```bibtex
@inproceedings{sun2021mamba,
  title     = {MAMBA: Multi-level Aggregation via Memory Bank for Video Object Detection},
  author    = {Sun, Guanxiong and Hua, Yang and Hu, Guosheng and Robertson, Neil},
  booktitle = {AAAI},
  year      = {2021}
}
@inproceedings{sun2022eovod,
  title     = {Efficient One-stage Video Object Detection by Exploiting Temporal Consistency},
  author    = {Sun, Guanxiong and Hua, Yang and Hu, Guosheng and Robertson, Neil},
  booktitle = {ECCV},
  year      = {2022}
}
@inproceedings{sun2023stpn,
  title     = {Spatio-temporal Prompting Network for Robust Video Feature Extraction},
  author    = {Sun, Guanxiong and Wang, Chi and Zhang, Zhaoyu and Deng, Jiankang
               and Zafeiriou, Stefanos and Hua, Yang},
  booktitle = {ICCV},
  year      = {2023}
}
```

`vfe/` is a derivative work of [MMDetection](https://github.com/open-mmlab/mmdetection)
2.19.1 and [MMCV](https://github.com/open-mmlab/mmcv) 1.3.17, Apache-2.0; see
[NOTICE](NOTICE).
