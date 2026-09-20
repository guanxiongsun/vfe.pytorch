# vfe.pytorch — Video Feature Enhancement in PyTorch

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Video object detection on ImageNet VID, in plain PyTorch. This repository holds
reference implementations of

- **[MAMBA](https://arxiv.org/abs/2401.09923)** — Multi-level Aggregation via Memory Bank (AAAI 2021)
- **[STPN](https://arxiv.org/abs/2402.02574)** — Spatio-temporal Prompting Network (ICCV 2023)

together with the ImageNet VID data and annotations needed to train and
evaluate them, since the official dataset links are no longer reachable.

**Version 2.0 is a rewrite.** Up to 1.x this was a fork of MMDetection 2.19.1
and needed `mmcv-full`, `mmdet` and a Python 3.8 environment pinned to PyTorch
1.10. The `vfe/` package now implements everything it uses — models, data
pipeline, evaluator, training loop, samplers, config system — on plain `torch`
and `torchvision.ops`. Nothing from mmcv, mmdet or mmengine is imported at
runtime, and the code runs on current PyTorch. See
[what changed in 2.0](#what-changed-in-20).

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

## Tests and parity

```bash
python -m pytest                       # 35 fast CPU tests, no data needed
python tools/checks/run_parity.py check --all
```

The second command re-runs every layer of the port against frozen outputs of
the original implementation and compares them tensor by tensor. See
[docs/parity.md](docs/parity.md) for how that works, what it does and does not
claim, and how to rebuild the oracle.

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
MMDetection model zoo this was forked from — all still in `v1.0.0`.

## Citation

```bibtex
@inproceedings{sun2021mamba,
  title     = {MAMBA: Multi-level Aggregation via Memory Bank for Video Object Detection},
  author    = {Sun, Guanxiong and Hua, Yang and Hu, Guosheng and Robertson, Neil},
  booktitle = {AAAI},
  year      = {2021}
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
