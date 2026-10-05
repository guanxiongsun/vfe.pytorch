# vfe.pytorch — Video Feature Enhancement in PyTorch

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Video object detection on ImageNet VID, in plain PyTorch. This repository holds
reference implementations of

- **[MAMBA](https://arxiv.org/abs/2401.09923)** — Multi-level Aggregation via Memory Bank (AAAI 2021)
- **[STPN](https://arxiv.org/abs/2402.02574)** — Spatio-temporal Prompting Network (ICCV 2023)
- **[EOVOD](https://arxiv.org/abs/2402.09241)** — Efficient One-stage Video Object Detection by
  Exploiting Temporal Consistency (ECCV 2022), on FCOS and YOLOX
- **[TDViT](https://arxiv.org/abs/2402.09257)** — Temporal Dilated Video Transformer for Dense
  Video Tasks (ECCV 2022)

together with the ImageNet VID data and annotations needed to train and
evaluate them, since the official dataset links are no longer reachable.

## News

- **2026-10-02** — TDViT, whose code was never released, implemented from its
  paper and reproduced ([configs/vid/tdvit](configs/vid/tdvit)).
- **2026-10-01** — EOVOD reproduced on FCOS and running on YOLOX; MAMBA's
  pixel level, which the original release left out, implemented
  ([#7](https://github.com/guanxiongsun/vfe.pytorch/pull/7)).
- **2026-09-27** — EOVOD implemented from its paper, on a ported FCOS
  ([#6](https://github.com/guanxiongsun/vfe.pytorch/pull/6)).
- **2026-09-20** — **v2.0**, a rewrite in plain PyTorch.
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
  ([tools/checks/](tools/checks)).
- **Retrained, not only ported.** Released checkpoints score within 0.02 AP50
  of their published results here, and every model is also trained from
  scratch with this code: STPN lands 0.6 AP50 below its paper, the others
  match or beat theirs.
- **What the original releases left out:** MAMBA's pixel level, EOVOD's
  location and size priors (LPN and SPN), EOVOD on YOLOX, and TDViT
  altogether, whose code was never published.
- **Exact multi-GPU training.** With BatchNorm frozen, `--accumulate k` makes
  `n` GPUs train exactly as `n * k` did, and `--resume-from auto` continues a
  run exactly.
- **Fast inference**, optional and with the same detections: CUDA graphs and
  fewer GPU syncs run FCOS at 105 FPS and YOLOX-M at over 180 FPS on one GH200.

## Results

ImageNet VID validation. AP50 is the dataset's standard metric; AP is
COCO-style AP over IoU 0.5–0.95, which the EOVOD and TDViT papers report.

| Method | Detector | Backbone | AP50 | AP | |
| :-- | :-- | :-- | :--: | :--: | :-- |
| [MAMBA](configs/vid/mamba) (AAAI 2021) | Faster R-CNN | ResNet-101 | 83.8 | — | [model](https://huggingface.co/guanxiongsun/vfe.pytorch/tree/main/work_dirs/mamba_r101_dc5_6x) |
| [STPN](configs/vid/stpn) (ICCV 2023) | Faster R-CNN | Swin-T | 85.2 | — | [model](https://huggingface.co/guanxiongsun/vfe.pytorch/tree/main/work_dirs/stpn_swint_adam_9x) |
| [EOVOD](configs/vid/eovod) (ECCV 2022) | FCOS | ResNet-101 | 79.7 | 54.0 | |
| | YOLOX-M | CSPDarknet | 76.3 | 56.1 | |
| [TDViT](configs/vid/tdvit) (ECCV 2022) | Faster R-CNN | TDViT-T | 80.8 | 50.6 | |
| | Faster R-CNN | TDViT-T+ | 81.4 | 51.4 | |

MAMBA and STPN are their released checkpoints, evaluated here; EOVOD (with
its location prior) and TDViT were trained with this code. Each method's
folder has its comparison with the paper, its baselines and variants, and
anything its configs do not say.

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
Method-specific steps, such as converting YOLOX's COCO weights for EOVOD, are
in each method's folder under [configs/vid/](configs/vid).

## Tests and parity

```bash
python -m pytest                       # fast unit tests, no data needed; GPU-only ones skip on CPU
python tools/checks/run_parity.py check --all
```

The second command re-runs every layer of the port against frozen outputs of
the original implementation and compares them tensor by tensor; the docstring
of [`run_parity.py`](tools/checks/run_parity.py) explains how. FCOS and YOLOX, ported for EOVOD, have
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
@inproceedings{sun2022tdvit,
  title     = {TDViT: Temporal Dilated Video Transformer for Dense Video Tasks},
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
