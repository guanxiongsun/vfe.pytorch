# vfe.pytorch 2.0

**The mmlab dependency is gone.** Up to 1.x this repository was a fork of
MMDetection 2.19.1: running MAMBA or STPN meant `mmcv-full` 1.3.17, Python 3.8
and PyTorch 1.10, a combination that is increasingly hard to install and cannot
use current GPUs. The `vfe/` package now implements everything these models
need — backbones, necks, heads, the memory bank, prompted Swin, the data
pipeline, the VID evaluator, the training loop, samplers and the config system
— on plain `torch` and `torchvision.ops`. Nothing from mmcv, mmdet or mmengine
is imported at runtime.

The rewrite is not a reinterpretation. Every layer was checked against the
original implementation by running identical inputs through both and comparing
the results tensor by tensor; most artifacts are bit-exact, and the rest sit on
the documented floating-point floor. `docs/parity.md` explains how to re-run
those checks, and `docs/rewrite-plan.md` records what they found.

## Results

| | released checkpoint | trained with 2.0 | originally published |
| :-- | :--: | :--: | :--: |
| MAMBA | 83.80 | 84.06 | 83.82 |
| STPN | 85.15 | 84.54 | 85.15 |

Released checkpoints load unchanged and evaluate to within 0.02 AP50. Training
from scratch reproduces MAMBA. STPN lands 0.61 low with per-epoch losses within
0.6% of the original run; a second seed was not run, so that gap is unattributed.

## Upgrading

| 1.x | 2.0 |
| :-- | :-- |
| `python tools/train.py CONFIG` | `python -m vfe.cli.train CONFIG` (or `vfe-train`) |
| `./tools/dist_train.sh CONFIG 8` | `torchrun --nproc_per_node=8 -m vfe.cli.train CONFIG --launcher pytorch` |
| `python tools/test.py CONFIG --checkpoint CKPT --eval bbox` | `python -m vfe.cli.test CONFIG CKPT` |
| `mmcv.Config` | `vfe.config.Config` |
| `mmdet.apis.train_detector` | `vfe.engine.train_detector` |

Configs and checkpoints carry over unchanged.

## Also new

- **Gradient accumulation that is exact, not approximate.** `--accumulate k`
  gives each micro-step its own loader and random streams, so `n` processes
  reproduce what `n * k` GPUs did, iteration for iteration. Eight-GPU recipes
  run on four.
- **Resumption that continues the random streams**, not just the weights.
- **`run_parity.py`** freezes the original implementation's side of all 27
  equivalence checks, so the port stays checkable after its reference
  environment is gone.
- **Slurm scripts for Isambard-AI** (aarch64 / GH200) in `tools/isambard/`.
- **A real STPN Swin-S config**, which was an empty file in 1.x. Untrained: no
  released checkpoint exists for it.

## Removed

SELSA and the single-frame baselines, fp16 training, the vendored `mmdet/`
tree, and the MMDetection model zoo and documentation this was forked from.
All of it remains at the `v1.0.0` tag, which is also the reference environment
the parity checks use.

## Requirements

Python 3.12, PyTorch 2.10 (CUDA 12.8 wheels). `uv.lock` resolves for x86_64 and
aarch64 alike.
