# Changelog

## 2.0.0

Rewritten on plain PyTorch; mmcv, mmdet and mmengine are no longer used.
Full notes: [docs/release-notes-2.0.md](docs/release-notes-2.0.md).

- `vfe/` implements the models, data pipeline, VID evaluator, training loop,
  samplers and config system that MAMBA and STPN use, on `torch` and
  `torchvision.ops`. Checked against mmdet 2.19.1 tensor by tensor.
- Python 3.12 / PyTorch 2.10; runs on current GPUs, x86_64 and aarch64.
- `python -m vfe.cli.{train,test}` and the `vfe-train` / `vfe-test` entry
  points replace `tools/{train,test}.py` and the `dist_*.sh` scripts.
- Exact gradient accumulation (`--accumulate k`): `n` processes reproduce
  `n * k` GPUs. Resuming restores the random streams as well as the weights.
- `tools/checks/run_parity.py` freezes and re-checks all 27 parity variants.
- A real STPN Swin-S config (`configs/vid/stpn/stpn_swins_adam_9x.py`), which
  was a 0-byte file; untrained.
- Removed: the vendored `mmdet/` tree, SELSA and single-frame configs, fp16
  training, and MMDetection's docs, demos and model zoo. All preserved at
  `v1.0.0`.
- Licence unchanged (Apache-2.0); attribution in `NOTICE`. The README's
  licence badge said BSD and was wrong.

## 1.0.0

The original implementation, on MMDetection 2.19.1 / MMCV 1.3.17 / PyTorch
1.10. Tagged from `main` as it stood before the rewrite.
