"""Sanity-check the legacy ``vfe`` env: print versions and build the VID detectors on CPU.

Builds MAMBA and STPN from their configs to confirm the mmlab registry + custom VID
modules import and construct correctly. Does not require a GPU.

Usage:
    conda run -n vfe --no-capture-output python tools/checks/verify_env.py
"""
import traceback
from pathlib import Path

import mmcv
import mmdet
import torch
from mmcv import Config
from mmdet.models import build_detector

REPO_ROOT = Path(__file__).resolve().parents[2]

print(f"torch       {torch.__version__}")
print(f"mmcv        {mmcv.__version__}")
print(f"mmdet       {mmdet.__version__}")
print(f"cuda(build) {torch.version.cuda}   cuda_available={torch.cuda.is_available()}")
print("-" * 60)

CONFIGS = [
    "configs/vid/mamba/mamba_r101_dc5_6x.py",
    "configs/vid/stpn/stpn_swint_adam_9x.py",
]

for rel in CONFIGS:
    path = REPO_ROOT / rel
    try:
        cfg = Config.fromfile(str(path))
        model = build_detector(cfg.model)
        n = sum(p.numel() for p in model.parameters())
        print(f"OK   {rel}")
        print(f"     -> {type(model).__name__}, {n / 1e6:.1f}M params")
    except Exception:
        print(f"FAIL {rel}")
        traceback.print_exc()
