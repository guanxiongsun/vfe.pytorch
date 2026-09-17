"""Parity check: end-to-end video inference on real val frames, vfe vs mmdet.

Runs a released checkpoint (MAMBA by default; ``--config`` selects another
model, e.g. STPN) over the first ``--videos`` val videos in
test order on each stack (mmdet: ``MMDataParallel`` + mmcv collate; vfe:
``vfe.apis.single_gpu_test`` + ``collate_video_test``) and compares every
frame's detections as a set (``parity_detector.match_detections``). This is
the first check with real images *and* trained weights, and the memory state
carries across hundreds of frames.

TF32 is disabled on both sides, and torch's CPU RNG (memory sampling) is
seeded identically.

Expect small differences to grow along a video. cuDNN differs by ~1e-6 across
versions, near-tied proposals can swap order, and the memory keeps each
frame's top-k rows by position, so later frames may read slightly different
references. ``--rtol`` is therefore looser than the unit harnesses. A port bug
in the loop or collation would show up at the first frames, not as drift.

Only detections scoring >= ``--min-score`` (1e-3) are compared. The config
keeps anything above 1e-4, and on the first val video a third of all
detections sit within 2x of that threshold, where float noise decides whether
they survive. Unfiltered, 462 of 464 frames matched; the other two differed
only by one detection scoring exactly 0.000100 (the threshold) and one NMS
pick between near-duplicates scoring 0.00027. The confident detections
(~0.9997) were identical. Neither affects AP50 measurably.

Usage (GPU):
    conda run -n vfe       --no-capture-output python tools/checks/parity_vid_test.py --impl mmdet --ckpt CKPT --out A.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_vid_test.py --impl vfe   --ckpt CKPT --out B.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_vid_test.py --compare A.pt B.pt
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import parity_detector as pd
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/vid/mamba/mamba_r101_dc5_6x.py"


def run(impl, config, ckpt, videos, min_score):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    if impl == "mmdet":
        from mmcv import Config
        from mmcv.parallel import MMDataParallel, collate
        from mmcv.runner import load_checkpoint

        from mmdet.datasets import build_dataset
        from mmdet.models import build_model

        cfg = Config.fromfile(str(config))
        dataset = build_dataset(cfg.data.test)
        collate_fn = partial(collate, samples_per_gpu=1)
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.config import Config
        from vfe.datasets import build_dataset, collate_video_test
        from vfe.models.builder import build_model
        from vfe.models.checkpoint import load_checkpoint

        cfg = Config.fromfile(str(config))
        dataset = build_dataset(cfg.data.test)
        collate_fn = collate_video_test

    starts = [i for i, info in enumerate(dataset.data_infos) if info["frame_id"] == 0]
    end = starts[videos] if videos < len(starts) else len(dataset)
    loader = DataLoader(Subset(dataset, range(end)), batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)

    model = build_model(cfg.model)
    load_checkpoint(model, ckpt, map_location="cpu")
    torch.manual_seed(0)
    start = time.time()
    if impl == "mmdet":
        model = MMDataParallel(model.cuda(), device_ids=[0])
        model.eval()
        results = []
        for data in loader:
            with torch.no_grad():
                results.extend(model(return_loss=False, rescale=True, **data))
    else:
        from vfe.apis import single_gpu_test

        results = single_gpu_test(model.cuda(), loader, torch.device("cuda"))
    print(f"{impl}: {len(results)} frames in {time.time() - start:.0f}s")

    out = {"frames": torch.tensor(len(results))}
    for i, res in enumerate(results):
        pd.put_result(out, f"frame{i:05d}", [cls[cls[:, 4] >= min_score] for cls in res])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--ckpt")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--videos", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--min-score", type=float, default=1e-3,
                    help="ignore detections below this score; see the module docstring")
    args = ap.parse_args()
    if args.compare:
        pd.compare(*args.compare, atol=1e-6, rtol=args.rtol, rtol_overrides={},
                   label="VID TEST")
    elif args.impl and args.out and args.ckpt:
        torch.save(run(args.impl, args.config, args.ckpt, args.videos, args.min_score), args.out)
        print(f"saved -> {args.out}")
    else:
        ap.error("pass --impl/--ckpt/--out, or --compare")
