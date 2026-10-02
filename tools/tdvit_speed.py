"""Time TDViT and its Swin baseline on one GPU, with the frames already on it.

The paper's speed claim (Table 2, V100): TDViT-T runs slightly *faster* than
Swin-T in Faster R-CNN (23.9 vs 22.8 FPS), because a TDTB computes only the
queries of each frame and reuses a reference's keys and values for ``D_t``
frames. This runs the first ``--frames`` frames of the first ``--videos`` val
videos in order, one at a time (TDViT's memory is stateful), and reports per
config:

* detector time per frame (CUDA-synchronised; data loading excluded), the
  videos' first frames apart -- their memories are empty;
* the backbone's share, timed in a second pass over the same frames;
* the peak GPU memory.

    python tools/tdvit_speed.py configs/vid/tdvit/frcnn_swint_fpn_3x.py \\
        configs/vid/tdvit/tdvit_t_frcnn_fpn_3x.py [--ckpts A.pth B.pth] --out speed.json

A config may carry its own overrides, URL-style:
``configs/vid/tdvit/tdvit_t_frcnn_fpn_3x.py?model.detector.backbone.attention=joint``
(several joined by ``&``), applied after ``--cfg-options``.

Without checkpoints the models keep their initial weights: the backbone's
time does not depend on them, the RoI head's barely (300 proposals a frame).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time

import torch
from torch.utils.data import DataLoader, Subset

from vfe.config import Config, parse_cfg_options
from vfe.datasets import build_dataset, collate_video_test
from vfe.engine.evaluator import to_device
from vfe.models.builder import build_model
from vfe.models.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--ckpts", nargs="*", default=[], help="one per config, in order")
    ap.add_argument("--videos", type=int, default=5)
    ap.add_argument("--frames", type=int, default=100, help="per video")
    ap.add_argument("--warmup", type=int, default=20, help="untimed frames before each config")
    ap.add_argument("--cfg-options", nargs="+", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", help="write the results here as JSON")
    return ap.parse_args()


def load_frames(cfg, n_videos: int, n_frames: int, device) -> list[list[dict]]:
    """The first frames of the first videos, as model inputs on ``device``."""
    dataset = build_dataset(cfg.data.test)
    keep, seen = [], 0
    for i, info in enumerate(dataset.data_infos):
        seen += info["frame_id"] == 0
        if seen > n_videos:
            break
        if info["frame_id"] < n_frames:
            keep.append(i)
    loader = DataLoader(Subset(dataset, keep), batch_size=1, num_workers=4,
                        collate_fn=collate_video_test)
    videos: list[list[dict]] = []
    for batch in loader:
        if batch["img_metas"][0][0]["frame_id"] == 0:
            videos.append([])
        videos[-1].append(to_device(batch, device))
    return videos


def sync_time(fn) -> float:
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000.0


def time_model(model, videos, warmup: int) -> dict:
    flat = [frame for video in videos for frame in video]
    with torch.no_grad():
        for frame in flat[:warmup]:  # cuDNN / allocator warm-up, on a throwaway video
            model(return_loss=False, rescale=True, **frame)
        torch.cuda.reset_peak_memory_stats()
        detector_ms, first_ms = [], []
        for video in videos:
            for frame in video:
                ms = sync_time(lambda f=frame: model(return_loss=False, rescale=True, **f))
                (first_ms if frame["img_metas"][0][0]["frame_id"] == 0 else detector_ms).append(ms)
        peak = torch.cuda.max_memory_allocated() / 2**20
        backbone = model.detector.backbone
        backbone_ms = []
        for video in videos:
            if hasattr(backbone, "reset_memory"):
                backbone.reset_memory()
            for i, frame in enumerate(video):
                img = frame["img"][0]
                ms = sync_time(lambda img=img: backbone(img))
                if i:
                    backbone_ms.append(ms)
    return dict(
        frames=len(detector_ms), detector_ms=st.mean(detector_ms),
        detector_ms_median=st.median(detector_ms), fps=1000.0 / st.mean(detector_ms),
        first_frame_ms=st.mean(first_ms) if first_ms else None,
        backbone_ms=st.mean(backbone_ms), peak_memory_mib=peak)


def main() -> None:
    args = parse_args()
    if args.ckpts and len(args.ckpts) != len(args.configs):
        raise SystemExit("give one checkpoint per config, or none")
    device = torch.device("cuda")
    results = {"gpu": torch.cuda.get_device_name(device), "videos": args.videos,
               "frames_per_video": args.frames, "models": {}}
    videos = None
    for k, path in enumerate(args.configs):
        file, _, query = path.partition("?")
        cfg = Config.fromfile(file)
        cfg.merge_from_dict(parse_cfg_options(args.cfg_options))
        cfg.merge_from_dict(parse_cfg_options([o for o in query.split("&") if o]))
        if videos is None:
            videos = load_frames(cfg, args.videos, args.frames, device)
        cfg.model.detector.backbone.init_cfg = None  # timing needs no pretrained download
        model = build_model(cfg.model)
        if args.ckpts:
            load_checkpoint(model, args.ckpts[k], map_location="cpu")
        model = model.to(device).eval()
        row = time_model(model, videos, args.warmup)
        results["models"][path] = row
        print(f"{path}: {row['detector_ms']:.1f} ms/frame ({row['fps']:.1f} FPS), backbone "
              f"{row['backbone_ms']:.1f} ms, first frames {row['first_frame_ms']:.1f} ms, "
              f"peak {row['peak_memory_mib']:.0f} MiB", flush=True)
        del model
        torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
