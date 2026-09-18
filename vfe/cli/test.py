"""Evaluate a checkpoint on a config's test set. Replaces ``tools/test.py``.

Single GPU:
    python -m vfe.cli.test CONFIG CHECKPOINT --work-dir DIR
One node, several GPUs:
    torchrun --standalone --nproc_per_node=4 -m vfe.cli.test CONFIG CHECKPOINT \\
        --launcher pytorch --work-dir DIR
Smoke test on the first videos only (no metric; the metric needs every frame):
    python -m vfe.cli.test CONFIG CHECKPOINT --work-dir DIR --max-videos 2

Writes ``DIR/eval_<timestamp>.json`` (metrics) and, with ``--out``, the raw
per-frame detections as a pickle, which ``--eval-only`` can re-score later.

Evaluation of MAMBA is randomised by its memory sampling (torch's CPU RNG),
and the original never seeded it. ``--seed`` makes a run repeatable; leave it
unset to match the original protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import time

import torch
from torch.utils.data import DataLoader, Subset

from vfe.apis import evaluation_kwargs, make_tmpdir, multi_gpu_test, single_gpu_test
from vfe.config import Config
from vfe.datasets import build_dataset, collate_video_test
from vfe.datasets.samplers import DistributedVideoSampler
from vfe.models.builder import build_model
from vfe.models.checkpoint import load_checkpoint
from vfe.utils import get_dist_info, init_dist


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("checkpoint", nargs="?", help="omit with --eval-only")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    ap.add_argument("--out", help="also save raw detections to this pickle")
    ap.add_argument("--eval-only", metavar="PKL", help="score saved detections; no inference")
    ap.add_argument("--max-videos", type=int, help="run only the first N videos (no metric)")
    ap.add_argument("--workers", type=int, help="override data.workers_per_gpu")
    ap.add_argument("--seed", type=int, help="seed torch's CPU RNG (memory sampling)")
    return ap.parse_args()


def first_n_videos(dataset, n: int) -> Subset:
    starts = [i for i, info in enumerate(dataset.data_infos) if info["frame_id"] == 0]
    end = starts[n] if n < len(starts) else len(dataset)
    return Subset(dataset, range(end))


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if not cfg.get("is_video_model", False):
        raise NotImplementedError("only video models (is_video_model=True) are wired up")

    distributed = args.launcher != "none"
    if distributed:
        init_dist()
    rank, world_size = get_dist_info()
    os.makedirs(args.work_dir, exist_ok=True)
    dataset = build_dataset(cfg.data.test)

    if args.eval_only:
        with open(args.eval_only, "rb") as f:
            outputs = pickle.load(f)
    else:
        if args.checkpoint is None:
            raise SystemExit("a checkpoint is required unless --eval-only is given")
        if args.seed is not None:
            torch.manual_seed(args.seed)
        test_set = first_n_videos(dataset, args.max_videos) if args.max_videos else dataset
        if distributed and args.max_videos:
            raise SystemExit("--max-videos is a single-process smoke test")
        sampler = DistributedVideoSampler(dataset, world_size, rank) if distributed else None
        loader = DataLoader(
            test_set,
            batch_size=1,
            sampler=sampler,
            shuffle=False,
            num_workers=args.workers if args.workers is not None else cfg.data.workers_per_gpu,
            collate_fn=collate_video_test,
        )

        device = (torch.device("cuda", torch.cuda.current_device())
                  if torch.cuda.is_available() else torch.device("cpu"))
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        model = model.to(device)

        if distributed:
            outputs = multi_gpu_test(model, loader, device, make_tmpdir(args.work_dir))
        else:
            outputs = single_gpu_test(model, loader, device)

    if rank != 0:
        return
    if args.out:
        with open(args.out, "wb") as f:
            pickle.dump(outputs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"detections -> {args.out}")
    if args.max_videos:
        n = sum(sum(len(cls) for cls in frame) for frame in outputs)
        print(f"smoke test: {len(outputs)} frames, {n} detections; no metric on a subset")
        return

    eval_kwargs = evaluation_kwargs(cfg.get("evaluation", {}))
    metric = {k: float(v) for k, v in dataset.evaluate(outputs, **eval_kwargs).items()}
    print(json.dumps(metric, indent=1))
    path = osp.join(args.work_dir, f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(dict(config=args.config, checkpoint=args.checkpoint, metric=metric), f,
                  indent=1)
    print(f"metrics -> {path}")


if __name__ == "__main__":
    main()
