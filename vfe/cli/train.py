"""Train a detector from a config. Replaces ``tools/train.py``.

Single process::

    python -m vfe.cli.train CONFIG --work-dir DIR --seed N

One node, several GPUs (``--accumulate k`` with ``n`` processes trains as
``n * k`` GPUs did; see :mod:`vfe.apis.train`)::

    torchrun --standalone --nproc_per_node=4 -m vfe.cli.train CONFIG \\
        --launcher pytorch --accumulate 2 --work-dir DIR --seed N

``--resume-from auto`` resumes from ``WORK_DIR/latest.pth`` when it exists, so
a chain of identical jobs continues a run. Setup follows mmdet's script, in
its order (it decides what the random streams produce): process group, logger
and environment, seed, model and ``init_weights``, datasets, training.

TF32 matrix multiplications and convolutions are on by default: the released
models trained with PyTorch 1.10 on A100s, where both were the default
(PyTorch 1.12 turned the matmul default off). ``--no-tf32`` disables both.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from vfe.apis.train import train_detector
from vfe.config import Config, parse_cfg_options
from vfe.datasets import build_dataset
from vfe.engine.train_log import collect_env, get_logger
from vfe.models.builder import build_model
from vfe.utils import get_dist_info, init_dist


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--work-dir", help="default: work_dirs/<config name>")
    ap.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    ap.add_argument("--seed", type=int, help="default: random, shared by all ranks")
    ap.add_argument("--accumulate", type=int, default=1,
                    help="micro-steps per iteration; n processes emulate n * k GPUs")
    ap.add_argument("--resume-from", help="training checkpoint, or 'auto' for WORK_DIR/latest.pth")
    ap.add_argument("--load-from", help="initialise from these weights (no optimiser state)")
    ap.add_argument("--no-validate", action="store_true", help="skip evaluation during training")
    ap.add_argument("--workers", type=int, help="override data.workers_per_gpu")
    ap.add_argument("--max-epochs", type=int, help="override runner.max_epochs")
    ap.add_argument("--max-iters-per-epoch", type=int, help="truncate epochs (short test runs)")
    ap.add_argument("--device", choices=["cuda", "cpu"],
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-tf32", action="store_true", help="disable TF32 (see module docstring)")
    ap.add_argument("--cfg-options", nargs="+", default=[], metavar="KEY=VALUE",
                    help="override config entries, e.g. data.train.0.ann_file=PATH")
    return ap.parse_args()


def init_random_seed(seed: int | None, device: torch.device) -> int:
    """``seed``, or a random one drawn on rank 0 and shared by every rank."""
    if seed is not None:
        return seed
    seed = int(np.random.randint(2**31))
    _, world_size = get_dist_info()
    if world_size == 1:
        return seed
    holder = torch.tensor(seed, dtype=torch.int64, device=device)
    dist.broadcast(holder, src=0)
    return int(holder.item())


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(parse_cfg_options(args.cfg_options))
    if args.workers is not None:
        cfg.data.workers_per_gpu = args.workers

    distributed = args.launcher != "none"
    if distributed:
        init_dist(backend="nccl" if args.device == "cuda" else "gloo")
    rank, world_size = get_dist_info()
    device = (torch.device("cuda", torch.cuda.current_device()) if args.device == "cuda"
              else torch.device("cpu"))
    tf32 = not args.no_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    work_dir = args.work_dir or osp.join("work_dirs", osp.splitext(osp.basename(args.config))[0])
    os.makedirs(work_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logger = get_logger(osp.join(work_dir, f"{timestamp}.log"), rank)
    env_info = collect_env()
    logger.info("Environment info:\n%s\n%s\n%s", "-" * 60, env_info, "-" * 60)
    config_text = json.dumps(cfg.to_dict(), indent=1, default=repr)
    if rank == 0:
        with open(osp.join(work_dir, osp.basename(args.config) + ".json"), "w") as f:
            f.write(config_text)
    logger.info("Distributed: %s; %d process(es), %d micro-step(s) each: trains as %d GPU(s)",
                distributed, world_size, args.accumulate, world_size * args.accumulate)
    logger.info("Config:\n%s", config_text)

    seed = init_random_seed(args.seed, device)
    logger.info("Set random seed to %d, deterministic: False", seed)
    set_random_seed(seed)

    model = build_model(cfg.model)
    model.init_weights()
    dataset = build_dataset(cfg.data.train)
    model.CLASSES = dataset.CLASSES

    resume_from = args.resume_from or cfg.get("resume_from")
    if resume_from == "auto":
        latest = osp.join(work_dir, "latest.pth")
        resume_from = latest if osp.exists(latest) else None
    meta = dict(env_info=env_info, config=config_text, seed=seed,
                exp_name=osp.basename(args.config))

    train_detector(model, dataset, cfg, work_dir=work_dir, timestamp=timestamp, meta=meta,
                   logger=logger, device=device, seed=seed, distributed=distributed,
                   validate=not args.no_validate, accumulate=args.accumulate,
                   resume_from=resume_from, load_from=args.load_from or cfg.get("load_from"),
                   max_epochs=args.max_epochs, max_iters_per_epoch=args.max_iters_per_epoch)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
