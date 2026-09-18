"""Parity check: training data loaders with worker processes, vfe vs mmdet.

``parity_vid_train_data`` compares samplers and single samples; this runs the
real loaders, whose worker processes seed ``random`` and numpy per rank and
worker, so it also covers which worker augments which sample, and the replay of
those streams after workers restart each epoch.

mmdet side: ``build_dataloader`` exactly as ``train_detector`` called it, as
rank ``r`` of the original 8-GPU run (``get_dist_info`` patched). vfe side:
``build_train_loaders`` for a 4-process, 2-micro-step layout, taking the loader
of the process and micro-step that plays virtual rank ``r``. For ranks 0, 5
and 7 and epochs 0 and 1, the first 8 batches (two per worker) must be
identical: a digest of every image, ground-truth tensor and meta. The
training set is VID + DET; the DET images these batches use must be present
locally (a few, copied from the staged data).

Usage:
    conda run -n vfe --no-capture-output python tools/checks/parity_train_loader.py --impl mmdet --out A.pt
    python tools/checks/parity_train_loader.py --impl vfe --out B.pt
    python tools/checks/parity_train_loader.py --compare A.pt B.pt
"""

import argparse
import itertools
import sys
from pathlib import Path

import parity_train_step as ts
import parity_vid_pipeline as pp
import parity_vid_train_data as td
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = td.SEED
WORKERS = 4
RANKS = (0, 5, 7)
EPOCHS = (0, 1)
BATCHES = 8
PROCESSES, ACCUMULATE = 4, 2  # vfe layout: 4 processes x 2 micro-steps = 8 virtual ranks


def loader_for(impl, dataset, rank):
    if impl == "mmdet":
        from mmcv.parallel import DataContainer

        import mmdet.datasets.builder as builder

        builder.get_dist_info = lambda: (rank, PROCESSES * ACCUMULATE)
        loader = builder.build_dataloader(dataset, 1, WORKERS, num_gpus=1, dist=True, seed=SEED,
                                          runner_type="EpochBasedRunner",
                                          persistent_workers=False)

        def unwrap(batch):
            return {k: td.to_cpu(v.data[0] if isinstance(v, DataContainer) else v)
                    for k, v in batch.items()}
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.datasets.loader import build_train_loaders

        process, micro_step = divmod(rank, ACCUMULATE)
        loader = build_train_loaders(dataset, 1, WORKERS, world_size=PROCESSES, rank=process,
                                     accumulate=ACCUMULATE, seed=SEED)[micro_step]

        def unwrap(batch):
            return batch
    return loader, unwrap


def run(impl):
    dataset = td.build(impl)[0]
    out = {}
    for rank in RANKS:
        loader, unwrap = loader_for(impl, dataset, rank)
        for epoch in EPOCHS:
            loader.sampler.set_epoch(epoch)
            names = []
            for i, batch in enumerate(itertools.islice(loader, BATCHES)):
                batch = unwrap(batch)
                out[f"rank{rank}/epoch{epoch}/batch{i}"] = ts.batch_digest(batch)
                meta = batch["img_metas"][0]
                names.append(f"{Path(meta['ori_filename']).stem}{'(f)' if meta['flip'] else ''}")
            print(f"{impl} rank {rank} epoch {epoch}: {' '.join(names)}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        pp.compare(*args.compare, label="TRAIN LOADER")
    elif args.impl and args.out:
        torch.save(run(args.impl), args.out)
        print(f"saved -> {args.out}")
    else:
        ap.error("pass either --impl/--out or --compare")
