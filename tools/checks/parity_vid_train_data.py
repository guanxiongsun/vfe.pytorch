"""Parity check: the MAMBA training data path, vfe vs mmdet.

Builds the training set from the MAMBA config (ImageNet VID train + the DET
30-class subset, concatenated) on each side and compares:

* ``data/*`` -- sizes and image ids of both datasets after training-mode
  filtering, and the concatenated aspect-ratio flags.
* ``sampler/*`` -- ``DistributedGroupSampler`` indices for two ranks and two
  epochs of an 8-GPU run, and its length, which must also be 13,711: the
  iterations per epoch in the original training logs, so the filtered training
  set has the original size. Also ``GroupSampler`` (one process) under a seeded
  numpy RNG.
* ``sample<idx>/*`` -- whole training samples as ``forward_train`` receives
  them (mmdet: ``collate`` then unwrap the single device chunk that scatter
  would select): structure, metas, ground truth, and
  a digest of every image tensor. Python's ``random`` (reference sampling) and
  numpy's (flips) are seeded identically before each sample. VID samples only
  by default: DET images are not on this machine (``--det`` once they are).

The legacy DET loader (``load_image_anns``) uses ``pycocotools.coco.COCO``
directly and calls the snake_case aliases (``get_cat_ids``, ``cat_img_map``, ...)
that the original env's ``mmpycocotools`` had. This env has official pycocotools,
so ``patch_legacy_det_coco`` points that loader at mmdet's own alias subclass,
the one the VID loader already uses.

Usage:
    conda run -n vfe       --no-capture-output python tools/checks/parity_vid_train_data.py --impl mmdet --out A.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_vid_train_data.py --impl vfe   --out B.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_vid_train_data.py --compare A.pt B.pt
"""

import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import parity_vid_pipeline as pp
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/vid/mamba/mamba_r101_dc5_6x.py"
SEED = 1466607766  # the seed in the original MAMBA training log
WORLD = 8
EXPECTED_ITERS_PER_EPOCH = 13711


def patch_legacy_det_coco():
    """Point the legacy DET loader at ``mmdet.datasets.api_wrappers.COCO``."""
    import mmdet.datasets.imagenet_vid_dataset as legacy
    from mmdet.datasets.api_wrappers import COCO

    legacy.COCO = COCO


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, list):
        return [to_cpu(x) for x in obj]
    return obj


def build(impl, select=None, config=CONFIG):
    """``select`` picks the dataset config out of ``data.train`` (default: all)."""
    if impl == "mmdet":
        patch_legacy_det_coco()
        from mmcv import Config
        from mmcv.parallel import DataContainer, collate

        from mmdet.datasets import build_dataset
        from mmdet.datasets.samplers import DistributedGroupSampler, GroupSampler

        train = Config.fromfile(str(config)).data.train
        dataset = build_dataset(select(train) if select else train)

        def batch_of(idx):
            # collate puts one chunk per target device in DataContainer.data.
            # With samples_per_gpu=1 there is exactly one chunk, and selecting
            # it reproduces single-GPU scatter's shapes without requiring CUDA.
            # mmcv's scatter-to-CPU path is not equivalent: it unsqueezes every
            # tensor, producing shapes training never sees.
            batch = collate([dataset[idx]], samples_per_gpu=1)
            return {
                key: to_cpu(value.data[0] if isinstance(value, DataContainer) else value)
                for key, value in batch.items()
            }
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.config import Config
        from vfe.datasets import build_dataset, collate_video_train
        from vfe.datasets.samplers import DistributedGroupSampler, GroupSampler

        train = Config.fromfile(str(config)).data.train
        dataset = build_dataset(select(train) if select else train)

        def batch_of(idx):
            return collate_video_train([dataset[idx]])
    return dataset, batch_of, GroupSampler, DistributedGroupSampler


def border_frames(vid):
    """The first downscaled VID frames (larger than 1000x600) with a box
    reaching the right / bottom edge. VID boxes end at most at size - 1; after
    downscaling they end within a pixel of the edge, so an off-by-one in
    clipping or flipping shows."""
    picks = []
    for column, size in ((2, "width"), (3, "height")):
        for idx, info in enumerate(vid.data_infos):
            w, h = info["width"], info["height"]
            if max(w, h) <= 1000 and min(w, h) <= 600:
                continue
            bboxes = vid.get_ann_info(idx)["bboxes"]
            if len(bboxes) and (bboxes[:, column] >= info[size] - 1).any():
                picks.append(idx)
                break
    return picks


def run(impl, det):
    dataset, batch_of, group_sampler, dist_sampler = build(impl)
    vid, det_ds = dataset.datasets
    out = {
        "data/vid/len": torch.tensor(len(vid)),
        "data/det/len": torch.tensor(len(det_ds)),
        "data/vid/img_ids": torch.tensor(vid.img_ids, dtype=torch.int64),
        "data/det/img_ids": torch.tensor(det_ds.img_ids, dtype=torch.int64),
        "data/flag": torch.from_numpy(dataset.flag.astype(np.int64)),
    }
    for rank in (0, WORLD - 1):
        sampler = dist_sampler(dataset, 1, WORLD, rank, seed=SEED)
        out["sampler/len"] = torch.tensor(len(sampler))
        for epoch in (0, 3):
            sampler.set_epoch(epoch)
            out[f"sampler/rank{rank}/epoch{epoch}"] = torch.tensor(list(sampler), dtype=torch.int64)
    np.random.seed(SEED)
    out["sampler/group"] = torch.tensor(list(group_sampler(dataset, 1)), dtype=torch.int64)
    print(f"{impl}: VID {len(vid)} + DET {len(det_ds)} images; "
          f"sampler length {int(out['sampler/len'])} per rank "
          f"(original log: {EXPECTED_ITERS_PER_EPOCH})")

    picks = [0, 1, 5000, 20000, 40000, len(vid) - 1, *border_frames(vid)]
    if det:
        picks += [len(vid), len(vid) + 1000, len(dataset) - 1]
    for idx in picks:
        random.seed(1000 + idx)
        np.random.seed(1000 + idx)
        batch = batch_of(idx)
        tag = f"sample{idx}"
        out[f"{tag}/structure"] = pp.encode_str(
            ";".join(f"{k}={pp.signature(v)}" for k, v in sorted(batch.items())))
        out[f"{tag}/metas"] = pp.encode_str(pp.canonical(
            {k: v for k, v in batch.items() if k.endswith("img_metas")}))
        for key, value in sorted(batch.items()):
            if key in ("img", "ref_img"):
                out[f"{tag}/digest/{key}"] = pp.encode_str(
                    hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest())
            elif not key.endswith("img_metas"):
                for i, t in enumerate(value):
                    out[f"{tag}/{key}{i}"] = t
        flip = batch["img_metas"][0]["flip"]
        print(f"  {tag}: {batch['img_metas'][0]['ori_filename']}, flip={flip}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--det", action="store_true", help="also compare DET samples (needs images)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        pp.compare(*args.compare, label="VID TRAIN DATA",
                   expect={"sampler/len": EXPECTED_ITERS_PER_EPOCH})
    elif args.impl and args.out:
        torch.save(run(args.impl, args.det), args.out)
        print(f"saved -> {args.out}")
    else:
        ap.error("pass either --impl/--out or --compare")
