"""Parity check: YOLOX's training pipeline (``MultiImageMixDataset`` with
Mosaic, RandomAffine, MixUp, HSV, flip, resize, square pad, the annotation
filter and the format bundle) in ``vfe`` vs mmdet 2.19.1.

A synthetic dataset of random images and boxes (numpy only, identical on both
sides) feeds both stacks; each sample runs after seeding numpy (which every
transform draws from, as in mmdet) with its index, so equal code gives equal
outputs.

    PYTHONPATH=/path/to/v1.0.0 legacy38/bin/python tools/checks/parity_yolox_pipeline.py --impl mmdet --out pipe_mmdet.pt
    python tools/checks/parity_yolox_pipeline.py --impl vfe --out pipe_vfe.pt
    python tools/checks/parity_yolox_pipeline.py --compare pipe_mmdet.pt pipe_vfe.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_SCALE = (640, 640)
PIPELINE = [
    dict(type="Mosaic", img_scale=IMG_SCALE, pad_val=114.0),
    dict(type="RandomAffine", scaling_ratio_range=(0.1, 2),
         border=(-IMG_SCALE[0] // 2, -IMG_SCALE[1] // 2)),
    dict(type="MixUp", img_scale=IMG_SCALE, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Resize", img_scale=IMG_SCALE, keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
NUM_SAMPLES = 24


class SyntheticDataset:
    """Random images of assorted sizes with 0-4 random boxes each: what
    ``LoadImageFromFile`` + ``LoadAnnotations`` would hand the mixer."""

    CLASSES = tuple(f"c{i}" for i in range(30))

    def __init__(self, n=10):
        self.n = n
        self.flag = np.zeros(n, dtype=np.uint8)

    def __len__(self):
        return self.n

    def _item(self, idx):
        rng = np.random.RandomState(1000 + idx)
        h, w = int(rng.randint(200, 700)), int(rng.randint(200, 900))
        img = rng.randint(0, 256, (h, w, 3)).astype(np.uint8)
        k = int(rng.randint(0, 5))
        xy = rng.uniform(0, 1, (k, 2)) * [w * 0.7, h * 0.7]
        wh = rng.uniform(0.05, 0.3, (k, 2)) * [w, h]
        bboxes = np.concatenate([xy, xy + wh], 1).astype(np.float32).reshape(-1, 4)
        labels = rng.randint(0, 30, k).astype(np.int64)
        return img, bboxes, labels

    def get_ann_info(self, idx):
        _, bboxes, labels = self._item(idx)
        return dict(bboxes=bboxes, labels=labels)

    def __getitem__(self, idx):
        img, bboxes, labels = self._item(idx)
        return dict(img=img, img_shape=img.shape, ori_shape=img.shape, img_fields=["img"],
                    bbox_fields=["gt_bboxes"], gt_bboxes=bboxes, gt_labels=labels,
                    filename=f"img{idx}.jpg", ori_filename=f"img{idx}.jpg",
                    img_info=dict(filename=f"img{idx}.jpg"))


def build(impl):
    dataset = SyntheticDataset()
    if impl == "mmdet":
        from mmdet.datasets import MultiImageMixDataset

        return MultiImageMixDataset(dataset, PIPELINE)
    sys.path.insert(0, str(REPO_ROOT))
    import vfe.datasets.pipelines  # noqa: F401  (registers the transforms)
    from vfe.datasets import MultiImageMixDataset

    return MultiImageMixDataset(dataset, PIPELINE)


def unwrap(value):
    return value.data if hasattr(value, "data") and not torch.is_tensor(value) else value


def run(impl):
    mixed = build(impl)
    out = {}
    for i in range(NUM_SAMPLES):
        np.random.seed(i)
        sample = mixed[i % len(mixed)]
        out[f"sample{i}/img"] = unwrap(sample["img"])
        out[f"sample{i}/gt_bboxes"] = unwrap(sample["gt_bboxes"])
        out[f"sample{i}/gt_labels"] = unwrap(sample["gt_labels"])
        meta = unwrap(sample["img_metas"])
        out[f"sample{i}/scale_factor"] = torch.as_tensor(np.asarray(meta["scale_factor"]))
        out[f"sample{i}/flip"] = torch.tensor(bool(meta["flip"]))
    # The last epochs: no Mosaic, RandomAffine or MixUp.
    mixed.update_skip_type_keys(("Mosaic", "RandomAffine", "MixUp"))
    for i in range(4):
        np.random.seed(100 + i)
        sample = mixed[i]
        out[f"plain{i}/img"] = unwrap(sample["img"])
        out[f"plain{i}/gt_bboxes"] = unwrap(sample["gt_bboxes"])
    return out


def compare(path_a, path_b, atol):
    a = torch.load(path_a, weights_only=False)
    b = torch.load(path_b, weights_only=False)
    if set(a) != set(b):
        print("artifact key sets differ:", sorted(set(a) ^ set(b)))
        return False
    ok, exact, worst = True, 0, 0.0
    for key in sorted(a):
        va, vb = a[key], b[key]
        if va.shape != vb.shape:
            print(f"FAIL {key}: shape {tuple(va.shape)} vs {tuple(vb.shape)}")
            ok = False
            continue
        if torch.equal(va, vb):
            exact += 1
            continue
        diff = (va.double() - vb.double()).abs().max().item()
        worst = max(worst, diff)
        if diff > atol:
            print(f"FAIL {key}: max |a-b| = {diff:.3e}")
            ok = False
    print(f"{len(a)} artifacts, {exact} bit-exact, worst difference {worst:.3e}")
    print("YOLOX PIPELINE PARITY OK" if ok else "YOLOX PIPELINE PARITY FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--atol", type=float, default=1e-3,
                    help="image values are 0-255; boxes are pixels")
    args = ap.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare, args.atol) else 1)
    if not args.impl or not args.out:
        ap.error("--impl and --out are required unless --compare is given")
    out = run(args.impl)
    torch.save(out, args.out)
    print(f"{args.impl}: {len(out)} artifacts -> {args.out}")


if __name__ == "__main__":
    main()
