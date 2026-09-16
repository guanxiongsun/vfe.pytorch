"""Parity check: ``vfe.ops`` vs the compiled ``mmcv.ops`` oracle.

Same two-process pattern as ``parity_config.py`` -- mmcv only runs in the py3.8
``vfe`` env -- but here the payload is tensors, so each side saves a ``.pt`` of
results computed from a *seeded* input generator and we compare numerically.

Usage:
    conda run -n vfe       --no-capture-output python tools/checks/parity_ops.py --impl mmcv --out /tmp/ops_mmcv.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_ops.py --impl vfe  --out /tmp/ops_vfe.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_ops.py --compare /tmp/ops_mmcv.pt /tmp/ops_vfe.pt

Add ``--device cuda`` to compare the CUDA kernels instead of the CPU ones.
Inputs are built on CPU from a fixed seed and only then moved, so both envs see
bit-identical data despite different torch versions.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- inputs: deterministic, version-independent -------------------------------
def make_boxes(n, seed, spread=200.0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * spread
    wh = 5.0 + torch.rand(n, 2, generator=g) * 40.0
    return torch.cat([xy, xy + wh], dim=1)


def make_scores(n, seed):
    g = torch.Generator().manual_seed(seed + 1)
    return torch.rand(n, generator=g)


def make_feat(shape, seed):
    g = torch.Generator().manual_seed(seed + 2)
    return torch.randn(*shape, generator=g)


CASES = [
    # (name, kind, kwargs)
    ("nms/basic", "nms", dict(n=200, seed=0, iou_threshold=0.5)),
    ("nms/tight", "nms", dict(n=200, seed=0, iou_threshold=0.1)),
    ("nms/loose", "nms", dict(n=200, seed=1, iou_threshold=0.9)),
    ("nms/rpn_0.7", "nms", dict(n=2000, seed=2, iou_threshold=0.7)),
    ("nms/dense", "nms", dict(n=500, seed=3, iou_threshold=0.5, spread=40.0)),
    ("nms/offset1", "nms", dict(n=200, seed=0, iou_threshold=0.5, offset=1)),
    ("nms/score_thr", "nms", dict(n=300, seed=4, iou_threshold=0.5, score_threshold=0.5)),
    ("nms/max_num", "nms", dict(n=300, seed=4, iou_threshold=0.5, max_num=20)),
    ("batched_nms/30cls", "batched_nms", dict(n=1000, seed=5, num_classes=30, iou_threshold=0.5)),
    ("batched_nms/agnostic", "batched_nms", dict(n=500, seed=6, num_classes=30, iou_threshold=0.5, class_agnostic=True)),
    ("batched_nms/split", "batched_nms", dict(n=1200, seed=7, num_classes=30, iou_threshold=0.5, split_thr=500)),
    ("roi_align/7x7_sr2", "roi_align", dict(seed=8, output_size=7, sampling_ratio=2, spatial_scale=1 / 16)),
    ("roi_align/7x7_sr0", "roi_align", dict(seed=9, output_size=7, sampling_ratio=0, spatial_scale=1 / 16)),
    ("roi_align/unaligned", "roi_align", dict(seed=8, output_size=7, sampling_ratio=2, spatial_scale=1 / 16, aligned=False)),
    ("roi_align/scale1", "roi_align", dict(seed=10, output_size=14, sampling_ratio=2, spatial_scale=1.0)),
]


def run(impl, device):
    if impl == "mmcv":
        from mmcv.ops import RoIAlign, batched_nms, nms
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.ops import RoIAlign, batched_nms, nms

    results = {}
    for name, kind, kw in CASES:
        kw = dict(kw)
        if kind == "nms":
            boxes = make_boxes(kw.pop("n"), kw["seed"], kw.pop("spread", 200.0)).to(device)
            scores = make_scores(boxes.shape[0], kw.pop("seed")).to(device)
            dets, inds = nms(boxes, scores, **kw)
            results[name] = {"dets": dets.cpu(), "inds": inds.cpu().long()}

        elif kind == "batched_nms":
            n, seed = kw.pop("n"), kw.pop("seed")
            num_classes = kw.pop("num_classes")
            class_agnostic = kw.pop("class_agnostic", False)
            boxes = make_boxes(n, seed).to(device)
            scores = make_scores(n, seed).to(device)
            g = torch.Generator().manual_seed(seed + 3)
            idxs = torch.randint(0, num_classes, (n,), generator=g).to(device)
            dets, keep = batched_nms(boxes, scores, idxs, dict(type="nms", **kw), class_agnostic)
            results[name] = {"dets": dets.cpu(), "inds": keep.cpu().long()}

        elif kind == "roi_align":
            seed = kw.pop("seed")
            feat = make_feat((2, 8, 38, 50), seed).to(device)
            rois = make_boxes(64, seed + 5, spread=600.0)
            batch_idx = (torch.arange(64) % 2).float().unsqueeze(1)
            rois = torch.cat([batch_idx, rois], dim=1).to(device)
            layer = RoIAlign(**kw).to(device)
            results[name] = {"out": layer(feat, rois).cpu()}

        print(f"RAN   {name}")
    return results


def compare(path_a, path_b, atol, rtol):
    a = torch.load(path_a, map_location="cpu", weights_only=True)
    b = torch.load(path_b, map_location="cpu", weights_only=True)

    failures = 0
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            failures += 1
            print(f"FAIL  {name}: only in {'A' if name in a else 'B'}")
            continue

        notes = []
        ok = True
        for key in sorted(set(a[name]) | set(b[name])):
            ta, tb = a[name][key], b[name][key]
            if ta.shape != tb.shape:
                ok = False
                notes.append(f"{key}: shape {tuple(ta.shape)} != {tuple(tb.shape)}")
                continue
            if ta.dtype.is_floating_point:
                if not torch.allclose(ta, tb, atol=atol, rtol=rtol):
                    ok = False
                    diff = (ta - tb).abs().max().item()
                    notes.append(f"{key}: max|diff| = {diff:.3e}")
            elif not torch.equal(ta, tb):
                ok = False
                n_diff = (ta != tb).sum().item()
                notes.append(f"{key}: {n_diff}/{ta.numel()} elements differ")

        if ok:
            print(f"OK    {name}")
        else:
            failures += 1
            print(f"FAIL  {name}")
            for note in notes:
                print(f"        {note}")

    print("-" * 70)
    print("OPS PARITY OK" if failures == 0 else f"{failures} op case(s) differ")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mmcv", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-5)
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare, atol=args.atol, rtol=args.rtol)
    elif args.impl and args.out:
        torch.save(run(args.impl, args.device), args.out)
        print(f"saved -> {args.out}")
    else:
        ap.error("pass either --impl/--out or --compare")
