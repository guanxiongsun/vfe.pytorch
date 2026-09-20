"""Parity check: ImageNet VID val annotations and the VID metric, vfe vs mmdet.

Builds the val dataset from the MAMBA test config on each side and compares:

* ``order/img_ids`` -- frame order after the first-frame-fixed shuffle. This is
  MAMBA's test-time processing order, so it must match exactly.
* ``gt/*`` -- boxes, labels, crowd boxes and instance ids of all 176,126 frames.
* ``motion/*`` -- the motion-IoU table as each side loads it (mmdet: the .mat via
  scipy; vfe: the converted .npz), per frame, placeholder zeros included.
* ``eval/*`` -- AP50 overall, fast / medium / slow and per class, for identical
  synthetic detections: ground-truth boxes jittered, some missed, some
  mislabelled, some duplicated, plus false positives. Scores are distinct
  (rank / (N + 1), exact in float32 for N below ~16M): the metric sorts by
  score, and tied scores could sort differently across numpy versions.

Everything is compared exactly except ``eval/values``, which may differ by
1e-12 relative. The first full run matched frame order, all ground truth, the
motion table and the four headline metrics bit for bit; three per-class APs
differed by one float64 ULP (~1e-16), from ``np.sum``'s summation order
changing between numpy 1.23 and 2.5.

Needs the val annotations at data/ILSVRC/annotations/imagenet_vid_val.json;
no images are read.

Usage:
    conda run -n vfe       --no-capture-output python tools/checks/parity_vid_eval.py --impl mmdet --out A.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_vid_eval.py --impl vfe   --out B.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_vid_eval.py --compare A.pt B.pt
"""

import argparse
import os
import sys
import time

import _legacy
import numpy as np
import torch

REPO_ROOT = _legacy.REPO_ROOT
CONFIG = REPO_ROOT / "configs/vid/mamba/mamba_r101_dc5_6x.py"
# FGFA's table, read from the legacy tree (see tools/checks/_legacy.py).
MOTION_IOU_MAT = "mmdet/datasets/mamba/vid_groundtruth_motion_iou.mat"
NUM_CLASSES = 30


def encode_str(s):
    return torch.tensor(list(s.encode()), dtype=torch.uint8)


def patch_legacy_motion_iou():
    """Point mmdet's VID evaluator at the legacy tree's motion-IoU table.

    ``eval_detection_vid`` loads it from the *relative* path
    "mmdet/datasets/mamba/vid_groundtruth_motion_iou.mat", which resolved only
    while the working directory happened to contain the mmdet tree. It is a
    local variable, not an argument, so the redirect goes through the one
    ``scipy.io`` call that reads it.
    """
    from mmdet.datasets.mamba import vid_eval as legacy

    real_loadmat = legacy.sio.loadmat

    class RedirectedScipyIO:
        @staticmethod
        def loadmat(path, *args, **kwargs):
            if not os.path.isabs(str(path)) and str(path).endswith(MOTION_IOU_MAT):
                path = str(_legacy.legacy_file(MOTION_IOU_MAT))
            return real_loadmat(path, *args, **kwargs)

    legacy.sio = RedirectedScipyIO


def build_dataset(impl):
    if impl == "mmdet":
        from mmcv import Config
        from mmdet.datasets import build_dataset as mm_build

        patch_legacy_motion_iou()
        cfg = Config.fromfile(str(CONFIG))
        return mm_build(cfg.data.test)

    sys.path.insert(0, str(REPO_ROOT))
    from vfe.config import Config
    from vfe.datasets import ImagenetVIDDataset

    test = Config.fromfile(str(CONFIG)).data.test
    return ImagenetVIDDataset(
        ann_file=test.ann_file,
        img_prefix=test.img_prefix,
        test_mode=test.test_mode,
        shuffle_video_frames=test.shuffle_video_frames,
    )


def load_motion(impl):
    """Per-frame motion IoUs as (values, lengths), however each side loads them."""
    if impl == "mmdet":
        import scipy.io as sio

        m = sio.loadmat(str(_legacy.legacy_file(MOTION_IOU_MAT)))
        # The construction in mmdet/datasets/mamba/vid_eval.py, minus np.array.
        frames = [
            [
                m["motion_iou"][i][0][j][0] if len(m["motion_iou"][i][0][j]) != 0 else 0
                for j in range(len(m["motion_iou"][i][0]))
            ]
            for i in range(len(m["motion_iou"]))
        ]
        values = np.array([float(v) for f in frames for v in f], dtype=np.float64)
        lengths = np.array([len(f) for f in frames], dtype=np.int64)
        return values, lengths

    from vfe.evaluation import load_motion_ious

    values, offsets = load_motion_ious()
    return values, np.diff(offsets)


def make_predictions(dataset, seed=2024):
    """Detections derived from each frame's ground truth; identical on both
    sides because RandomState's streams are frozen across numpy versions."""
    rs = np.random.RandomState(seed)
    raw = []
    for idx in range(len(dataset)):
        info = dataset.data_infos[idx]
        w, h = info["width"], info["height"]
        gt = dataset.get_ann_info(info)
        dets = []
        for i in range(len(gt["bboxes"])):
            box = gt["bboxes"][i].astype(np.float64)
            size = np.array([box[2] - box[0], box[3] - box[1]] * 2)
            if rs.rand() < 0.15:
                continue  # missed
            label = int(gt["labels"][i]) if rs.rand() < 0.9 else int(rs.randint(NUM_CLASSES))
            dets.append((label, box + rs.normal(0, 0.08, size=4) * size))
            if rs.rand() < 0.1:  # a second, near-duplicate detection
                dets.append((label, box + rs.normal(0, 0.03, size=4) * size))
        for _ in range(rs.randint(0, 3)):  # false positives
            x1, y1 = rs.uniform(0, 0.8 * w), rs.uniform(0, 0.8 * h)
            x2, y2 = x1 + rs.uniform(10, 0.5 * w), y1 + rs.uniform(10, 0.5 * h)
            dets.append((int(rs.randint(NUM_CLASSES)), np.array([x1, y1, x2, y2])))
        raw.append(dets)

    total = sum(len(d) for d in raw)
    scores = (rs.permutation(total) + 1) / float(total + 1)
    results, k = [], 0
    for dets in raw:
        per_class = [[] for _ in range(NUM_CLASSES)]
        for label, box in dets:
            per_class[label].append(np.concatenate([box, [scores[k]]]))
            k += 1
        results.append([np.array(p, dtype=np.float32).reshape(-1, 5) for p in per_class])
    return results, total


def run(impl):
    out = {}
    t0 = time.time()
    dataset = build_dataset(impl)
    out["order/img_ids"] = torch.tensor(dataset.img_ids, dtype=torch.int64)
    print(f"built dataset: {len(dataset)} frames ({time.time() - t0:.0f}s)")

    boxes, labels, ignores, inst, counts, ignore_counts = [], [], [], [], [], []
    for idx in range(len(dataset)):
        gt = dataset.get_ann_info(dataset.data_infos[idx])
        boxes.append(gt["bboxes"])
        labels.append(gt["labels"])
        ignores.append(gt["bboxes_ignore"])
        inst.append(gt["instance_ids"].astype(np.int64))
        counts.append(len(gt["bboxes"]))
        ignore_counts.append(len(gt["bboxes_ignore"]))
    out["gt/bboxes"] = torch.from_numpy(np.concatenate(boxes))
    out["gt/labels"] = torch.from_numpy(np.concatenate(labels))
    out["gt/bboxes_ignore"] = torch.from_numpy(np.concatenate(ignores))
    out["gt/instance_ids"] = torch.from_numpy(np.concatenate(inst))
    out["gt/counts"] = torch.tensor(counts, dtype=torch.int64)
    out["gt/ignore_counts"] = torch.tensor(ignore_counts, dtype=torch.int64)

    values, lengths = load_motion(impl)
    out["motion/values"] = torch.from_numpy(values)
    out["motion/lengths"] = torch.from_numpy(lengths)

    t0 = time.time()
    predictions, total = make_predictions(dataset)
    out["eval/num_detections"] = torch.tensor(total)
    result = dataset.evaluate(predictions, vid_style=True)
    keys = list(result)
    out["eval/keys"] = encode_str("\n".join(keys))
    out["eval/values"] = torch.tensor([float(result[k]) for k in keys], dtype=torch.float64)
    print(f"evaluated {total} detections ({time.time() - t0:.0f}s): "
          + ", ".join(f"{k}={float(result[k]):.4f}" for k in keys[:4]))
    return out


# Only the metric values get a tolerance; see the module docstring.
EVAL_RTOL = 1e-12


def compare(path_a, path_b, rtol):
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)
    failures = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            failures.append(f"{key}: only in {'A' if key in a else 'B'}")
            continue
        va, vb = a[key], b[key]
        if va.shape != vb.shape:
            failures.append(f"{key}: shape {tuple(va.shape)} != {tuple(vb.shape)}")
        elif torch.equal(va, vb):
            continue
        elif (va.dtype.is_floating_point
              and torch.allclose(va, vb, rtol=max(rtol, EVAL_RTOL if key == "eval/values" else 0),
                                 atol=0)):
            print(f"NOTE  {key}: equal within tolerance, not bit-exact "
                  f"(max |diff| {(va - vb).abs().max().item():.3e})")
        else:
            n = (va != vb).sum().item()
            failures.append(f"{key}: {n} of {va.numel()} entries differ")
    if "eval/keys" in b:
        names = bytes(b["eval/keys"].tolist()).decode().split("\n")
        print("metrics (vfe): " + ", ".join(
            f"{n}={v:.4f}" for n, v in zip(names[:4], b["eval/values"][:4].tolist(), strict=True)))
    for note in failures:
        print(f"FAIL  {note}")
    print("-" * 70)
    print(f"{len(failures)} of {len(set(a) | set(b))} artifact(s) differ" if failures
          else f"VID EVAL PARITY OK ({len(a)} artifacts)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--rtol", type=float, default=0.0,
                    help="extra tolerance for all float artifacts (default: exact, "
                         "except eval/values)")
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare, rtol=args.rtol)
    elif args.impl and args.out:
        torch.save(run(args.impl), args.out)
        print(f"saved -> {args.out}")
    else:
        ap.error("pass either --impl/--out or --compare")
