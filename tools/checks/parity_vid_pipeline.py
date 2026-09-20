"""Parity check: the ImageNet VID test-time data path, vfe vs mmdet.

Builds the test dataset from the MAMBA config on each side, takes real val
samples, and compares what the model would receive: mmdet's
``collate`` + ``scatter`` against ``vfe.datasets.collate_video_test``.

Samples: from videos of different resolutions (so resize rounding varies),
the first frame, which carries 14 reference frames spread over the whole
video, and the next frame in the shuffled test order, which carries none.

Per sample:
* ``structure`` -- the nesting of lists, tensors and dicts, with shapes.
* ``metas`` -- every image meta (shapes, scale factors, normalisation, frame
  ids, filenames), serialised exactly.
* ``digest/*`` -- SHA-256 of each image tensor's bytes: bit-exact or not.
* ``thumb/*`` -- every 8th pixel, to see *where* a digest mismatch comes from.

Needs the val images under data/ILSVRC/Data/VID.

Usage:
    conda run -n vfe       --no-capture-output python tools/checks/parity_vid_pipeline.py --impl mmdet --out A.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_vid_pipeline.py --impl vfe   --out B.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_vid_pipeline.py --compare A.pt B.pt
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/vid/mamba/mamba_r101_dc5_6x.py"
NUM_VIDEOS = 6


def encode_str(s):
    return torch.tensor(list(s.encode()), dtype=torch.uint8)


def signature(x):
    if isinstance(x, torch.Tensor):
        return f"Tensor{tuple(x.shape)}:{x.dtype}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}]<{signature(x[0]) if len(x) else ''}>"
    if isinstance(x, dict):
        return "dict{" + ",".join(sorted(x)) + "}"
    return type(x).__name__


def canonical(obj):
    def default(o):
        if isinstance(o, np.ndarray):
            return {"ndarray": str(o.dtype), "shape": list(o.shape), "data": o.tolist()}
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"cannot serialise {type(o)}")
    return json.dumps(obj, sort_keys=True, default=default)


def build(impl):
    if impl == "mmdet":
        from mmcv import Config
        from mmcv.parallel import collate, scatter
        from mmdet.datasets import build_dataset

        dataset = build_dataset(Config.fromfile(str(CONFIG)).data.test)

        def batch_of(idx):
            return scatter(collate([dataset[idx]], samples_per_gpu=1), [-1])[0]
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.config import Config
        from vfe.datasets import build_dataset, collate_video_test

        dataset = build_dataset(Config.fromfile(str(CONFIG)).data.test)

        def batch_of(idx):
            return collate_video_test([dataset[idx]])
    return dataset, batch_of


def pick_samples(dataset):
    """First frame + the following index, for NUM_VIDEOS videos of distinct sizes."""
    picked, sizes = [], set()
    for idx, info in enumerate(dataset.data_infos):
        size = (info["width"], info["height"])
        if info["frame_id"] == 0 and size not in sizes:
            sizes.add(size)
            picked.extend([idx, idx + 1])
            if len(sizes) == NUM_VIDEOS:
                break
    return picked


def run(impl):
    dataset, batch_of = build(impl)
    out = {}
    samples = pick_samples(dataset)
    out["samples"] = torch.tensor(samples)
    for idx in samples:
        batch = batch_of(idx)
        tag = f"idx{idx}"
        out[f"{tag}/structure"] = encode_str(
            ";".join(f"{k}={signature(v)}" for k, v in sorted(batch.items())))
        metas = {k: v for k, v in batch.items() if k.endswith("img_metas")}
        out[f"{tag}/metas"] = encode_str(canonical(metas))
        for key in ("img", "ref_img"):
            if key not in batch:
                continue
            tensor = batch[key][0].contiguous()
            out[f"{tag}/digest/{key}"] = encode_str(
                hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
            out[f"{tag}/thumb/{key}"] = tensor[..., ::8, ::8].clone()
        print(f"{tag}: frame {dataset.data_infos[idx]['frame_id']}, "
              f"{dataset.data_infos[idx]['width']}x{dataset.data_infos[idx]['height']}, "
              f"{'with' if 'ref_img' in batch else 'without'} references")
    return out


def compare(path_a, path_b, label="VID PIPELINE", expect=None):
    """Bit-exact comparison; ``expect`` maps keys to values both sides must hold."""
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)
    failures = []
    for key, value in (expect or {}).items():
        for side, artifacts in (("A", a), ("B", b)):
            if key in artifacts and artifacts[key].tolist() != value:
                failures.append(f"{key}: {side} has {artifacts[key].tolist()}, expected {value}")
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            failures.append(f"{key}: only in {'A' if key in a else 'B'}")
        elif a[key].shape != b[key].shape or not torch.equal(a[key], b[key]):
            detail = ""
            if "/thumb/" in key and a[key].shape == b[key].shape:
                detail = f" (max |diff| {(a[key] - b[key]).abs().max().item():.3e})"
            elif key.endswith(("/metas", "/structure")):
                sa, sb = bytes(a[key].tolist()).decode(), bytes(b[key].tolist()).decode()
                first = next((i for i in range(min(len(sa), len(sb))) if sa[i] != sb[i]),
                             min(len(sa), len(sb)))
                detail = f"\n      A: ...{sa[max(0, first - 60):first + 60]}\n      B: ...{sb[max(0, first - 60):first + 60]}"
            failures.append(f"{key}: differs{detail}")
    for note in failures:
        print(f"FAIL  {note}")
    print("-" * 70)
    print(f"{len(failures)} of {len(set(a) | set(b))} artifact(s) differ" if failures
          else f"{label} PARITY OK ({len(a)} artifacts, all bit-exact)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.impl and args.out:
        torch.save(run(args.impl), args.out)
        print(f"saved -> {args.out}")
    else:
        ap.error("pass either --impl/--out or --compare")
