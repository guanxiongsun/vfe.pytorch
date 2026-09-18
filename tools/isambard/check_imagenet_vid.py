"""Check that an extracted ImageNet VID / DET tree has every image its
annotation files reference.

Usage:
    python tools/isambard/check_imagenet_vid.py --root /projects/b5cs/imagenet_vid/ILSVRC
    python tools/isambard/check_imagenet_vid.py --root data/ILSVRC --splits vid_val --limit 2000

``--root`` is the directory the configs call ``data/ILSVRC``. Exit status is 0
only if nothing is missing, so a staging job can stop on it.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# split -> (annotation file, directory its file_name entries are relative to)
SPLITS = {
    "vid_train": ("annotations/imagenet_vid_train.json", "Data/VID"),
    "vid_val": ("annotations/imagenet_vid_val.json", "Data/VID"),
    "det_30plus1cls": ("annotations/imagenet_det_30plus1cls.json", "Data/DET"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    ap.add_argument("--limit", type=int, default=0, help="check only the first N images per split")
    ap.add_argument("--workers", type=int, default=32, help="parallel stat() calls")
    args = ap.parse_args()

    complete = True
    for split in args.splits:
        ann_file, prefix = SPLITS[split]
        with open(os.path.join(args.root, ann_file)) as f:
            images = json.load(f)["images"]
        names = [img["file_name"] for img in images]
        if args.limit:
            names = names[: args.limit]
        paths = [os.path.join(args.root, prefix, name) for name in names]

        # Parallel stat(): on Lustre each lookup is a metadata round trip, and
        # the three splits reference ~1.6M files.
        with ThreadPoolExecutor(args.workers) as pool:
            found = list(pool.map(os.path.isfile, paths, chunksize=512))
        missing = [p for p, ok in zip(paths, found, strict=True) if not ok]

        print(f"{split:16s} {len(paths):>9,d} images referenced, {len(missing):,d} missing")
        for path in missing[:5]:
            print(f"    missing: {path}")
        complete = complete and not missing

    print("DATA COMPLETE" if complete else "DATA INCOMPLETE")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
