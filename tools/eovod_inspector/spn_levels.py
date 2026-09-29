"""How the size prior loses objects: from a ``dump.py`` run made with the size
prior on, count the ground-truth objects on partial frames (the head ran only
some levels) whose FCOS level -- the one their size assigns them to -- was
not run, and compare how often objects are detected on partial and full
frames.

    python tools/eovod_inspector/spn_levels.py ROOT/runs/NAME.json [...]
"""

from __future__ import annotations

import json
import sys

# FCOS's regress ranges on max(l, t, r, b); at a box's centre that is half
# its longer side, in network-input pixels.
BOUNDS = (64, 128, 256, 512)


def fcos_level(box, sf) -> int:
    half = max((box[2] - box[0]) * sf[0], (box[3] - box[1]) * sf[1]) / 2
    return next((i for i, b in enumerate(BOUNDS) if half <= b), len(BOUNDS))


def main(paths):
    for path in paths:
        with open(path) as f:
            run = json.load(f)
        stats = {k: [0, 0] for k in ("full", "partial_level_run", "partial_level_skipped")}
        by_level = {}  # level -> [skipped, objects on partial frames]
        for video in run["videos"]:
            sf = next((fr["sf"] for fr in video["frames"] if "sf" in fr), None)
            if sf is None:
                continue
            for fr in video["frames"]:
                for g, hit in zip(fr["gt"], fr["gt_hit"], strict=True):
                    lvl = fcos_level(g, sf)
                    if fr["full"]:
                        key = "full"
                    else:
                        key = ("partial_level_run" if lvl in fr["levels"]
                               else "partial_level_skipped")
                        row = by_level.setdefault(lvl, [0, 0])
                        row[0] += key == "partial_level_skipped"
                        row[1] += 1
                    stats[key][0] += 1
                    stats[key][1] += hit >= 0.05  # a same-class detection at IoU >= 0.5
        print(f"== {run['run']}")
        for key, (n, found) in stats.items():
            print(f"  {key:22s} objects {n:5d}  detected (score >= 0.05) {found / max(n, 1):.3f}")
        partial = sum(v[1] for v in by_level.values())
        skipped = sum(v[0] for v in by_level.values())
        print(f"  objects on partial frames whose level was skipped: {skipped}/{partial} "
              f"({skipped / max(partial, 1):.1%})")
        for lvl in sorted(by_level):
            s, n = by_level[lvl]
            print(f"    P{lvl + 3}: skipped {s}/{n}")


if __name__ == "__main__":
    main(sys.argv[1:])
