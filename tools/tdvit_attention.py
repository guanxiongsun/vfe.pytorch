"""Where TDViT's joint attention looks: the share of each TDTB's attention that
goes to the reference rather than to the frame's own window.

Runs the first ``--frames`` frames of the first ``--videos`` val videos in
order, online as at test time, and for every TDTB averages, over frames,
windows, heads and queries, the softmax mass on the reference's keys (the
second half of a joint window's keys). On a video's first frame the
reference is the frame itself and the block runs as Swin, so those frames are
left out; 0.5 would be no preference, 0 a block that ignores its reference.

    python tools/tdvit_attention.py CONFIG CKPT [--cfg-options ...] [--out shares.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict

import torch
from tdvit_speed import load_frames  # tools/, on sys.path when run as a script

from vfe.config import Config, parse_cfg_options
from vfe.models.backbones.tdvit import TDTB
from vfe.models.builder import build_model
from vfe.models.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("ckpt")
    ap.add_argument("--videos", type=int, default=20)
    ap.add_argument("--frames", type=int, default=100, help="per video")
    ap.add_argument("--cfg-options", nargs="+", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out", help="write the shares here as JSON")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(parse_cfg_options(args.cfg_options))
    if cfg.model.detector.backbone.get("attention", "cross") != "joint":
        raise SystemExit("the shares are defined for attention='joint'")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.model.detector.backbone.init_cfg = None
    model = build_model(cfg.model)
    load_checkpoint(model, args.ckpt, map_location="cpu")
    model = model.to(device).eval()

    shares: dict[str, list[float]] = defaultdict(list)
    for i, stage in enumerate(model.detector.backbone.stages):
        for j, block in enumerate(stage.blocks):
            if not isinstance(block, TDTB):
                continue
            name = f"stage{i + 1}.block{j}"

            def hook(module, inputs, out, name=name):
                n = out.shape[-2]
                if out.shape[-1] == 2 * n:  # a joint window; first frames run as Swin
                    shares[name].append(float(out[..., n:].sum(-1).mean()))

            block.attn.w_msa.softmax.register_forward_hook(hook)

    videos = load_frames(cfg, args.videos, args.frames, device)
    with torch.no_grad():
        for video in videos:
            for frame in video:
                model(return_loss=False, rescale=True, **frame)

    result = {name: dict(share=st.mean(v), frames=len(v)) for name, v in shares.items()}
    for name, row in result.items():
        print(f"{name}: {row['share']:.3f} of the attention on the reference ({row['frames']} frames)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(config=args.config, ckpt=args.ckpt, options=args.cfg_options,
                           videos=len(videos), shares=result), f, indent=1)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
