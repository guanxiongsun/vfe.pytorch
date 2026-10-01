"""Convert an official YOLOX checkpoint (Megvii-BaseDetection/YOLOX, the
0.1.1rc0 release and later) to ``vfe``'s YOLOX parameter names.

The official model keeps the PAFPN's backbone inside it and the 1x1 "stems"
in the head; mmdet, and so ``vfe``, puts the stems in the neck as its output
convs. The layers are otherwise the same, in the same order: Megvii's
``CSPLayer`` ``conv1`` / ``conv2`` / ``conv3`` / ``m`` are mmdet's
``main_conv`` / ``short_conv`` / ``final_conv`` / ``blocks``.

The released weights take raw BGR pixels in 0-255, letterboxed and padded
with 114 -- mmdet's YOLOX pipeline, no normalisation.

    python tools/convert_yolox_megvii.py yolox_m.pth yolox_m_coco_vfe.pth
    # For EOVOD on ImageNet VID: 30 classes, and the detector inside EOVOD
    python tools/convert_yolox_megvii.py yolox_m.pth yolox_m_coco_eovod.pth \
        --drop-classifier --prefix detector.
"""

import argparse
import re
from collections import OrderedDict

import torch

# Megvii prefix -> vfe prefix, first match wins.
STAGES = {"dark2": "stage1", "dark3": "stage2", "dark4": "stage3", "dark5": "stage4"}
NECK = {
    "lateral_conv0": "reduce_layers.0",
    "C3_p4": "top_down_blocks.0",
    "reduce_conv1": "reduce_layers.1",
    "C3_p3": "top_down_blocks.1",
    "bu_conv2": "downsamples.0",
    "C3_n3": "bottom_up_blocks.0",
    "bu_conv1": "downsamples.1",
    "C3_n4": "bottom_up_blocks.1",
}
CSP = {"conv1": "main_conv", "conv2": "short_conv", "conv3": "final_conv", "m": "blocks"}
HEAD = {
    "cls_convs": "bbox_head.multi_level_cls_convs",
    "reg_convs": "bbox_head.multi_level_reg_convs",
    "cls_preds": "bbox_head.multi_level_conv_cls",
    "reg_preds": "bbox_head.multi_level_conv_reg",
    "obj_preds": "bbox_head.multi_level_conv_obj",
    "stems": "neck.out_convs",
}


def _csp(rest: str) -> str:
    """``conv1.conv.weight`` -> ``main_conv.conv.weight``; ``m.0.conv1...``
    -> ``blocks.0.conv1...`` (a bottleneck's own conv1/conv2 keep their names)."""
    first, _, tail = rest.partition(".")
    return f"{CSP[first]}.{tail}"


def convert_key(key: str) -> str:
    m = re.match(r"backbone\.backbone\.stem\.(.*)", key)
    if m:
        return f"backbone.stem.{m.group(1)}"
    m = re.match(r"backbone\.backbone\.(dark\d)\.(\d+)\.(.*)", key)
    if m:
        dark, index, rest = m.groups()
        stage = STAGES[dark]
        is_csp = index == ("2" if dark == "dark5" else "1")
        if is_csp:
            rest = _csp(rest)
        elif dark == "dark5" and index == "1":  # SPP: conv1 / conv2 keep their names
            pass
        return f"backbone.{stage}.{index}.{rest}"
    m = re.match(r"backbone\.(\w+?)\.(.*)", key)
    if m and m.group(1) in NECK:
        name, rest = m.groups()
        if name.startswith("C3_"):
            rest = _csp(rest)
        return f"neck.{NECK[name]}.{rest}"
    m = re.match(r"head\.(\w+?)\.(.*)", key)
    if m and m.group(1) in HEAD:
        return f"{HEAD[m.group(1)]}.{m.group(2)}"
    raise KeyError(f"no rule for {key}")


def convert(state: dict, drop_classifier: bool = False, prefix: str = "") -> OrderedDict:
    out = OrderedDict()
    for key, value in state.items():
        new = convert_key(key)
        if drop_classifier and new.startswith("bbox_head.multi_level_conv_cls."):
            continue
        out[prefix + new] = value
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--drop-classifier", action="store_true",
                    help="leave out the class predictors (fine-tuning to other classes)")
    ap.add_argument("--prefix", default="",
                    help="prepended to every key, e.g. 'detector.' for a video model")
    args = ap.parse_args()
    checkpoint = torch.load(args.src, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    converted = convert(state, args.drop_classifier, args.prefix)
    torch.save({"state_dict": converted,
                "meta": {"converted_from": args.src, "drop_classifier": args.drop_classifier,
                         "prefix": args.prefix}},
               args.dst)
    print(f"{len(state)} tensors -> {len(converted)} -> {args.dst}")


if __name__ == "__main__":
    main()
