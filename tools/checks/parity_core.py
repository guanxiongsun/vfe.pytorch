"""Parity check: ``vfe.core`` vs the ``mmdet.core`` oracle.

Anchors, box coding, IoU, assignment, sampling and multiclass NMS. Same
two-process pattern as ``parity_ops.py`` / ``parity_backbone.py``.

Usage:
    conda run -n vfe       --no-capture-output python tools/checks/parity_core.py --impl mmdet --out ~/core_mmdet.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_core.py --impl vfe   --out ~/core_vfe.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_core.py --compare ~/core_mmdet.pt ~/core_vfe.pt

Everything here is either exact integer/index work or a handful of elementwise
float ops, so the default tolerance is **zero** -- any difference at all is a
porting bug, not arithmetic noise. The one seeded case (``RandomSampler``) is
made reproducible by seeding the CPU RNG immediately before the call; both
torch versions draw the same permutation from ``torch.randperm``, which is the
only stochastic step.

One known, benign divergence, deliberately kept out of the fixtures rather than
papered over in the comparison: **when two candidates have exactly equal
scores, mmcv's NMS and torchvision's can return them in either order.** The
kept set and the scores are identical; only the order within the tie differs.
The score fixtures below are therefore constructed to be tie-free, so that a
failure in an NMS case means something real. If a future case does produce
ties, expect pairs of adjacent transpositions and check the kept set as a
multiset before assuming a bug.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

# The RPN's anchor layout in configs/_base_/models/vid/faster_rcnn_r50_dc5.py:
# one stride-16 level, 4 scales x 3 ratios.
DC5_ANCHOR = dict(strides=[16], ratios=[0.5, 1.0, 2.0], scales=[4, 8, 16, 32])
# An FPN-style multi-level generator, to exercise the per-level bookkeeping
# that the single-level DC5 config never reaches.
FPN_ANCHOR = dict(strides=[4, 8, 16, 32, 64], ratios=[0.5, 1.0, 2.0], scales=[8])
# RetinaNet's octave parameterisation, the other way to specify scales.
OCTAVE_ANCHOR = dict(
    strides=[8, 16], ratios=[0.5, 1.0, 2.0], octave_base_scale=4, scales_per_octave=3
)

# 37x62 is the feature-map size of a 600x1000 image at stride 16, and
# deliberately odd so `valid_flags` has real padding to mask out.
DC5_FEATMAP = [(37, 62)]
FPN_FEATMAPS = [(152, 100), (76, 50), (38, 25), (19, 13), (10, 7)]
OCTAVE_FEATMAPS = [(76, 50), (38, 25)]
PAD_SHAPE = (600, 992)


def boxes(n, seed, scale=600.0, device="cpu"):
    """``n`` random valid ``(x1, y1, x2, y2)`` boxes, from a fixed seed."""
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(n, 2, generator=g) * scale
    wh = torch.rand(n, 2, generator=g) * (scale / 4) + 1.0
    return torch.cat([xy, xy + wh], dim=1).to(device)


def run(impl, device):
    if impl == "mmdet":
        from mmdet.core import (
            AnchorGenerator,
            MaxIoUAssigner,
            RandomSampler,
            anchor_inside_flags,
            bbox2result,
            bbox2roi,
            bbox_overlaps,
            build_bbox_coder,
            images_to_levels,
            multiclass_nms,
        )

        # Not re-exported from mmdet.core, unlike everything above.
        from mmdet.core.bbox.coder.delta_xywh_bbox_coder import bbox2delta, delta2bbox
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.core import (
            AnchorGenerator,
            MaxIoUAssigner,
            RandomSampler,
            anchor_inside_flags,
            bbox2delta,
            bbox2result,
            bbox2roi,
            bbox_overlaps,
            build_bbox_coder,
            delta2bbox,
            images_to_levels,
            multiclass_nms,
        )

    out = {}

    # ---- anchors -----------------------------------------------------------
    for name, cfg, featmaps in [
        ("anchor/dc5", DC5_ANCHOR, DC5_FEATMAP),
        ("anchor/fpn", FPN_ANCHOR, FPN_FEATMAPS),
        ("anchor/octave", OCTAVE_ANCHOR, OCTAVE_FEATMAPS),
    ]:
        gen = AnchorGenerator(**cfg)
        priors = gen.grid_priors(featmaps, device=device)
        flags = gen.valid_flags(featmaps, PAD_SHAPE, device=device)
        out[f"{name}/num_base_priors"] = torch.tensor(gen.num_base_priors)
        for i in range(len(priors)):
            out[f"{name}/priors{i}"] = priors[i].cpu()
            out[f"{name}/flags{i}"] = flags[i].cpu()

    # scale_major=False reorders the (ratio, scale) grid; a pretrained rpn_cls
    # conv's channel order depends on getting this right.
    gen = AnchorGenerator(**DC5_ANCHOR, scale_major=False)
    out["anchor/scale_minor/priors0"] = gen.grid_priors(DC5_FEATMAP, device=device)[0].cpu()
    # center_offset=0.5 is the pre-2.0 convention, still used by some configs.
    gen = AnchorGenerator(**DC5_ANCHOR, center_offset=0.5)
    out["anchor/offset/priors0"] = gen.grid_priors(DC5_FEATMAP, device=device)[0].cpu()

    flat = AnchorGenerator(**DC5_ANCHOR).grid_priors(DC5_FEATMAP, device=device)[0]
    valid = AnchorGenerator(**DC5_ANCHOR).valid_flags(DC5_FEATMAP, PAD_SHAPE, device=device)[0]
    for border in (-1, 0, 32):
        inside = anchor_inside_flags(flat, valid, (600, 992), allowed_border=border)
        out[f"anchor/inside{border}"] = inside.cpu()

    # images_to_levels: [per-image] -> [per-level]
    num_lvl = [152 * 100 * 3, 76 * 50 * 3]
    per_img = [torch.arange(sum(num_lvl)) + 1000 * i for i in range(2)]
    for i, lvl in enumerate(images_to_levels(per_img, num_lvl)):
        out[f"images_to_levels/{i}"] = lvl.cpu()

    # ---- IoU ---------------------------------------------------------------
    a, b = boxes(64, 1, device=device), boxes(48, 2, device=device)
    for mode in ("iou", "iof", "giou"):
        out[f"iou/{mode}"] = bbox_overlaps(a, b, mode=mode).cpu()
        out[f"iou/{mode}/aligned"] = bbox_overlaps(a, b[:0].new_zeros(64, 4) + a, mode=mode,
                                                   is_aligned=True).cpu()
    # Degenerate inputs: zero-area boxes hit the `eps` floor, empty sets hit the
    # early return. Both are reachable from real data (a clipped box, an image
    # with no annotations).
    zero = torch.zeros(4, 4, device=device)
    out["iou/zero_area"] = bbox_overlaps(zero, a[:4], mode="iou").cpu()
    out["iou/empty_rows"] = bbox_overlaps(a[:0], b, mode="iou").cpu()
    out["iou/empty_cols"] = bbox_overlaps(a, b[:0], mode="iou").cpu()

    # ---- box coding --------------------------------------------------------
    proposals, gts = boxes(100, 3, device=device), boxes(100, 4, device=device)
    for means, stds in [
        ((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)),      # RPN
        ((0.0, 0.0, 0.0, 0.0), (0.2, 0.2, 0.2, 0.2)),      # R-CNN
        ((0.1, -0.1, 0.2, -0.2), (0.1, 0.2, 0.3, 0.4)),    # asymmetric, to catch
    ]:                                                      # a means/stds mix-up
        coder = build_bbox_coder(
            dict(type="DeltaXYWHBBoxCoder", target_means=list(means), target_stds=list(stds))
        )
        tag = f"coder/{stds[0]}_{means[0]}"
        deltas = coder.encode(proposals, gts)
        out[f"{tag}/encode"] = deltas.cpu()
        out[f"{tag}/decode"] = coder.decode(proposals, deltas, max_shape=(600, 1000)).cpu()
        out[f"{tag}/decode_noclip"] = coder.decode(proposals, deltas).cpu()
        # Class-specific regression: (N, num_classes * 4).
        g = torch.Generator().manual_seed(5)
        multi = (torch.randn(100, 30 * 4, generator=g) * 0.5).to(device)
        out[f"{tag}/decode_multi"] = coder.decode(proposals, multi, max_shape=(600, 1000)).cpu()

    out["coder/fn/bbox2delta"] = bbox2delta(proposals, gts).cpu()
    # Extreme deltas: without wh_ratio_clip these would exp() to infinity.
    g = torch.Generator().manual_seed(6)
    wild = (torch.randn(100, 4, generator=g) * 20).to(device)
    out["coder/fn/delta2bbox_clipped"] = delta2bbox(proposals, wild).cpu()
    out["coder/fn/delta2bbox_empty"] = delta2bbox(proposals[:0], wild[:0]).cpu()

    # ---- assignment --------------------------------------------------------
    anchors = flat
    gt_bboxes = boxes(8, 7, device=device)
    gt_labels = torch.arange(8, device=device) % 30
    ignore = boxes(2, 8, device=device)
    assign_cases = {
        # RPN settings, then R-CNN settings (which also disable low-quality matching).
        "rpn": dict(pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, ignore_iof_thr=-1),
        "rcnn": dict(pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5, ignore_iof_thr=-1),
        "no_lowq": dict(
            pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, match_low_quality=False
        ),
        "band": dict(pos_iou_thr=0.7, neg_iou_thr=(0.0, 0.3), min_pos_iou=0.3),
        "ignore": dict(pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5, ignore_iof_thr=0.5),
        "gt_max_one": dict(
            pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, gt_max_assign_all=False
        ),
    }
    results = {}
    for name, cfg in assign_cases.items():
        assigner = MaxIoUAssigner(**cfg)
        res = assigner.assign(
            anchors,
            gt_bboxes,
            gt_bboxes_ignore=ignore if cfg.get("ignore_iof_thr", -1) > 0 else None,
            gt_labels=gt_labels,
        )
        results[name] = res
        out[f"assign/{name}/gt_inds"] = res.gt_inds.cpu()
        out[f"assign/{name}/max_overlaps"] = res.max_overlaps.cpu()
        out[f"assign/{name}/labels"] = res.labels.cpu()
        out[f"assign/{name}/num_gts"] = torch.tensor(res.num_gts)

    # No ground truth at all -> everything background, nothing ignored.
    empty_res = MaxIoUAssigner(pos_iou_thr=0.7, neg_iou_thr=0.3).assign(
        anchors, gt_bboxes[:0], gt_labels=gt_labels[:0]
    )
    out["assign/no_gt/gt_inds"] = empty_res.gt_inds.cpu()
    out["assign/no_gt/labels"] = empty_res.labels.cpu()

    # ---- sampling ----------------------------------------------------------
    for name, sampler_cfg, add_gt in [
        ("rpn", dict(num=256, pos_fraction=0.5, neg_pos_ub=-1), False),
        ("rcnn", dict(num=256, pos_fraction=0.25, neg_pos_ub=-1), True),
        ("ub", dict(num=256, pos_fraction=0.5, neg_pos_ub=3), False),
    ]:
        # Re-assign per case: `sample` mutates the AssignResult when add_gt is on.
        res = MaxIoUAssigner(pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5).assign(
            anchors, gt_bboxes, gt_labels=gt_labels
        )
        sampler = RandomSampler(add_gt_as_proposals=add_gt, **sampler_cfg)
        torch.manual_seed(1234)
        sr = sampler.sample(res, anchors, gt_bboxes, gt_labels)
        out[f"sample/{name}/pos_inds"] = sr.pos_inds.cpu()
        out[f"sample/{name}/neg_inds"] = sr.neg_inds.cpu()
        out[f"sample/{name}/pos_gt_bboxes"] = sr.pos_gt_bboxes.cpu()
        out[f"sample/{name}/pos_gt_labels"] = sr.pos_gt_labels.cpu()
        out[f"sample/{name}/pos_assigned_gt_inds"] = sr.pos_assigned_gt_inds.cpu()
        out[f"sample/{name}/pos_is_gt"] = sr.pos_is_gt.cpu()
        out[f"sample/{name}/bboxes"] = sr.bboxes.cpu()

    # ---- transforms --------------------------------------------------------
    bbox_list = [boxes(5, 20, device=device), boxes(0, 21, device=device),
                 boxes(3, 22, device=device)]
    out["transforms/bbox2roi"] = bbox2roi(bbox_list).cpu()
    dets = torch.cat([boxes(40, 23), torch.rand(40, 1, generator=torch.Generator().manual_seed(24))],
                     dim=1)
    lbls = torch.arange(40) % 30
    res_arrays = bbox2result(dets, lbls, 30)
    out["transforms/bbox2result"] = torch.cat([torch.from_numpy(r) for r in res_arrays])
    out["transforms/bbox2result/counts"] = torch.tensor([len(r) for r in res_arrays])

    # ---- multiclass_nms ----------------------------------------------------
    g = torch.Generator().manual_seed(30)
    n, num_classes = 300, 30
    shared = boxes(n, 31, device=device)
    per_class = boxes(n, 32, device=device).repeat(1, num_classes)
    # Scores are built to be *distinct and exactly representable*, for two
    # separate reasons -- see the module docstring's note on NMS:
    #   - not `softmax(randn)`, because torch 1.10's and 2.10's softmax kernels
    #     disagree by ~1.8e-7, which reorders near-tied candidates;
    #   - not `rand`, because ~9k fp32 draws collide often enough (birthday
    #     bound) to produce exact ties, which the two NMS sorts break
    #     differently.
    # Multiplying by an odd constant mod 2**24 is a bijection, so every score
    # is unique; dividing by 2**24 is exact.
    idx = torch.arange(n * (num_classes + 1), dtype=torch.int64)
    scores = ((idx * 2654435761) % (1 << 24)).double().div(1 << 24).float()
    scores = scores.view(n, num_classes + 1).to(device)
    out["nms/input_scores"] = scores.cpu()
    nms_cfg = dict(type="nms", iou_threshold=0.5)
    for name, bb, thr, max_num in [
        ("shared", shared, 0.0001, 100),      # the rcnn test_cfg of the VID configs
        ("per_class", per_class, 0.0001, 100),
        ("highthr", per_class, 0.5, 100),     # most candidates filtered out
        ("nocap", per_class, 0.0001, -1),
        ("empty", per_class, 1.5, 100),       # nothing survives the threshold
    ]:
        d, lab, keep = multiclass_nms(bb, scores, thr, nms_cfg, max_num, return_inds=True)
        out[f"nms/{name}/dets"] = d.cpu()
        out[f"nms/{name}/labels"] = lab.cpu()
        out[f"nms/{name}/inds"] = keep.cpu()
    factors = torch.rand(n, generator=torch.Generator().manual_seed(33)).to(device)
    d, lab = multiclass_nms(per_class, scores, 0.0001, nms_cfg, 100, score_factors=factors)
    out["nms/factors/dets"] = d.cpu()
    out["nms/factors/labels"] = lab.cpu()

    print(f"RAN   {len(out)} artifacts on {device}")
    return out


def compare(path_a, path_b, atol, rtol):
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
        elif va.dtype != vb.dtype:
            failures.append(f"{key}: dtype {va.dtype} != {vb.dtype}")
        elif not va.dtype.is_floating_point:
            if not torch.equal(va, vb):
                n = int((va != vb).sum())
                failures.append(f"{key}: {n}/{va.numel()} elements differ")
        elif not torch.allclose(va, vb, atol=atol, rtol=rtol):
            failures.append(
                f"{key}: max|diff| = {(va - vb).abs().max().item():.3e} "
                f"(scale {va.abs().max().item():.3e})"
            )

    for note in failures:
        print(f"FAIL  {note}")
    print("-" * 70)
    if failures:
        print(f"{len(failures)} of {len(set(a) | set(b))} artifact(s) differ")
    else:
        print(f"CORE PARITY OK ({len(a)} artifacts)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    # Zero by default: none of this is supposed to be approximate.
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, default=0.0)
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare, atol=args.atol, rtol=args.rtol)
    elif args.impl and args.out:
        torch.save(run(args.impl, args.device), args.out)
        print(f"saved -> {args.out}")
    else:
        ap.error("pass either --impl/--out or --compare")
