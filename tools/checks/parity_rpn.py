"""Parity check: ``vfe.models.dense_heads.RPNHead`` vs the mmdet oracle.

Covers all three jobs the head does, since they fail independently:

* ``forward`` -- the convs. Weights are generated on CPU from a fixed seed and
  loaded into both implementations, so this is a pure numerics check.
* ``get_bboxes`` -- anchor decode, per-level top-k, per-level NMS. The MAMBA
  config (single DC5 level, stride 16) and the STPN config (5 FPN levels) are
  both exercised, because the multi-level path is where the level-id trick in
  ``batched_nms`` matters and the single-level path never touches it.
* ``loss`` -- assignment, sampling, and the two losses, including the
  ``avg_factor`` denominator. Sampling draws from the CPU RNG (see
  ``RandomSampler``), so both runs seed ``torch.manual_seed`` identically right
  before the call and draw the same anchors.

Usage:
    conda run -n vfe       --no-capture-output python tools/checks/parity_rpn.py --impl mmdet --out ~/rpn_mmdet.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_rpn.py --impl vfe   --out ~/rpn_vfe.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_rpn.py --compare ~/rpn_mmdet.pt ~/rpn_vfe.pt

At ``--atol 0 --rtol 0`` on CPU, 75 of the 80 artifacts are bit-exact --
including every integer target tensor and the whole single-level DC5 path. The
5 that are not are all downstream of FPN level 4, the 10x7 feature map:
``fpn/{cls4,reg4}``, one of its losses, and the two proposal sets that decode
from it. This is torch, not the port: a bare ``F.conv2d`` given bit-identical
input and weights differs by 1.3e-6 on a 10x7 map between torch 1.10 and 2.10
on CPU, while the 19x13 map above it is bit-exact -- the two versions pick
different blocking for very small spatial sizes. The proposal coordinates then
differ by 3.1e-5 on a 600-pixel scale, which is one fp32 ULP at that magnitude;
crucially the *ordering* survives, so NMS kept the same boxes. Hence the
defaults below are 1e-6 / 1e-5 rather than 0.

On CUDA the picture is cleaner: every conv output, every proposal and every
target tensor is bit-exact, and the only divergence is 7 classification losses
at <=6e-8 -- the fused-BCE noise floor already documented in ``parity_losses``.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

# MAMBA: ResNet-101-DC5 + ChannelMapper, one level at stride 16, 12 anchors.
DC5_HEAD = dict(
    type="RPNHead",
    in_channels=512,
    feat_channels=512,
    anchor_generator=dict(
        type="AnchorGenerator", scales=[4, 8, 16, 32], ratios=[0.5, 1.0, 2.0], strides=[16]
    ),
    bbox_coder=dict(
        type="DeltaXYWHBBoxCoder",
        target_means=[0.0, 0.0, 0.0, 0.0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
    ),
    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
)
DC5_FEATMAPS = [(1, 512, 37, 62)]

# STPN: Swin-T + FPN, five levels, 3 anchors each.
FPN_HEAD = dict(
    type="RPNHead",
    in_channels=256,
    feat_channels=256,
    anchor_generator=dict(
        type="AnchorGenerator", scales=[8], ratios=[0.5, 1.0, 2.0], strides=[4, 8, 16, 32, 64]
    ),
    bbox_coder=dict(
        type="DeltaXYWHBBoxCoder",
        target_means=[0.0, 0.0, 0.0, 0.0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
    ),
    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
)
FPN_FEATMAPS = [(1, 256, 152, 100), (1, 256, 76, 50), (1, 256, 38, 25), (1, 256, 19, 13),
                (1, 256, 10, 7)]

IMG_SHAPE = (600, 992, 3)
PAD_SHAPE = (608, 992, 3)

TRAIN_CFG = dict(
    assigner=dict(
        type="MaxIoUAssigner",
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        min_pos_iou=0.3,
        match_low_quality=True,
        ignore_iof_thr=-1,
    ),
    sampler=dict(
        type="RandomSampler", num=256, pos_fraction=0.5, neg_pos_ub=-1,
        add_gt_as_proposals=False
    ),
    allowed_border=0,
    pos_weight=-1,
    debug=False,
)
TEST_CFG = dict(nms_pre=6000, max_per_img=300, nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0)

# Two images' worth of boxes, in the padded coordinate frame. Deliberately
# includes a small box and a near-duplicate pair so the assigner's
# match_low_quality path and the sampler both have something to do.
GT_BBOXES = [
    [[23.0, 41.0, 310.0, 520.0], [400.0, 60.0, 900.0, 430.0], [700.0, 300.0, 740.0, 350.0]],
    [[100.0, 100.0, 500.0, 500.0], [110.0, 105.0, 505.0, 495.0]],
]


def fixed(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def seeded_state_dict(head, seed):
    """Overwrite every parameter from a fixed CPU seed, so both implementations
    start from bit-identical weights regardless of their default init."""
    state = {}
    g = torch.Generator().manual_seed(seed)
    for name, param in sorted(head.state_dict().items()):
        state[name] = torch.randn(*param.shape, generator=g) * 0.02
    return state


def run(impl, device):
    # train_cfg/test_cfg reach the head as a ConfigDict in real use, and mmdet
    # reads them by attribute (`cfg.min_bbox_size`), so a plain dict does not
    # survive the mmdet side. Each impl gets its own.
    if impl == "mmdet":
        from mmcv import ConfigDict
        from mmdet.models import build_head
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.config import ConfigDict
        from vfe.models.builder import build_head

    img_metas = [dict(img_shape=IMG_SHAPE, pad_shape=PAD_SHAPE, scale_factor=1.0)]

    if device == "cuda":
        # See parity_backbone: TF32 hides ~1e-4 of error in conv outputs.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    out = {}

    for name, head_cfg, featmaps, seed in [
        ("dc5", DC5_HEAD, DC5_FEATMAPS, 401),
        ("fpn", FPN_HEAD, FPN_FEATMAPS, 402),
    ]:
        feats = [fixed(shape, seed + i, scale=0.5).to(device) for i, shape in enumerate(featmaps)]

        # ---- forward ------------------------------------------------------
        head = build_head(dict(head_cfg, train_cfg=None, test_cfg=ConfigDict(TEST_CFG)))
        head.load_state_dict(seeded_state_dict(head, seed))
        head = head.to(device).eval()

        with torch.no_grad():
            cls_scores, bbox_preds = head(feats)
        # Indexed rather than zip(): this runs under py3.8 too, which has no
        # zip(strict=), and ruff's B905 rightly flags a bare zip.
        for i in range(len(cls_scores)):
            out[f"{name}/cls{i}"] = cls_scores[i].cpu()
            out[f"{name}/reg{i}"] = bbox_preds[i].cpu()

        # ---- get_bboxes ---------------------------------------------------
        with torch.no_grad():
            proposals = head.get_bboxes(cls_scores, bbox_preds, img_metas=img_metas)
        out[f"{name}/proposals"] = proposals[0].cpu()

        # nms_pre below the anchor count forces the sort-and-truncate branch,
        # which is skipped entirely at the default 6000 for the small levels.
        small_cfg = ConfigDict(dict(TEST_CFG, nms_pre=1000, max_per_img=100))
        with torch.no_grad():
            proposals = head.get_bboxes(cls_scores, bbox_preds, img_metas=img_metas,
                                        cfg=small_cfg)
        out[f"{name}/proposals_small"] = proposals[0].cpu()

        # ---- loss ---------------------------------------------------------
        train_head = build_head(dict(head_cfg, train_cfg=ConfigDict(TRAIN_CFG),
                                     test_cfg=ConfigDict(TEST_CFG)))
        train_head.load_state_dict(seeded_state_dict(train_head, seed))
        train_head = train_head.to(device).train()

        for gt_name, boxes in [("gt0", GT_BBOXES[0]), ("gt1", GT_BBOXES[1])]:
            gt_bboxes = [torch.tensor(boxes, dtype=torch.float32, device=device)]
            cls_scores, bbox_preds = train_head(feats)
            # RandomSampler draws from the CPU RNG; reseed so both runs pick
            # the same anchors.
            torch.manual_seed(seed)
            losses = train_head.loss(cls_scores, bbox_preds, gt_bboxes, img_metas)
            for key in sorted(losses):
                for i, value in enumerate(losses[key]):
                    out[f"{name}/{gt_name}/{key}{i}"] = value.detach().float().cpu()

        # An image with no ground truth at all: every anchor is background, the
        # regression loss must still be a gradient-connected zero.
        cls_scores, bbox_preds = train_head(feats)
        torch.manual_seed(seed)
        losses = train_head.loss(
            cls_scores, bbox_preds, [torch.zeros(0, 4, device=device)], img_metas
        )
        for key in sorted(losses):
            for i, value in enumerate(losses[key]):
                out[f"{name}/nogt/{key}{i}"] = value.detach().float().cpu()

        # ---- targets, checked directly ------------------------------------
        # The loss is a scalar and can hide a compensating pair of errors in
        # assignment; the label/weight tensors cannot.
        #
        # Twice: MAMBA trains with allowed_border=0 (anchors crossing the image
        # edge are ignored), STPN with allowed_border=-1 (they are kept), which
        # is a separate branch of anchor_inside_flags.
        border_head = build_head(dict(head_cfg, train_cfg=ConfigDict(dict(TRAIN_CFG,
                                      allowed_border=-1)), test_cfg=ConfigDict(TEST_CFG)))
        border_head.load_state_dict(seeded_state_dict(border_head, seed))
        border_head = border_head.to(device).train()

        gt0 = [torch.tensor(GT_BBOXES[0], dtype=torch.float32, device=device)]
        featmap_sizes = [f.shape[-2:] for f in feats]
        for tag, head in [("tgt", train_head), ("border", border_head)]:
            if tag == "border":
                cls_scores, bbox_preds = head(feats)
                torch.manual_seed(seed)
                losses = head.loss(cls_scores, bbox_preds, gt0, img_metas)
                for key in sorted(losses):
                    for i, value in enumerate(losses[key]):
                        out[f"{name}/border/{key}{i}"] = value.detach().float().cpu()

            anchor_list, valid_flag_list = head.get_anchors(
                featmap_sizes, img_metas, device=device
            )
            torch.manual_seed(seed)
            targets = head.get_targets(
                anchor_list, valid_flag_list, gt0, img_metas, gt_labels_list=None,
                label_channels=1,
            )
            labels, label_weights, bbox_targets, bbox_weights, num_pos, num_neg = targets
            prefix = f"{name}/tgt" if tag == "tgt" else f"{name}/border/tgt"
            for i in range(len(labels)):
                out[f"{prefix}/labels{i}"] = labels[i].cpu()
                out[f"{prefix}/label_weights{i}"] = label_weights[i].cpu()
                out[f"{prefix}/bbox_targets{i}"] = bbox_targets[i].cpu()
                out[f"{prefix}/bbox_weights{i}"] = bbox_weights[i].cpu()
            out[f"{prefix}/num_pos"] = torch.tensor(num_pos)
            out[f"{prefix}/num_neg"] = torch.tensor(num_neg)

    print(f"RAN   {len(out)} artifacts on {device}")
    return out


def compare(path_a, path_b, atol, rtol):
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)

    failures, worst = [], 0.0
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            failures.append(f"{key}: only in {'A' if key in a else 'B'}")
            continue
        va, vb = a[key], b[key]
        if va.shape != vb.shape:
            failures.append(f"{key}: shape {tuple(va.shape)} != {tuple(vb.shape)}")
        elif not va.dtype.is_floating_point:
            if not torch.equal(va, vb):
                n = (va != vb).sum().item()
                failures.append(f"{key}: {n} of {va.numel()} entries differ")
        else:
            diff = (va - vb).abs().max().item()
            worst = max(worst, diff)
            if not torch.allclose(va, vb, atol=atol, rtol=rtol):
                failures.append(
                    f"{key}: max|diff| = {diff:.3e} (scale {va.abs().max().item():.3e})"
                )

    for note in failures:
        print(f"FAIL  {note}")
    print("-" * 70)
    print(f"largest difference across all artifacts: {worst:.3e}")
    if failures:
        print(f"{len(failures)} of {len(set(a) | set(b))} artifact(s) differ")
    else:
        print(f"RPN PARITY OK ({len(a)} artifacts)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-5)
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare, atol=args.atol, rtol=args.rtol)
    elif args.impl and args.out:
        torch.save(run(args.impl, args.device), args.out)
        print(f"saved -> {args.out}")
    else:
        ap.error("pass either --impl/--out or --compare")
