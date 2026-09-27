"""Parity check: ``vfe.models.dense_heads.FCOSHead`` vs the mmdet oracle.

FCOS is the detector EOVOD is built on, and none of it existed in 2.0. The
check covers the head's three jobs, since they fail independently:

* ``forward`` -- the GroupNorm towers, the per-level ``Scale`` and ``exp``.
  Weights are generated on CPU from a fixed seed and loaded into both heads.
* ``get_bboxes`` -- point decode, score threshold and top-k, centerness as
  score factor, NMS. Compared before NMS (deterministic order) and after.
* ``loss`` -- the size-range assignment, centerness targets, focal / IoU /
  BCE losses and every parameter gradient, on a two-image batch.

Also the pieces under it: ``MlvlPointGenerator`` grids and valid flags, the
distance coder, and FocalLoss / IoULoss on their own.

Usage:
    conda run -n vfe       --no-capture-output python tools/checks/parity_fcos.py --impl mmdet --out ~/fcos_mmdet.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_fcos.py --impl vfe   --out ~/fcos_vfe.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_fcos.py --compare ~/fcos_mmdet.pt ~/fcos_vfe.pt

Both sides run on CPU here (gradients are only trusted on CPU, see
docs/parity.md); ``--device cuda`` compares forward passes on a GPU too. The
mmdet side needs ``mmdet`` importable: the v1.0.0 checkout on ``PYTHONPATH``.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

HEAD = dict(
    type="FCOSHead",
    num_classes=30,
    in_channels=256,
    stacked_convs=4,
    feat_channels=256,
    strides=[8, 16, 32, 64, 128],
    loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
    loss_bbox=dict(type="IoULoss", loss_weight=1.0),
    loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
)
TRAIN_CFG = dict(allowed_border=-1, pos_weight=-1, debug=False)
TEST_CFG = dict(nms_pre=1000, min_bbox_size=0, score_thr=0.05,
                nms=dict(type="nms", iou_threshold=0.5), max_per_img=100)

# A 600x992 image padded to 608x992, through FPN P3-P7.
IMG_SHAPE = (600, 992, 3)
PAD_SHAPE = (608, 992, 3)
FEATMAPS = [(2, 256, 76, 124), (2, 256, 38, 62), (2, 256, 19, 31), (2, 256, 10, 16),
            (2, 256, 5, 8)]

GT_BBOXES = [
    [[23.0, 41.0, 310.0, 520.0], [400.0, 60.0, 900.0, 430.0], [700.0, 300.0, 740.0, 350.0]],
    [[100.0, 100.0, 500.0, 500.0], [110.0, 105.0, 505.0, 495.0], [0.0, 0.0, 30.0, 25.0]],
]
GT_LABELS = [[3, 17, 29], [0, 0, 12]]


def fixed(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def seeded_state_dict(head, seed):
    """One deterministic weight per parameter name; the names match across
    stacks, so both heads receive identical weights. The classifier's weights
    are larger so scores spread out and NMS has ties to break rarely."""
    state = {}
    g = torch.Generator().manual_seed(seed)
    for name, param in sorted(head.state_dict().items()):
        std = 0.3 if name.startswith("conv_cls.weight") else 0.02
        state[name] = torch.randn(tuple(param.shape), generator=g) * std
        if name.startswith("scales."):
            state[name] = torch.ones_like(param)
        if name.endswith("gn.weight") or name.endswith("norm.weight"):
            state[name] = 1.0 + state[name]
    return state


def build(impl, cfg_train, cfg_test):
    if impl == "mmdet":
        from mmcv import ConfigDict
        from mmdet.models import build_head

        head = build_head(dict(HEAD, train_cfg=ConfigDict(cfg_train), test_cfg=ConfigDict(cfg_test)))
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.models.builder import build_head

        head = build_head(dict(HEAD, train_cfg=cfg_train, test_cfg=cfg_test))
    return head


def sort_dets(det_bboxes, det_labels):
    """Detections in a stack-independent order: label, then score (descending),
    then x1. Sorting by score alone lets near-tied scores land either way."""
    import numpy as np

    b = det_bboxes.detach().cpu().numpy()
    lbl = det_labels.detach().cpu().numpy()
    order = torch.as_tensor(np.lexsort((b[:, 0], -b[:, 4], lbl)))
    return det_bboxes[order], det_labels[order]


def run(impl, device, dtype="float32"):
    out = {}
    device = torch.device(device)
    dtype = getattr(torch, dtype)
    head = build(impl, TRAIN_CFG, TEST_CFG)
    head.load_state_dict(seeded_state_dict(head, 1))
    head = head.to(device=device, dtype=dtype).eval()

    feats = [fixed(shape, 100 + i).to(device=device, dtype=dtype)
             for i, shape in enumerate(FEATMAPS)]
    metas = [dict(img_shape=IMG_SHAPE, pad_shape=PAD_SHAPE, ori_shape=IMG_SHAPE,
                  scale_factor=torch.tensor([1.6, 1.6, 1.6, 1.6]).numpy(), flip=False)] * 2

    # ---- forward ----
    with torch.no_grad():
        cls_scores, bbox_preds, centernesses = head(feats)
    for i in range(len(FEATMAPS)):
        out[f"forward/cls{i}"] = cls_scores[i].cpu()
        out[f"forward/reg{i}"] = bbox_preds[i].cpu()
        out[f"forward/ctr{i}"] = centernesses[i].cpu()

    # ---- points ----
    points = head.prior_generator.grid_priors([f.shape[-2:] for f in feats], device=device)
    for i, p in enumerate(points):
        out[f"points/level{i}"] = p.cpu()
    flags = head.prior_generator.valid_flags([f.shape[-2:] for f in feats], PAD_SHAPE, device=device)
    for i, f in enumerate(flags):
        out[f"points/valid{i}"] = f.cpu()

    # ---- decode ----
    with torch.no_grad():
        raw = head.get_bboxes(cls_scores, bbox_preds, centernesses, img_metas=metas,
                              rescale=False, with_nms=False)
        for img_id, (bboxes, scores, labels) in enumerate(raw):
            out[f"decode/nonms{img_id}/bboxes"] = bboxes.cpu()
            out[f"decode/nonms{img_id}/scores"] = scores.cpu()
            out[f"decode/nonms{img_id}/labels"] = labels.cpu()
        # mmcv's compiled NMS is float32-only, so the post-NMS detections are
        # compared in float32 runs only; float64 is for the gradients.
        if dtype == torch.float32:
            dets = head.get_bboxes(cls_scores, bbox_preds, centernesses, img_metas=metas,
                                   rescale=True, with_nms=True)
            for img_id, (det_bboxes, det_labels) in enumerate(dets):
                det_bboxes, det_labels = sort_dets(det_bboxes, det_labels)
                out[f"decode/dets{img_id}/bboxes"] = det_bboxes.cpu()
                out[f"decode/dets{img_id}/labels"] = det_labels.cpu()

    # ---- loss and gradients ----
    head.train()
    head.zero_grad()
    gt_bboxes = [torch.tensor(b, device=device, dtype=dtype) for b in GT_BBOXES]
    gt_labels = [torch.tensor(lbl, device=device) for lbl in GT_LABELS]
    cls_scores, bbox_preds, centernesses = head(feats)
    labels, bbox_targets = head.get_targets(points, gt_bboxes, gt_labels)
    for i in range(len(FEATMAPS)):
        out[f"targets/labels{i}"] = labels[i].cpu()
        out[f"targets/bbox{i}"] = bbox_targets[i].cpu()
    losses = head.loss(cls_scores, bbox_preds, centernesses, gt_bboxes, gt_labels, metas)
    total = sum(losses.values())
    total.backward()
    for k, v in losses.items():
        out[f"loss/{k}"] = v.detach().cpu()
    for name, param in head.named_parameters():
        if param.grad is not None:
            out[f"grad/{name}"] = param.grad.detach().cpu()

    # ---- the losses alone ----
    if impl == "mmdet":
        from mmdet.models import build_loss
    else:
        from vfe.models.builder import build_loss
    focal = build_loss(HEAD["loss_cls"])
    iou = build_loss(HEAD["loss_bbox"])
    pred = fixed((64, 30), 7, 2.0).to(device=device, dtype=dtype)
    target = torch.arange(64, device=device) % 31  # 30 == background
    out["losses/focal_mean"] = focal(pred, target).cpu()
    out["losses/focal_avg"] = focal(pred, target, avg_factor=10.0).cpu()
    boxes_a = fixed((16, 4), 8, 50.0).abs().to(device=device, dtype=dtype)
    boxes_a[:, 2:] += boxes_a[:, :2] + 1
    boxes_b = boxes_a + fixed((16, 4), 9, 5.0).to(device=device, dtype=dtype)
    boxes_b[:, 2:] = torch.maximum(boxes_b[:, 2:], boxes_b[:, :2] + 1)
    weight = fixed((16,), 10).abs().to(device=device, dtype=dtype)
    out["losses/iou"] = iou(boxes_a, boxes_b, weight=weight, avg_factor=weight.sum()).cpu()
    return out


def compare(path_a, path_b, atol, rtol):
    """Each float artifact is judged against its own scale:
    ``max|a-b| <= atol + rtol * max(|a|, |b|)``. Integer and bool artifacts
    (labels, valid flags, targets) must be identical."""
    a = torch.load(path_a, weights_only=False)
    b = torch.load(path_b, weights_only=False)
    if set(a) != set(b):
        print("artifact key sets differ:", sorted(set(a) ^ set(b)))
        return False
    ok, exact = True, 0
    worst = {}  # group -> (relative difference, key)
    for key in sorted(a):
        va, vb = a[key], b[key]
        group = key.split("/")[0]
        if va.shape != vb.shape:
            print(f"FAIL {key}: shape {tuple(va.shape)} vs {tuple(vb.shape)}")
            ok = False
            continue
        if va.dtype in (torch.bool, torch.int64, torch.int32):
            if not torch.equal(va, vb):
                print(f"FAIL {key}: integer/bool tensors differ")
                ok = False
            else:
                exact += 1
            continue
        va, vb = va.double(), vb.double()
        if torch.equal(va, vb):
            exact += 1
            continue
        diff = (va - vb).abs().max().item()
        scale = max(va.abs().max().item(), vb.abs().max().item(), 1e-12)
        rel = diff / scale
        if rel > worst.get(group, (0.0, ""))[0]:
            worst[group] = (rel, key)
        if diff > atol + rtol * scale:
            print(f"FAIL {key}: max |a-b| = {diff:.3e} on scale {scale:.3e} ({rel:.2e} relative)")
            ok = False
    print(f"{len(a)} artifacts, {exact} bit-exact")
    for group, (rel, key) in sorted(worst.items()):
        print(f"  {group:8s} worst relative difference {rel:.2e} ({key})")
    print("FCOS PARITY OK" if ok else "FCOS PARITY FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="float64 separates porting errors from float32 accumulation noise")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-3,
                    help="relative to each artifact's own scale; the training-step convention")
    args = ap.parse_args()
    if args.compare:
        sys.exit(0 if compare(args.compare[0], args.compare[1], args.atol, args.rtol) else 1)
    if not args.impl or not args.out:
        ap.error("--impl and --out are required unless --compare is given")
    out = run(args.impl, args.device, args.dtype)
    torch.save(out, args.out)
    print(f"{args.impl}: {len(out)} artifacts -> {args.out}")


if __name__ == "__main__":
    main()
