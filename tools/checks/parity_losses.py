"""Parity check: ``vfe.models.losses`` vs the ``mmdet.models.losses`` oracle.

Same two-process pattern as the other harnesses. Both the loss *value* and its
*gradient w.r.t. the prediction* are compared -- a loss can be right and still
train wrong if the weighting is applied after reduction, and only the gradient
notices.

Usage:
    conda run -n vfe       --no-capture-output python tools/checks/parity_losses.py --impl mmdet --out ~/loss_mmdet.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_losses.py --impl vfe   --out ~/loss_vfe.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_losses.py --compare ~/loss_mmdet.pt ~/loss_vfe.pt

Tolerance defaults to 1e-6 rather than 0: ``F.cross_entropy`` and
``binary_cross_entropy_with_logits`` are fused kernels whose internals changed
between torch 1.10 and 2.10, so a few ULP of difference here is the framework,
not the port. Everything built from plain elementwise arithmetic
(``smooth_l1``, ``l1``) does come out bit-exact, and the comparison prints the
observed difference so a real regression stands out from the noise floor.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

NUM_ROIS = 512
NUM_CLASSES = 30


def fixed(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def run(impl, device):
    if impl == "mmdet":
        from mmdet.models import build_loss
        from mmdet.models.losses import accuracy
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.models import accuracy
        from vfe.models.builder import build_loss

    out = {}

    def record(tag, loss, pred):
        """Store the loss and d(loss)/d(pred). Both matter; see module docstring."""
        out[f"{tag}/value"] = loss.detach().float().cpu()
        if loss.requires_grad and loss.numel() == 1:
            (grad,) = torch.autograd.grad(loss, pred, retain_graph=False)
            out[f"{tag}/grad"] = grad.detach().float().cpu()

    # ---- softmax cross entropy (the RoI head's loss_cls) -------------------
    # (num_classes + 1) channels: the last is background.
    labels = (torch.arange(NUM_ROIS) * 7919 % (NUM_CLASSES + 1)).to(device)
    # Mimic the real label distribution: mostly background, some foreground.
    labels = torch.where(torch.arange(NUM_ROIS, device=device) % 4 == 0, labels,
                         torch.full_like(labels, NUM_CLASSES))
    ce_weights = (torch.arange(NUM_ROIS, device=device) % 3 != 0).float()

    for tag, cfg, kwargs in [
        ("ce/plain", dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0), {}),
        ("ce/weighted", dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
         dict(weight=ce_weights)),
        # avg_factor is the RoI head's actual call: divide by the number of
        # sampled RoIs, not by the element count.
        ("ce/avgfactor", dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
         dict(weight=ce_weights, avg_factor=float(ce_weights.sum()))),
        ("ce/lw2", dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=2.5), {}),
        ("ce/sum", dict(type="CrossEntropyLoss", use_sigmoid=False, reduction="sum"), {}),
        ("ce/none", dict(type="CrossEntropyLoss", use_sigmoid=False, reduction="none"), {}),
        ("ce/classw", dict(
            type="CrossEntropyLoss", use_sigmoid=False,
            class_weight=[1.0 + 0.5 * (i % 3) for i in range(NUM_CLASSES + 1)]), {}),
    ]:
        pred = fixed((NUM_ROIS, NUM_CLASSES + 1), 101, scale=3.0).to(device).requires_grad_(True)
        record(tag, build_loss(dict(cfg))(pred, labels, **kwargs), pred)

    # ignore_index: -100 labels must contribute neither loss nor gradient.
    pred = fixed((NUM_ROIS, NUM_CLASSES + 1), 101, scale=3.0).to(device).requires_grad_(True)
    ignored = torch.where(torch.arange(NUM_ROIS, device=device) % 5 == 0,
                          torch.full_like(labels, -100), labels)
    loss_fn = build_loss(dict(type="CrossEntropyLoss", use_sigmoid=False, ignore_index=-100))
    record("ce/ignore", loss_fn(pred, ignored), pred)

    # ---- sigmoid BCE (the RPN's loss_cls) ----------------------------------
    num_anchors = 37 * 62 * 12
    bin_labels = (torch.arange(num_anchors, device=device) % 11 == 0).long()
    # The RPN weights anchors 1/0 by whether they were sampled at all.
    bin_weights = (torch.arange(num_anchors, device=device) % 3 != 0).float()
    for tag, cfg, kwargs in [
        ("bce/plain", dict(type="CrossEntropyLoss", use_sigmoid=True), {}),
        ("bce/weighted", dict(type="CrossEntropyLoss", use_sigmoid=True),
         dict(weight=bin_weights)),
        ("bce/avgfactor", dict(type="CrossEntropyLoss", use_sigmoid=True),
         dict(weight=bin_weights, avg_factor=256.0)),
        ("bce/none", dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="none"), {}),
    ]:
        pred = fixed((num_anchors, 1), 102, scale=2.0).to(device).requires_grad_(True)
        record(tag, build_loss(dict(cfg))(pred, bin_labels, **kwargs), pred)

    # Labels already one-hot (N, C) rather than (N,), the other BCE entry point.
    pred = fixed((NUM_ROIS, NUM_CLASSES), 103, scale=2.0).to(device).requires_grad_(True)
    onehot = torch.zeros(NUM_ROIS, NUM_CLASSES, device=device)
    onehot[torch.arange(NUM_ROIS, device=device), labels.clamp(max=NUM_CLASSES - 1)] = 1
    record("bce/onehot", build_loss(dict(type="CrossEntropyLoss", use_sigmoid=True))(
        pred, onehot), pred)

    # ---- smooth L1 / L1 (the regression losses) ----------------------------
    target = fixed((NUM_ROIS, 4), 201, scale=0.5).to(device)
    # Per-element weight, zero on negatives: exactly how the heads mask out
    # background boxes, which have no regression target at all.
    reg_weight = torch.zeros(NUM_ROIS, 4, device=device)
    reg_weight[: NUM_ROIS // 4] = 1.0

    for tag, cfg, kwargs in [
        # beta=1/9 is the RPN's, beta=1.0 the RoI head's.
        ("sl1/rpn", dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0), {}),
        ("sl1/rcnn", dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0), {}),
        ("sl1/weighted", dict(type="SmoothL1Loss", beta=1.0), dict(weight=reg_weight)),
        ("sl1/avgfactor", dict(type="SmoothL1Loss", beta=1.0),
         dict(weight=reg_weight, avg_factor=float(NUM_ROIS // 4))),
        ("sl1/none", dict(type="SmoothL1Loss", beta=1.0, reduction="none"), {}),
        ("sl1/sum", dict(type="SmoothL1Loss", beta=1.0, reduction="sum"), {}),
        ("l1/plain", dict(type="L1Loss"), {}),
        ("l1/weighted", dict(type="L1Loss"), dict(weight=reg_weight)),
    ]:
        # scale 3.0 straddles beta in both directions, so both arms of the
        # piecewise function are exercised.
        pred = fixed((NUM_ROIS, 4), 202, scale=3.0).to(device).requires_grad_(True)
        record(tag, build_loss(dict(cfg))(pred, target, **kwargs), pred)

    # An image with no positives: the loss must be a gradient-connected zero,
    # not a detached constant, or DDP reports unused parameters.
    pred = fixed((0, 4), 203).to(device).requires_grad_(True)
    empty_loss = build_loss(dict(type="SmoothL1Loss", beta=1.0))(pred, target[:0])
    out["sl1/empty/value"] = empty_loss.detach().float().cpu()
    out["sl1/empty/requires_grad"] = torch.tensor(empty_loss.requires_grad)

    # ---- accuracy ----------------------------------------------------------
    pred = fixed((NUM_ROIS, NUM_CLASSES + 1), 301, scale=3.0).to(device)
    out["acc/top1"] = accuracy(pred, labels).float().cpu()
    out["acc/top5"] = torch.stack(accuracy(pred, labels, topk=(1, 5))).float().cpu()
    out["acc/thresh"] = accuracy(pred, labels, topk=1, thresh=0.5).float().cpu()
    out["acc/empty"] = accuracy(pred[:0], labels[:0]).float().cpu()

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
                failures.append(f"{key}: {va.tolist()} != {vb.tolist()}")
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
        print(f"LOSS PARITY OK ({len(a)} artifacts)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-6)
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare, atol=args.atol, rtol=args.rtol)
    elif args.impl and args.out:
        torch.save(run(args.impl, args.device), args.out)
        print(f"saved -> {args.out}")
    else:
        ap.error("pass either --impl/--out or --compare")
