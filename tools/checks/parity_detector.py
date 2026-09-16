"""Parity check: ``vfe.models.detectors`` (``FasterRCNN``) vs the mmdet oracle.

The detectors are built from the real VID configs, loaded by each side's own
config loader, so config routing is under test too (``train_cfg.rpn`` ->
RPN head, ``rpn_proposal`` -> training proposals, ``rcnn`` -> RoI head):

* ``dc5``  -- MAMBA's detector: ResNet-101-DC5 + ChannelMapper. The RoI head is
  swapped for the plain ``StandardRoIHead`` / ``ConvFCBBoxHead`` that MAMBA's
  own heads subclass; those come in Phase 4f.
* ``swin`` -- STPN's detector: Swin-T + FPN, with the prompted
  ``STPNSwinTransformer`` replaced by the plain ``SwinTransformer`` until it is
  ported.

Three groups of artifacts:

* ``init/*`` -- the model after ``init_weights()``. The pretrained ResNet must
  load exactly (checked by float64 checksums plus leading values). Every other
  parameter is randomly initialised, so it is compared by *distribution*:
  constants exactly; random tensors by robust quantiles (see ``tensor_stats``
  for why not moments), which also separate a uniform init from a normal one.
  Plus ``requires_grad`` for every parameter (``frozen_stages``) and
  train/eval mode for every BatchNorm after ``.train()`` (``norm_eval``). A wrong init here would pass every inference
  check and still change training.
* ``train/*`` -- ``forward_train`` losses, the training proposals behind them,
  and ``parse_losses``. Sampling is seeded identically on both sides. CPU only;
  see the comment in ``run``.
* ``test/*`` -- ``simple_test_rpn`` proposals and ``simple_test`` detections
  with ``rescale=True``, compared as sets rather than position by position:
  detections whose scores are closer than the float noise can legitimately
  come out of NMS in either order.

For ``train`` and ``test`` the backbone is the pretrained ResNet (``dc5``) and
every other parameter is overwritten from a fixed CPU seed, scaled by fan-in
so activations stay O(1). Norm weights are ~1, not ~0: small norm weights
shrink every residual branch and the features collapse; features that are
too large saturate the sigmoid instead. Either way scores tie en masse, and
``run`` refuses training proposals with tied scores (see
``require_distinct``).

Tolerance is relative to each artifact's scale, 1e-5 by default, except
``test/swin/*`` at 1e-4. That is not slack for the port; it is measured. The
Swin backbone's ``nn.Linear`` layers put ~1.5e-6 of cross-version MKL noise
into the FPN features (the same floor as in ``parity_roi_head``). The RPN
turns that into proposal boxes ~2e-3 px apart, and RoIAlign sampling those
shifted boxes on random, spatially rough features gives RoI logits ~3e-5
apart. Injecting 1.5e-6 of noise into the features in a *single* env
reproduces it: ~2e-3 px and 3.8e-5, with the proposal order unchanged.
Features alone, with fixed proposals, move the logits only 1e-6 -- the
amplification is all in the box shift. DC5 has no such path: its backbone and
RPN are bit-exact.

Usage:
    conda run -n vfe       --no-capture-output python tools/checks/parity_detector.py --impl mmdet --out A.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_detector.py --impl vfe   --out B.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_detector.py --compare A.pt B.pt
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MAMBA_CFG = REPO_ROOT / "configs/vid/mamba/mamba_r101_dc5_3x.py"
STPN_CFG = REPO_ROOT / "configs/vid/stpn/stpn_swint_adam_9x.py"

IMG_SHAPE = (480, 640, 3)
IMG_METAS = [
    dict(
        img_shape=IMG_SHAPE,
        ori_shape=(384, 512, 3),
        pad_shape=IMG_SHAPE,
        scale_factor=np.array([1.25, 1.25, 1.25, 1.25], dtype=np.float32),
        flip=False,
        batch_input_shape=IMG_SHAPE[:2],
    )
    for _ in range(2)
]
GT = [
    ([[23.0, 41.0, 310.0, 420.0], [400.0, 60.0, 630.0, 330.0], [300.0, 300.0, 340.0, 350.0]],
     [3, 17, 29]),
    ([[100.0, 100.0, 400.0, 400.0], [110.0, 105.0, 405.0, 395.0], [500.0, 10.0, 620.0, 90.0]],
     [0, 0, 11]),
]


def fixed(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def encode_str(s):
    return torch.tensor(list(s.encode()), dtype=torch.uint8)


def detector_cfg(config_cls, name):
    """The ``model.detector`` sub-config, adapted as the module docstring says."""
    if name == "dc5":
        det = config_cls.fromfile(str(MAMBA_CFG)).model.detector
        det.roi_head.type = "StandardRoIHead"
        det.roi_head.bbox_head.type = "ConvFCBBoxHead"
        det.roi_head.bbox_head.pop("aggregator")
        det.roi_head.bbox_head.pop("topk")
    else:
        det = config_cls.fromfile(str(STPN_CFG)).model.detector
        det.backbone.type = "SwinTransformer"
        det.backbone.pop("prompt_cfg")
        # Random weights instead of downloading the Swin checkpoint.
        det.backbone.init_cfg = None
    return det


def seed_params(model, seed, skip_prefix=None, rpn_cls_scale=3.0):
    """Overwrite parameters (not buffers) from a fixed CPU seed; see the module
    docstring for the scaling."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            if skip_prefix is not None and name.startswith(skip_prefix):
                continue
            if param.ndim >= 2:
                std = 1.0 / math.sqrt(param.numel() // param.shape[0])
                if "fc_cls" in name:
                    # Spread the scores out, for the same reason as in
                    # parity_roi_head: near-ties make NMS order arbitrary.
                    std *= 3.0
                elif "rpn_cls" in name:
                    std *= rpn_cls_scale
                value = torch.randn(*param.shape, generator=g) * std
            elif name.endswith("weight"):
                value = 1.0 + 0.1 * torch.randn(*param.shape, generator=g)
            else:
                value = 0.1 * torch.randn(*param.shape, generator=g)
            param.copy_(value)


def tensor_stats(t):
    """``[numel, is_const, const_value, median, q50, q90]`` in float64, where
    q50 / q90 are quantiles of ``|t - median|``.

    Quantiles, not moments: ``trunc_normal_(std=0.02, a=-2, b=2)`` -- Swin's
    init, on both sides -- occasionally emits a weight of exactly +-2 (a
    uniform draw of exactly -1 goes through erfinv to -inf and is clamped to
    the absolute bound). One such 100-sigma value moves the std by a few
    percent and the kurtosis by hundreds, at random, so moments flag noise.
    The q90/q50 ratio still separates a normal init (2.44) from a uniform one
    (1.80).
    """
    t = t.detach().double().flatten().cpu()
    n = t.numel()
    if n == 0 or bool((t == t[0]).all()):
        return torch.tensor([n, 1.0, t[0].item() if n else 0.0, 0.0, 0.0, 0.0],
                            dtype=torch.float64)
    # kthvalue rather than torch.quantile, which refuses tensors over 16M
    # elements (the RoI head's first FC has 25M).
    median = t.kthvalue((n + 1) // 2).values
    dev = (t - median).abs()
    q50 = dev.kthvalue(max(1, round(0.5 * n))).values
    q90 = dev.kthvalue(max(1, round(0.9 * n))).values
    return torch.tensor([n, 0.0, 0.0, median.item(), q50.item(), q90.item()],
                        dtype=torch.float64)


def checksum(t):
    t = t.detach().double().flatten().cpu()
    head = t[:16]
    return torch.cat([torch.stack([t.sum(), t.abs().sum(), (t * t).sum()]), head])


def require_distinct(scores, what):
    """Refuse training proposals whose scores tie.

    mmcv's and torchvision's NMS break ties in different orders, and the RoI
    sampler draws proposals *by position*, so a tie would change which RoIs
    get sampled on the two sides. Test-time outputs don't need this: they
    are compared as sets (see ``match_detections``)."""
    scores = torch.as_tensor(scores).flatten()
    if len(torch.unique(scores)) != len(scores):
        raise RuntimeError(
            f"{what}: {len(scores)} scores but only {len(torch.unique(scores))} distinct; "
            "rescale the fixture so the logits do not saturate"
        )


def put_dets(out, prefix, dets, labels):
    dets = torch.as_tensor(dets).cpu()
    out[f"{prefix}/boxes"] = dets[:, :4]
    out[f"{prefix}/scores"] = dets[:, 4]
    out[f"{prefix}/labels"] = torch.as_tensor(labels).long().cpu()


def put_result(out, prefix, bbox_result):
    dets = torch.cat([torch.from_numpy(arr) for arr in bbox_result], 0)
    labels = torch.cat(
        [torch.full((len(arr),), c, dtype=torch.long) for c, arr in enumerate(bbox_result)]
    )
    put_dets(out, prefix, dets, labels)


def run(impl, device):
    if impl == "mmdet":
        from mmcv import Config

        from mmdet.core import bbox2roi
        from mmdet.models import build_detector
        from mmdet.models.detectors.base import BaseDetector

        def parse_losses(losses):
            return BaseDetector._parse_losses(None, losses)
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.config import Config
        from vfe.core import bbox2roi
        from vfe.models.builder import build_detector
        from vfe.models.detectors import parse_losses

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    out = {}
    gt_bboxes = [torch.tensor(b, dtype=torch.float32, device=device) for b, _ in GT]
    gt_labels = [torch.tensor(lab, dtype=torch.long, device=device) for _, lab in GT]

    for name in ("dc5", "swin"):
        seed = 900 if name == "dc5" else 950
        torch.manual_seed(0)
        model = build_detector(detector_cfg(Config, name))

        # ---- init -----------------------------------------------------------
        model.init_weights()
        named = sorted(model.named_parameters())
        out[f"init/{name}/param_names"] = encode_str("\n".join(n for n, _ in named))
        out[f"init/{name}/requires_grad"] = torch.tensor([p.requires_grad for _, p in named])
        for pname, param in named:
            if name == "dc5" and pname.startswith("backbone."):
                out[f"init/{name}/exact/{pname}"] = checksum(param)
            else:
                out[f"init/{name}/stats/{pname}"] = tensor_stats(param)
        if name == "dc5":
            for bname, buf in sorted(model.named_buffers()):
                if bname.startswith("backbone.") and buf.dtype.is_floating_point:
                    out[f"init/{name}/exact/{bname}"] = checksum(buf)

        model.train()
        bn_modes = [
            (mname, m.training)
            for mname, m in sorted(model.named_modules())
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        ]
        out[f"init/{name}/bn_names"] = encode_str("\n".join(n for n, _ in bn_modes))
        out[f"init/{name}/bn_training"] = torch.tensor([t for _, t in bn_modes], dtype=torch.bool)

        # ---- train / test weights --------------------------------------------
        # dc5 keeps the pretrained backbone just loaded; swin has none.
        # Random Swin + FPN features are large enough to saturate the RPN's
        # sigmoid at the dc5 scale, so its classifier is scaled down instead.
        seed_params(model, seed, skip_prefix="backbone." if name == "dc5" else None,
                    rpn_cls_scale=3.0 if name == "dc5" else 0.1)
        model = model.to(device)
        img = fixed((2, 3) + IMG_SHAPE[:2], seed + 1).to(device)

        # Training parity only on CPU, where DC5's backbone and RPN are
        # bit-exact across torch versions. On CUDA, cuDNN differs by ~1e-6,
        # which reorders near-tied proposals; the RoI sampler draws by
        # position, so it would sample different RoIs and the losses would
        # differ for reasons that have nothing to do with the port.
        if name == "dc5" and device == "cpu":
            model.train()
            with torch.no_grad():
                x = model.extract_feat(img)
                proposals = model.rpn_head.get_bboxes(
                    *model.rpn_head(x), img_metas=IMG_METAS,
                    cfg=model.train_cfg.rpn_proposal,
                )
                for i, props in enumerate(proposals):
                    require_distinct(props[:, 4], f"{name} training proposals, image {i}")
                    out[f"train/{name}/proposals{i}"] = props.cpu()

                torch.manual_seed(seed)
                losses = model.forward_train(img, IMG_METAS, gt_bboxes, gt_labels)
            for key in sorted(losses):
                values = losses[key] if isinstance(losses[key], list) else [losses[key]]
                for i, value in enumerate(values):
                    out[f"train/{name}/{key}{i}"] = value.detach().float().reshape(-1).cpu()
            loss, log_vars = parse_losses(losses)
            out[f"train/{name}/parsed/loss"] = loss.detach().float().reshape(-1).cpu()
            out[f"train/{name}/parsed/keys"] = encode_str("\n".join(log_vars))
            out[f"train/{name}/parsed/values"] = torch.tensor(list(log_vars.values()))

        model.eval()
        with torch.no_grad():
            x = model.extract_feat(img)
            # Intermediates, so a difference in the detections can be traced to
            # the stage where it opened up. Subsampled: the full FPN maps are
            # tens of MB.
            for i, feat in enumerate(x):
                out[f"test/{name}/feat{i}"] = feat[:, :, ::4, ::4].cpu()
            proposal_list = model.rpn_head.simple_test_rpn(x, IMG_METAS)
            for i, props in enumerate(proposal_list):
                put_dets(out, f"test/{name}/rpn{i}", props, torch.zeros(len(props)))
            # The RoI head's raw outputs, one row per proposal. Stored as a
            # matched set keyed by (image, proposal box), like the detections,
            # because proposal order is not stable across versions either.
            rois = bbox2roi(proposal_list)
            bbox_results = model.roi_head._bbox_forward(x, rois)
            out[f"test/{name}/roi_head/boxes"] = rois[:, 1:].cpu()
            out[f"test/{name}/roi_head/scores"] = torch.cat([q[:, 4] for q in proposal_list]).cpu()
            out[f"test/{name}/roi_head/labels"] = rois[:, 0].long().cpu()
            out[f"test/{name}/roi_head/payload"] = torch.cat(
                [bbox_results["cls_score"], bbox_results["bbox_pred"]], 1
            ).cpu()
            results = model.simple_test(img, IMG_METAS, rescale=True)
        for i, res in enumerate(results):
            put_result(out, f"test/{name}/det{i}", res)

        del model, x

    print(f"RAN   {len(out)} artifacts on {device}")
    return out


def compare_stats(key, sa, sb):
    """Distributional comparison for ``init/*/stats`` artifacts."""
    n = sa[0].item()
    if sa[0] != sb[0] or sa[1] != sb[1]:
        return f"{key}: numel/constness differ ({sa[:2].tolist()} vs {sb[:2].tolist()})"
    if sa[1] == 1:
        if sa[2] != sb[2]:
            return f"{key}: constant {sa[2].item()} vs {sb[2].item()}"
        return None
    if n < 1000:
        return None  # too few samples for the quantiles to mean anything
    med_a, q50_a, q90_a = sa[3:].tolist()
    med_b, q50_b, q90_b = sb[3:].tolist()
    # Quantile standard errors are O(1/sqrt(n)); 8/sqrt(n) keeps a false alarm
    # vanishingly rare while still resolving a 10% change in scale from ~6k
    # elements up.
    tol = 8.0 / math.sqrt(n)
    problems = []
    if abs(q50_a - q50_b) > tol * q50_b:
        problems.append(f"median |dev| {q50_a:.4g} vs {q50_b:.4g}")
    if abs(q90_a - q90_b) > tol * q90_b:
        problems.append(f"90th pct |dev| {q90_a:.4g} vs {q90_b:.4g}")
    if abs(med_a - med_b) > tol * 1.4826 * q50_b:
        problems.append(f"median {med_a:.4g} vs {med_b:.4g}")
    return f"{key}: " + ", ".join(problems) if problems else None


# Measured amplification, not slack; see the module docstring.
RTOL_OVERRIDES = {"test/swin/": 1e-4}


def match_detections(prefix, a, b, rtol, atol):
    """Compare one ``{prefix}/boxes|scores|labels[|payload]`` group as a *set*.

    NMS output is sorted by score, so near-tied detections -- scores closer
    than the cross-version float noise -- legitimately come out in either
    order. Each detection on side A must pair with a distinct one on side B
    that has the same label and a box and score within tolerance. An optional
    ``payload`` (extra per-row values, e.g. logits) must then agree too.
    Returns ``(failure message or None, worst relative difference)``.
    """
    ba, sa, la = a[f"{prefix}/boxes"], a[f"{prefix}/scores"], a[f"{prefix}/labels"]
    bb, sb, lb = b[f"{prefix}/boxes"], b[f"{prefix}/scores"], b[f"{prefix}/labels"]
    if len(ba) != len(bb):
        return f"{prefix}: {len(ba)} vs {len(bb)} detections", 0.0
    if len(ba) == 0:
        return None, 0.0
    box_scale = ba.abs().max().item()
    score_scale = sa.abs().max().item()
    box_tol = atol + rtol * box_scale
    score_tol = atol + rtol * score_scale

    box_dist = (ba[:, None, :] - bb[None, :, :]).abs().amax(-1)
    score_dist = (sa[:, None] - sb[None, :]).abs()
    ok = (la[:, None] == lb[None, :]) & (box_dist <= box_tol) & (score_dist <= score_tol)
    payload_key = f"{prefix}/payload"
    payload_dist, payload_scale = None, 1.0
    if payload_key in a:
        pa, pb = a[payload_key], b[payload_key]
        payload_scale = pa.abs().max().item()
        payload_dist = (pa[:, None, :] - pb[None, :, :]).abs().amax(-1)
        ok &= payload_dist <= atol + rtol * payload_scale
    # Greedy is enough: at these tolerances a detection has at most a
    # handful of candidates, and they are its own near-duplicates.
    used = torch.zeros(len(bb), dtype=torch.bool)
    unmatched, worst = [], 0.0
    for i in torch.argsort(sa, descending=True).tolist():
        candidates = (ok[i] & ~used).nonzero().flatten()
        if len(candidates) == 0:
            unmatched.append(i)
            continue
        j = candidates[torch.argmin(box_dist[i, candidates])].item()
        used[j] = True
        worst = max(worst, box_dist[i, j].item() / box_scale, score_dist[i, j].item() / score_scale)
        if payload_dist is not None:
            worst = max(worst, payload_dist[i, j].item() / payload_scale)
    if unmatched:
        return (f"{prefix}: {len(unmatched)} of {len(ba)} detections have no match within "
                f"{rtol:.0e} of scale (first: A[{unmatched[0]}])"), worst
    return None, worst


def compare(path_a, path_b, atol, rtol):
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)

    failures, exact, worst, worst_key = [], 0, 0.0, None
    keys = sorted(set(a) | set(b))
    det_prefixes = sorted({k[: -len("/boxes")] for k in keys if k.endswith("/boxes")})
    det_keys = {f"{p}/{f}" for p in det_prefixes for f in ("boxes", "scores", "labels", "payload")}
    for prefix in det_prefixes:
        if not all(k in a and k in b for k in (f"{prefix}/boxes", f"{prefix}/scores",
                                                f"{prefix}/labels")):
            failures.append(f"{prefix}: detection group incomplete on one side")
            continue
        prefix_rtol = max([rtol] + [v for k, v in RTOL_OVERRIDES.items() if prefix.startswith(k)])
        problem, ratio = match_detections(prefix, a, b, prefix_rtol, atol)
        if problem:
            failures.append(problem)
        if ratio > worst:
            worst, worst_key = ratio, prefix
    for key in keys:
        if key in det_keys:
            continue
        if key not in a or key not in b:
            failures.append(f"{key}: only in {'A' if key in a else 'B'}")
            continue
        va, vb = a[key], b[key]
        if va.shape != vb.shape:
            failures.append(f"{key}: shape {tuple(va.shape)} != {tuple(vb.shape)}")
        elif "/stats/" in key:
            problem = compare_stats(key, va, vb)
            if problem:
                failures.append(problem)
        elif not va.dtype.is_floating_point or "/exact/" in key:
            if torch.equal(va, vb):
                exact += 1
            else:
                n = (va != vb).sum().item()
                failures.append(f"{key}: {n} of {va.numel()} entries differ (exact required)")
        elif va.numel() == 0 or torch.equal(va, vb):
            exact += 1
        else:
            diff = (va - vb).abs().max().item()
            scale = va.abs().max().item()
            ratio = diff / scale if scale > 0 else float("inf")
            if ratio > worst:
                worst, worst_key = ratio, key
            key_rtol = max([rtol] + [v for k, v in RTOL_OVERRIDES.items() if key.startswith(k)])
            if diff > atol + key_rtol * scale:
                failures.append(f"{key}: max|diff| = {diff:.3e} = {ratio:.2e} of scale {scale:.3e}")

    for note in failures:
        print(f"FAIL  {note}")
    print("-" * 70)
    print(f"bit-exact: {exact} of {len(keys) - len(det_keys & set(keys))} "
          f"(init/*/stats compared by distribution, {len(det_prefixes)} detection sets matched)")
    print(f"largest difference relative to its artifact's scale: {worst:.2e} ({worst_key})")
    if failures:
        print(f"{len(failures)} of {len(keys)} artifact(s) differ")
    else:
        print(f"DETECTOR PARITY OK ({len(keys)} artifacts)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--rtol", type=float, default=1e-5, help="relative to each artifact's max")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare, atol=args.atol, rtol=args.rtol)
    elif args.impl and args.out:
        torch.save(run(args.impl, args.device), args.out)
        print(f"saved -> {args.out}")
    else:
        ap.error("pass either --impl/--out or --compare")
