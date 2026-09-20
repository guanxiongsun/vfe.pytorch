"""Parity check: ``vfe.models.roi_heads`` vs the mmdet oracle.

Two RoI heads, taken from the real configs: MAMBA's (one DC5 level at stride
16, 512 channels, ``sampling_ratio=2``) and STPN's (four FPN levels, 256
channels, ``sampling_ratio=0``). Both run a batch of two images, so the
``bbox2roi`` batch index and the per-image split on the way out are exercised.

What is compared, and why each is separate:

* ``meta/*/state_keys`` -- the state-dict key list, byte for byte. Matching
  keys is what lets the released checkpoints load, and every numeric check
  below silently depends on it.
* ``ext/*`` -- ``SingleRoIExtractor`` forward *and backward*. The backward pass
  goes through ``torchvision.ops.roi_align`` on the vfe side and mmcv's kernel
  on the other; the Phase 2 ops check only covered forward. Includes the FPN
  level routing, ``roi_scale_factor``, and a batch whose RoIs all land on one
  level, which is where the zero-valued keep-in-graph term matters.
* ``bbox/*`` -- ``Shared2FCBBoxHead`` forward, class-specific and agnostic.
* ``dec/*`` -- ``get_bboxes`` with and without NMS, and with ``rescale``.
* ``train/*`` -- ``forward_train`` losses, the targets behind them, and the
  gradient w.r.t. every head parameter and every feature level.
* ``test/*`` -- ``simple_test`` end to end, including an image with no
  proposals.

Weights are generated on CPU from a fixed seed and loaded into both sides. The
classifier's weights are scaled up so the logits spread out: at the default
scale softmax scores all sit near 1/31, thousands of them within 1e-7 of each
other, and ``torch.softmax``'s known ~2e-7 drift between torch 1.10 and 2.10
would reorder NMS output for reasons that have nothing to do with the port.

Tolerance is judged per artifact against that artifact's own magnitude:
``max|a - b| <= atol + rtol * max|a|``. Gradients here span 1e-4 to 1e+1, so a
single absolute tolerance is either blind on the small ones or failing on the
large ones. Detections are stored as separate boxes / scores / labels for the
same reason -- a score error must not hide behind 640-pixel coordinates.

What the observed differences are (all verified outside the model):

* **CPU:** every integer artifact, every extractor artifact (RoIAlign forward
  *and* backward) and every target is bit-exact. The rest differ by <=1e-6 of
  their scale, all downstream of ``nn.Linear``: a bare ``F.linear`` on
  bit-identical inputs differs by ~9e-7 relative between torch 1.10's and
  2.10's bundled MKL. CPU is the reference for gradients.
* **CUDA:** the same ~1e-6 floor, except that the *legacy* stack gets some
  backward passes wrong. On an RTX 4060 (sm_89, newer than the cu113 build it
  runs on), torch 1.10.1+cu113 computes the ``dc5_agn`` head's ``shared_fcs``
  gradients up to 1.3% off. A plain functional replay with no mmdet or mmcv code
  reproduces the error exactly on torch 1.10 and not on 2.10, and the vfe
  values match a float64 CPU reference to <1e-6. The trigger depends on
  allocation history -- every op involved is correct in isolation -- so it is
  documented rather than pinned to one kernel. Hence ``--skip-train-grads``
  for CUDA comparisons: gradients are checked on CPU, where the oracle is sound.

Usage:
    conda run -n vfe       --no-capture-output python tools/checks/parity_roi_head.py --impl mmdet --out A.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_roi_head.py --impl vfe   --out B.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_roi_head.py --compare A.pt B.pt
    # on CUDA (add --device cuda to both runs):
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_roi_head.py --compare A.pt B.pt --skip-train-grads
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

NUM_CLASSES = 30
IMG_SHAPE = (480, 640, 3)
# mmdet stores scale_factor as a float32 array of (w, h, w, h).
SCALE_FACTOR = np.array([1.25, 1.25, 1.25, 1.25], dtype=np.float32)


def bbox_head_cfg(in_channels, target_stds, beta, agnostic=False):
    return dict(
        type="Shared2FCBBoxHead",
        in_channels=in_channels,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=NUM_CLASSES,
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder", target_means=[0.0, 0.0, 0.0, 0.0], target_stds=target_stds
        ),
        reg_class_agnostic=agnostic,
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=beta, loss_weight=1.0),
    )


HEADS = [
    # (name, extractor cfg, bbox head cfg, feature shapes)
    (
        "dc5",
        dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=2),
            out_channels=512,
            featmap_strides=[16],
        ),
        bbox_head_cfg(512, [0.2, 0.2, 0.2, 0.2], 1.0),
        [(2, 512, 30, 40)],
    ),
    (
        "fpn",
        dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head_cfg(256, [0.2, 0.2, 0.2, 0.2], 1.0 / 9.0),
        [(2, 256, 120, 160), (2, 256, 60, 80), (2, 256, 30, 40), (2, 256, 15, 20)],
    ),
    (
        # Not used by any config, but it is a separate branch in both the loss
        # (one box per RoI instead of one per class) and the decode.
        "dc5_agn",
        dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=2),
            out_channels=512,
            featmap_strides=[16],
        ),
        bbox_head_cfg(512, [0.2, 0.2, 0.2, 0.2], 1.0, agnostic=True),
        [(2, 512, 30, 40)],
    ),
]
SEEDS = {"dc5": 600, "fpn": 700, "dc5_agn": 800}

TRAIN_CFG = dict(
    assigner=dict(
        type="MaxIoUAssigner",
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.5,
        match_low_quality=True,
        ignore_iof_thr=-1,
    ),
    sampler=dict(
        type="RandomSampler", num=256, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True
    ),
    pos_weight=-1,
    debug=False,
)
TEST_CFG = dict(score_thr=0.0001, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100)

GT = [
    ([[23.0, 41.0, 310.0, 420.0], [400.0, 60.0, 630.0, 330.0], [300.0, 300.0, 340.0, 350.0]],
     [3, 17, 29]),
    ([[100.0, 100.0, 400.0, 400.0], [110.0, 105.0, 405.0, 395.0], [500.0, 10.0, 620.0, 90.0]],
     [0, 0, 11]),
]


def fixed(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def make_proposals(num, seed, gt_boxes):
    """``(num, 5)`` proposals: half jittered around the ground truth (so there
    are positives to sample), half anywhere, with sizes from 16 to 600 px so
    the FPN extractor routes RoIs to all four levels."""
    g = torch.Generator().manual_seed(seed)
    h, w = IMG_SHAPE[:2]
    gt = torch.tensor(gt_boxes)
    n_near = num // 2
    base = gt[torch.arange(n_near) % len(gt)]
    wh = (base[:, 2:] - base[:, :2]).clamp(min=1)
    jitter = (torch.rand(n_near, 4, generator=g) - 0.5) * 0.4 * torch.cat([wh, wh], 1)
    near = base + jitter

    n_far = num - n_near
    # 16 + u^2 * 584: skewed towards small boxes like real proposals, and built
    # from + and * only. torch.exp / torch.pow go through libm, which is not
    # guaranteed bit-identical across torch versions, and a one-ULP difference
    # in the *inputs* would show up as a failure everywhere downstream.
    u = torch.rand(n_far, 2, generator=g)
    size = 16.0 + u * u * 584.0
    ctr = torch.rand(n_far, 2, generator=g) * torch.tensor([w, h], dtype=torch.float32)
    far = torch.cat([ctr - size / 2, ctr + size / 2], 1)

    boxes = torch.cat([near, far], 0)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h - 1)
    # Keep every proposal non-degenerate.
    boxes[:, 2:] = torch.max(boxes[:, 2:], boxes[:, :2] + 2)
    scores = torch.rand(num, 1, generator=g)
    return torch.cat([boxes, scores], 1)


def seeded_state_dict(module, seed):
    """Every parameter from a fixed CPU seed. ``fc_cls`` gets 10x the scale;
    see the module docstring for why."""
    g = torch.Generator().manual_seed(seed)
    state = {}
    for name, param in sorted(module.state_dict().items()):
        scale = 0.2 if "fc_cls" in name else 0.02
        state[name] = torch.randn(*param.shape, generator=g) * scale
    return state


def make_feats(shapes, seed, device):
    return [
        fixed(shape, seed + i, scale=0.5).to(device).requires_grad_(True)
        for i, shape in enumerate(shapes)
    ]


def grad_or_marker(grad):
    """``allow_unused`` grads come back as None; store a sentinel instead so a
    None on one side and a tensor on the other shows up as a shape mismatch."""
    return torch.tensor([-1.0]) if grad is None else grad.cpu()


def encode_str(s):
    return torch.tensor(list(s.encode()), dtype=torch.uint8)


def put_dets(out, prefix, dets, labels):
    """``(k, 5)`` detections + ``(k,)`` labels, stored as three artifacts."""
    dets = torch.as_tensor(dets).cpu()
    out[f"{prefix}/boxes"] = dets[:, :4]
    out[f"{prefix}/scores"] = dets[:, 4]
    out[f"{prefix}/labels"] = torch.as_tensor(labels).long().cpu()


def put_result(out, prefix, bbox_result):
    """``bbox2result``'s list of per-class numpy arrays, flattened into
    :func:`put_dets` form with the class index as the label."""
    dets = torch.cat([torch.from_numpy(arr) for arr in bbox_result], 0)
    labels = torch.cat(
        [torch.full((len(arr),), c, dtype=torch.long) for c, arr in enumerate(bbox_result)]
    )
    put_dets(out, prefix, dets, labels)


def run(impl, device):
    if impl == "mmdet":
        from mmcv import ConfigDict
        from mmdet.core import bbox2roi
        from mmdet.models import build_head
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.config import ConfigDict
        from vfe.core import bbox2roi
        from vfe.models.builder import build_head

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    img_metas = [
        dict(img_shape=IMG_SHAPE, pad_shape=IMG_SHAPE, scale_factor=SCALE_FACTOR)
        for _ in range(2)
    ]
    gt_bboxes = [torch.tensor(b, dtype=torch.float32, device=device) for b, _ in GT]
    gt_labels = [torch.tensor(lab, dtype=torch.long, device=device) for _, lab in GT]
    proposals = [make_proposals(300, 500 + i, GT[i][0]).to(device) for i in range(2)]

    out = {}
    for name, ext_cfg, head_cfg, feat_shapes in HEADS:
        seed = SEEDS[name]
        roi_head = build_head(
            dict(
                type="StandardRoIHead",
                bbox_roi_extractor=dict(ext_cfg),
                bbox_head=dict(head_cfg),
                train_cfg=ConfigDict(TRAIN_CFG),
                test_cfg=ConfigDict(TEST_CFG),
            )
        )
        state = seeded_state_dict(roi_head, seed)
        out[f"meta/{name}/state_keys"] = encode_str("\n".join(sorted(state)))
        roi_head.load_state_dict(state)
        roi_head = roi_head.to(device)
        extractor = roi_head.bbox_roi_extractor
        bbox_head = roi_head.bbox_head

        # ---- extractor, forward and backward ------------------------------
        rois = bbox2roi(proposals)
        upstream = fixed((rois.size(0), ext_cfg["out_channels"], 7, 7), seed + 50).to(device)

        feats = make_feats(feat_shapes, seed, device)
        roi_feats = extractor(feats, rois)
        out[f"ext/{name}/feats"] = roi_feats.detach().cpu()
        grads = torch.autograd.grad((roi_feats * upstream).sum(), feats)
        for i, grad in enumerate(grads):
            out[f"ext/{name}/grad{i}"] = grad.cpu()

        if len(feat_shapes) > 1:
            out[f"ext/{name}/levels"] = extractor.map_roi_levels(rois, len(feat_shapes)).cpu()

            feats = make_feats(feat_shapes, seed, device)
            rescaled = extractor(feats, rois, roi_scale_factor=1.5)
            out[f"ext/{name}/rescaled"] = rescaled.detach().cpu()

            # Only the small RoIs: every one maps to level 0, so levels 1-3
            # take the "no RoIs here" branch and must still get a (zero) grad.
            small = rois[(rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]) < 100.0 ** 2]
            feats = make_feats(feat_shapes, seed, device)
            one_level = extractor(feats, small)
            out[f"ext/{name}/one_level"] = one_level.detach().cpu()
            grads = torch.autograd.grad(one_level.sum(), feats, allow_unused=True)
            for i, grad in enumerate(grads):
                out[f"ext/{name}/one_level_grad{i}"] = grad_or_marker(grad)

        feats = make_feats(feat_shapes, seed, device)
        out[f"ext/{name}/empty_shape"] = torch.tensor(extractor(feats, rois[:0]).shape)

        # ---- bbox head forward --------------------------------------------
        with torch.no_grad():
            feats = [f.detach() for f in make_feats(feat_shapes, seed, device)]
            roi_feats = extractor(feats, rois)
            cls_score, bbox_pred = bbox_head(roi_feats)
        out[f"bbox/{name}/cls"] = cls_score.cpu()
        out[f"bbox/{name}/reg"] = bbox_pred.cpu()

        # ---- get_bboxes ---------------------------------------------------
        n0 = len(proposals[0])
        with torch.no_grad():
            bboxes, scores = bbox_head.get_bboxes(
                rois[:n0], cls_score[:n0], bbox_pred[:n0], IMG_SHAPE, SCALE_FACTOR, cfg=None
            )
            out[f"dec/{name}/nocfg/bboxes"] = bboxes.cpu()
            out[f"dec/{name}/nocfg/scores"] = scores.cpu()
            for case, rescale in [("nms", False), ("rescale", True)]:
                det_bboxes, det_labels = bbox_head.get_bboxes(
                    rois[:n0], cls_score[:n0], bbox_pred[:n0], IMG_SHAPE, SCALE_FACTOR,
                    rescale=rescale, cfg=ConfigDict(TEST_CFG),
                )
                put_dets(out, f"dec/{name}/{case}", det_bboxes, det_labels)

        # ---- training -----------------------------------------------------
        roi_head.train()
        for case, case_gt_bboxes, case_gt_labels in [
            ("both", gt_bboxes, gt_labels),
            # Second image has no ground truth: every one of its RoIs is a negative.
            ("nogt1", [gt_bboxes[0], gt_bboxes[1][:0]], [gt_labels[0], gt_labels[1][:0]]),
            # No ground truth anywhere: no positives, so loss_bbox takes the
            # gradient-connected-zero branch.
            ("nogt", [b[:0] for b in gt_bboxes], [lab[:0] for lab in gt_labels]),
        ]:
            feats = make_feats(feat_shapes, seed, device)
            torch.manual_seed(seed)
            losses = roi_head.forward_train(
                feats, img_metas, proposals, case_gt_bboxes, case_gt_labels
            )
            for key in sorted(losses):
                out[f"train/{name}/{case}/{key}"] = losses[key].detach().float().reshape(-1).cpu()
            total = losses["loss_cls"] + losses["loss_bbox"]
            named = sorted(roi_head.named_parameters())
            grads = torch.autograd.grad(
                total, feats + [p for _, p in named], allow_unused=True
            )
            for i in range(len(feats)):
                out[f"train/{name}/{case}/grad_feat{i}"] = grad_or_marker(grads[i])
            for j in range(len(named)):
                out[f"train/{name}/{case}/grad/{named[j][0]}"] = grad_or_marker(
                    grads[len(feats) + j]
                )

            # The targets the loss above was computed from, checked directly:
            # a scalar loss can hide compensating errors in them.
            torch.manual_seed(seed)
            sampling_results = []
            for i in range(2):
                assign_result = roi_head.bbox_assigner.assign(
                    proposals[i], case_gt_bboxes[i], None, case_gt_labels[i]
                )
                sampling_results.append(
                    roi_head.bbox_sampler.sample(
                        assign_result, proposals[i], case_gt_bboxes[i], case_gt_labels[i]
                    )
                )
            labels, label_weights, bbox_targets, bbox_weights = bbox_head.get_targets(
                sampling_results, case_gt_bboxes, case_gt_labels, ConfigDict(TRAIN_CFG)
            )
            out[f"train/{name}/{case}/tgt/rois"] = bbox2roi(
                [res.bboxes for res in sampling_results]
            ).cpu()
            out[f"train/{name}/{case}/tgt/labels"] = labels.cpu()
            out[f"train/{name}/{case}/tgt/label_weights"] = label_weights.cpu()
            out[f"train/{name}/{case}/tgt/bbox_targets"] = bbox_targets.cpu()
            out[f"train/{name}/{case}/tgt/bbox_weights"] = bbox_weights.cpu()
        roi_head.eval()

        # ---- simple_test ----------------------------------------------------
        with torch.no_grad():
            feats = [f.detach() for f in make_feats(feat_shapes, seed, device)]
            for case, props, rescale in [
                ("plain", proposals, False),
                ("rescale", proposals, True),
                ("empty1", [proposals[0], proposals[1][:0]], False),
            ]:
                results = roi_head.simple_test(feats, props, img_metas, rescale=rescale)
                for i, res in enumerate(results):
                    put_result(out, f"test/{name}/{case}/img{i}", res)

    print(f"RAN   {len(out)} artifacts on {device}")
    return out


def compare(path_a, path_b, atol, rtol, skip_train_grads=False):
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)

    failures, checked, exact, worst, worst_key = [], 0, 0, 0.0, None
    for key in sorted(set(a) | set(b)):
        if skip_train_grads and key.startswith("train/") and "/grad" in key:
            continue
        checked += 1
        if key not in a or key not in b:
            failures.append(f"{key}: only in {'A' if key in a else 'B'}")
            continue
        va, vb = a[key], b[key]
        if va.shape != vb.shape:
            failures.append(f"{key}: shape {tuple(va.shape)} != {tuple(vb.shape)}")
        elif not va.dtype.is_floating_point:
            if torch.equal(va, vb):
                exact += 1
            else:
                n = (va != vb).sum().item()
                failures.append(f"{key}: {n} of {va.numel()} entries differ")
        elif va.numel() == 0 or torch.equal(va, vb):
            exact += 1
        else:
            diff = (va - vb).abs().max().item()
            scale = va.abs().max().item()
            ratio = diff / scale if scale > 0 else float("inf")
            if ratio > worst:
                worst, worst_key = ratio, key
            if diff > atol + rtol * scale:
                failures.append(f"{key}: max|diff| = {diff:.3e} = {ratio:.2e} of scale {scale:.3e}")

    for note in failures:
        print(f"FAIL  {note}")
    print("-" * 70)
    print(f"bit-exact: {exact} of {checked}")
    print(f"largest difference relative to its artifact's scale: {worst:.2e} ({worst_key})")
    if skip_train_grads:
        print(f"skipped {len(set(a) | set(b)) - checked} training-gradient artifacts")
    if failures:
        print(f"{len(failures)} of {checked} artifact(s) differ")
    else:
        print(f"ROI HEAD PARITY OK ({checked} artifacts)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--rtol", type=float, default=1e-5, help="relative to each artifact's max")
    ap.add_argument("--skip-train-grads", action="store_true",
                    help="for CUDA runs; see the module docstring")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare, atol=args.atol, rtol=args.rtol,
                skip_train_grads=args.skip_train_grads)
    elif args.impl and args.out:
        torch.save(run(args.impl, args.device), args.out)
        print(f"saved -> {args.out}")
    else:
        ap.error("pass either --impl/--out or --compare")
