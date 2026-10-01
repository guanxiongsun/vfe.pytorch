"""Parity check: the ``vfe`` YOLOX (CSPDarknet, YOLOXPAFPN, YOLOXHead,
SimOTAAssigner, the YOLOX detector) vs the mmdet 2.19.1 oracle.

YOLOX is the second one-stage detector EOVOD is evaluated on, and none of it
existed in 2.0. Weights are generated from each parameter's name and shape
(both stacks name them alike), with the scale of a Kaiming init so a
YOLOX-S-sized network keeps its activations in range; BatchNorm statistics
are seeded too. Artifacts:

* ``forward`` -- the backbone's three outputs, the neck's, and the head's
  class / box / objectness maps, in eval mode.
* ``priors`` / ``decode`` -- the stride-carrying point priors, the decoded
  boxes, and the post-NMS detections of ``get_bboxes`` (rescaled), compared
  as sets since random weights tie thousands of scores.
* ``targets`` -- SimOTA's assignment through ``_get_target_single`` (the
  foreground mask, soft class targets, objectness, box and L1 targets).
* ``loss`` / ``grad`` -- the four losses with the L1 term on, and every
  parameter's gradient, in train mode (BatchNorm on batch statistics).
* ``resize`` -- the detector's multi-scale step on a batch and its boxes.

Usage (the legacy side needs the v1.0.0 tree on ``PYTHONPATH``; see
docs/eovod-plan.md for building that environment):
    PYTHONPATH=/path/to/v1.0.0 legacy38/bin/python tools/checks/parity_yolox.py --impl mmdet --out yolox_mmdet.pt
    python tools/checks/parity_yolox.py --impl vfe --out yolox_vfe.pt
    python tools/checks/parity_yolox.py --compare yolox_mmdet.pt yolox_vfe.pt
Add ``--dtype float64`` to both runs to separate porting errors from float32
accumulation order (NMS runs in float32 only).
"""

import argparse
import sys
import zlib
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

# YOLOX-S with 30 classes (ImageNet VID's).
MODEL = dict(
    type="YOLOX",
    input_size=(320, 384),
    random_size_range=(8, 12),
    random_size_interval=10,
    backbone=dict(type="CSPDarknet", deepen_factor=0.33, widen_factor=0.5),
    neck=dict(type="YOLOXPAFPN", in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=1),
    bbox_head=dict(type="YOLOXHead", num_classes=30, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65)),
)
IMG = (2, 3, 320, 384)
GT_BBOXES = [
    [[23.0, 41.0, 210.0, 300.0], [200.0, 60.0, 370.0, 230.0], [150.0, 250.0, 180.0, 290.0]],
    [[100.0, 100.0, 300.0, 310.0], [5.0, 8.0, 60.0, 50.0]],
]
GT_LABELS = [[3, 17, 29], [0, 12]]


def fixed(shape, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def seeded_state_dict(model):
    """A deterministic value for every entry, from its name alone."""
    state = {}
    for name, value in model.state_dict().items():
        g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
        if not value.is_floating_point():
            state[name] = torch.zeros_like(value)
            continue
        noise = torch.randn(tuple(value.shape), generator=g)
        if name.endswith("running_var"):
            state[name] = 1.0 + 0.2 * noise.abs()
        elif name.endswith("running_mean"):
            state[name] = 0.1 * noise
        elif value.dim() == 1 and ".bn." in name + ".":
            state[name] = (1.0 if name.endswith("weight") else 0.0) + 0.1 * noise
        elif value.dim() == 4:
            fan_in = value.shape[1] * value.shape[2] * value.shape[3]
            state[name] = noise / fan_in ** 0.5
        elif "conv_cls" in name or "conv_obj" in name:  # predictor biases
            state[name] = -2.0 + 0.5 * noise
        else:
            state[name] = 0.1 * noise
    return state


def build(impl):
    if impl == "mmdet":
        from mmcv import ConfigDict
        from mmdet.models import build_detector

        return build_detector(ConfigDict(MODEL))
    sys.path.insert(0, str(REPO_ROOT))
    from vfe.models.builder import build_detector

    return build_detector(MODEL)


def sort_dets(det_bboxes, det_labels):
    """Detections in a stack-independent order: label, score (descending),
    then every coordinate. Random weights give many exact ties in label,
    score and x1, so all four coordinates break them; rounding keeps float
    noise from reordering near-equal keys."""
    import numpy as np

    b = det_bboxes.detach().cpu().double().numpy()
    lbl = det_labels.detach().cpu().numpy()
    keys = [np.round(b[:, k], 3) for k in (3, 2, 1, 0)] + [np.round(-b[:, 4], 5), lbl]
    order = torch.as_tensor(np.lexsort(keys))
    return det_bboxes[order], det_labels[order]


def float32_assigner(head):
    """mmdet's SimOTA builds float32 one-hot targets, so it cannot run in
    float64; in float64 runs both stacks assign in float32 and continue in
    float64."""
    assign = head.assigner.assign

    def assign32(pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels, *args, **kwargs):
        result = assign(pred_scores.float(), priors.float(), decoded_bboxes.float(),
                        gt_bboxes.float(), gt_labels, *args, **kwargs)
        result.max_overlaps = result.max_overlaps.to(pred_scores.dtype)
        return result

    head.assigner.assign = assign32


def run(impl, dtype="float32"):
    import numpy as np

    out = {}
    dtype = getattr(torch, dtype)
    model = build(impl)
    model.load_state_dict(seeded_state_dict(model))
    model = model.to(dtype=dtype).eval()
    head = model.bbox_head
    if dtype == torch.float64:
        float32_assigner(head)
    img = fixed(IMG, 5, 1.0).to(dtype)
    metas = [dict(img_shape=(320, 384, 3), pad_shape=(320, 384, 3), ori_shape=(200, 240, 3),
                  scale_factor=np.array([1.6, 1.6, 1.6, 1.6], dtype=np.float32), flip=False,
                  batch_input_shape=(320, 384))] * 2

    # ---- forward ----
    with torch.no_grad():
        backbone_outs = model.backbone(img)
        neck_outs = model.neck(backbone_outs)
        cls_scores, bbox_preds, objectnesses = head(neck_outs)
    for i in range(3):
        out[f"forward/backbone{i}"] = backbone_outs[i]
        out[f"forward/neck{i}"] = neck_outs[i]
        out[f"forward/cls{i}"] = cls_scores[i]
        out[f"forward/reg{i}"] = bbox_preds[i]
        out[f"forward/obj{i}"] = objectnesses[i]

    # ---- priors and decoding ----
    sizes = [c.shape[2:] for c in cls_scores]
    priors = head.prior_generator.grid_priors(sizes, dtype=dtype, device="cpu", with_stride=True)
    for i, p in enumerate(priors):
        out[f"priors/level{i}"] = p
    flat_preds = torch.cat([b.permute(0, 2, 3, 1).reshape(2, -1, 4) for b in bbox_preds], dim=1)
    out["decode/boxes"] = head._bbox_decode(torch.cat(priors), flat_preds)
    if dtype == torch.float32:
        with torch.no_grad():
            dets = head.get_bboxes(cls_scores, bbox_preds, objectnesses, img_metas=metas,
                                   rescale=True)
        for img_id, (det_bboxes, det_labels) in enumerate(dets):
            det_bboxes, det_labels = sort_dets(det_bboxes, det_labels)
            out[f"decode/dets{img_id}/bboxes"] = det_bboxes
            out[f"decode/dets{img_id}/labels"] = det_labels

    # ---- targets, losses and gradients ----
    model.train()
    head.use_l1 = True
    model.zero_grad()
    gt_bboxes = [torch.tensor(b, dtype=dtype) for b in GT_BBOXES]
    gt_labels = [torch.tensor(lbl) for lbl in GT_LABELS]
    feats = model.extract_feat(img)
    cls_scores, bbox_preds, objectnesses = head(feats)
    flat_cls = torch.cat([c.permute(0, 2, 3, 1).reshape(2, -1, 30) for c in cls_scores], dim=1)
    flat_reg = torch.cat([b.permute(0, 2, 3, 1).reshape(2, -1, 4) for b in bbox_preds], dim=1)
    flat_obj = torch.cat([o.permute(0, 2, 3, 1).reshape(2, -1) for o in objectnesses], dim=1)
    flat_priors = torch.cat(head.prior_generator.grid_priors(
        [c.shape[2:] for c in cls_scores], dtype=dtype, device="cpu", with_stride=True))
    boxes = head._bbox_decode(flat_priors, flat_reg)
    for img_id in range(2):
        fg, cls_t, obj_t, bbox_t, l1_t, num_pos = head._get_target_single(
            flat_cls[img_id].detach(), flat_obj[img_id].detach(), flat_priors,
            boxes[img_id].detach(), gt_bboxes[img_id], gt_labels[img_id])
        out[f"targets/img{img_id}/foreground"] = fg
        out[f"targets/img{img_id}/cls"] = cls_t
        out[f"targets/img{img_id}/obj"] = obj_t
        out[f"targets/img{img_id}/bbox"] = bbox_t
        out[f"targets/img{img_id}/l1"] = l1_t
        out[f"targets/img{img_id}/num_pos"] = torch.tensor(num_pos)
    losses = head.loss(cls_scores, bbox_preds, objectnesses, gt_bboxes, gt_labels, metas)
    sum(losses.values()).backward()
    for k, v in losses.items():
        out[f"loss/{k}"] = v.detach()
    for name, param in model.named_parameters():
        if param.grad is not None:
            out[f"grad/{name}"] = param.grad.detach()

    # ---- the multi-scale step ----
    model._input_size = (256, 320)
    resized, resized_boxes = model._preprocess(img.clone(), [b.clone() for b in gt_bboxes])
    out["resize/img"] = resized
    for i, b in enumerate(resized_boxes):
        out[f"resize/boxes{i}"] = b
    return {k: v.detach().cpu() for k, v in out.items()}


def compare(path_a, path_b, atol, rtol, skip_grads=False):
    """Each float artifact against its own scale: ``max|a-b| <= atol + rtol *
    max(|a|, |b|)``. Integer and bool artifacts must be identical."""
    a = torch.load(path_a, weights_only=False)
    b = torch.load(path_b, weights_only=False)
    if set(a) != set(b):
        print("artifact key sets differ:", sorted(set(a) ^ set(b)))
        return False
    ok, exact, worst = True, 0, {}
    for key in sorted(a):
        va, vb = a[key], b[key]
        group = key.split("/")[0]
        if key.startswith("decode/dets") and key.endswith("/labels"):
            continue  # judged with the boxes
        if skip_grads and group == "grad":
            continue
        if key.startswith("decode/dets") and key.endswith("/bboxes"):
            # Random weights give thousands of near-tied scores, so no sort
            # order is stable across stacks: compare as sets, each detection
            # against the nearest one of the same label on the other side.
            if va.shape != vb.shape:
                print(f"FAIL {key}: {len(va)} vs {len(vb)} detections")
                ok = False
                continue
            la, lb = a[key.replace("bboxes", "labels")], b[key.replace("bboxes", "labels")]
            dist = torch.cdist(va.double(), vb.double()) + 1e9 * (la[:, None] != lb[None, :])
            diff = max(dist.min(1).values.max().item(), dist.min(0).values.max().item())
            scale = va.abs().max().item()
            if diff / scale > worst.get(group, (0.0, ""))[0]:
                worst[group] = (diff / scale, key)
            if diff > atol + rtol * scale:
                print(f"FAIL {key}: a detection is {diff:.3e} from its nearest match")
                ok = False
            continue
        if va.shape != vb.shape:
            print(f"FAIL {key}: shape {tuple(va.shape)} vs {tuple(vb.shape)}")
            ok = False
            continue
        if not va.is_floating_point():
            if torch.equal(va, vb):
                exact += 1
            else:
                print(f"FAIL {key}: integer/bool tensors differ")
                ok = False
            continue
        va, vb = va.double(), vb.double()
        if torch.equal(va, vb):
            exact += 1
            continue
        diff = (va - vb).abs().max().item()
        scale = max(va.abs().max().item(), vb.abs().max().item(), 1e-12)
        if diff / scale > worst.get(group, (0.0, ""))[0]:
            worst[group] = (diff / scale, key)
        if diff > atol + rtol * scale:
            print(f"FAIL {key}: max |a-b| = {diff:.3e} on scale {scale:.3e}")
            ok = False
    print(f"{len(a)} artifacts, {exact} bit-exact")
    for group, (rel, key) in sorted(worst.items()):
        print(f"  {group:8s} worst relative difference {rel:.2e} ({key})")
    print("YOLOX PARITY OK" if ok else "YOLOX PARITY FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--skip-grads", action="store_true",
                    help="float32 runs: torch 1.10's CPU backward through the whole network is "
                         "off by ~1e-2 against float64 (vfe's by ~1e-4), so gradients are "
                         "judged in float64 runs")
    args = ap.parse_args()
    if args.compare:
        sys.exit(0 if compare(args.compare[0], args.compare[1], args.atol, args.rtol,
                              args.skip_grads) else 1)
    if not args.impl or not args.out:
        ap.error("--impl and --out are required unless --compare is given")
    torch.manual_seed(0)
    out = run(args.impl, args.dtype)
    torch.save(out, args.out)
    print(f"{args.impl}: {len(out)} artifacts -> {args.out}")


if __name__ == "__main__":
    main()
