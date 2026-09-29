"""Dump EOVOD's per-frame state on a few ImageNet VID val videos for the
frame inspector (``server.py`` beside this file).

Runs the model's own ``simple_test`` frame by frame (so the detections are the
evaluator's) and captures its internals through thin wrappers: the location
prior and its masks, the levels run, the key set and where each key came from.
On every frame it also scores the head with and without aggregation; on every
``--heavy-every``-th frame it saves the image, per-level score maps
(plain / aggregated / mask as one RGB atlas) and, for two queries (the top
detection and the first ground-truth box), the reference pixels they attend
to most.

    python tools/eovod_inspector/dump.py --ckpt CKPT --run NAME --out DIR \
        [--cfg-options model.location_prior.validate_on=cls_score ...]

Writes ``DIR/runs/NAME.json`` and the frame images under ``DIR/img/``.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import glob
import io
import json
import os
import os.path as osp
import time

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Subset

from vfe.config import Config, parse_cfg_options
from vfe.datasets import build_dataset, collate_video_test
from vfe.engine.evaluator import to_device
from vfe.models.builder import build_model
from vfe.models.checkpoint import load_checkpoint
from vfe.models.vid.eovod import boxes_to_level_masks, scale_boxes

# FCOS's regress ranges: the level a box of this half-size would be assigned to.
REGRESS_RANGES = (64, 128, 256, 512)
THUMB_W, REF_THUMB_W = 640, 256


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vid/eovod/eovod_fcos_r101_fpn_3x.py")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", required=True, help="run name shown on the page")
    ap.add_argument("--note", default="", help="one line shown under the run name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", default="0,45,120,230,340,480", help="val video indices")
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--heavy-every", type=int, default=15)
    ap.add_argument("--cfg-options", nargs="+", default=[])
    ap.add_argument("--config-model-only", action="store_true",
                    help="build the model from --config alone, not the checkpoint's training config")
    return ap.parse_args()


def iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None] - inter)


def score_maps(head, feats, reg_feats=None):
    """Per level ``(H, W)`` of max-class sigmoid x centerness sigmoid: the
    score a detection at that cell would get."""
    cls, _, ctr = head(feats, reg_feats=reg_feats)
    return [(c[0].sigmoid().max(0).values * t[0, 0].sigmoid()) for c, t in zip(cls, ctr, strict=True)]


def atlas_png(plain, enh, masks):
    """Levels stacked vertically; R = plain score, G = aggregated score,
    B = mask. Returns (data URL, [[y, h, w] per level])."""
    width = plain[0].shape[1]
    layout, y = [], 0
    for m in plain:
        layout.append([y, int(m.shape[0]), int(m.shape[1])])
        y += m.shape[0]
    rgb = np.zeros((y, width, 3), np.uint8)
    for (y0, h, w), p, e, k in zip(layout, plain, enh, masks, strict=True):
        rgb[y0:y0 + h, :w, 0] = (p.clamp(0, 1) * 255).round().byte().cpu().numpy()
        rgb[y0:y0 + h, :w, 1] = (e.clamp(0, 1) * 255).round().byte().cpu().numpy()
        if k is not None:
            rgb[y0:y0 + h, :w, 2] = k.byte().cpu().numpy() * 255
    ok, buf = cv2.imencode(".png", rgb[:, :, ::-1])  # cv2 writes BGR
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(), layout


def save_thumb(path, filename, width):
    if osp.exists(path):
        return
    img = cv2.imread(filename)
    h, w = img.shape[:2]
    img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def save_ref_sprite(path, filenames, cols=7):
    """Reference frames as one sprite, ``cols`` per row, each REF_THUMB_W wide."""
    thumbs = []
    for f in filenames:
        img = cv2.imread(f)
        h, w = img.shape[:2]
        thumbs.append(cv2.resize(img, (REF_THUMB_W, round(h * REF_THUMB_W / w)),
                                 interpolation=cv2.INTER_AREA))
    th = max(t.shape[0] for t in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    sprite = np.zeros((rows * th, cols * REF_THUMB_W, 3), np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sprite[r * th:r * th + t.shape[0], c * REF_THUMB_W:(c + 1) * REF_THUMB_W] = t
    cv2.imwrite(path, sprite, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return th, cols


class Recorder:
    """Wraps the model's internals to capture one frame's state."""

    def __init__(self, model):
        self.model = model
        self.frame: dict = {}
        self.video: dict = {}
        self.in_gather = False
        m, head = model, model.detector.bbox_head
        orig_enhance, orig_levels = m._enhance, m._levels_to_run
        orig_gather, orig_write = m._gather_reference_keys, m._write_memory
        orig_head_test = head.simple_test

        def enhance(feats, masks, keys):
            out = orig_enhance(feats, masks, keys)
            self.frame.update(feats=feats, masks=masks, keys=keys, enhanced=out)
            return out

        def levels_to_run():
            levels, full = orig_levels()
            prev = m._prev_boxes
            self.frame.update(levels=levels, full=full,
                              prior=None if prev is None else prev.detach().clone())
            return levels, full

        def gather(refs, ref_metas):
            self.in_gather = True
            self.video = dict(origins=[[] for _ in m.strides], ref_boxes=[], ref_metas=ref_metas,
                              ref_count=0)
            try:
                return orig_gather(refs, ref_metas)
            finally:
                self.in_gather = False

        def write_memory(feats, boxes):
            if self.in_gather:
                r = self.video["ref_count"]
                self.video["ref_count"] += 1
                self.video["ref_boxes"].append(boxes.detach().cpu().numpy())
                if boxes.numel():
                    masks = boxes_to_level_masks(boxes, [f.shape[-2:] for f in feats],
                                                 m.agg_strides)
                    for lvl, mask in enumerate(masks):
                        yx = mask.nonzero().cpu().numpy()  # row-major: the memory's order
                        self.video["origins"][lvl].append(np.c_[np.full(len(yx), r), yx])
            return orig_write(feats, boxes)

        def head_test(*args, **kwargs):
            out = orig_head_test(*args, **kwargs)
            if not self.in_gather:
                self.frame["head_out"] = out[0]
            return out

        def sample(level):
            # PixelMemory.sample, drawing the same random subset, but keeping its
            # indices so a key can be traced to its reference pixel.
            bank, n = m.memory.banks[level], m.memory.num_keys
            if bank is None or len(bank) == 0 or n is None or len(bank) <= n:
                self.frame.setdefault("key_idx", {})[level] = None
                return orig_sample(level)
            idx = torch.randperm(len(bank), device=bank.device)[:n]
            self.frame.setdefault("key_idx", {})[level] = idx
            return bank[idx]

        orig_sample = m.memory.sample
        m.memory.sample = sample
        m._enhance, m._levels_to_run = enhance, levels_to_run
        m._gather_reference_keys, m._write_memory = gather, write_memory
        head.simple_test = head_test


def attention(model, rec, level, cy, cx, top=48):
    """Where the query at cell (cy, cx) of ``level`` attends: mean over heads.
    None when the keys cannot be traced to reference pixels: a memory that
    takes every frame's pixels (``memory.update``) mixes them with the
    references' and replaces them at random."""
    if model.memory.update:
        return None
    keys = rec.frame["keys"][level]
    origins = rec.video.get("origins", [[]] * (level + 1))[level]
    if keys is None or len(keys) == 0 or not origins:
        return None
    origins = np.concatenate(origins)
    idx = rec.frame.get("key_idx", {}).get(level)
    if idx is not None:  # a capped key set is a random subset of the bank
        origins = origins[idx.cpu().numpy()]
    assert len(origins) == len(keys), (len(origins), len(keys))
    agg = model._aggregator(level)
    x = rec.frame["feats"][level][0][:, cy, cx][None]
    nh = agg.num_heads
    q = agg.fc_embed(x).view(1, nh, -1).permute(1, 0, 2)
    k = agg.ref_fc_embed(keys).view(len(keys), nh, -1).permute(1, 2, 0)
    w = (torch.bmm(q, k) / (q.shape[-1] ** 0.5)).softmax(dim=2).mean(0)[0]  # (M,)
    n = min(top, len(w))
    val, idx = w.topk(n)
    stride = model.agg_strides[level]
    pts = []
    for v, i in zip(val.tolist(), idx.tolist(), strict=True):
        r, y, xx = origins[i]
        sf = to_np(rec.video["ref_metas"][r]["scale_factor"])
        pts.append([int(r), round(float((xx + 0.5) * stride / sf[0]), 1),
                    round(float((y + 0.5) * stride / sf[1]), 1), round(v, 5)])
    ent = float(-(w * (w + 1e-12).log()).sum())
    mask = rec.frame["masks"]
    in_mask = bool(mask is not None and mask[level][cy, cx])
    delta = (rec.frame["enhanced"][level][0][:, cy, cx] - rec.frame["feats"][level][0][:, cy, cx])
    return dict(level=level, cell=[cy, cx], in_mask=in_mask, num_keys=len(keys),
                max_w=round(float(val[0]), 5), entropy=round(ent, 3),
                uniform_entropy=round(float(np.log(len(keys))), 3),
                delta_ratio=round(float(delta.norm() / x.norm()), 4), top=pts)


def to_np(x):
    return x.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(x) \
        else np.asarray(x, dtype=np.float32)


def level_for_box(box):
    half = max(box[2] - box[0], box[3] - box[1]) / 2
    for lvl, hi in enumerate(REGRESS_RANGES):
        if half <= hi:
            return lvl
    return len(REGRESS_RANGES)


@torch.no_grad()
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # The model as it was trained: vfe.cli.train saves the merged config next to
    # its checkpoints. The test-time options below still apply on top.
    saved = sorted(glob.glob(osp.join(osp.dirname(args.ckpt), "*.py.json")))
    if saved and not args.config_model_only:
        with open(saved[-1]) as f:
            cfg.merge_from_dict({"model": json.load(f)["model"]})
        print(f"model config from {saved[-1]}", flush=True)
    cfg.merge_from_dict(parse_cfg_options(args.cfg_options))
    ds = build_dataset(cfg.data.test)
    model = build_model(cfg.model)
    load_checkpoint(model, args.ckpt, map_location="cpu")
    device = torch.device("cuda")
    model = model.to(device).eval()
    head = model.detector.bbox_head
    rec = Recorder(model)
    os.makedirs(osp.join(args.out, "img"), exist_ok=True)
    os.makedirs(osp.join(args.out, "runs"), exist_ok=True)

    starts = [i for i, info in enumerate(ds.data_infos) if info["frame_id"] == 0]
    ends = starts[1:] + [len(ds)]
    videos, coco_dets, img_ids = [], [], []
    t0 = time.time()
    video_ids = [int(v) for v in args.videos.split(",")]
    for n, vi in enumerate(video_ids, 1):
        s, e = starts[vi], min(ends[vi], starts[vi] + args.max_frames)
        vname = osp.basename(osp.dirname(ds.data_infos[s]["filename"]))
        loader = DataLoader(Subset(ds, range(s, e)), batch_size=1, shuffle=False,
                            num_workers=4, collate_fn=collate_video_test)
        frames, video_refs = [], {}
        rec.video = {}
        for k, batch in enumerate(loader):
            idx = s + k
            rec.frame = {}
            batch = to_device(batch, device)
            result = model(return_loss=False, rescale=True, **batch)[0]
            meta = batch["img_metas"][0][0] if isinstance(batch["img_metas"][0], list) \
                else batch["img_metas"][0]
            sf = to_np(meta["scale_factor"])
            fid = int(meta["frame_id"])
            if fid == 0 and "ref_metas" in rec.video:
                refs = rec.video["ref_metas"]
                sprite = f"img/{vname}_refs.jpg"
                th, cols = save_ref_sprite(osp.join(args.out, sprite), [r["filename"] for r in refs])
                ref_info = []
                for rm, rb in zip(refs, rec.video["ref_boxes"], strict=True):
                    rsf = to_np(rm["scale_factor"])
                    oh, ow = rm["ori_shape"][:2]
                    ref_info.append(dict(frame_id=int(rm["frame_id"]), w=int(ow), h=int(oh),
                                         boxes=(rb / rsf).round(1).tolist() if len(rb) else []))
                video_refs = dict(sprite=sprite, thumb_w=REF_THUMB_W, thumb_h=th, cols=cols,
                                  refs=ref_info,
                                  keys=[int(sum(len(o) for o in lv)) for lv in rec.video["origins"]])

            det_b, det_l, det_lv, det_cls = rec.frame["head_out"]
            det = det_b.detach().cpu().numpy()
            det[:, :4] /= sf
            labels, lvls, clsn = det_l.cpu().numpy(), det_lv.cpu().numpy(), det_cls.cpu().numpy()
            order = np.argsort(-det[:, 4])
            ann = ds.get_ann_info(idx)
            gtb, gtl = ann["bboxes"], ann["labels"]
            # Per ground-truth box: best score of a same-class detection at IoU >= 0.5.
            ious = iou(gtb, det[:, :4])
            hit = [float(det[(ious[g] >= 0.5) & (labels == gtl[g]), 4].max(initial=0.0))
                   for g in range(len(gtb))]
            valid = model._validation_scores(det_b, det_cls).cpu().numpy() > model.score_thr
            fp = 0
            for j in np.nonzero(valid)[0]:
                same = gtl == labels[j]
                if not same.any() or iou(det[j:j + 1, :4], gtb[same]).max() < 0.5:
                    fp += 1
            feats, enhanced, masks, keys = (rec.frame[k] for k in ("feats", "enhanced", "masks", "keys"))
            aggregated = masks is not None and any(
                m is not None and bool(m.any()) and kk is not None and len(kk) for m, kk in zip(masks, keys, strict=True))
            # The maps the head reads: through the FPN when aggregation sits
            # before it; the regression tower on the plain maps if cls-only.
            plain_fpn = model._stage_two(feats)
            enh_fpn = model._stage_two(enhanced) if aggregated else plain_fpn
            reg = None if model.aggregate_reg else plain_fpn
            plain_maps = score_maps(head, plain_fpn, reg)
            enh_maps = score_maps(head, enh_fpn, reg) if aggregated else plain_maps
            prior = rec.frame.get("prior")
            prior_boxes = []
            if prior is not None and len(prior):
                pb = scale_boxes(prior, model.box_ratio).cpu().numpy() / sf
                prior_boxes = pb.round(1).tolist()
            fr = dict(
                idx=idx, fid=fid, full=bool(rec.frame["full"]),
                levels=list(rec.frame["levels"]), aggregated=aggregated,
                prior=prior_boxes,
                keys=[0 if kk is None else int(len(kk)) for kk in keys],
                mask_frac=[0.0 if masks is None or m is None else round(float(m.float().mean()), 4)
                           for m in (masks or [None] * len(feats))],
                top=round(float(det[:, 4].max(initial=0.0)), 4),
                plain_max=round(float(max(m.max() for m in plain_maps)), 4),
                enh_max=round(float(max(m.max() for m in enh_maps)), 4),
                n_valid=int(valid.sum()), n_fp=fp,
                gt=[[*b.round(1).tolist(), int(c)] for b, c in zip(gtb, gtl, strict=True)],
                gt_hit=[round(h, 4) for h in hit],
                dets=[[*det[j, :4].round(1).tolist(), round(float(det[j, 4]), 4), int(labels[j]),
                       int(lvls[j]), round(float(clsn[j]), 4)] for j in order[:30]],
            )
            if fid % args.heavy_every == 0:
                img_rel = f"img/{vname}_{fid:04d}.jpg"
                save_thumb(osp.join(args.out, img_rel), meta["filename"], THUMB_W)
                oh, ow = meta["ori_shape"][:2]
                # Masks live on the aggregation levels, which are the head's
                # levels only when aggregation follows the FPN.
                atlas_masks = (masks if masks is not None and not model.aggregate_backbone
                               else [None] * len(plain_maps))
                url, layout = atlas_png(plain_maps, enh_maps, atlas_masks)
                queries = {}
                if len(order) and not model.aggregate_backbone:
                    j = order[0]
                    lv = int(lvls[j])
                    bx = det_b[j, :4].cpu().numpy()
                    st = model.strides[lv]
                    h, w = feats[lv].shape[-2:]
                    cy = min(int((bx[1] + bx[3]) / 2 // st), h - 1)
                    cx = min(int((bx[0] + bx[2]) / 2 // st), w - 1)
                    queries["top_det"] = attention(model, rec, lv, cy, cx)
                if len(gtb) and not model.aggregate_backbone:
                    gb = gtb[0] * np.r_[sf[:2], sf[:2]]  # network coords
                    lv = level_for_box(gb)
                    st = model.strides[lv]
                    h, w = feats[lv].shape[-2:]
                    cy = min(int((gb[1] + gb[3]) / 2 // st), h - 1)
                    cx = min(int((gb[0] + gb[2]) / 2 // st), w - 1)
                    queries["gt0"] = attention(model, rec, lv, cy, cx)
                fr.update(img=img_rel, w=int(ow), h=int(oh), sf=[float(sf[0]), float(sf[1])],
                          strides=list(model.strides), atlas=url, layout=layout,
                          queries={k: v for k, v in queries.items() if v is not None})
            frames.append(fr)
            img_ids.append(int(ds.img_ids[idx]))
            for label, b in enumerate(result):
                for x1, y1, x2, y2, sc in np.asarray(b).tolist():
                    coco_dets.append(dict(image_id=int(ds.img_ids[idx]), bbox=[x1, y1, x2 - x1, y2 - y1],
                                          score=sc, category_id=int(ds.cat_ids[label])))
        videos.append(dict(name=vname, index=vi, frames=frames, **video_refs))
        print(f"video {n}/{len(video_ids)} {vname}: {len(frames)} frames, "
              f"{time.time() - t0:.0f} s", flush=True)

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(ds.ann_file)
        dt = gt.loadRes(coco_dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = img_ids
        ev.params.catIds = list(ds.cat_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    summary = dict(AP=round(float(ev.stats[0]), 4), AP50=round(float(ev.stats[1]), 4),
                   AP75=round(float(ev.stats[2]), 4), frames=len(img_ids))
    run = dict(run=args.run, note=args.note, ckpt=args.ckpt, config=args.config,
               cfg_options=args.cfg_options, score_thr=model.score_thr,
               validate_on=model.validate_on, box_ratio=model.box_ratio,
               branches="all" if model.aggregate_reg else "cls",
               position="backbone" if model.aggregate_backbone else "fpn",
               queries="all" if model.queries_all else "prior",
               classes=list(ds.CLASSES), summary=summary, created=time.strftime("%Y-%m-%d %H:%M"),
               videos=videos)
    path = osp.join(args.out, "runs", f"{args.run}.json")
    with open(path, "w") as f:
        json.dump(run, f, separators=(",", ":"))
    print(f"subset AP {summary['AP']:.3f} AP50 {summary['AP50']:.3f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
