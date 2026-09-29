"""Time EOVOD's inference on one GPU, with the frames already on the device.

For each checkpoint and each test-time setting, runs the first frames of a few
ImageNet VID val videos in order (the priors are stateful) and reports:

* model time per frame (CUDA-synchronised; data loading excluded), with the
  first frame of each video -- which also detects on the 14 reference frames
  that give the key set -- reported apart from the rest;
* where the time goes: backbone + FPN, aggregation, head, and the per-video
  key gathering;
* what the priors did: the share of frames aggregated, query and key counts,
  the levels run; the peak GPU memory; and COCO AP on the timed frames.

    python tools/eovod_speed.py --ckpt A.pth [B.pth ...] --out speed.json

The model is built from the checkpoint's saved training config (as
``tools/eovod_inspector/dump.py`` does) with each setting's options on top.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os.path as osp
import statistics as st
import time

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Subset

from vfe.config import Config, parse_cfg_options
from vfe.datasets import build_dataset, collate_video_test
from vfe.engine.cuda_graphs import enable_cuda_graphs
from vfe.engine.evaluator import to_device
from vfe.models.builder import build_model
from vfe.models.checkpoint import load_checkpoint

VAL_MEAN_LEN = 176126 / 555
# Every setting names its key cap and size-prior margin: the configs' own
# defaults changed over time, and a checkpoint's saved training config may
# carry either.
UNCAPPED, CAP = "model.memory.num_keys=None", "model.memory.num_keys=4096"
PLAIN = ["model.location_prior.score_thr=1.1", "model.size_prior=None", UNCAPPED]
LPN = ["model.location_prior.score_thr=0.3", "model.size_prior=None"]
SPN = ["model.location_prior.score_thr=0.3", "model.size_prior.interval=7"]
# (name, cfg options, matmul TF32)
SETTINGS = [
    ("plain", PLAIN, False),
    ("lpn_det0.3", [*LPN, UNCAPPED], False),
    ("lpn_det0.3_T7", [*SPN, "model.size_prior.margin=0", UNCAPPED], False),
    ("lpn_det0.3_keys4096", [*LPN, CAP], False),
    ("lpn_det0.3_keys1024", [*LPN, "model.memory.num_keys=1024"], False),
    ("lpn_cls0.5", ["model.location_prior.validate_on=cls_score",
                    "model.location_prior.score_thr=0.5", "model.size_prior=None", UNCAPPED],
     False),
    ("plain_tf32", PLAIN, True),
    ("lpn_det0.3_tf32", [*LPN, UNCAPPED], True),
    ("lpn_det0.3_T7_keys4096", [*SPN, "model.size_prior.margin=0", CAP], False),
    ("lpn_det0.3_T7_margin1_keys4096", [*SPN, "model.size_prior.margin=1", CAP], False),
    ("lpn_det0.3_T7_margin1_floor1_keys4096",
     [*SPN, "model.size_prior.margin=1", "model.size_prior.margin_min_level=1", CAP], False),
    ("lpn_det0.3_T7_down1_keys4096",
     [*SPN, "model.size_prior.margin=1", "model.size_prior.margin_up=0", CAP], False),
    ("lpn_det0.2_r1.2_keys4096", ["model.location_prior.score_thr=0.2",
                                  "model.location_prior.box_ratio=1.2",
                                  "model.size_prior=None", CAP], False),
    ("lpn_det0.2_r1.5_keys4096", ["model.location_prior.score_thr=0.2",
                                  "model.location_prior.box_ratio=1.5",
                                  "model.size_prior=None", CAP], False),
    ("spn_m1_det0.2_r1.2_keys4096", ["model.location_prior.score_thr=0.2",
                                     "model.location_prior.box_ratio=1.2",
                                     "model.size_prior.interval=7", "model.size_prior.margin=1",
                                     CAP], False),
    # the first setting again, last: the GPU's clock should not have drifted
    ("plain_again", PLAIN, False),
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/vid/eovod/eovod_fcos_r101_fpn_3x.py")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", default="0,70,140,210,280,350,420,490")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--settings", help="comma-separated subset of: "
                    + ", ".join(s[0] for s in SETTINGS))
    ap.add_argument("--warmup-s", type=float, default=120,
                    help="untimed plain inference before anything is timed, so the GPU "
                         "reaches its steady clock (the first settings ran slow without it)")
    ap.add_argument("--note", default="", help="shown with the results on the inspector page")
    ap.add_argument("--engines", default="eager",
                    help="comma-separated: eager (PyTorch defaults) and/or fast (cuDNN "
                         "autotuning, CUDA graphs for backbone+FPN and the head's per-level "
                         "forward, one GPU sync per frame in post-processing)")
    return ap.parse_args()


def build(cfg_path: str, ckpt: str, options: list[str], engine: str = "eager"):
    cfg = Config.fromfile(cfg_path)
    saved = sorted(glob.glob(osp.join(osp.dirname(ckpt), "*.py.json")))
    if saved:
        with open(saved[-1]) as f:
            cfg.merge_from_dict({"model": json.load(f)["model"]})
    cfg.merge_from_dict(parse_cfg_options(options))
    model = build_model(cfg.model)
    load_checkpoint(model, ckpt, map_location="cpu")
    model = model.cuda().eval()
    torch.backends.cudnn.benchmark = engine == "fast"
    if engine == "fast":
        enable_cuda_graphs(model.detector)
        model.detector.bbox_head.one_sync_postprocess = True
    return model


class Clock:
    """Wraps model methods to record each frame.

    ``instrument=False`` (the timed pass): no synchronisation added -- query
    and key counts are kept as GPU tensors and read after the frame's time is
    taken. ``instrument=True`` (the breakdown pass): each component is timed
    between CUDA synchronisations, which themselves cost time.
    """

    def __init__(self, model, instrument: bool):
        self.model = model
        self.frame: dict = {}
        self.depth = 0  # >0 inside key gathering, whose inner calls it owns
        if instrument:
            det, head = model.detector, model.detector.bbox_head
            self._wrap(model, "_gather_reference_keys", "gather", outer=True)
            self._wrap(det, "extract_feat", "backbone_fpn")
            self._wrap(model, "_enhance", "aggregation")
            self._wrap(head, "simple_test", "head")
            return
        original_levels, original_enhance = model._levels_to_run, model._enhance

        def levels_to_run():
            levels, full = original_levels()
            self.frame.update(levels=len(levels), full=full)
            return levels, full

        def enhance(feats, masks, keys):
            out = original_enhance(feats, masks, keys)
            if masks is not None:
                nk = [0 if k is None else len(k) for k in keys]
                zero = feats[0].new_zeros((), dtype=torch.long)
                sums = torch.stack([m.sum() if m is not None and n else zero
                                    for m, n in zip(masks, nk, strict=True)])
                self.frame["counts"] = (sums, nk)
            return out

        model._levels_to_run, model._enhance = levels_to_run, enhance

    def resolve(self) -> None:
        """Turn the frame's GPU counts into numbers (after it was timed)."""
        counts = self.frame.pop("counts", None)
        if counts is None:
            self.frame.update(engaged=False, queries=0, keys=0)
            return
        sums, nk = counts
        q = sums.tolist()
        self.frame.update(engaged=sum(q) > 0, queries=sum(q),
                          keys=sum(n for qi, n in zip(q, nk, strict=True) if qi))

    def _wrap(self, obj, name, key, outer=False):
        original = getattr(obj, name)

        def timed(*args, **kwargs):
            if self.depth and not outer:
                return original(*args, **kwargs)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.depth += outer
            try:
                out = original(*args, **kwargs)
            finally:
                self.depth -= outer
            torch.cuda.synchronize()
            self.frame[key] = self.frame.get(key, 0.0) + time.perf_counter() - t0
            return out

        setattr(obj, name, timed)


def run(model, clock, batches, img_ids, dataset):
    frames, dets = [], []
    base = torch.cuda.memory_allocated()  # the model and the preloaded frames
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch, img_id in zip(batches, img_ids, strict=True):
            clock.frame = {}
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = model(return_loss=False, rescale=True, **batch)[0]
            torch.cuda.synchronize()
            total = time.perf_counter() - t0
            if hasattr(clock, "resolve"):
                clock.resolve()
            frames.append(dict(clock.frame, total=total,
                               first=batch["img_metas"][0][0]["frame_id"] == 0))
            for label, b in enumerate(result):
                for x1, y1, x2, y2, s in np.asarray(b).tolist():
                    dets.append(dict(image_id=img_id, bbox=[x1, y1, x2 - x1, y2 - y1], score=s,
                                     category_id=int(dataset.cat_ids[label])))
    return frames, dets, (torch.cuda.max_memory_allocated() - base) / 2**30


def coco_ap(gt, dets, img_ids, dataset):
    if not dets:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.params.imgIds = img_ids
        ev.params.catIds = list(dataset.cat_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])


def ms(xs) -> float:
    return 1000 * st.mean(xs) if xs else 0.0


def summarise(frames, instrumented):
    rest = [f for f in frames if not f["first"]]
    inst_rest = [f for f in instrumented if not f["first"]]
    engaged = [f for f in rest if f.get("engaged")]
    first_ms, other_ms = ms([f["total"] for f in frames if f["first"]]), ms([f["total"] for f in rest])
    return dict(
        # FPS over a whole video of ImageNet VID val's mean length (176,126 / 555 frames).
        fps_mean_video=round(1000 * VAL_MEAN_LEN / (first_ms + (VAL_MEAN_LEN - 1) * other_ms), 2),
        frames=len(frames), videos=sum(f["first"] for f in frames),
        ms_per_frame=round(ms([f["total"] for f in frames]), 2),
        fps=round(len(frames) / sum(f["total"] for f in frames), 2),
        ms_first_frame=round(ms([f["total"] for f in frames if f["first"]]), 2),
        ms_other_frames=round(ms([f["total"] for f in rest]), 2),
        ms_p90_other=round(1000 * float(np.quantile([f["total"] for f in rest], 0.9)), 2),
        # From the instrumented pass, whose synchronisations add time: its
        # total is reported beside the clean one.
        ms_other_frames_instrumented=round(ms([f["total"] for f in inst_rest]), 2),
        breakdown_ms=dict(
            {k: round(ms([f.get(k, 0.0) for f in inst_rest]), 2)
             for k in ("backbone_fpn", "aggregation", "head")},
            other=round(ms([f["total"] - sum(f.get(k, 0.0) for k in
                                             ("backbone_fpn", "aggregation", "head"))
                            for f in inst_rest]), 2)),
        gather_ms_per_video=round(ms([f.get("gather", 0.0) for f in instrumented
                                      if f["first"]]), 2),
        engaged_share=round(len(engaged) / max(len(rest), 1), 3),
        levels_mean=round(st.mean(f.get("levels", 5) for f in rest), 2),
        queries_mean=round(st.mean(f["queries"] for f in engaged), 1) if engaged else 0,
        keys_mean=round(st.mean(f["keys"] for f in engaged), 1) if engaged else 0,
        ms_aggregation_engaged=round(ms([f.get("aggregation", 0.0) for f in inst_rest
                                         if f.get("aggregation", 0.0) > 1e-4]), 2),
    )


def main():
    args = parse_args()
    torch.manual_seed(0)
    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    starts = [i for i, info in enumerate(dataset.data_infos) if info["frame_id"] == 0]
    ends = starts[1:] + [len(dataset)]
    index = [i for v in (int(x) for x in args.videos.split(","))
             for i in range(starts[v], min(ends[v], starts[v] + args.max_frames))]
    loader = DataLoader(Subset(dataset, index), batch_size=1, shuffle=False, num_workers=8,
                        collate_fn=collate_video_test)
    device = torch.device("cuda")
    batches = [to_device(b, device) for b in loader]  # every frame on the GPU up front
    img_ids = [int(dataset.img_ids[i]) for i in index]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(dataset.ann_file)
    warm = sum(1 for i in index if i < ends[int(args.videos.split(",")[0])])
    print(f"{len(batches)} frames of {len(args.videos.split(','))} videos on the GPU; "
          f"{torch.cuda.get_device_name()}", flush=True)

    wanted = set(args.settings.split(",")) if args.settings else None
    if args.warmup_s > 0:
        model = build(args.config, args.ckpt[0], SETTINGS[0][1])
        t_end = time.perf_counter() + args.warmup_s
        with torch.no_grad():
            while time.perf_counter() < t_end:
                for b in batches[:warm]:
                    model(return_loss=False, rescale=True, **b)  # no Clock: nothing timed
        torch.cuda.synchronize()
        del model
        print(f"warmed up for {args.warmup_s:.0f} s", flush=True)
    out = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, note=args.note,
               cudnn_tf32=torch.backends.cudnn.allow_tf32, videos=args.videos,
               max_frames=args.max_frames, results=[])
    # The first three frames of every video: with CUDA graphs, each new input
    # shape (a video's frames, its reference chunks) is captured on first
    # sight, and that must not land in the timed pass.
    first_frames = [i for i, b in enumerate(batches)
                    if b["img_metas"][0][0]["frame_id"] < 3]
    for ckpt in args.ckpt:
        for engine in args.engines.split(","):
            for name, options, tf32 in SETTINGS:
                if wanted and name not in wanted:
                    continue
                torch.backends.cuda.matmul.allow_tf32 = tf32
                model = build(args.config, ckpt, options, engine)
                clock = Clock(model, instrument=False)  # once per model: wrappers must not stack
                run(model, clock, batches[:warm], img_ids[:warm], dataset)  # warm-up: one video
                run(model, clock, [batches[i] for i in first_frames],
                    [img_ids[i] for i in first_frames], dataset)
                frames, dets, peak = run(model, clock, batches, img_ids, dataset)
                ap, ap50 = coco_ap(gt, dets, img_ids, dataset)
                clock.frame = {}  # its wrappers stay underneath; keep them off the last frame
                timers = Clock(model, instrument=True)  # the breakdown, in a second pass
                instrumented, _, _ = run(model, timers, batches, img_ids, dataset)
                row = dict(ckpt=ckpt, setting=name if engine == "eager" else f"{name}@{engine}",
                           engine=engine, options=options, matmul_tf32=tf32,
                           peak_mem_gb_above_inputs=round(peak, 2), AP=round(ap, 4),
                           AP50=round(ap50, 4), **summarise(frames, instrumented))
                out["results"].append(row)
                print(json.dumps(row), flush=True)
                del model
                torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
