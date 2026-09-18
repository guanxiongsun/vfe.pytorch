"""Parity check: vfe's MAMBA (memory bank, aggregator, RoI head, video
detector) vs the mmdet oracle.

Builds on ``parity_detector`` (imported for its fixtures, init checks and
set-matching comparison) and adds what is new in MAMBA:

* ``memory/exact/*`` -- ``MemoryBank`` driven through init / sample / update
  with small capacities, so the random-subset read and the random-replacement
  write both run. Compared bit for bit (the ``/exact/`` in the key enforces
  it): both are ``randperm`` + indexing + ``cat``, so any difference at all
  means a different permutation or a different kept count.
* ``agg/*`` -- ``MambaAggregator.forward_with_ref_x`` forward, and on CPU its
  gradients w.r.t. inputs and every parameter.
* ``model/init/*`` -- the full MAMBA from its real config after
  ``init_weights()``: pretrained ResNet exact, everything else by
  distribution. The aggregators are the interesting part: mmdet's init
  recursion never reaches them, so they keep torch's default ``nn.Linear``
  init, and a port that "fixed" that would change training.
* ``model/train/*`` -- ``forward_train`` losses for a key frame with two
  reference frames (CPU only, as in ``parity_detector``).
* ``model/video_*`` -- ``simple_test`` over short videos, one call per
  frame, so the memory state carries across calls:
  ``video_adaptive`` (the released test setting), ``video_smallmem`` (same,
  with a memory small enough that sampling and replacement are random), and
  ``video_fixed`` (the fixed-stride sliding window).

Everything random in MAMBA draws from the global CPU RNG, which is seeded
identically on both sides before each sequence.

Usage:
    conda run -n vfe       --no-capture-output python tools/checks/parity_mamba.py --impl mmdet --out A.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_mamba.py --impl vfe   --out B.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_mamba.py --compare A.pt B.pt
"""

import argparse
import sys
from pathlib import Path

import parity_detector as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_HW = pd.IMG_SHAPE[:2]
NUM_REFS = 6  # the released config uses 14; 6 keeps the CPU run short
NUM_LEFT = NUM_REFS // 2


def frame_meta(frame_id, frame_stride, num_left=NUM_LEFT):
    meta = dict(pd.IMG_METAS[0])
    meta.update(frame_id=frame_id, frame_stride=frame_stride, num_left_ref_imgs=num_left)
    return meta


def run(impl, device):
    if impl == "mmdet":
        from mmcv import Config

        from mmdet.models import build_model
        from mmdet.models.aggregators import MambaAggregator
        from mmdet.models.memory import MemoryBank
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.config import Config
        from vfe.models.aggregators import MambaAggregator
        from vfe.models.builder import build_model
        from vfe.models.memory import MemoryBank

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    out = {}

    # ---- memory bank ---------------------------------------------------------
    torch.manual_seed(1000)
    bank = MemoryBank(max_length=50, key_length=20)
    rows = pd.fixed((200, 8), 1001).to(device)
    steps = [
        ("init30", lambda: bank.init_memory(rows[:30])),
        ("sample_random", None),                 # 30 >= key_length: random subset
        ("update_append", lambda: bank.update(rows[30:45])),   # 45 < max_length: append
        ("update_overshoot", lambda: bank.update(rows[45:60])),  # 45 < 50: append to 60
        ("update_replace", lambda: bank.update(rows[60:70])),  # full: random replacement
        ("sample_after_replace", None),
        ("update_replace2", lambda: bank.update(rows[70:90])),
        ("reset_then_update", lambda: (bank.reset(), bank.update(rows[90:95]))),
        ("sample_all", None),                    # 5 < key_length: everything
    ]
    for tag, action in steps:
        if action is None:
            out[f"memory/exact/{tag}"] = bank.sample().cpu()
        else:
            action()
            out[f"memory/exact/{tag}/feat"] = bank.feat.cpu()
            out[f"memory/exact/{tag}/len"] = torch.tensor(len(bank))

    # ---- aggregator ----------------------------------------------------------
    agg = MambaAggregator(in_channels=1024, num_attention_blocks=16)
    pd.seed_params(agg, 1100)
    agg = agg.to(device)
    x = pd.fixed((300, 1024), 1101).to(device).requires_grad_(True)
    ref_x = pd.fixed((525, 1024), 1102).to(device).requires_grad_(True)
    y = agg.forward_with_ref_x(x, ref_x)
    out["agg/out"] = y.detach().cpu()
    if device == "cpu":
        named = sorted(agg.named_parameters())
        upstream = pd.fixed(tuple(y.shape), 1103)
        grads = torch.autograd.grad((y * upstream).sum(), [x, ref_x] + [p for _, p in named])
        out["agg/grad/x"] = grads[0]
        out["agg/grad/ref_x"] = grads[1]
        grad_by_name = {named[j][0]: grads[2 + j] for j in range(len(named))}
        for j in range(len(named)):
            if named[j][0] == "ref_fc_embed.bias":
                # Exactly zero in exact arithmetic: this bias shifts every
                # attention logit in a row by the same amount, and softmax is
                # shift-invariant. Both sides return roundoff (~1e-6), which
                # cannot be compared relatively -- so check that it *is*
                # roundoff. That also confirms the softmax runs over the
                # references, the only axis that makes the gradient vanish.
                ratio = (grad_by_name["ref_fc_embed.bias"].abs().max()
                         / grad_by_name["ref_fc_embed.weight"].abs().max())
                out["agg/grad/ref_fc_embed.bias/is_roundoff"] = torch.tensor(bool(ratio < 1e-5))
            else:
                out[f"agg/grad/{named[j][0]}"] = grads[2 + j]

    # ---- the model ---------------------------------------------------------------
    cfg = Config.fromfile(str(REPO_ROOT / "configs/vid/mamba/mamba_r101_dc5_3x.py"))
    torch.manual_seed(0)
    model = build_model(cfg.model)
    model.init_weights()
    pd.record_init(out, "model/init", model, exact_prefix="detector.backbone.")
    pd.seed_params(model, 1200, skip_prefix="detector.backbone.")
    model = model.to(device)
    bbox_head = model.detector.roi_head.bbox_head

    # Seed chosen so no training proposal scores tie; require_distinct checks.
    frames = pd.fixed((8, 3) + IMG_HW, 1211).to(device)

    if device == "cpu":
        model.train()
        img = frames[[0]]
        ref_img = frames[1:3][None]
        img_metas = [frame_meta(0, -1)]
        ref_img_metas = [[frame_meta(1, -1), frame_meta(2, -1)]]
        gt_bboxes = [torch.tensor(pd.GT[0][0], dtype=torch.float32)]
        gt_labels = [torch.tensor(pd.GT[0][1], dtype=torch.long)]
        with torch.no_grad():
            # The proposals the losses depend on, checked for ties first:
            # both the RoI sampler and the reference top-k select by position.
            all_x = model.detector.extract_feat(torch.cat((img, ref_img[0]), 0))
            x_key = [lvl[[0]] for lvl in all_x]
            x_ref = [lvl[1:] for lvl in all_x]
            props = model.detector.rpn_head.get_bboxes(
                *model.detector.rpn_head(x_key), img_metas=img_metas,
                cfg=model.detector.train_cfg.rpn_proposal,
            )
            ref_props = model.detector.rpn_head.simple_test_rpn(x_ref, ref_img_metas[0])
            for i, p in enumerate(props + ref_props):
                pd.require_distinct(p[:, 4], f"training proposals {i}")
            torch.manual_seed(1202)
            # ref_gt_* are required positionals in mmdet but unused by MAMBA.
            losses = model.forward_train(
                img, img_metas, gt_bboxes, gt_labels, ref_img, ref_img_metas,
                [torch.zeros(0, 5)], [torch.zeros(0, 2)],
            )
        for key in sorted(losses):
            values = losses[key] if isinstance(losses[key], list) else [losses[key]]
            for i, value in enumerate(values):
                out[f"model/train/{key}{i}"] = value.detach().float().reshape(-1).cpu()

    model.eval()

    def record_memory(tag):
        for i, aggregator in enumerate(bbox_head.aggregator):
            out[f"{tag}/memory{i}/len"] = torch.tensor(len(aggregator.memory_bank))

    def run_video(tag, stride, schedule):
        """``schedule``: (frame_id, key frame index, reference frame indices)."""
        torch.manual_seed(1300)
        for frame_id, key, refs in schedule:
            img_metas = [frame_meta(frame_id, stride)]
            ref_img = ref_img_metas = None
            if refs:
                ref_img = [frames[refs][None]]
                ref_img_metas = [[[frame_meta(r, stride) for r in refs]]]
            with torch.no_grad():
                results = model.simple_test(
                    frames[[key]], img_metas, ref_img=ref_img, ref_img_metas=ref_img_metas,
                    rescale=True,
                )
            pd.put_result(out, f"model/{tag}/frame{frame_id}", results[0])
            record_memory(f"model/{tag}/frame{frame_id}")

    adaptive = [(0, 0, list(range(1, 1 + NUM_REFS)))] + [(f, f, []) for f in (1, 2, 3)]
    run_video("video_adaptive", -1, adaptive)

    # A memory small enough that frame 0's 7 x 75 reference RoIs overflow it:
    # every later read samples and every write replaces, at random.
    for aggregator in bbox_head.aggregator:
        aggregator.memory_bank.max_length = 200
        aggregator.memory_bank.key_length = 100
    run_video("video_smallmem", -1, adaptive)
    for aggregator in bbox_head.aggregator:
        aggregator.memory_bank.max_length = 20000
        aggregator.memory_bank.key_length = 2000

    # Fixed stride 2: frame 0 brings the whole window (key frame at index
    # NUM_LEFT), even frames bring the next reference, odd frames nothing.
    window = list(range(NUM_REFS + 1))
    fixed_stride = [(0, NUM_LEFT, window), (1, 1, []), (2, 2, [NUM_REFS + 1]), (3, 3, [])]
    run_video("video_fixed", 2, fixed_stride)

    print(f"RAN   {len(out)} artifacts on {device}")
    return out


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
        pd.compare(*args.compare, atol=args.atol, rtol=args.rtol, rtol_overrides={},
                   label="MAMBA")
    elif args.impl and args.out:
        torch.save(run(args.impl, args.device), args.out)
        print(f"saved -> {args.out}")
    else:
        ap.error("pass either --impl/--out or --compare")
