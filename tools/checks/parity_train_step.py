"""Parity check: training steps on real data, vfe vs mmdet (Phases 6c, 7).

Each side builds the model from ``--config`` (MAMBA 6x by default; STPN with
``configs/vid/stpn/stpn_swint_adam_9x.py``), loads the released checkpoint and
takes two optimiser steps on real VID training samples, built by its own data
path from ``--data-config`` (default: the same config; ``parity_vid_train_data``
shows the two data paths are bit-identical). CPU only: the legacy CUDA stack
mis-computes some gradients (see docs/rewrite-plan.md).

A step is what mmcv's runner did per iteration: ``train_step`` (forward_train
and parse_losses), then ``OptimizerHook``: zero_grad, backward, gradient
clipping if the config has it (MAMBA: 35, L2; STPN: none), optimiser step
(MAMBA: SGD; STPN: AdamW with per-parameter decay). Swin's DropPath is active
in training; its draws come from the seeded CPU generator.

After each step's results are recorded the weights are restored, so every step
runs forward and backward on the released weights on both stacks, while the
optimiser state carries over (the second step still exercises momentum or
moment accumulation and bias correction). Without the restore, AdamW's first
step would already leave the stacks on slightly different weights: Adam
normalises every element, and elements whose gradient is within the stacks'
roundoff (sums over thousands of tokens differ by ~1e-6 absolute) take steps of
different size or sign. STPN's second step then saw different RPN proposals.

Recorded per step, under ``step<k>/``:

* ``exact/batch`` -- digest of the batch: both sides must train on identical
  inputs. ``exact/with_grad`` -- names of the parameters that received a
  gradient (the legacy DDP ran with ``find_unused_parameters=False``, so this
  must be every trainable parameter). ``exact/sampled`` -- the RoI sampler's
  positive and negative picks. ``exact/buffers`` -- digest of all buffers, and
  ``exact/buffers_unchanged`` (frozen BatchNorm statistics must not move).
* ``loss/*`` (the logged values), ``grad_norm`` (total L2 norm before any
  clipping), ``proposals/*`` (key frame, and MAMBA's reference frames).
* ``grad/<param>`` (before clipping), ``state/<param>.<name>`` (the optimiser's
  state tensors after the step: ``momentum_buffer``; ``exp_avg``,
  ``exp_avg_sq``), ``exact/hparams/<param>`` (lr, weight decay and the rest).
* ``residual/<param>``: on each stack, the change the step made against the
  optimiser's formula applied to that stack's own state and hyperparameters, in
  float64 (SGD: ``-lr * momentum_buffer``; AdamW: decoupled decay and the
  bias-corrected moment ratio). Every element's deviation, divided by the
  float32 spacing at the old and new value (plus 1e-6 of the step), must be at
  most 1: exactly the rounding of an in-place float32 update. Comparing the
  changes across stacks instead would mostly measure that rounding (a step of
  ~1e-5 on a weight near 1 is quantised in ~1e-7 increments) and, for AdamW,
  Adam's amplification of roundoff-sized gradients; the state is what is
  compared across stacks.

Parameters up to ``SMALL`` elements are saved whole; larger ones as a sketch
(key suffix ``:sketch``), ``[norm, 16 bins]``, where
  each bin sums a hash-chosen subset of the elements with hash-chosen signs.
  The sketch is linear and identical on both sides, so the bins of ``a - b``
  are the difference of the bins, and their root sum of squares estimates
  ``||a - b||`` (within tens of percent, enough to judge orders of magnitude).

Every float comparison is ``||a - b|| / ||a||`` against a per-kind tolerance.

Usage:
    conda run -n vfe --no-capture-output python tools/checks/parity_train_step.py --impl mmdet --checkpoint CKPT --out A.pt
    python tools/checks/parity_train_step.py --impl vfe --checkpoint CKPT --out B.pt
    python tools/checks/parity_train_step.py --compare A.pt B.pt
    (STPN: add --config configs/vid/stpn/stpn_swint_adam_9x.py to both runs)
"""

import argparse
import gc
import hashlib
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import parity_vid_pipeline as pp
import parity_vid_train_data as td
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs/vid/mamba/mamba_r101_dc5_6x.py"
SAMPLES = (711, 40000)  # a downscaled, flipped frame with a box on the edge; an unflipped one
SEED = 1466607766
SMALL = 4096
BINS = 16
HPARAMS = ("lr", "weight_decay", "momentum", "dampening", "nesterov", "betas", "eps", "amsgrad")


T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---- sketches ----------------------------------------------------------------------

_HASH_CACHE = {}


def _hash(n):
    """A well-mixed 32-bit hash of each position, in int64 arithmetic that no
    step can overflow, so torch 1.10 and 2.10 compute identical values."""
    if n not in _HASH_CACHE:
        h = torch.arange(n, dtype=torch.int64)
        for mult in (2654435761, 73244475):
            h = torch.remainder(h * mult, 1 << 32)
            h = h ^ torch.div(h, 1 << 16, rounding_mode="floor")
        _HASH_CACHE.clear()  # one size at a time: parameters are visited in order
        _HASH_CACHE[n] = (torch.remainder(h, BINS),
                          torch.remainder(torch.div(h, BINS, rounding_mode="floor"), 2) * 2 - 1)
    return _HASH_CACHE[n]


def put(out, key, t):
    """Save ``t`` whole if small, else its sketch under ``key + ':sketch'``."""
    t = t.detach().reshape(-1).double()
    if t.numel() <= SMALL:
        out[key] = t.clone()
        return
    bins, signs = _hash(t.numel())
    sk = torch.zeros(BINS + 1, dtype=torch.float64)
    sk[0] = t.norm()
    sk[1:].index_add_(0, bins, t * signs.double())
    out[key + ":sketch"] = sk


def digest_tensors(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().contiguous().numpy().tobytes())
    return pp.encode_str(h.hexdigest())


def batch_digest(batch):
    h = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        if key.endswith("img_metas"):
            h.update(pp.canonical(value).encode())
            continue
        for t in value if isinstance(value, list) else [value]:
            h.update(t.contiguous().numpy().tobytes())
    return pp.encode_str(h.hexdigest())


# ---- the two stacks ------------------------------------------------------------------

def load_batches(impl, data_config):
    """Build only the VID training set, take the samples, and free the set (its
    annotation index is larger than the model)."""
    dataset, batch_of, _, _ = td.build(impl, select=lambda train: train[0], config=data_config)
    batches = []
    for idx in SAMPLES:
        random.seed(1000 + idx)
        np.random.seed(1000 + idx)
        batches.append(batch_of(idx))
    del dataset, batch_of
    gc.collect()
    return batches


def build(impl, config, checkpoint):
    if impl == "mmdet":
        from mmcv import Config
        from mmcv.runner import OptimizerHook, build_optimizer, load_checkpoint

        from mmdet.models import build_model

        cfg = Config.fromfile(str(config))
        model = build_model(cfg.model)
        load_checkpoint(model, checkpoint, map_location="cpu")
        optimizer = build_optimizer(model, cfg.optimizer)
        hook = OptimizerHook(**cfg.optimizer_config)

        def forward(batch):
            outputs = model.train_step(batch, optimizer)
            return outputs["loss"], outputs["log_vars"]

        def clip():
            return hook.clip_grads(model.parameters()) if hook.grad_clip else None
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.config import Config
        from vfe.engine import build_optimizer, clip_grads
        from vfe.models.builder import build_model
        from vfe.models.checkpoint import load_checkpoint
        from vfe.models.detectors.base import parse_losses

        cfg = Config.fromfile(str(config))
        model = build_model(cfg.model)
        load_checkpoint(model, checkpoint, map_location="cpu")
        optimizer = build_optimizer(model, cfg.optimizer)
        grad_clip = cfg.optimizer_config.get("grad_clip")

        def forward(batch):
            return parse_losses(model(**batch))

        def clip():
            return clip_grads(model.parameters(), **grad_clip) if grad_clip else None
    return model, optimizer, forward, clip


def hparams(group):
    """The optimiser settings a parameter trains with, as a float64 vector."""
    values = []
    for key in HPARAMS:
        if key in group:
            item = group[key]
            values.extend(float(x) for x in (item if isinstance(item, (tuple, list)) else [item]))
    return torch.tensor(values, dtype=torch.float64)


def ulp32(x):
    """The float32 spacing at each element of ``x``, in float64."""
    x = x.detach().float().abs()
    return (torch.nextafter(x, torch.full_like(x, float("inf"))) - x).double()


def optimizer_update(param, state, group):
    """The step's change in float64, from the state the optimiser holds after
    the step: what PyTorch 1.10 and 2.10 both compute for SGD with momentum and
    for AdamW (decoupled decay)."""
    if group.get("maximize") or group.get("amsgrad") or group.get("nesterov"):
        raise NotImplementedError("maximize / amsgrad / nesterov")
    if "momentum_buffer" in state:
        return -group["lr"] * state["momentum_buffer"].double()
    if "exp_avg_sq" not in state:
        raise NotImplementedError("only SGD with momentum and AdamW are checked")
    beta1, beta2 = group["betas"]
    step = float(state["step"])
    lr, eps = group["lr"], group["eps"]
    denom = state["exp_avg_sq"].double().sqrt() / math.sqrt(1 - beta2**step) + eps
    new = param * (1 - lr * group["weight_decay"])
    new = new - lr / (1 - beta1**step) * state["exp_avg"].double() / denom
    return new - param


def capture(obj, name, sink):
    """Wrap ``obj.name`` on the instance so each call's result is appended to ``sink``."""
    original = getattr(obj, name)

    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        sink.append(result)
        return result

    setattr(obj, name, wrapper)


def run(impl, config, data_config, checkpoint, threads, debug_param=None):
    if threads:
        torch.set_num_threads(threads)
    log(f"{impl}: torch {torch.__version__}, {torch.get_num_threads()} threads")
    batches = load_batches(impl, data_config)
    log("batches built")
    model, optimizer, forward, clip = build(impl, config, checkpoint)
    model.train()
    log("model built, checkpoint loaded")

    detector = model.detector
    rpn_calls, ref_rpn_calls, samples = [], [], []
    capture(detector.rpn_head, "forward_train", rpn_calls)
    capture(detector.rpn_head, "simple_test_rpn", ref_rpn_calls)
    capture(detector.roi_head.bbox_sampler, "sample", samples)

    params = dict(model.named_parameters())
    out = {"exact/trainable": pp.encode_str(",".join(sorted(
        name for name, p in params.items() if p.requires_grad)))}

    for k, batch in enumerate(batches):
        tag = f"step{k}"
        rpn_calls.clear()
        ref_rpn_calls.clear()
        samples.clear()
        buffers_before = digest_tensors([b for _, b in sorted(model.named_buffers())])
        out[f"{tag}/exact/batch"] = batch_digest(batch)

        torch.manual_seed(SEED + k)
        loss, log_vars = forward(batch)
        log(f"{tag}: forward, loss {float(loss.detach()):.6f}")
        optimizer.zero_grad()
        loss.backward()
        log(f"{tag}: backward")

        for name, value in log_vars.items():
            out[f"{tag}/loss/{name}"] = torch.tensor(float(value), dtype=torch.float64)
        (proposals,) = rpn_calls[0][1]
        out[f"{tag}/proposals/key"] = proposals.detach()
        # MAMBA feeds only topk reference proposals to the RoI head. The legacy
        # implementation truncates the RPN's returned list in place, whereas
        # vfe returns a new truncated list; slicing here records the effective
        # RoI-head input on both sides rather than that implementation detail.
        # (STPN has no reference proposals.)
        topk = getattr(detector.roi_head.bbox_head, "topk", None)
        for i, ref in enumerate(ref_rpn_calls[0] if ref_rpn_calls else []):
            out[f"{tag}/proposals/ref{i}"] = ref[:topk].detach()
        (sampled,) = samples
        out[f"{tag}/exact/sampled"] = torch.cat([sampled.pos_inds, sampled.neg_inds]).long()
        out[f"{tag}/exact/num_pos"] = torch.tensor(len(sampled.pos_inds))

        with_grad = sorted(name for name, p in params.items() if p.grad is not None)
        out[f"{tag}/exact/with_grad"] = pp.encode_str(",".join(with_grad))
        for name in with_grad:
            put(out, f"{tag}/grad/{name}", params[name].grad)

        grad_norm = torch.stack([params[n].grad.detach().double().norm() for n in with_grad]).norm()
        out[f"{tag}/grad_norm"] = grad_norm
        clipped_norm = clip()
        before = {name: params[name].detach().clone() for name in with_grad}
        if debug_param:
            out[f"{tag}/debug/grad"] = params[debug_param].grad.detach().clone()
            out[f"{tag}/debug/before"] = before[debug_param].clone()
        optimizer.step()
        log(f"{tag}: grad norm {float(grad_norm):.4f} "
            f"({'clipping configured' if clipped_norm is not None else 'no clipping'}), step taken")

        for name in with_grad:
            p = params[name]
            state = optimizer.state[p]
            group = next(g for g in optimizer.param_groups if any(q is p for q in g["params"]))
            out[f"{tag}/exact/hparams/{name}"] = hparams(group)
            for key, value in sorted(state.items()):
                if key != "step" and isinstance(value, torch.Tensor):
                    put(out, f"{tag}/state/{name}.{key}", value)
            update = p.detach().double() - before[name].double()
            expected = optimizer_update(before[name].double(), state, group)
            bound = ulp32(before[name]) + ulp32(p.detach()) + 1e-6 * expected.abs()
            residual = ((update - expected).abs() / bound).max().item()
            out[f"{tag}/residual/{name}"] = torch.tensor(residual, dtype=torch.float64)
            with torch.no_grad():
                p.copy_(before[name])  # the next step starts from the same weights
        if debug_param:
            out[f"{tag}/debug/after"] = params[debug_param].detach().clone()
            for key, value in optimizer.state[params[debug_param]].items():
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    out[f"{tag}/debug/state.{key}"] = value.detach().clone()
        del before
        buffers = digest_tensors([b for _, b in sorted(model.named_buffers())])
        out[f"{tag}/exact/buffers"] = buffers
        out[f"{tag}/exact/buffers_unchanged"] = torch.tensor(bool(torch.equal(buffers, buffers_before)))
        log(f"{tag}: recorded")
        gc.collect()
    return out


# ---- comparison ------------------------------------------------------------------------

# ||a - b|| / ||a|| allowed per kind. See docs/rewrite-plan.md (Phase 6c) for how
# these were set: the stacks' CPU kernels differ by ~1e-6 relative per op, and
# a ResNet-101 backward pass amplifies that.
TOLERANCES = {
    "loss": 1e-4,
    "grad_norm": 1e-4,
    "proposals": 1e-4,
    "grad": 1e-3,
    "state": 2e-3,  # exp_avg_sq squares the gradient, doubling its relative difference
    "residual": 1.0,  # in units of float32 spacing: two rounding steps, half a spacing each
}


def relative_diff(a, b, is_sketch):
    a, b = a.double(), b.double()
    if is_sketch:
        diff = (a[1:] - b[1:]).norm().item()
        scale = a[0].item()
        diff = max(diff, abs(a[0].item() - b[0].item()))
    else:
        diff = (a - b).norm().item()
        scale = a.norm().item()
    if diff == 0:
        return 0.0
    return diff / scale if scale > 0 else math.inf


def compare(path_a, path_b):
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)
    failures, worst, roundoff_grads = [], {}, []
    for side, artifacts in (("A", a), ("B", b)):
        trainable = artifacts.get("exact/trainable")
        for key in sorted(k for k in artifacts if k.endswith("/exact/with_grad")):
            if not torch.equal(artifacts[key], trainable):
                failures.append(f"{key}: {side}'s parameters with gradients are not "
                                "exactly its trainable parameters")
        for key in sorted(k for k in artifacts if k.endswith("/grad_norm")):
            print(f"{side} {key}: {artifacts[key].item():.4f}")
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            failures.append(f"{key}: only in {'A' if key in a else 'B'}")
            continue
        if "/debug/" in key:
            continue
        if "/exact/" in key or key.startswith("exact/"):
            if a[key].shape != b[key].shape or not torch.equal(a[key], b[key]):
                failures.append(f"{key}: differs")
            continue
        kind = key.split("/")[1]
        if kind == "residual":  # per stack, against AdamW's formula
            rel = max(a[key].item(), b[key].item())
            worst.setdefault(kind, []).append((rel, key.split("/", 2)[-1]))
            if rel > TOLERANCES[kind]:
                failures.append(f"{key}: update deviates from AdamW's formula by {rel:.3g} "
                                "float32 spacings")
            continue
        if a[key].shape != b[key].shape:
            failures.append(f"{key}: shape {tuple(a[key].shape)} vs {tuple(b[key].shape)}")
            continue
        name = key.split("/", 2)[-1]
        step = key.split("/")[0]
        adaptive = any(k.startswith(f"{step}/state/") and ".exp_avg" in k for k in a)
        if adaptive and kind == "state" and "ref_fc_embed.bias" in name:
            # Adam divides this roundoff-level gradient (see below) by its own
            # magnitude, turning noise into an lr-sized step of random sign.
            roundoff_grads.append((name, 0.0))
            continue
        if name.split(".")[-2:] == ["qkv", "bias"] and not key.endswith(":sketch"):
            # Swin's key bias adds q . b_k to every logit in a query's row, which
            # softmax ignores: its gradient is exactly zero and both stacks
            # return roundoff (like ref_fc_embed.bias below). Compare the query
            # and value thirds; require the key third's gradient to be
            # roundoff; leave the key third out of Adam's state and update,
            # which scale that roundoff up to lr-sized steps of random sign.
            third = a[key].numel() // 3
            q_a, k_a, v_a = a[key].split(third)
            q_b, k_b, v_b = b[key].split(third)
            if kind == "grad":
                magnitude = max(k_a.abs().max().item(), k_b.abs().max().item())
                if magnitude > 1e-7:
                    failures.append(f"{key}: key-bias gradient is not roundoff (max {magnitude:.3e})")
                roundoff_grads.append((name, magnitude))
            kept = (torch.cat([q_a, v_a]), torch.cat([q_b, v_b])) if adaptive or kind == "grad" \
                else (a[key], b[key])
            rel = relative_diff(*kept, False)
            worst.setdefault(kind, []).append((rel, name))
            if rel > TOLERANCES[kind]:
                failures.append(f"{key}: relative difference {rel:.3e} > {TOLERANCES[kind]:.0e}")
            continue
        if kind == "grad" and name.endswith("ref_fc_embed.bias"):
            # This bias shifts every attention logit in a row equally, so its
            # gradient is exactly zero by softmax shift-invariance. Different
            # kernels leave different roundoff; a relative comparison is
            # meaningless when both norms are about 1e-9. This same invariant
            # is independently checked in parity_mamba.py.
            magnitude = max(a[key].abs().max().item(), b[key].abs().max().item())
            if magnitude <= 1e-7:
                roundoff_grads.append((name, magnitude))
                continue
            failures.append(f"{key}: shift-invariant bias gradient is not roundoff "
                            f"(max {magnitude:.3e})")
            continue
        rel = relative_diff(a[key], b[key], key.endswith(":sketch"))
        worst.setdefault(kind, []).append((rel, name))
        if rel > TOLERANCES[kind]:
            failures.append(f"{key}: relative difference {rel:.3e} > {TOLERANCES[kind]:.0e}")

    for kind, values in sorted(worst.items()):
        values.sort(reverse=True)
        top = ", ".join(f"{name} {rel:.2e}" for rel, name in values[:3])
        print(f"{kind:10s} n={len(values):4d}  max {values[0][0]:.2e}  (tol {TOLERANCES[kind]:.0e})  {top}")
    if roundoff_grads:
        print(f"NOTE  {len(roundoff_grads)} shift-invariant bias artifacts (ref_fc_embed.bias, the "
              "key third of qkv.bias) hold only roundoff and are checked as such (largest "
              f"gradient {max(x[1] for x in roundoff_grads):.2e})")
    for note in failures:
        print(f"FAIL  {note}")
    print("-" * 70)
    print(f"{len(failures)} of {len(set(a) | set(b))} artifact(s) differ" if failures
          else f"TRAIN STEP PARITY OK ({len(a)} artifacts)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--checkpoint", help="the released checkpoint for --config")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--data-config", help="config whose training data to use (default: --config)")
    ap.add_argument("--debug-param", help="also save this parameter's full tensors (not compared)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (default: torch's)")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.impl and args.out and args.checkpoint:
        torch.save(run(args.impl, args.config, args.data_config or args.config, args.checkpoint,
                       args.threads, args.debug_param), args.out)
        log(f"saved -> {args.out}")
    else:
        ap.error("pass --impl/--checkpoint/--out, or --compare")
