"""Parity check: optimiser construction and LR schedule, vfe vs mmcv.

* ``optim/<recipe>`` -- every parameter's effective optimiser settings (lr,
  weight decay, momentum / betas, ...), keyed by parameter name, as JSON.
  Compared exactly. Group layout is deliberately not compared: vfe merges
  parameters with identical settings, which is equivalent for SGD and AdamW.
  Recipes: MAMBA's SGD on the real MAMBA model; STPN's AdamW (``custom_keys``:
  no decay on norms and position tables) on the Swin-T detector, standing in
  for the prompted Swin until it is ported.
* ``lr/<recipe>`` -- min and max LR over all param groups at every training
  iteration of the full schedule (82,266 iterations for MAMBA 6x, 123,399 for
  STPN 9x). The mmcv side drives the real ``StepLrUpdaterHook`` through a
  minimal runner, so hook semantics (warmup boundary, epoch steps) are under
  test, not re-derived. Compared exactly.

Usage:
    conda run -n vfe       --no-capture-output python tools/checks/parity_optim.py --impl mmdet --out A.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_optim.py --impl vfe   --out B.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_optim.py --compare A.pt B.pt
"""

import argparse
import json
import sys
from pathlib import Path

import parity_detector as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MAMBA = REPO_ROOT / "configs/vid/mamba/mamba_r101_dc5_6x.py"
STPN = REPO_ROOT / "configs/vid/stpn/stpn_swint_adam_9x.py"
ITERS_PER_EPOCH = 13711  # from the original logs: 8 GPUs x 1 image

SETTING_KEYS = ("lr", "weight_decay", "momentum", "dampening", "nesterov", "betas", "eps",
                "amsgrad")


def settings_by_name(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    out = {}
    for group in optimizer.param_groups:
        setting = {k: group[k] for k in SETTING_KEYS if k in group}
        for param in group["params"]:
            out[names[id(param)]] = setting
    return json.dumps(out, sort_keys=True)


def run(impl):
    if impl == "mmdet":
        from mmcv import Config
        from mmcv.runner import build_optimizer
        from mmcv.runner.hooks.lr_updater import StepLrUpdaterHook

        from mmdet.models import build_detector, build_model

        class Runner:
            pass

        def schedule(optimizer, lr_config, epochs):
            cfg = dict(lr_config)
            assert cfg.pop("policy") == "step"
            hook = StepLrUpdaterHook(**cfg)
            runner = Runner()
            runner.optimizer, runner.epoch, runner.iter = optimizer, 0, 0
            hook.before_run(runner)
            lows, highs = [], []
            for epoch in range(epochs):
                runner.epoch = epoch
                hook.before_train_epoch(runner)
                for _ in range(ITERS_PER_EPOCH):
                    hook.before_train_iter(runner)
                    lrs = [g["lr"] for g in optimizer.param_groups]
                    lows.append(min(lrs))
                    highs.append(max(lrs))
                    runner.iter += 1
            return lows, highs
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.config import Config
        from vfe.engine import build_lr_scheduler, build_optimizer
        from vfe.models.builder import build_detector, build_model

        def schedule(optimizer, lr_config, epochs):
            scheduler = build_lr_scheduler(optimizer, lr_config)
            lows, highs, global_iter = [], [], 0
            for epoch in range(epochs):
                scheduler.before_epoch(epoch)
                for _ in range(ITERS_PER_EPOCH):
                    scheduler.before_iter(global_iter)
                    lrs = [g["lr"] for g in optimizer.param_groups]
                    lows.append(min(lrs))
                    highs.append(max(lrs))
                    global_iter += 1
            return lows, highs

    out = {}
    mamba_cfg = Config.fromfile(str(MAMBA))
    stpn_cfg = Config.fromfile(str(STPN))
    recipes = [
        ("mamba", build_model(mamba_cfg.model), mamba_cfg),
        ("stpn_swin", build_detector(pd.detector_cfg(Config, "swin")), stpn_cfg),
    ]
    for name, model, cfg in recipes:
        optimizer = build_optimizer(model, cfg.optimizer)
        out[f"optim/{name}"] = pd.encode_str(settings_by_name(model, optimizer))
        lows, highs = schedule(optimizer, cfg.lr_config, cfg.total_epochs)
        out[f"lr/{name}/min"] = torch.tensor(lows, dtype=torch.float64)
        out[f"lr/{name}/max"] = torch.tensor(highs, dtype=torch.float64)
        print(f"{name}: {len(optimizer.param_groups)} param groups, {len(lows)} iterations, "
              f"lr {lows[0]:.3e} -> {lows[499]:.3e} -> {lows[500]:.3e} -> {lows[-1]:.3e}")
    return out


def compare(path_a, path_b):
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)
    failures = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            failures.append(f"{key}: only in {'A' if key in a else 'B'}")
        elif a[key].shape != b[key].shape or not torch.equal(a[key], b[key]):
            if key.startswith("optim/"):
                sa = json.loads(bytes(a[key].tolist()).decode())
                sb = json.loads(bytes(b[key].tolist()).decode())
                diff = [n for n in sorted(set(sa) | set(sb)) if sa.get(n) != sb.get(n)]
                failures.append(f"{key}: {len(diff)} parameters differ, e.g. {diff[:3]}: "
                                f"{[sa.get(n) for n in diff[:1]]} vs {[sb.get(n) for n in diff[:1]]}")
            else:
                n = (a[key] != b[key]).sum().item() if a[key].shape == b[key].shape else "?"
                failures.append(f"{key}: {n} iterations differ")
    for note in failures:
        print(f"FAIL  {note}")
    print("-" * 70)
    print(f"{len(failures)} of {len(set(a) | set(b))} artifact(s) differ" if failures
          else f"OPTIM PARITY OK ({len(a)} artifacts, all exact)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.impl and args.out:
        torch.save(run(args.impl), args.out)
        print(f"saved -> {args.out}")
    else:
        ap.error("pass either --impl/--out or --compare")
