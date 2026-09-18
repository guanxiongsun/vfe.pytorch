"""Summarise an mmcv-format training log (``*.log.json``) and check its
learning rates against the config's schedule.

Works on logs from ``vfe.cli.train`` and from the original mmdet runs alike::

    python tools/analyze_train_log.py WORK_DIR/20260917_093000.log.json \\
        --config configs/vid/mamba/mamba_r101_dc5_6x.py [--iters-per-epoch 13711]

Prints, per logged interval, the averaged losses, gradient norm, time and memory
(``--no-rows`` skips them); then whether every logged LR equals the schedule's
value for that iteration (at the log's 5-decimal precision).
``--iters-per-epoch`` defaults to the length the original 8-GPU run had; pass
the truncated length for short runs.

``--compare OTHER.log.json`` adds per-epoch means of the losses, accuracy and
gradient norm for both logs side by side, over the epochs both contain, and
their evaluations: how a reproduction run is overlaid on the original log.
"""

from __future__ import annotations

import argparse
import json

from vfe.config import Config

KEYS = ("loss_rpn_cls", "loss_rpn_bbox", "loss_cls", "acc", "loss_bbox", "loss", "grad_norm",
        "time", "data_time", "memory")


def scheduled_lr(cfg, epoch: int, global_iter: int) -> float:
    """``StepLrScheduler``'s LR (first group) for 0-based ``epoch`` and ``global_iter``."""
    lr_cfg = cfg.lr_config
    steps = lr_cfg["step"] if isinstance(lr_cfg["step"], (list, tuple)) else [lr_cfg["step"]]
    regular = cfg.optimizer["lr"] * lr_cfg.get("gamma", 0.1) ** sum(epoch >= s for s in steps)
    if lr_cfg.get("warmup") and global_iter < lr_cfg["warmup_iters"]:
        if lr_cfg["warmup"] != "linear":
            raise NotImplementedError(f"warmup {lr_cfg['warmup']!r}")
        k = (1 - global_iter / lr_cfg["warmup_iters"]) * (1 - lr_cfg["warmup_ratio"])
        return regular * (1 - k)
    return regular


CURVE_KEYS = ("loss", "loss_cls", "loss_bbox", "loss_rpn_cls", "loss_rpn_bbox", "acc", "grad_norm")


def load(path: str) -> tuple[list[dict], list[dict]]:
    with open(path) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    return ([e for e in entries if e.get("mode") == "train"],
            [e for e in entries if e.get("mode") == "val"])


def epoch_means(train: list[dict]) -> dict[int, dict[str, float]]:
    by_epoch: dict[int, list[dict]] = {}
    for e in train:
        by_epoch.setdefault(e["epoch"], []).append(e)
    return {epoch: {k: sum(e[k] for e in es if k in e) / n
                    for k in CURVE_KEYS if (n := sum(k in e for e in es))}
            for epoch, es in by_epoch.items()}


def compare(train: list[dict], val: list[dict], other_path: str) -> None:
    other_train, other_val = load(other_path)
    mine, theirs = epoch_means(train), epoch_means(other_train)
    common = sorted(set(mine) & set(theirs))
    print(f"\nper-epoch means, this log vs {other_path} (relative difference):")
    for epoch in common:
        cells = [f"{k} {mine[epoch][k]:.4g}/{theirs[epoch][k]:.4g} "
                 f"({(mine[epoch][k] - theirs[epoch][k]) / theirs[epoch][k]:+.1%})"
                 for k in CURVE_KEYS if k in mine[epoch] and k in theirs[epoch]]
        print(f"  epoch {epoch}: " + ", ".join(cells))
    theirs_val = {e["epoch"]: e for e in other_val}
    for e in val:
        if e["epoch"] in theirs_val:
            o = theirs_val[e["epoch"]]
            print(f"  val epoch {e['epoch']}: AP50 " + ", ".join(
                f"{k} {100 * e[k]:.2f}/{100 * o[k]:.2f}" for k in ("all", "fast", "medium", "slow")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--config", required=True)
    ap.add_argument("--iters-per-epoch", type=int, default=13711)
    ap.add_argument("--no-rows", action="store_true", help="skip the per-interval rows")
    ap.add_argument("--compare", metavar="OTHER_LOG", help="overlay per-epoch means on this log")
    args = ap.parse_args()
    cfg = Config.fromfile(args.config)
    train, val = load(args.log)

    if not args.no_rows:
        print("epoch  iter " + " ".join(f"{k:>12s}" for k in KEYS))
        for e in train:
            print(f"{e['epoch']:5d} {e['iter']:5d} "
                  + " ".join(f"{e[k]:12.5g}" if k in e else f"{'-':>12s}" for k in KEYS))
    for e in val:
        print(f"val epoch {e['epoch']}: AP50 all {e.get('all')}, fast {e.get('fast')}, "
              f"medium {e.get('medium')}, slow {e.get('slow')}")

    mismatches = []
    for e in train:
        global_iter = (e["epoch"] - 1) * args.iters_per_epoch + e["iter"] - 1
        expected = round(scheduled_lr(cfg, e["epoch"] - 1, global_iter), 5)
        if e["lr"] != expected:
            mismatches.append((e["epoch"], e["iter"], e["lr"], expected))
    print(f"LR check: {len(train) - len(mismatches)} of {len(train)} logged LRs match the schedule")
    for epoch, it, got, expected in mismatches[:10]:
        print(f"  epoch {epoch} iter {it}: logged {got}, schedule {expected}")
    if train:
        steady = [e for e in train if e["iter"] > 50]  # the first interval includes worker start-up
        if steady:
            mean_time = sum(e["time"] for e in steady) / len(steady)
            print(f"mean time per iteration (after the first interval): {mean_time:.3f} s; "
                  f"peak memory {max(e.get('memory', 0) for e in train)} MiB")
    if args.compare:
        compare(train, val, args.compare)


if __name__ == "__main__":
    main()
