"""Parity check: ``vfe.config.Config`` vs the legacy ``mmcv.Config`` oracle.

The two loaders live in different Python versions (mmcv needs the py3.8 ``vfe``
env), so they cannot be imported into one process. Instead each env dumps the
fully-resolved config to canonical JSON and we diff the two files.

Usage:
    conda run -n vfe       --no-capture-output python tools/checks/parity_config.py --loader mmcv --out /tmp/cfg_mmcv
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_config.py --loader vfe  --out /tmp/cfg_vfe
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_config.py --diff /tmp/cfg_mmcv /tmp/cfg_vfe
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIGS = [
    "configs/vid/mamba/mamba_r101_dc5_3x.py",
    "configs/vid/mamba/mamba_r101_dc5_6x.py",
    "configs/vid/stpn/stpn_swint_adam_9x.py",
    "configs/vid/stpn/stpn_swins_adam_9x.py",
]


def canonical(obj):
    """Normalise so the two loaders' outputs are comparable.

    Tuples become lists (JSON has no tuples) and dict ordering is dropped by
    ``sort_keys`` at dump time. Anything not JSON-native is repr'd.
    """
    if isinstance(obj, dict):
        return {str(k): canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def dump(loader, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if loader == "mmcv":
        from mmcv import Config
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.config import Config

    for rel in CONFIGS:
        path = REPO_ROOT / rel
        if not path.exists():
            raise SystemExit(f"{rel} is missing: every config listed in CONFIGS must exist")
        cfg = Config.fromfile(str(path))
        data = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg._cfg_dict)
        # An empty file loads as {} in both stacks, so the diff would compare
        # nothing and report OK. Each of these configs defines a model and a
        # dataset; anything less means the file is a stub.
        missing = [key for key in ("model", "data") if not data.get(key)]
        if missing:
            raise SystemExit(f"{rel}: loaded config has no {' or '.join(missing)} "
                             f"-- empty or stub file?")
        target = out_dir / (rel.replace("/", "__") + ".json")
        target.write_text(json.dumps(canonical(data), sort_keys=True, indent=2))
        print(f"DUMP  {rel} -> {target.name}")


def diff(dir_a, dir_b):
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    names = sorted({p.name for p in dir_a.glob("*.json")} | {p.name for p in dir_b.glob("*.json")})
    if not names:
        raise SystemExit("nothing to diff -- run the two --loader passes first")

    failures = 0
    for name in names:
        pa, pb = dir_a / name, dir_b / name
        if not (pa.exists() and pb.exists()):
            failures += 1
            print(f"FAIL  {name}: only in {'A' if pa.exists() else 'B'}")
            continue
        a, b = json.loads(pa.read_text()), json.loads(pb.read_text())
        if a == b:
            print(f"OK    {name}")
        else:
            failures += 1
            print(f"FAIL  {name}")
            for line in _describe(a, b):
                print(f"        {line}")

    print("-" * 70)
    print("CONFIG PARITY OK" if failures == 0 else f"{failures} config(s) differ")
    raise SystemExit(1 if failures else 0)


def _describe(a, b, path=""):
    """Yield human-readable leaf differences between two nested structures."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            sub = f"{path}.{key}" if path else key
            if key not in a:
                yield f"{sub}: missing in A (B={b[key]!r})"
            elif key not in b:
                yield f"{sub}: missing in B (A={a[key]!r})"
            else:
                yield from _describe(a[key], b[key], sub)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield f"{path}: length {len(a)} != {len(b)}"
        else:
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                yield from _describe(x, y, f"{path}[{i}]")
    elif a != b:
        yield f"{path}: A={a!r}  B={b!r}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loader", choices=["mmcv", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--diff", nargs=2, metavar=("DIR_A", "DIR_B"))
    args = ap.parse_args()

    if args.diff:
        diff(*args.diff)
    elif args.loader and args.out:
        dump(args.loader, args.out)
    else:
        ap.error("pass either --loader/--out or --diff")
