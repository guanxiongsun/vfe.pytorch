"""Freeze the legacy side of every parity check, then re-check against it.

Each harness in this directory runs the same inputs through both stacks and
compares them, which needs the legacy mmdet environment. From vfe.pytorch 2.0
that environment is a separate checkout (see docs/parity.md) and will not
outlive this machine, so its side of every check is *frozen*: run once, saved
under .parity-golden/, and compared against from then on.

    python tools/checks/run_parity.py list
    python tools/checks/run_parity.py freeze --all      # needs the legacy env
    python tools/checks/run_parity.py check  --all      # needs neither
    python tools/checks/run_parity.py verify            # hashes only, seconds

This file schedules and bookkeeps; it never compares tensors itself. Every
harness already ends in `SystemExit(1 if failures else 0)`, so a variant passes
exactly when its own comparison passes.

A frozen artifact is only evidence while the code that produced it is
unchanged, so every entry records the sha256 of its harness *and every harness
it imports*, of every config it loads, and of the artifact. A mismatch is
reported as STALE and fails: the honest answer is "re-freeze", not "probably
fine". Kept parsable by Python 3.8, which runs the legacy side.
"""

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import _legacy

REPO_ROOT = _legacy.REPO_ROOT
GOLDEN_DIR = REPO_ROOT / ".parity-golden"
MANIFEST = GOLDEN_DIR / "manifest.json"
PUBLISHED_MANIFEST = REPO_ROOT / "tools/checks/golden-manifest.json"
CHECKS_DIR = Path(__file__).resolve().parent

MAMBA_CFG = "configs/vid/mamba/mamba_r101_dc5_6x.py"
STPN_CFG = "configs/vid/stpn/stpn_swint_adam_9x.py"

DEFAULT_CKPTS = {
    "mamba": Path.home() / "ckpt/mamba_epoch_6_model.pth",
    "stpn": Path.home() / "ckpt/stpn_epoch_9_model.pth",
}

# The legacy interpreter. `conda run` keeps the env's LD_LIBRARY_PATH, which
# mmcv's compiled ops need.
LEGACY_PYTHON = os.environ.get("VFE_LEGACY_PYTHON", "").split() or [
    "conda", "run", "-n", "vfe", "--no-capture-output", "python"]
VFE_PYTHON = [sys.executable]

# Legacy Swin inference fragments the allocator enough to OOM on a 8 GB card
# that is also driving a desktop; this is what made it fit.
LEGACY_CUDA_ENV = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64"}


class Variant:
    """One frozen comparison: how to produce each side, and how to compare them."""

    def __init__(self, name, harness, legacy, vfe, gen=(), compare="--compare",
                 compare_extra=(), artifact="file", requires=(), configs=(), note=""):
        self.name = name
        self.harness = harness
        self.legacy = list(legacy)          # flags selecting the legacy implementation
        self.vfe = list(vfe)                # flags selecting the new one
        self.gen = list(gen)                # flags applied to *both* sides
        self.compare = compare
        self.compare_extra = list(compare_extra)  # comparison-only, e.g. --skip-train-grads
        self.artifact = artifact            # "file" (.pt) or "dir" (config dumps)
        self.requires = tuple(requires)
        self.configs = tuple(configs)
        self.note = note

    @property
    def script(self):
        return CHECKS_DIR / (self.harness + ".py")

    @property
    def golden(self):
        return GOLDEN_DIR / (self.name if self.artifact == "dir" else self.name + ".pt")

    def uses_cuda(self):
        return "cuda" in self.requires


def _both_devices(name, harness, legacy, requires=(), configs=(), compare_extra_cuda=(),
                  note=""):
    """A harness checked on CPU and on CUDA: the same run, `--device` apart."""
    return [
        Variant(name + "_cpu", harness, legacy, ["--impl", "vfe"], ["--device", "cpu"],
                requires=requires, configs=configs, note=note),
        Variant(name + "_cuda", harness, legacy, ["--impl", "vfe"], ["--device", "cuda"],
                compare_extra=compare_extra_cuda, requires=("cuda",) + tuple(requires),
                configs=configs, note=note),
    ]


MMDET = ["--impl", "mmdet"]
BOTH_CFGS = (MAMBA_CFG, STPN_CFG)

VARIANTS = (
    [Variant("config", "parity_config", ["--loader", "mmcv"], ["--loader", "vfe"],
             compare="--diff", artifact="dir",
             configs=(MAMBA_CFG, "configs/vid/mamba/mamba_r101_dc5_3x.py", STPN_CFG,
                      "configs/vid/stpn/stpn_swins_adam_9x.py"),
             note="both config loaders, fully resolved")]
    + _both_devices("ops", "parity_ops", ["--impl", "mmcv"], note="nms, roi_align")
    + _both_devices("core", "parity_core", MMDET, note="anchors, coders, assigners, samplers")
    + _both_devices("losses", "parity_losses", MMDET)
    + _both_devices("backbone", "parity_backbone", MMDET, note="ResNet/DC5, Swin, FPN")
    + _both_devices("rpn", "parity_rpn", MMDET, configs=BOTH_CFGS)
    + _both_devices("roi_head", "parity_roi_head", MMDET, configs=BOTH_CFGS,
                    # The legacy stack miscomputes some FC backward passes on this
                    # GPU, so CUDA gradients are excluded at comparison time -- the
                    # golden keeps them.
                    compare_extra_cuda=["--skip-train-grads"])
    + _both_devices("detector", "parity_detector", MMDET, requires=("hub_resnet101",),
                    configs=BOTH_CFGS, note="FasterRCNN from the real configs")
    + _both_devices("mamba", "parity_mamba", MMDET, requires=("hub_resnet101",),
                    configs=(MAMBA_CFG,), note="memory bank, aggregators, stateful test")
    + [
        Variant("optim", "parity_optim", MMDET, ["--impl", "vfe"],
                requires=("hub_resnet101",), configs=BOTH_CFGS,
                note="per-parameter optimiser settings"),
        Variant("vid_eval", "parity_vid_eval", MMDET, ["--impl", "vfe"],
                requires=("vid_ann", "legacy_scipy"), configs=(MAMBA_CFG,),
                note="176,126 frames, VID metric"),
        Variant("vid_pipeline", "parity_vid_pipeline", MMDET, ["--impl", "vfe"],
                requires=("vid_ann", "vid_images"), configs=(MAMBA_CFG,),
                note="test pipeline on real frames"),
        Variant("vid_train_data", "parity_vid_train_data", MMDET, ["--impl", "vfe"],
                requires=("vid_ann", "vid_images"), configs=(MAMBA_CFG,),
                note="VID training samples and samplers"),
        Variant("vid_train_data_det", "parity_vid_train_data", MMDET, ["--impl", "vfe"],
                gen=["--det"], requires=("vid_ann", "vid_images", "det_images"),
                configs=(MAMBA_CFG,), note="same, including DET samples"),
        Variant("train_loader", "parity_train_loader", MMDET, ["--impl", "vfe"],
                requires=("vid_ann", "vid_images", "det_images"), configs=(MAMBA_CFG,),
                note="virtual-rank loaders vs mmdet's, 2 epochs"),
        Variant("train_step_mamba", "parity_train_step", MMDET, ["--impl", "vfe"],
                gen=["--config", MAMBA_CFG, "--checkpoint", "{mamba_ckpt}"],
                requires=("vid_ann", "vid_images", "mamba_ckpt"), configs=(MAMBA_CFG,),
                note="two SGD steps from the released checkpoint"),
        Variant("train_step_stpn", "parity_train_step", MMDET, ["--impl", "vfe"],
                gen=["--config", STPN_CFG, "--checkpoint", "{stpn_ckpt}"],
                requires=("vid_ann", "vid_images", "stpn_ckpt"), configs=(STPN_CFG,),
                note="two AdamW steps from the released checkpoint"),
        Variant("vid_test_mamba", "parity_vid_test", MMDET, ["--impl", "vfe"],
                gen=["--config", MAMBA_CFG, "--ckpt", "{mamba_ckpt}"],
                requires=("cuda", "vid_ann", "vid_images", "mamba_ckpt"), configs=(MAMBA_CFG,),
                note="end-to-end inference on a whole val video"),
        Variant("vid_test_stpn", "parity_vid_test", MMDET, ["--impl", "vfe"],
                gen=["--config", STPN_CFG, "--ckpt", "{stpn_ckpt}"],
                requires=("cuda", "vid_ann", "vid_images", "stpn_ckpt"), configs=(STPN_CFG,),
                note="same, STPN"),
    ]
)

BY_NAME = OrderedDict((v.name, v) for v in VARIANTS)


# ---------------------------------------------------------------------------
# Hashing: what a frozen artifact is evidence *about*
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_path(path):
    """A file's digest, or a directory's: its sorted (name, digest) listing."""
    path = Path(path)
    if path.is_dir():
        h = hashlib.sha256()
        for sub in sorted(p for p in path.rglob("*") if p.is_file()):
            h.update(str(sub.relative_to(path)).encode())
            h.update(sha256_file(sub).encode())
        return h.hexdigest()
    return sha256_file(path)


def tree_size(path):
    path = Path(path)
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def harness_sources(harness):
    """`harness` and every module in this directory it imports, transitively.

    parity_train_step imports parity_vid_train_data, which imports
    parity_vid_pipeline: editing the innermost one invalidates all three, and
    only a real import graph knows that.
    """
    seen, queue = set(), [harness]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        path = CHECKS_DIR / (name + ".py")
        if not path.exists():
            continue
        seen.add(name)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                queue += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                queue.append(node.module.split(".")[0])
    return sorted(seen)


def config_closure(rel):
    """`rel` and every config it inherits through `_base_`, repo-relative."""
    out, queue = set(), [rel]
    while queue:
        cur = queue.pop()
        if cur in out:
            continue
        path = REPO_ROOT / cur
        if not path.exists():
            continue
        out.add(cur)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == "_base_" for t in node.targets):
                bases = ast.literal_eval(node.value)
                if isinstance(bases, str):
                    bases = [bases]
                for base in bases:
                    resolved = (path.parent / base).resolve().relative_to(REPO_ROOT)
                    queue.append(str(resolved))
    return sorted(out)


def fingerprint(variant):
    """Everything the golden depends on, as {kind: {name: sha256}}."""
    sources = {}
    for name in harness_sources(variant.harness):
        sources[name + ".py"] = sha256_file(CHECKS_DIR / (name + ".py"))
    configs = {}
    for rel in variant.configs:
        for member in config_closure(rel):
            configs[member] = sha256_file(REPO_ROOT / member)
    return {"sources": sources, "configs": configs}


# ---------------------------------------------------------------------------
# Requirements: what a variant needs before it can say anything
# ---------------------------------------------------------------------------

_CACHE = {}


def _mamba_cfg():
    if "cfg" not in _CACHE:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.config import Config
        _CACHE["cfg"] = Config.fromfile(str(REPO_ROOT / MAMBA_CFG))
    return _CACHE["cfg"]


def _dataset_dir(path):
    """True when `path` is a readable directory. `resolve()` first: the image
    trees are symlinks to an external drive, and an unmounted drive leaves a
    dangling link that `is_dir()` alone reports as existing."""
    return (REPO_ROOT / path).resolve().is_dir()


def probe(name, ckpts):
    """(ok, detail) for one requirement."""
    if name == "cuda":
        import torch
        if not torch.cuda.is_available():
            return False, "no CUDA device"
        return True, torch.cuda.get_device_name(0)
    if name == "vid_ann":
        ann = REPO_ROOT / _mamba_cfg().data.test.ann_file
        return ann.is_file(), str(ann)
    if name == "vid_images":
        d = _mamba_cfg().data.test.img_prefix
        return _dataset_dir(d), str((REPO_ROOT / d).resolve())
    if name == "det_images":
        d = _mamba_cfg().data.train[1].img_prefix
        return _dataset_dir(d), str((REPO_ROOT / d).resolve())
    if name == "hub_resnet101":
        import torch
        cache = Path(torch.hub.get_dir()) / "checkpoints"
        hits = sorted(cache.glob("resnet101-*.pth")) if cache.is_dir() else []
        return bool(hits), str(hits[0]) if hits else "no resnet101-*.pth in " + str(cache)
    if name in ("mamba_ckpt", "stpn_ckpt"):
        path = Path(ckpts[name.split("_")[0]])
        return path.is_file(), str(path)
    if name == "legacy_scipy":
        rc = subprocess.call(LEGACY_PYTHON + ["-c", "import scipy"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return rc == 0, "scipy in the legacy env"
    raise KeyError("unknown requirement: " + name)


def requirements_for(variant, ckpts, needed_for_freeze):
    """('', '') when satisfied, else (requirement, why not)."""
    for name in variant.requires:
        if name == "legacy_scipy" and not needed_for_freeze:
            continue  # only the legacy side reads the .mat
        key = (name, needed_for_freeze)
        if key not in _CACHE:
            _CACHE[key] = probe(name, ckpts)
        ok, detail = _CACHE[key]
        if not ok:
            return name, detail
    return "", ""


# ---------------------------------------------------------------------------
# Running one side, and comparing
# ---------------------------------------------------------------------------

def expand(args, ckpts):
    return [str(a).format(mamba_ckpt=ckpts["mamba"], stpn_ckpt=ckpts["stpn"]) for a in args]


def out_flag(variant):
    return "--out"


def run(cmd, log_path, extra_env=None):
    """Run `cmd`, streaming to `log_path`. Returns (returncode, last line)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(extra_env or {})
    with open(str(log_path), "w") as log:
        log.write("$ " + " ".join(str(c) for c in cmd) + "\n\n")
        log.flush()
        rc = subprocess.call([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT,
                             cwd=str(REPO_ROOT), env=env)
    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
    return rc, (lines[-1] if lines else "")


def side_command(variant, side, out_path, ckpts):
    python = LEGACY_PYTHON if side == "legacy" else VFE_PYTHON
    flags = variant.legacy if side == "legacy" else variant.vfe
    return python + [variant.script] + flags + expand(variant.gen, ckpts) + \
        [out_flag(variant), str(out_path)]


def compare_command(variant, golden, candidate):
    return VFE_PYTHON + [variant.script, variant.compare, str(golden), str(candidate)] + \
        variant.compare_extra


def tail(log_path, n=25):
    lines = log_path.read_text().splitlines()
    return "\n".join("    | " + ln for ln in lines[-n:])


# ---------------------------------------------------------------------------
# The manifest: what was frozen, from what, when
# ---------------------------------------------------------------------------

def load_manifest():
    if MANIFEST.is_file():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(manifest):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True)
    tmp = MANIFEST.with_suffix(".json.partial")
    tmp.write_text(text)
    os.replace(str(tmp), str(MANIFEST))
    # The tracked copy: provenance without the 3 GB.
    PUBLISHED_MANIFEST.write_text(text)


def git_commit(root):
    try:
        out = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def env_versions():
    if "env" not in _CACHE:
        code = ("import torch, platform; "
                "print(torch.__version__, platform.python_version(), end='')")
        try:
            legacy = subprocess.check_output(LEGACY_PYTHON + ["-c", code]).decode().strip()
        except (subprocess.CalledProcessError, OSError):
            legacy = "unavailable"
        vfe = subprocess.check_output(VFE_PYTHON + ["-c", code]).decode().strip()
        _CACHE["env"] = {"legacy": legacy, "vfe": vfe}
    return _CACHE["env"]


def staleness(variant, entry):
    """Why `entry` can no longer speak for `variant` -- empty when it still can."""
    reasons = []
    current = fingerprint(variant)
    for kind in ("sources", "configs"):
        was, now = entry.get("fingerprint", {}).get(kind, {}), current[kind]
        for name in sorted(set(was) | set(now)):
            if was.get(name) != now.get(name):
                reasons.append(f"{kind[:-1]} changed: {name}")
    if entry.get("compare_flags", []) != variant.compare_extra:
        reasons.append("comparison flags changed")
    golden = variant.golden
    if not golden.exists():
        reasons.append("artifact is gone")
    elif sha256_path(golden) != entry.get("artifact", {}).get("sha256"):
        reasons.append("artifact was modified since it was frozen")
    return reasons


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(args):
    manifest = load_manifest()
    print("{:<22} {:<22} {:<9} {}".format("VARIANT", "HARNESS", "FROZEN", "NEEDS"))
    print("-" * 100)
    for v in VARIANTS:
        entry = manifest.get(v.name)
        state = "-"
        if entry:
            state = "stale" if staleness(v, entry) else "yes"
        print("{:<22} {:<22} {:<9} {}".format(
            v.name, v.harness, state, ",".join(v.requires) or "-"))
    print("-" * 100)
    print(f"{len(VARIANTS)} variants, {len(manifest)} frozen")
    if args.check_manifest:
        missing = [v.name for v in VARIANTS if v.name not in manifest]
        extra = [name for name in manifest if name not in BY_NAME]
        for name in missing:
            print(f"MISSING  {name}: never frozen")
        for name in extra:
            print(f"ORPHAN   {name}: frozen but no longer in the matrix")
        return 1 if (missing or extra) else 0
    return 0


def cmd_verify(args):
    manifest = load_manifest()
    bad = 0
    for v in selected(args):
        entry = manifest.get(v.name)
        if entry is None:
            print(f"MISSING  {v.name}")
            bad += 1
            continue
        reasons = staleness(v, entry)
        if reasons:
            bad += 1
            print(f"STALE    {v.name}")
            for reason in reasons:
                print(f"           {reason}")
        else:
            size = entry["artifact"]["bytes"] / 1e6
            print("ok       {:<22} {:>9.1f} MB  {}".format(v.name, size, entry["frozen_at"]))
    print("-" * 70)
    print(f"{len(selected(args)) - bad} verified, {bad} unusable")
    return 1 if bad else 0


def selected(args):
    if args.all:
        return list(VARIANTS)
    chosen = []
    for name in args.variants:
        if name in BY_NAME:
            chosen.append(BY_NAME[name])
        else:
            matches = [v for v in VARIANTS if v.harness == name or v.name.startswith(name)]
            if not matches:
                raise SystemExit("unknown variant or harness: " + name)
            chosen += matches
    return chosen


def preflight(args, variants):
    """Refuse to start a freeze that would be wasted, or dangerous."""
    problems = []
    rc = subprocess.call(["git", "-C", str(REPO_ROOT), "check-ignore", "-q", ".parity-golden"])
    if rc != 0:
        problems.append(".parity-golden/ is not git-ignored -- it holds gigabytes")
    free_gb = shutil.disk_usage(str(REPO_ROOT)).free / 1e9
    if free_gb < 15:
        problems.append(f"only {free_gb:.1f} GB free; a full freeze needs ~15")
    status = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"]).decode().splitlines()
    # The freeze writes the tracked manifest itself, so a run that froze
    # anything leaves it modified; that is not the kind of drift this guards.
    dirty = [ln for ln in status if PUBLISHED_MANIFEST.name not in ln]
    if dirty and not args.allow_dirty:
        problems.append("working tree is dirty; the goldens would record a commit that is not "
                        "what ran (pass --allow-dirty to accept that)")
    if problems:
        for p in problems:
            print("REFUSING: " + p)
        raise SystemExit(2)
    if args.skip_env_check:
        return
    print("preflight: legacy environment ...")
    log = GOLDEN_DIR / "logs/preflight_legacy.log"
    rc, last = run(LEGACY_PYTHON + [CHECKS_DIR / "verify_env.py"], log)
    if rc != 0:
        print(tail(log))
        raise SystemExit("the legacy environment is not usable; see " + str(log))
    print("  " + last)
    if any(v.uses_cuda() for v in variants):
        log = GOLDEN_DIR / "logs/preflight_gpu.log"
        rc, last = run(VFE_PYTHON + [CHECKS_DIR / "gpu_smoke.py"], log)
        if rc != 0:
            print(tail(log))
            raise SystemExit("GPU smoke test failed; see " + str(log))
        print("  " + last)


def freeze_one(variant, args, ckpts, manifest, scratch):
    name = variant.name
    entry = manifest.get(name)
    if entry and not staleness(variant, entry) and not args.force:
        print(f"SKIP     {name:<22} already frozen and current (--force to redo)")
        return "skip"

    req, detail = requirements_for(variant, ckpts, needed_for_freeze=True)
    if req:
        print(f"SKIP     {name:<22} needs {req} ({detail})")
        return "skip"

    golden, partial = variant.golden, variant.golden.with_name(variant.name + ".partial")
    if partial.exists():
        shutil.rmtree(str(partial)) if partial.is_dir() else partial.unlink()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"freeze   {name:<22} legacy side ...", end=" ")
    sys.stdout.flush()
    legacy_env = dict(LEGACY_CUDA_ENV) if variant.uses_cuda() else {}
    rc, last = run(side_command(variant, "legacy", partial, ckpts),
                   GOLDEN_DIR / f"logs/{name}.legacy.log", legacy_env)
    legacy_s = time.time() - started
    if rc != 0 or not partial.exists():
        print(f"FAILED after {legacy_s:.0f}s")
        print(tail(GOLDEN_DIR / f"logs/{name}.legacy.log"))
        return "fail"
    print(f"{legacy_s:.0f}s")

    if golden.exists():
        shutil.rmtree(str(golden)) if golden.is_dir() else golden.unlink()
    os.replace(str(partial), str(golden))

    status, timings, summary = compare_against_golden(variant, args, ckpts, scratch, "freeze")
    manifest[name] = {
        "status": status,
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(str(c) for c in side_command(variant, "legacy", golden, ckpts)),
        "compare_flags": variant.compare_extra,
        "artifact": {"path": str(golden.relative_to(REPO_ROOT)),
                     "sha256": sha256_path(golden), "bytes": tree_size(golden)},
        "fingerprint": fingerprint(variant),
        "checkpoints": {k: sha256_file(p) for k, p in ckpts.items()
                        if (k + "_ckpt") in variant.requires},
        "repo_commit": git_commit(REPO_ROOT),
        "legacy_commit": git_commit(_legacy.legacy_root()),
        "legacy_root": str(_legacy.legacy_root()),
        "env": env_versions(),
        "seconds": dict(timings, legacy=round(legacy_s, 1)),
        "summary": summary,
        "requires": list(variant.requires),
        "note": variant.note,
    }
    save_manifest(manifest)
    return "ok" if status == "OK" else "fail"


def compare_against_golden(variant, args, ckpts, scratch, mode):
    """Run the vfe side and compare it with the frozen legacy artifact."""
    name = variant.name
    candidate = Path(scratch) / (name if variant.artifact == "dir" else name + ".pt")
    print(f"{mode:<8} {name:<22} vfe side ...", end=" ")
    sys.stdout.flush()
    started = time.time()
    rc, _ = run(side_command(variant, "vfe", candidate, ckpts),
                GOLDEN_DIR / f"logs/{name}.vfe.log")
    vfe_s = time.time() - started
    if rc != 0 or not candidate.exists():
        print(f"FAILED after {vfe_s:.0f}s")
        print(tail(GOLDEN_DIR / f"logs/{name}.vfe.log"))
        return "ERROR", {"vfe": round(vfe_s, 1)}, "the vfe side did not produce an artifact"
    print(f"{vfe_s:.0f}s; comparing ...", end=" ")
    sys.stdout.flush()
    started = time.time()
    rc, last = run(compare_command(variant, variant.golden, candidate),
                   GOLDEN_DIR / f"logs/{name}.compare.log")
    compare_s = time.time() - started
    print("OK" if rc == 0 else "FAIL")
    if rc != 0:
        print(tail(GOLDEN_DIR / f"logs/{name}.compare.log"))
    timings = {"vfe": round(vfe_s, 1), "compare": round(compare_s, 1)}
    return ("OK" if rc == 0 else "FAIL"), timings, last


def cmd_freeze(args):
    variants = selected(args)
    ckpts = checkpoints(args)
    preflight(args, variants)
    manifest = load_manifest()
    scratch = make_scratch(args)
    counts = {"ok": 0, "fail": 0, "skip": 0}
    for v in variants:
        counts[freeze_one(v, args, ckpts, manifest, scratch)] += 1
    return report(counts, args, "frozen")


def cmd_check(args):
    variants = selected(args)
    ckpts = checkpoints(args)
    manifest = load_manifest()
    scratch = make_scratch(args)
    counts = {"ok": 0, "fail": 0, "skip": 0}
    for v in variants:
        entry = manifest.get(v.name)
        if entry is None:
            print(f"MISSING  {v.name:<22} never frozen")
            counts["skip" if not args.strict else "fail"] += 1
            continue
        reasons = staleness(v, entry)
        if reasons:
            print("STALE    {:<22} {}".format(v.name, "; ".join(reasons)))
            if not args.allow_stale:
                counts["fail"] += 1
                continue
            print("           (--allow-stale: comparing anyway, the result proves nothing)")
        req, detail = requirements_for(v, ckpts, needed_for_freeze=False)
        if req:
            print(f"SKIP     {v.name:<22} needs {req} ({detail})")
            counts["fail" if args.strict else "skip"] += 1
            continue
        status, _, _ = compare_against_golden(v, args, ckpts, scratch, "check")
        counts["ok" if status == "OK" else "fail"] += 1
    return report(counts, args, "checked")


def report(counts, args, verb):
    print("=" * 70)
    print("{}: {} ok, {} failed, {} skipped".format(
        verb, counts["ok"], counts["fail"], counts["skip"]))
    if args.allow_stale:
        print("UNTRUSTED: --allow-stale was used")
    return 1 if counts["fail"] else 0


def checkpoints(args):
    return {"mamba": Path(args.mamba_ckpt or os.environ.get("VFE_MAMBA_CKPT")
                          or DEFAULT_CKPTS["mamba"]),
            "stpn": Path(args.stpn_ckpt or os.environ.get("VFE_STPN_CKPT")
                         or DEFAULT_CKPTS["stpn"])}


def make_scratch(args):
    import tempfile
    path = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix="vfe-parity-"))
    path.mkdir(parents=True, exist_ok=True)
    print("vfe-side artifacts: {}\nlogs: {}\n".format(path, GOLDEN_DIR / "logs"))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command")

    def common(p, all_default=False):
        p.add_argument("variants", nargs="*", help="variant or harness names (default: --all)")
        p.add_argument("--all", action="store_true", default=all_default)
        p.add_argument("--mamba-ckpt")
        p.add_argument("--stpn-ckpt")
        p.add_argument("--scratch", help="where vfe-side artifacts go (default: a temp dir)")
        p.add_argument("--allow-stale", action="store_true",
                       help="compare against goldens whose inputs changed (proves nothing)")

    p_list = sub.add_parser("list", help="the matrix and what is frozen")
    p_list.add_argument("--check-manifest", action="store_true",
                        help="exit non-zero if any variant is unfrozen or orphaned")

    p_freeze = sub.add_parser("freeze", help="run the legacy side and save it")
    common(p_freeze)
    p_freeze.add_argument("--force", action="store_true", help="re-freeze current goldens")
    p_freeze.add_argument("--allow-dirty", action="store_true")
    p_freeze.add_argument("--skip-env-check", action="store_true")

    p_check = sub.add_parser("check", help="run the vfe side against the frozen goldens")
    common(p_check)
    p_check.add_argument("--strict", action="store_true",
                         help="a skipped variant is a failure")

    p_verify = sub.add_parser("verify", help="hashes only: are the goldens intact and current?")
    common(p_verify, all_default=True)

    args = ap.parse_args()
    if not args.command:
        ap.error("pick a command: list, freeze, check, verify")
    if args.command != "list" and not args.all and not args.variants:
        args.all = True
    return {"list": cmd_list, "freeze": cmd_freeze,
            "check": cmd_check, "verify": cmd_verify}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
