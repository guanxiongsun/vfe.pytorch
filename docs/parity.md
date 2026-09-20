# Parity: checking `vfe/` against the original implementation

`vfe/` is a reimplementation of code that already existed. Every part of it was
checked against the original — mmdetection 2.19.1 / mmcv-full 1.3.17 / PyTorch
1.10 — by running the same inputs through both and comparing the results tensor
by tensor. This document is how to run those checks.

The original implementation is not in this repository any more. It is preserved
at the `v1.0.0` tag, and the checks reach it in one of two ways:

* **frozen** — its side of every check was run once and saved under
  `.parity-golden/`. Re-checking needs only this repository.
* **live** — a checkout of `v1.0.0` plus its conda environment, which is what
  produced the frozen artifacts and what you need to freeze anything new.

`.parity-golden/` is git-ignored and machine-local: several gigabytes of tensors
that are evidence, not source. It is not published, and a fresh clone has none
of it. `tools/checks/golden-manifest.json` *is* tracked, and records what was
frozen, from which commits, with which hashes.

## The harnesses

`tools/checks/parity_*.py` each cover one layer. They share a shape:

```bash
conda run -n vfe --no-capture-output python tools/checks/parity_rpn.py --impl mmdet --out A.pt
python tools/checks/parity_rpn.py --impl vfe --out B.pt
python tools/checks/parity_rpn.py --compare A.pt B.pt      # exits non-zero on failure
```

The first two runs happen in different Python versions (mmcv needs 3.8), which
is why they communicate through files rather than a single process. The
comparison is file-only and runs in either environment.

`tools/checks/run_parity.py` drives all of them.

## Re-checking against the frozen artifacts

```bash
python tools/checks/run_parity.py list             # the matrix, and what is frozen
python tools/checks/run_parity.py verify           # hashes only: intact and current?
python tools/checks/run_parity.py check --all      # run the vfe side, compare
python tools/checks/run_parity.py check rpn_cpu train_step_mamba
```

`check` runs only this repository's side, so it needs no legacy environment —
but it does need whatever that variant reads: a GPU, the ImageNet VID
annotations and images, a released checkpoint. Anything missing is reported as
`SKIP` with the reason; `--strict` makes a skip a failure, which is what a
release check wants.

A frozen artifact is evidence only while the code that produced it is
unchanged, so every entry records the sha256 of its harness *and every harness
that one imports*, of every config in its `_base_` closure, and of the artifact
itself. If any of those changed, the variant is `STALE` and fails: re-freeze it,
because nobody can say what the old artifact means. `--allow-stale` compares
anyway and stamps the run `UNTRUSTED`.

## Setting up the live oracle

Needed to freeze a new variant, or to re-freeze a stale one.

```bash
git worktree add ~/code/vfe.legacy v1        # or: git clone --branch v1.0.0 ...

conda create --name vfe -y python=3.8
conda activate vfe
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install -r ~/code/vfe.legacy/requirements.txt mmpycocotools

cd ~/code/vfe.legacy && python setup.py develop --no-deps
cd ~ && python -c "import mmdet; print(mmdet.__file__)"   # must print the worktree
```

Run that last check from a directory that is neither repository: from inside a
repository, `mmdet` may be found by path rather than by installation, which
hides a broken install until the day you rely on it.

`python setup.py develop` must be run **from the v1 checkout**. Running it from
this repository picks up `pyproject.toml` instead, which describes `vfe`, not
`mmdet`, and silently does the wrong thing.

The harnesses find the legacy tree through `tools/checks/_legacy.py`, which
looks at `$VFE_LEGACY_ROOT`, then `../vfe.legacy`, then this repository. Only
two of them need it at all — both read FGFA's motion-IoU `.mat` table.

Verify the environment before trusting it:

```bash
conda run -n vfe --no-capture-output python tools/checks/verify_env.py
# -> mmdet 2.19.1, mmcv 1.3.17, and both detectors built from this repo's configs
```

## Freezing

```bash
python tools/checks/run_parity.py freeze --all \
    --mamba-ckpt ~/ckpt/mamba_epoch_6_model.pth \
    --stpn-ckpt  ~/ckpt/stpn_epoch_9_model.pth
```

A full freeze takes a few hours and writes a few gigabytes. It refuses to start
if `.parity-golden/` is not ignored, if the disk is nearly full, or if the
working tree is dirty — a golden that records a commit which is not what ran is
worse than no golden. It runs the legacy environment check first, and the GPU
check when a CUDA variant is selected.

Each variant is frozen *and* immediately re-checked, so a freeze is also a full
re-verification of the port against the live original.

## What the comparisons do and do not claim

* **Gradients are judged on CPU only.** On this machine's RTX 4060 (sm_89) the
  legacy stack's PyTorch 1.10 / CUDA 11.3 build computes some fully-connected
  backward passes incorrectly — the *old* stack, not the new one. The CUDA
  RoI-head variant therefore compares everything except training gradients
  (`--skip-train-grads`); the frozen artifact still contains them.
* **TF32 is off inside the harnesses**, on both sides, so CUDA comparisons are
  not measuring a precision mode. Training and evaluation outside the harnesses
  leave TF32 on, as the original runs had it.
* **Detections are compared as sets**, each matched to one with the same label
  and a box and score within tolerance. Scores closer together than the float
  noise legitimately leave NMS in either order.
* **Tolerances are per artifact, against its own scale**, and each harness
  prints the observed difference so a real regression stands out above the
  floor. Most artifacts are bit-exact; the documented exceptions, and the
  reasoning behind every tolerance, are in [rewrite-plan.md](rewrite-plan.md).
* The `config` variant's artifact is a **directory** of resolved configs, not a
  tensor file, and is compared with `--diff`.

## If a check fails

1. `verify` first. `STALE` means the inputs moved, not that the port broke.
2. Read `.parity-golden/logs/<variant>.compare.log`: harnesses name the
   artifact, the tolerance and the observed difference.
3. Reproduce the two sides by hand with the command in the manifest entry —
   it is the exact legacy invocation, recorded verbatim.
