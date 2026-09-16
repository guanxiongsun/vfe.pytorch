# vfe.pytorch — Pure-PyTorch Rewrite Plan

Tracking doc for migrating this codebase off the mmlab stack (mmdet / mmcv / mmengine)
to a **pure-PyTorch** implementation. Update the checkboxes and the Progress Log as work proceeds.

- **Goal:** rewrite MAMBA / STPN (and later TDViT, EOVOD) as a self-contained PyTorch codebase.
  Only a small part of mmlab is actually used — replace the framework scaffolding, keep the model logic.
- **Strategy:** two environments side by side.
  - `vfe` (legacy) — faithful reproduction, used **only** as a reference oracle. ✅ built & verified.
  - `vfe-torch` (new) — modern pure-PyTorch target, built incrementally with parity checks against `vfe`.
- **Method:** port bottom-up; after each module, feed identical inputs through original (`vfe`) and
  new (`vfe-torch`) and diff the tensors before moving on.

---

## Machine / GPU

- Pop!_OS 24.04, kernel 7.x. Miniforge at `~/miniforge3`.
- **GPU: NVIDIA RTX 4060, sm_89 (Ada), 8 GB VRAM**, driver 580.173.02 (CUDA 13 capable).
- ⚠️ 8 GB VRAM is likely too small to reproduce original *training* batch sizes. Fine for
  inference / parity checks; full training reproduction happens on Isambard-AI (below).

## Training target — Isambard-AI (aarch64 / Hopper)

Login: `ssh b5cs.aip2.isambard` (key-based). Probed 2026-09-15.

- **Platform:** HPE Cray EX, SLES 15-SP6, **aarch64 (Grace ARM)**, 64k-page kernel → x86_64 wheels won't run.
- **GPU:** NVIDIA GH200 Grace-Hopper, **Hopper sm_90** (H100/H200-class), on Slurm compute nodes (login node has no GPU).
- **Modules:** `cuda/12.6`, `cudatoolkit/24.11_12.6` (default), `cuda/11.8`; `nvhpc/24.11`; `gcc-native/{12.3,13.2,14.2}`; `cray-python/3.11.7`; Cray PE `cpe/25.03`.
- **Containers:** apptainer, singularity, podman, podman-hpc all present (no images pre-pulled) — **this is the delivery mechanism.**
- **Internet (login node):** PyPI (200), conda-forge/linux-aarch64 (200), `download.pytorch.org/whl/*` wheel index (200), and `nvcr.io` are **all reachable**. (Only the *bare root* of download.pytorch.org 403s — the wheel paths work.) PyPI now ships CUDA **aarch64** torch wheels (2.6+). ⇒ pip/uv, conda, and containers are all viable.
- **Storage:** `HOME` on VAST (quota ~101 GB / ~10M inodes); `PROJECTDIR=/projects/b5cs` + `SCRATCH=/scratch/...` on Lustre. Keep the `.venv` on `$HOME` or `$PROJECTDIR`; stage datasets to `$SCRATCH`.

**Chosen approach: `uv` venv + pinned pip wheels** (matches BriCS's distributed-PyTorch tutorial; lightest fit for a pure-PyTorch project). NGC container kept only as a fallback if we later need Hopper-tuned kernels (APEX/TransformerEngine); conda-forge is a secondary fallback.

- **Multi-GPU/node training:** NCCL backend + `module load brics/nccl brics/aws-ofi-nccl` + `srun --mpi=pmi2` (Slingshot 11). Single GH200 has ~96 GB, so few-GPU single-node training already dwarfs the original 8-GPU memory budget.
- **Portability rule:** pure-PyTorch package uses only `torch` + `torchvision.ops` + pure-Python deps. No mmcv/detectron2/install-time CUDA compilation (keeps a single spec working on both arches).

### Pinned versions (decided 2026-09-16, verified on both machines)

| | value | why |
|---|---|---|
| Python | **3.12** | `requires-python = ">=3.12,<3.13"`; uv-provided on Isambard |
| torch | **2.10.0+cu128** | newest release with CUDA **12.8** wheels for *both* arches; 2.11+ is CUDA 13, which Isambard's 565.57.01 driver (CUDA 12.7 native) can't run without compat shims |
| torchvision | **0.25.0+cu128** | pins `torch==2.10.0` |
| index | `https://download.pytorch.org/whl/cu128` | **required** — see gotcha below |

> ⚠️ **Gotcha (found the hard way):** PyPI's *default* `torch` aarch64 wheel is **CPU-only** (`2.10.0+cpu`). A plain `uv pip install torch==2.10.0` on Isambard silently yields `torch.cuda.is_available() == False`. `pyproject.toml` therefore routes `torch`/`torchvision` through an `explicit = true` `[[tool.uv.index]]` pointing at `whl/cu128`, and pins the `+cu128` local version so uv won't consider the CPU wheel "already satisfied".

Arch coverage confirmed at runtime: x86_64 wheel ships `sm_70…sm_120` (RTX 4060 = sm_89 runs on the sm_86 cubin via minor-version binary compat); aarch64 wheel ships `sm_80, sm_90, sm_100, sm_120` (GH200 = sm_90 native).

**Phase 1b — Isambard bring-up ✅ DONE (2026-09-16)**
- [x] `srun --account=brics.b5cs --gpus=1 --ntasks=1 --time=00:01:00 nvidia-smi` → GH200 120GB, driver **565.57.01**, CUDA 12.7, 97871 MiB. Partition `workq` (gpu:4/node), account `brics.b5cs`, QOS `normal`.
- [x] Installed `uv` 0.12.15 (aarch64 tarball → `~/.local/bin/uv`).
- [x] Repo cloned to `~/code/vfe.pytorch`; `uv venv --python 3.12 .venv`.
- [x] Installed the pinned stack from the cu128 index; **GPU smoke test passes** (`tools/checks/torch_smoke.py`: matmul, conv2d, bf16, autocast, nms, batched_nms, roi_align, deform_conv2d, NCCL — all OK on sm_90).
- [x] Bootstrap script committed: `tools/isambard/setup_env.sh`.
- [x] Repo synced via git: branch `pure-pytorch-rewrite` checked out at `~/code/vfe.pytorch`, `.venv` alongside it.
- [ ] **Run `uv pip install --python .venv/bin/python -e ".[log,dev]"` there** — the deps are installed but the `vfe` package itself is not yet editable-installed, so scripts currently rely on `sys.path`/`PYTHONPATH`.
- [ ] Confirm data staging path on `$SCRATCH=/scratch/b5cs/$USER`; plan ImageNet-VID transfer.
- [ ] (Later, multi-GPU/node) `module load brics/nccl brics/aws-ofi-nccl`; launch with `srun --mpi=pmi2 --ntasks-per-node=<gpus>`, NCCL backend, `MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)` — see `isambard:nccl` + `isambard:slurm`.
- [ ] *(Fallback only)* If Hopper-tuned kernels are ever needed: `apptainer build vfe.sif docker://nvcr.io/nvidia/pytorch:<arm64 tag>` (or `podman-hpc pull` + `podman-hpc migrate`), run with `--nv`/`--gpu`.

---

## Phase 0 — Legacy reference env `vfe` ✅ DONE

Verified working end-to-end (CPU model build + GPU ops on the RTX 4060 via PTX-JIT).

**Stack:** Python 3.8.20 · torch 1.10.1+cu113 · torchvision 0.11.2+cu113 · mmcv-full 1.3.17 · mmdet 2.19.1 (editable).

**Deviations from README install (intentional):**
- `pycocotools` instead of the pinned `mmpycocotools` (abandoned, won't build on modern gcc).
- `opencv-python<5` (pip pulled 5.0, which breaks 2021-era mmcv transforms) → 4.14.
- `numpy<1.24` (1.23.5) — old code uses removed `np.float` aliases.
- Editable install with `--no-deps` so `setup.py` doesn't re-pull `mmpycocotools` / bump numpy.
- Skipped `torchaudio` and the `onnx==1.7.0` + dev/test deps (export/test only, build-prone).

**Use it:**
```bash
conda activate vfe          # or: conda run -n vfe --no-capture-output python ...
```

**Re-verify (sanity):**
```bash
# builds MAMBA (89.6M) + STPN (45.0M) on CPU
conda run -n vfe --no-capture-output python tools/checks/verify_env.py
# runs torch + mmcv compiled ops on GPU
conda run -n vfe --no-capture-output python tools/checks/gpu_smoke.py
```

---

## mmlab dependency inventory (what the rewrite must replace)

The **algorithmic core is already ~pure PyTorch** — the coupling is mostly scaffolding.

| mmlab surface | Where / how much | Rewrite plan |
|---|---|---|
| **`mmcv.ops`** (compiled CUDA) | RoIAlign, nms, batched_nms (+ DeformConv if a DCN backbone is used) | **Easy win:** `torchvision.ops` has `roi_align`, `nms`, `batched_nms`, `deform_conv2d`. ~1:1 swap → removes the compiled dependency entirely. |
| **`mmcv.runner`** (~122 uses) | `EpochBasedRunner`, hooks, optimizer/LR build, checkpoint IO, `auto_fp16`; `mmdet.apis.train_detector` / `single_gpu_test` | **Biggest task:** plain train/eval loop (optimizer, `torch.cuda.amp`, DDP, checkpointing, logging). Replaces Runner + hooks. |
| **Registry + `build_*` + `Config`** | `build_detector/dataset/dataloader`, `type=...` configs, `ConfigDict` | Replace config-driven construction with plain Python instantiation (or a tiny ~30-line registry). Pervasive but mechanical. |
| **`mmcv.cnn`** (~101 uses) | `ConvModule`, `build_norm_layer`, weight init | Pure-`nn.Module` equivalents. |
| **Data pipeline** (`Seq*` transforms) | `datasets/pipelines/mmtrack/`, `datasets/mamba/` | Plain `Dataset` + transforms. Eval (`datasets/mamba/vid_eval.py`) is self-contained; keep `pycocotools`. |
| **`mmcv.parallel`** | `MMDataParallel`, `DataContainer` collate | torch DDP + custom `collate_fn` (drop the `DataContainer` wrapper). |

### Key source files (VID-specific, the code to port)
- Detectors: `mmdet/models/vid/{base,mamba,selsa}.py`, `mmdet/models/vid/stpn/{stpn,dvp_predictor}.py`
- Aggregators: `mmdet/models/aggregators/{mamba,selsa,embed}_aggregator.py`
- RoI heads: `mmdet/models/roi_heads/vid/{mamba,selsa}_roi_head.py` (+ `bbox_heads/`)
- Datasets: `mmdet/datasets/{imagenet_vid_dataset,coco_video_dataset}.py`, `mmdet/datasets/mamba/`
- Pipeline: `mmdet/datasets/pipelines/mmtrack/`
- Configs: `configs/vid/mamba/mamba_r101_dc5_{3x,6x}.py`, `configs/vid/stpn/stpn_swin{s,t}_adam_9x.py`

---

## Porting roadmap

### Phase 1 — New env + scaffolding ✅ MOSTLY DONE (2026-09-16)
- [x] Created `vfe-torch` env locally (x86_64): `conda create -n vfe-torch -c conda-forge python=3.12 uv`, then `uv pip install -e ".[log,dev]"`. Smoke test passes on RTX 4060.
- [x] Project layout: new package is **`vfe/`**; legacy `mmdet/` stays as the reference oracle. `setup.py`→`setup_mmdet_legacy.py` and `requirements.txt`→`requirements-mmdet-legacy.txt` so setuptools picks up the new `pyproject.toml` (the legacy `vfe` env is an old-style egg-link install and is unaffected — re-verified).
- [x] `pyproject.toml` with hard pins + cu128 index (see *Pinned versions* above).
- [x] `vfe/config.py` — `Config`/`ConfigDict` replacing `mmcv.Config` (~190 lines vs mmcv's ~700): `_base_` inheritance, `_delete_`, attribute access, dotted overrides.
- [x] `vfe/registry.py` — `Registry` + `build_from_cfg` replacing `mmcv.utils` equivalents (no scopes/parents/build_func).
- [x] Parity harness established (see below).

### Phase 2 — Ops ✅ DONE (2026-09-16)
- [x] Confirmed `torchvision.ops` provides working CUDA `nms`, `batched_nms`, `roi_align(aligned=True)`, `deform_conv2d` on **both** sm_89 and sm_90 (`tools/checks/torch_smoke.py`).
- [x] `vfe/ops/` — mmcv-signature-compatible `nms`, `batched_nms`, `roi_align`, `RoIAlign` over `torchvision.ops`. Handles mmcv's `offset=1` (emulated by growing `x2`/`y2`), `score_threshold`, `max_num`, and `batched_nms`'s `split_thr` chunking path.
- [x] **Parity verified bit-exact vs compiled `mmcv.ops`** — 15 cases (incl. rpn iou=0.7, dense boxes, 30-class batched, agnostic, split path, sampling_ratio 0/2, aligned/unaligned) on **both CPU and CUDA**. `soft_nms`/`max` pooling deliberately not ported: no config uses them, and `vfe/ops` raises rather than silently differing.
- [ ] Swap `mmdet` call sites over as each module is ported (happens per-phase, not up front).

### Parity harness (the method for everything below)

The legacy oracle is py3.8-only, so the two stacks can't share a process. Each harness therefore runs *twice* — once per env — dumping canonical artifacts, then diffs them:

```bash
# configs -> JSON
conda run -n vfe       --no-capture-output python tools/checks/parity_config.py --loader mmcv --out /tmp/cfg_mmcv
conda run -n vfe-torch --no-capture-output python tools/checks/parity_config.py --loader vfe  --out /tmp/cfg_vfe
conda run -n vfe-torch --no-capture-output python tools/checks/parity_config.py --diff /tmp/cfg_mmcv /tmp/cfg_vfe

# ops -> tensors  (add --device cuda to compare the CUDA kernels)
conda run -n vfe       --no-capture-output python tools/checks/parity_ops.py --impl mmcv --out /tmp/ops_mmcv.pt
conda run -n vfe-torch --no-capture-output python tools/checks/parity_ops.py --impl vfe  --out /tmp/ops_vfe.pt
conda run -n vfe-torch --no-capture-output python tools/checks/parity_ops.py --compare /tmp/ops_mmcv.pt /tmp/ops_vfe.pt
```

Inputs are generated on CPU from a fixed seed *before* any device move, so both torch versions see bit-identical data. Extend this pattern for backbones (Phase 3) and full detectors (Phase 4).

Because the artifacts are portable, this also works *across machines*. The strongest check run so far compares the compiled mmcv kernels on the **RTX 4060 (sm_89, torch 1.10/cu113)** against `vfe.ops` on an **Isambard GH200 (sm_90, torch 2.10/cu128)** — all 15 cases bit-identical, i.e. the op swap is invariant to architecture *and* torch version:

```bash
ssh b5cs.aip2.isambard 'cd ~/code/vfe.pytorch && srun --account=brics.b5cs --gpus=1 --ntasks=1 --time=00:05:00 \
    .venv/bin/python tools/checks/parity_ops.py --impl vfe --device cuda --out $HOME/ops_gh200.pt'
ssh b5cs.aip2.isambard 'cat $HOME/ops_gh200.pt' > /tmp/ops_gh200.pt
conda run -n vfe-torch --no-capture-output python tools/checks/parity_ops.py --compare /tmp/ops_mmcv_cu.pt /tmp/ops_gh200.pt
```

### Phase 3 — Backbone / neck (+ checkpoint loading) ✅ DONE (2026-09-16)
- [x] Port ResNet-101-DC5 (dilated C5) and Swin-T backbones — `vfe/models/backbones/{resnet,swin}.py`, plus `vfe/models/builder.py` (registries aliased to one `MODELS` registry, as mmdet does) and `vfe/layers/{weight_init,drop,transformer}.py`.
- [x] Port FPN neck + `ChannelMapper` — `vfe/models/necks/`.
- [x] Checkpoint loading — `vfe/models/checkpoint.py`: `load_checkpoint`/`load_state_dict`, `torchvision://` URI resolution (verified byte-identical URLs vs old mmcv), `swin_convert` for the released Swin weights. Partial loads are *logged*, never silent.
- [x] Parity: `tools/checks/parity_backbone.py`, 9 cases. **All pass on CPU and CUDA.** The 6 ResNet/FPN cases are bit-exact (`--atol 0 --rtol 0`) on CPU and still bit-exact on CUDA at 1e-6; Swin differs by ≤1.8e-6 absolute on ~1.0 scale; FPN on CUDA ~2.6e-6 relative.
- [ ] **Deferred:** `STPNSwinTransformer` (prompted Swin, `mmdet/models/backbones/sptn_swin.py`, ~1019 lines) — only STPN needs it, and MAMBA (ResNet-101-DC5 + ChannelMapper, already verified) is the first reproduction target. Port before Phase 7's STPN run.

Two things the harness caught that are worth remembering:
- **`act_cfg=None` means "no activation", not "default to ReLU".** mmcv puts `dict(type='ReLU')` in the *signature*, so an explicit `None` is meaningful — FPN's lateral and output convs rely on it. `ConvModule` had been coercing `None` → ReLU, producing completely wrong FPN outputs. Fixed with a `DEFAULT_ACT_CFG` sentinel in `vfe/layers/conv_module.py`.
- **TF32 must be off for parity.** It is on by default for convs in *both* torch 1.10 and 2.10, and its ~10-bit mantissa yields ~1e-4 relative noise — enough to hide a real porting bug. `parity_backbone.run()` disables it (plus `cudnn.benchmark`) on CUDA.

Two divergences that forced reimplementation rather than wrapping:
- **mmdet vs torchvision dilation.** mmdet's `ResLayer` passes `dilation` to *every* block in a stage; torchvision gives a dilated stage's first block `previous_dilation`. For DC5 that makes `layer4.0` dilated in mmdet but not in torchvision. (Key naming still matches torchvision exactly — loading `torchvision://resnet101` leaves only `fc.*` unexpected, zero missing.)
- **mmdet's Swin ≠ upstream Microsoft Swin.** mmdet merges patches with `nn.Unfold` (row-major 2×2) vs upstream's (TL, BL, TR, BR) gather, hence `swin_convert`'s 4-group `[0, 2, 1, 3]` permutation. mmcv's `FFN` also nests each hidden block in its own `Sequential` (`ffn.layers.0.0.*`) — flattening it breaks the released checkpoints.

### Phase 4 — Detector heads + VID modules
MAMBA first: it is the primary reproduction target and its backbone/neck are already verified.
- [ ] Anchor generator, `DeltaXYWHBBoxCoder`, assigner/sampler, `CrossEntropyLoss` / `SmoothL1Loss`.
- [ ] RPN head + Standard RoI head (Shared2FCBBoxHead, SingleRoIExtractor).
- [ ] MAMBA aggregator + MAMBA RoI head.
- [ ] STPN + DVP predictor.
- [ ] SELSA (baseline) if useful for cross-checking.
- [ ] Parity: full-model `simple_test` detections match `vfe`.

### Phase 5 — Data pipeline + eval
- [ ] ImageNet VID dataset + `Seq*` transforms as plain `Dataset`/transforms.
- [ ] `collate_fn` replacing `DataContainer`.
- [ ] Wire up `vid_eval.py` / COCO eval; reproduce eval numbers from released checkpoints.

### Phase 6 — Training loop
- [ ] Optimizer + LR schedule (SGD for MAMBA r101; AdamW for STPN swin).
- [ ] AMP (`torch.cuda.amp`), grad clipping, checkpoint save/resume, logging.
- [ ] DDP launcher replacing `dist_train.sh` / `MMDistributedDataParallel`.

### Phase 7 — Reproduce & finish
- [ ] Reproduce reported metrics (targets below) from scratch or fine-tune.
- [ ] Then tackle unfinished models: **TDViT**, **EOVOD**.

### Reproduction targets (ImageNet VID, from README)
| Model | Backbone | AP50 | Checkpoint |
|---|---|---|---|
| MAMBA | ResNet-101-DC5 | 83.8 | HF: `guanxiongsun/vfe.pytorch` → `work_dirs/mamba_r101_dc5_6x` |
| STPN  | Swin-T | 85.2 | HF: `guanxiongsun/vfe.pytorch` → `work_dirs/stpn_swint_adam_9x` |

---

## Open questions / risks
- 8 GB VRAM vs original training batch sizes (see Machine note) — may need gradient accumulation or a bigger GPU for full training.
- Exact weight-init & BN-eval details must match to get numerical parity — verify against `vfe` rather than assuming.
- `DataContainer` semantics (padded collation of variable-size images/metas) — replicate carefully in the new `collate_fn`.
- Dataset not yet downloaded locally — needed for Phase 5 eval parity (see README data-prep).

## Progress log
- **2026-09-16 (later still)** — Phase 3 complete. ResNet (incl. DC5), Swin-T, FPN, `ChannelMapper` and the checkpoint loader ported to `vfe/models/`; supporting layers in `vfe/layers/`. `tools/checks/parity_backbone.py` (9 cases) passes on CPU and CUDA, ResNet/FPN bit-exact. The harness caught a real bug (`ConvModule` coercing `act_cfg=None` to ReLU, silently breaking FPN) — the case for writing the check before trusting the port. `STPNSwinTransformer` deferred; MAMBA's backbone path is done, so Phase 4 proceeds on MAMBA first.
- **2026-09-16 (later)** — Phases 1–2 complete. `vfe/config.py` + `vfe/registry.py` written; config loader is **bit-exact vs `mmcv.Config`** on all 4 VID configs. `vfe/ops/` written; **bit-exact vs compiled `mmcv.ops`** across 15 cases on CPU *and* CUDA. Parity-harness pattern established for the remaining phases.
- **2026-09-16** — Versions pinned (torch 2.10.0+cu128 / torchvision 0.25.0+cu128 / py3.12); `pyproject.toml` + `vfe/` package skeleton created; legacy `setup.py`/`requirements.txt` renamed aside. Local `vfe-torch` env built and GPU-verified on RTX 4060. Phase 1b done: uv installed on Isambard, repo cloned to `~/code/vfe.pytorch`, `.venv` built, `tools/checks/torch_smoke.py` passes on a GH200 (sm_90) — all `torchvision.ops` we need work on both arches. Caught and fixed the PyPI-aarch64-is-CPU-only trap.
- **2026-09-15** — Phase 0 complete: Miniforge installed; `vfe` legacy env built & GPU-verified (MAMBA/STPN build; torch + mmcv ops run on RTX 4060 via PTX-JIT). mmlab dependency surface inventoried. Plan drafted.
