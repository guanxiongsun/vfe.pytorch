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
- [x] **4a** Anchor generator, `DeltaXYWHBBoxCoder`, `MaxIoUAssigner`/`RandomSampler`, bbox transforms, `multiclass_nms` — `vfe/core/`. Parity: `tools/checks/parity_core.py`, **118 artifacts bit-exact** (`--atol 0 --rtol 0`) on CPU *and* CUDA.
- [x] **4b** `CrossEntropyLoss` (softmax + sigmoid), `SmoothL1Loss`, `L1Loss`, `accuracy` — `vfe/models/losses/`. Parity: `tools/checks/parity_losses.py`, **45 artifacts**, CPU and CUDA.
- [x] **4c** RPN head (`AnchorHead` + `RPNHead`) — `vfe/models/dense_heads/`. Parity: `tools/checks/parity_rpn.py`, **120 artifacts** covering forward, `get_bboxes` (both the single-level DC5 and five-level FPN anchor layouts) and `loss`/`get_targets` under both `allowed_border=0` (MAMBA) and `-1` (STPN). CUDA: everything bit-exact bar the classification losses at ≤6e-8. CPU: the only non-exact artifacts are downstream of a torch CPU conv difference on the 10×7 feature map (see below).
- [x] **4d** Standard RoI head — `vfe/models/roi_heads/` (`SingleRoIExtractor`, `BBoxHead` / `ConvFCBBoxHead` / `Shared2FCBBoxHead`, `StandardRoIHead`). Parity: `tools/checks/parity_roi_head.py`, **268 artifacts** on MAMBA's and STPN's RoI-head configs plus a class-agnostic variant: state-dict keys, extractor forward *and* backward (first check of RoIAlign's backward), head forward, decode ± NMS ± rescale, `forward_train` losses / targets / gradients for every parameter and feature level, and `simple_test` including an image with no proposals. **CPU: passes in full**, 125 bit-exact, the rest ≤1.3e-6 of scale. **CUDA: passes with `--skip-train-grads`** (178 artifacts, ≤3.2e-6), because the *legacy* stack gets some FC backward passes wrong on this GPU — see below.
- [x] **4e** `BaseDetector` / `TwoStageDetector` / `FasterRCNN` + `parse_losses` — `vfe/models/detectors/`. Parity: `tools/checks/parity_detector.py`, ~800 artifacts on detectors built from the **real MAMBA and STPN configs** through each side's own config loader (so cfg routing is under test): `init_weights()` results, `forward_train` losses + `parse_losses` (CPU), and `simple_test` proposals/detections. **Passes on CPU and CUDA.** DC5 is bit-exact through backbone, neck and RPN; the rest sits on the MKL/cuDNN noise floor. Also fixed two Phase 3 gaps it exposed: **vfe `ResNet` had no `init_weights`**, so `init_cfg=Pretrained('torchvision://resnet101')` was silently ignored and MAMBA would have trained from random weights; and Swin's pretrained load bypassed the logging loader.
- [x] **4f** MAMBA — `vfe/models/{memory,aggregators}.py`, `vfe/models/roi_heads/mamba.py` (`MambaBBoxHead`, `MambaRoIHead`), `vfe/models/vid/` (`BaseVideoDetector`, `MAMBA`). Parity: `tools/checks/parity_mamba.py`, **~650 artifacts, passes on CPU and CUDA**: memory bank bit-exact through every branch, including random sample/replace; aggregator forward and CPU gradients; `init_weights()` on the real config (the aggregators keep torch's default `nn.Linear` init because mmcv's init recursion never reaches them, and the check pins that); `forward_train` losses; and 4-frame `simple_test` videos in adaptive-stride, small-memory and fixed-stride modes, with memory state carried across calls.
  - Behaviour kept from the original but worth knowing: at test time the memory stores **pre-ReLU** features; fixed-stride mode writes the current frame into the window's centre slot *in place*, so it persists; memory reads/writes draw from the global **CPU** RNG. One deliberate change: sampling an empty memory raises a clear error instead of returning `[]` and failing obscurely inside the aggregator.
  - **Phase 5 must answer:** the test config sets `shuffle_video_frames=True`, but MAMBA's memory is only seeded on `frame_id == 0`. If a shuffled video doesn't start with frame 0, the memory is either stale (from the previous video) or empty (now an error). Check how the original dataset orders frames before porting it.
- [ ] STPN + DVP predictor.
- [ ] SELSA (baseline) if useful for cross-checking.

Notes on what the parity harnesses measure, and where the noise floor sits:
- **Bit-exactness is achievable for anything built from plain elementwise arithmetic**, and worth insisting on — all 118 core artifacts and every `smooth_l1`/`l1`/`accuracy` artifact match exactly across torch 1.10 → 2.10. Divergence is confined to the **fused kernels**: `F.cross_entropy` and `binary_cross_entropy_with_logits` differ by ≤9.5e-7 on a ~1.7e+1 scale (≈6e-8 relative, a few ULP). That is the framework, not the port, so `parity_losses.py` defaults to 1e-6 and *prints the observed difference* so a real regression stands out above the floor.
- **A third fixture trap, found in 4c: `F.conv2d` on CPU is not version-stable at small spatial sizes.** Given bit-identical input and weights, torch 1.10 and 2.10 differ by 1.3e-6 on a 10×7 feature map while the 19×13 map above it is bit-exact — they pick different blocking below some threshold. It only shows up on FPN's top level, and only on CPU; CUDA is bit-exact throughout. Verified with a standalone `F.conv2d` probe before touching the port, which is the habit worth keeping: **reproduce the divergence outside the model first.** All three traps so far were in the harness, not the code under test.
- **Dense matmul on CPU is not version-stable either (found in 4d).** A bare `F.linear` on bit-identical inputs differs by ~9e-7 relative between the MKL bundled with torch 1.10 and with 2.10, at every size tried down to 64×64, and each is deterministic across thread counts. Anything downstream of an FC layer therefore sits on a ~1e-6 floor on CPU. `parity_roi_head.py` judges each artifact against **its own scale** (`max|a−b| ≤ rtol·max|a|`), because its gradients span 1e-4 to 1e+1 and no single absolute tolerance is both tight on the small ones and quiet on the large ones.
- **The legacy oracle computes some CUDA backward passes wrong on the RTX 4060.** torch 1.10.1+cu113 predates the GPU (sm_89) and runs on it via PTX JIT. For the class-agnostic RoI head it returns `shared_fcs` gradients up to **1.3% off**, while the forward pass, the losses and the `fc_cls`/`fc_reg` gradients are all fine. How this was pinned down, since the first reading is "the port is broken": (1) each env is deterministic run to run, so it's not scatter noise; (2) a plain functional replay with no mmdet or mmcv code reproduces the error *exactly* on torch 1.10 and not on 2.10; (3) against a float64 CPU reference, the vfe gradients are right to <1e-6 and the mmdet ones are the ones that are off. Every op involved is correct in isolation, and the error comes and goes with a 1e-5 change in the inputs, so it depends on allocation history and wasn't pinned to one kernel. **Policy from here on: gradients are checked against the CPU oracle; CUDA comparisons cover forward passes, losses and targets.** This matters most for 4f, where the MAMBA aggregator is all matmuls.
- **Detector-level parity needs three more techniques (4e), each forced by a real failure.** (1) *Initialisation is compared by distribution*: constants exactly, random tensors by robust quantiles within 8/√n. Not by moments: `trunc_normal_(std=0.02, a=-2, b=2)` occasionally emits a weight of exactly ±2, a 100σ outlier that moves the kurtosis by hundreds at random. (2) *Detections are compared as sets*: each must match one with the same label and box/score within tolerance, in any order. Scores closer than the float noise legitimately come out of NMS in either order, and on CUDA that happens in almost every run. (3) *Training parity is CPU-only* and refuses tied proposal scores, because the RoI sampler draws by position. Plus one measured, documented tolerance: STPN-style Swin detections get 1e-4, since 1.5e-6 of feature noise shifts proposal boxes ~2e-3 px and RoIAlign on random features amplifies that to ~4e-5, reproduced by noise injection in a *single* env. The comparison logic was mutation-tested: a 1-px box shift, a changed label, a missing detection, a 10× init std, normal-vs-uniform init, a flipped `requires_grad` and a 1e-9 change in a pretrained weight are all caught, and pure reordering passes.
- **Two more comparison rules from 4f.** (1) *NMS selection flips are real and rare*: two same-class candidates overlapping above the NMS threshold, with scores equal to within float noise, survive NMS on different sides (seen once: boxes clipped to the same image corner, 1.5 px apart, scores 1e-7 apart). The set matcher accepts a detection whose partner could have suppressed it (same label, equal score, IoU ≥ 0.5), caps that at 2% of the set, and always prints a NOTE. (2) *A gradient that is zero in exact arithmetic can't be compared relatively*: the aggregator's `ref_fc_embed.bias` gradient vanishes by softmax shift-invariance, so both sides report ~1e-6 of roundoff. The harness asserts it *is* roundoff instead, which also proves the softmax runs over the right axis. Both rules were mutation-tested, and that testing caught a docstring-vs-code gap: the "exact" memory-bank artifacts had been compared with a float tolerance.
- **Losses are compared on their gradient too, not just their value.** A loss can be numerically right and still train wrong if `weight` is applied after reduction instead of before; only `d(loss)/d(pred)` notices. Two fixture traps cost real time and are documented in the harness docstrings: `torch.softmax` drifts ~1.8e-7 between torch versions (so don't build test inputs with it), and exactly-tied NMS scores come back in either order from mmcv vs torchvision (so make the scores tie-free). Both initially looked exactly like porting bugs.

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
- **2026-09-16 (late night)** — Phase 4f complete: MAMBA — the primary reproduction target — matches mmdet across ~650 artifacts on CPU and CUDA, including its stateful multi-frame inference. The full MAMBA model path (ResNet-101-DC5 → ChannelMapper → RPN → MAMBA RoI head with memory) is now pure PyTorch and verified. Remaining for Phase 4: STPN (+ `STPNSwinTransformer`, DVP predictor); SELSA optional. Next: Phase 5 (data pipeline + eval), which MAMBA's reproduction needs before STPN does.
- **2026-09-16 (night)** — Phase 4e complete: `FasterRCNN` built from the real MAMBA/STPN configs matches mmdet at initialisation, in training losses and in detections, on CPU and CUDA. The most consequential find of the day came from checking `init_weights()` rather than inference: vfe's `ResNet` never loaded its ImageNet checkpoint. It is fixed now, and every inference-only parity check had passed right over it. Next: 4f, MAMBA itself.
- **2026-09-16 (evening)** — Phases 4c + 4d complete: `RPNHead` and `StandardRoIHead` ported and verified (120 and 268 artifacts). Two findings change how parity is judged: CPU `F.linear` differs by ~9e-7 between the torch versions' MKL builds, and the **legacy torch 1.10/cu113 stack returns FC gradients up to 1.3% off on the RTX 4060** — proven against a float64 reference with a mmdet-free replay, so CUDA parity now covers forward passes only and gradients are judged on CPU. Also found that `ruff check tools/checks/` had never linted anything: the `"tools/*.py"` exclude glob matched the harness directory too. Fixed in `pyproject.toml`, and the one finding it had hidden is fixed. Next: 4e, the two-stage detector.
- **2026-09-16 (afternoon)** — Phases 4a + 4b complete. `vfe/core/` (anchors, bbox coder/assigner/sampler/transforms, `multiclass_nms`) is **bit-exact vs mmdet across 118 artifacts** on CPU and CUDA; `vfe/models/losses/` passes 45 value-and-gradient artifacts, with divergence only in the fused CE/BCE kernels (≤9.5e-7, ~6e-8 relative). Two apparent "bugs" turned out to be bad test *fixtures* — softmax drift and NMS tie ordering — both now documented in the harnesses so the next person doesn't re-derive them. Next: 4c, the RPN head.
- **2026-09-16 (later still)** — Phase 3 complete. ResNet (incl. DC5), Swin-T, FPN, `ChannelMapper` and the checkpoint loader ported to `vfe/models/`; supporting layers in `vfe/layers/`. `tools/checks/parity_backbone.py` (9 cases) passes on CPU and CUDA, ResNet/FPN bit-exact. The harness caught a real bug (`ConvModule` coercing `act_cfg=None` to ReLU, silently breaking FPN) — the case for writing the check before trusting the port. `STPNSwinTransformer` deferred; MAMBA's backbone path is done, so Phase 4 proceeds on MAMBA first.
- **2026-09-16 (later)** — Phases 1–2 complete. `vfe/config.py` + `vfe/registry.py` written; config loader is **bit-exact vs `mmcv.Config`** on all 4 VID configs. `vfe/ops/` written; **bit-exact vs compiled `mmcv.ops`** across 15 cases on CPU *and* CUDA. Parity-harness pattern established for the remaining phases.
- **2026-09-16** — Versions pinned (torch 2.10.0+cu128 / torchvision 0.25.0+cu128 / py3.12); `pyproject.toml` + `vfe/` package skeleton created; legacy `setup.py`/`requirements.txt` renamed aside. Local `vfe-torch` env built and GPU-verified on RTX 4060. Phase 1b done: uv installed on Isambard, repo cloned to `~/code/vfe.pytorch`, `.venv` built, `tools/checks/torch_smoke.py` passes on a GH200 (sm_90) — all `torchvision.ops` we need work on both arches. Caught and fixed the PyPI-aarch64-is-CPU-only trap.
- **2026-09-15** — Phase 0 complete: Miniforge installed; `vfe` legacy env built & GPU-verified (MAMBA/STPN build; torch + mmcv ops run on RTX 4060 via PTX-JIT). mmlab dependency surface inventoried. Plan drafted.
