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

## Status at a glance (audit, 2026-09-16)

- **MAMBA's model code is complete and verified.** A from-scratch rerun of every harness (config + 8 harnesses, both envs, CPU and CUDA) passed **17/17**. The **released `epoch_6_model.pth` loads with 0 missing / 0 unexpected keys** and matches the legacy code on a 3-frame CPU video, largest difference 2.7e-6. Caveat: the synthetic frames produced only low-confidence detections.
- **No mmlab code at runtime:** after building MAMBA, no `mmcv`/`mmdet`/`mmengine` module is loaded. `mmdet` in the new env resolves only to the in-repo legacy tree, and `mmcv` is absent, so a stray import fails loudly.
- **Verified only partially:** backward through backbone and neck at model level (only head-level gradients so far); Swin in train mode (DropPath); anything on real images; the full model on Isambard (only ops were cross-checked there).
- **Not started:** data pipeline, evaluator, training loop, STPN.

## Reproduction facts (from the original training logs)

Source: `guanxiongsun/vfe.pytorch` on Hugging Face, `work_dirs/{mamba_r101_dc5_6x,stpn_swint_adam_9x}/`: logs, dumped configs, checkpoints.

| | MAMBA r101-dc5 6x | STPN Swin-T 9x |
|---|---|---|
| Hardware | 8× A100-40GB, DDP (NCCL), 1 image/GPU → batch 8 | same |
| Iterations/epoch | 13,711 (VID train frames + DET 30-class subset) | same |
| Optimiser | SGD lr 1e-3, momentum 0.9, wd 1e-4, grad clip 35 (L2) | AdamW lr 2.5e-5, betas (0.9, 0.999), wd 0.05 with `decay_mult=0` for `norm`, `relative_position_bias_table`, `absolute_pos_embed`; no clip |
| LR schedule | linear warmup 500 iters (ratio 1/3); ×0.1 after epoch 4; 6 epochs | warmup 500; ×0.1 after epoch 6; 9 epochs |
| Precision | fp32 (no fp16 anywhere) | fp32 |
| Speed | 0.165 s/iter ≈ 38 min/epoch; full val eval ≈ 20 min on 8 GPUs | 0.168 s/iter |
| Result (AP50; fast / medium / slow) | **83.82** (65.3 / 83.8 / 89.5); per epoch 82.0 → 83.6 → 83.8 | **85.15** (64.1 / 84.1 / 91.4) |
| Caveat | **Resumed from `epoch_3.pth`**; epochs 1–3 are not in the published log. Its JSON env row says 4 GPUs and seed 1485688156, yet the resume point, 41,133 = 3 × 13,711 iterations, implies batch 8. | complete run in log |

- **The evaluation protocol is itself stochastic.** VID val has 176,126 frames. Within each video the frames are shuffled with Python's `random`, *except the first*, so the memory is always seeded at `frame_id == 0` (this settles the frame-order question from 4f). At frame 0 the 14 references are spread over the *whole* video, and MAMBA's memory sampling is random, so repeated evaluations differ slightly.
- **Data** (`guanxiongsun/imagenetvid`): VID 92.1 GB + DET 60.9 GB (split `tar.gz`), annotations 60 MB. The evaluator also needs `mmdet/datasets/mamba/vid_groundtruth_motion_iou.mat` (in repo) and `scipy`.

## Rethinking the method (after Phase 4)

- **What worked:** bottom-up parity caught real bugs no inference check would have: `act_cfg=None` coerced to ReLU, ResNet never loading its pretrained weights, a lint config that excluded the harnesses.
- **What it cost:** at detector and video level, most of the effort went to *fixture* noise compounding through discrete ops (NMS, top-k, sampling), made worse by random weights (saturation, ties). Precision there is still cheap to *keep*, but expensive to *extend*.
- **So, from here:**
  1. Remaining model-level parity (STPN) uses **released checkpoints and real frames**, not random weights.
  2. **Deterministic pieces are held to exact equality**, with no tolerances: data transforms (both envs ship OpenCV 4.14.0), samplers under fixed seeds, LR schedules, optimiser parameter groups.
  3. The whole stack is judged by **task metrics**: released-checkpoint mAP on VID val, then training curves overlaid on the original logs, then reproduced mAP.
  4. **Gradients are compared on CPU only** (see the legacy CUDA backward bug in Phase 4 notes).
  5. Before the legacy tree is deleted, legacy outputs are **frozen as golden files**, so regressions stay detectable without the legacy env, including on Isambard.

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
- [x] Data staging path decided: `/projects/b5cs/imagenet_vid/` (the project's shared-dataset area, rather than `$SCRATCH`); see *Data*.
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
- [x] ~~Swap `mmdet` call sites over as each module is ported.~~ Not applicable: `vfe/` is a separate package and nothing under `mmdet/` is modified.

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
- [x] Parity: `tools/checks/parity_backbone.py`, 9 cases. **All pass on CPU and CUDA.** The 6 ResNet/FPN cases are bit-exact (`--atol 0 --rtol 0`) on CPU and within 1e-6 on CUDA; Swin differs by ≤1.8e-6 absolute on ~1.0 scale; FPN on CUDA ~2.6e-6 relative.
- [ ] **Deferred:** `STPNSwinTransformer` (prompted Swin, `mmdet/models/backbones/sptn_swin.py`, ~1019 lines) — only STPN needs it, and MAMBA (ResNet-101-DC5 + ChannelMapper, already verified) is the first reproduction target. Port before Phase 7's STPN run.

Two things the harness caught that are worth remembering:
- **`act_cfg=None` means "no activation", not "default to ReLU".** mmcv puts `dict(type='ReLU')` in the *signature*, so an explicit `None` is meaningful — FPN's lateral and output convs rely on it. `ConvModule` had been coercing `None` → ReLU, producing completely wrong FPN outputs. Fixed with a `DEFAULT_ACT_CFG` sentinel in `vfe/layers/conv_module.py`.
- **TF32 must be off for parity.** It is on by default for convs in *both* torch 1.10 and 2.10, and its ~10-bit mantissa yields ~1e-4 relative noise — enough to hide a real porting bug. `parity_backbone.run()` disables it (plus `cudnn.benchmark`) on CUDA.

Two divergences that forced reimplementation rather than wrapping:
- **mmdet vs torchvision dilation.** mmdet's `ResLayer` passes `dilation` to *every* block in a stage; torchvision gives a dilated stage's first block `previous_dilation`. For DC5 that makes `layer4.0` dilated in mmdet but not in torchvision. (Key naming still matches torchvision exactly — loading `torchvision://resnet101` leaves only `fc.*` unexpected, zero missing.)
- **mmdet's Swin ≠ upstream Microsoft Swin.** mmdet merges patches with `nn.Unfold` (row-major 2×2) vs upstream's (TL, BL, TR, BR) gather, hence `swin_convert`'s 4-group `[0, 2, 1, 3]` permutation. mmcv's `FFN` also nests each hidden block in its own `Sequential` (`ffn.layers.0.0.*`) — flattening it breaks the released checkpoints.

### Phase 4 — Detector heads + VID modules ✅ DONE for MAMBA (2026-09-16)
MAMBA first: it is the primary reproduction target and its backbone/neck are already verified. STPN moved to Phase 7 of the revised roadmap; SELSA dropped.
- [x] **4a** Anchor generator, `DeltaXYWHBBoxCoder`, `MaxIoUAssigner`/`RandomSampler`, bbox transforms, `multiclass_nms` — `vfe/core/`. Parity: `tools/checks/parity_core.py`, **118 artifacts bit-exact** (`--atol 0 --rtol 0`) on CPU *and* CUDA.
- [x] **4b** `CrossEntropyLoss` (softmax + sigmoid), `SmoothL1Loss`, `L1Loss`, `accuracy` — `vfe/models/losses/`. Parity: `tools/checks/parity_losses.py`, **45 artifacts**, CPU and CUDA.
- [x] **4c** RPN head (`AnchorHead` + `RPNHead`) — `vfe/models/dense_heads/`. Parity: `tools/checks/parity_rpn.py`, **120 artifacts** covering forward, `get_bboxes` (both the single-level DC5 and five-level FPN anchor layouts) and `loss`/`get_targets` under both `allowed_border=0` (MAMBA) and `-1` (STPN). CUDA: everything bit-exact bar the classification losses at ≤6e-8. CPU: the only non-exact artifacts are downstream of a torch CPU conv difference on the 10×7 feature map (see below).
- [x] **4d** Standard RoI head — `vfe/models/roi_heads/` (`SingleRoIExtractor`, `BBoxHead` / `ConvFCBBoxHead` / `Shared2FCBBoxHead`, `StandardRoIHead`). Parity: `tools/checks/parity_roi_head.py`, **268 artifacts** on MAMBA's and STPN's RoI-head configs plus a class-agnostic variant: state-dict keys, extractor forward *and* backward (first check of RoIAlign's backward), head forward, decode ± NMS ± rescale, `forward_train` losses / targets / gradients for every parameter and feature level, and `simple_test` including an image with no proposals. **CPU: passes in full**, 125 bit-exact, the rest ≤1.3e-6 of scale. **CUDA: passes with `--skip-train-grads`** (178 artifacts, ≤3.2e-6), because the *legacy* stack gets some FC backward passes wrong on this GPU — see below.
- [x] **4e** `BaseDetector` / `TwoStageDetector` / `FasterRCNN` + `parse_losses` — `vfe/models/detectors/`. Parity: `tools/checks/parity_detector.py`, ~800 artifacts on detectors built from the **real MAMBA and STPN configs** through each side's own config loader (so cfg routing is under test): `init_weights()` results, `forward_train` losses + `parse_losses` (CPU), and `simple_test` proposals/detections. **Passes on CPU and CUDA.** DC5 is bit-exact through backbone, neck and RPN; the rest sits on the MKL/cuDNN noise floor. Also fixed two Phase 3 gaps it exposed: **vfe `ResNet` had no `init_weights`**, so `init_cfg=Pretrained('torchvision://resnet101')` was silently ignored and MAMBA would have trained from random weights; and Swin's pretrained load bypassed the logging loader.
- [x] **4f** MAMBA — `vfe/models/{memory,aggregators}.py`, `vfe/models/roi_heads/mamba.py` (`MambaBBoxHead`, `MambaRoIHead`), `vfe/models/vid/` (`BaseVideoDetector`, `MAMBA`). Parity: `tools/checks/parity_mamba.py`, **~650 artifacts, passes on CPU and CUDA**: memory bank bit-exact through every branch, including random sample/replace; aggregator forward and CPU gradients; `init_weights()` on the real config (the aggregators keep torch's default `nn.Linear` init because mmcv's init recursion never reaches them, and the check pins that); `forward_train` losses; and 4-frame `simple_test` videos in adaptive-stride, small-memory and fixed-stride modes, with memory state carried across calls.
  - Behaviour kept from the original but worth knowing: at test time the memory stores **pre-ReLU** features; fixed-stride mode writes the current frame into the window's centre slot *in place*, so it persists; memory reads/writes draw from the global **CPU** RNG. One deliberate change: sampling an empty memory raises a clear error instead of returning `[]` and failing obscurely inside the aggregator.
  - ~~Phase 5 must answer: does `shuffle_video_frames=True` break MAMBA's memory seeding?~~ **Answered in the audit:** the dataset shuffles every frame *except the first*, so each video still starts at `frame_id == 0`.
- [ ] ~~STPN + DVP predictor.~~ → Phase 7 of the revised roadmap.
- [ ] ~~SELSA (baseline) if useful for cross-checking.~~ Dropped: not a reproduction target, and MAMBA is verified without it.

Notes on what the parity harnesses measure, and where the noise floor sits:
- **Bit-exactness is achievable for anything built from plain elementwise arithmetic**, and worth insisting on — all 118 core artifacts and every `smooth_l1`/`l1`/`accuracy` artifact match exactly across torch 1.10 → 2.10. Divergence is confined to the **fused kernels**: `F.cross_entropy` and `binary_cross_entropy_with_logits` differ by ≤9.5e-7 on a ~1.7e+1 scale (≈6e-8 relative, a few ULP). That is the framework, not the port, so `parity_losses.py` defaults to 1e-6 and *prints the observed difference* so a real regression stands out above the floor.
- **A third fixture trap, found in 4c: `F.conv2d` on CPU is not version-stable at small spatial sizes.** Given bit-identical input and weights, torch 1.10 and 2.10 differ by 1.3e-6 on a 10×7 feature map while the 19×13 map above it is bit-exact — they pick different blocking below some threshold. It only shows up on FPN's top level, and only on CPU; CUDA is bit-exact throughout. Verified with a standalone `F.conv2d` probe before touching the port, which is the habit worth keeping: **reproduce the divergence outside the model first.** All three traps found up to 4c were in the harness, not the code under test.
- **Dense matmul on CPU is not version-stable either (found in 4d).** A bare `F.linear` on bit-identical inputs differs by ~9e-7 relative between the MKL bundled with torch 1.10 and with 2.10, at every size tried down to 64×64, and each is deterministic across thread counts. Anything downstream of an FC layer therefore sits on a ~1e-6 floor on CPU. `parity_roi_head.py` judges each artifact against **its own scale** (`max|a−b| ≤ rtol·max|a|`), because its gradients span 1e-4 to 1e+1 and no single absolute tolerance is both tight on the small ones and quiet on the large ones.
- **The legacy oracle computes some CUDA backward passes wrong on the RTX 4060.** torch 1.10.1+cu113 predates the GPU (sm_89) and runs on it via PTX JIT. For the class-agnostic RoI head it returns `shared_fcs` gradients up to **1.3% off**, while the forward pass, the losses and the `fc_cls`/`fc_reg` gradients are all fine. How this was pinned down, since the first reading is "the port is broken": (1) each env is deterministic run to run, so it's not scatter noise; (2) a plain functional replay with no mmdet or mmcv code reproduces the error *exactly* on torch 1.10 and not on 2.10; (3) against a float64 CPU reference, the vfe gradients are right to <1e-6 and the mmdet ones are the ones that are off. Every op involved is correct in isolation, and the error comes and goes with a 1e-5 change in the inputs, so it depends on allocation history and wasn't pinned to one kernel. **Policy from here on: gradients are checked against the CPU oracle; CUDA comparisons cover forward passes, losses and targets.** This matters most for 4f, where the MAMBA aggregator is all matmuls.
- **Detector-level parity needs three more techniques (4e), each forced by a real failure.** (1) *Initialisation is compared by distribution*: constants exactly, random tensors by robust quantiles within 8/√n. Not by moments: `trunc_normal_(std=0.02, a=-2, b=2)` occasionally emits a weight of exactly ±2, a 100σ outlier that moves the kurtosis by hundreds at random. (2) *Detections are compared as sets*: each must match one with the same label and box/score within tolerance, in any order. Scores closer than the float noise legitimately come out of NMS in either order, and on CUDA that happens in almost every run. (3) *Training parity is CPU-only* and refuses tied proposal scores, because the RoI sampler draws by position. Plus one measured, documented tolerance: STPN-style Swin detections get 1e-4, since 1.5e-6 of feature noise shifts proposal boxes ~2e-3 px and RoIAlign on random features amplifies that to ~4e-5, reproduced by noise injection in a *single* env. The comparison logic was mutation-tested: a 1-px box shift, a changed label, a missing detection, a 10× init std, normal-vs-uniform init, a flipped `requires_grad` and a 1e-9 change in a pretrained weight are all caught, and pure reordering passes.
- **Two more comparison rules from 4f.** (1) *NMS selection flips are real and rare*: two same-class candidates overlapping above the NMS threshold, with scores equal to within float noise, survive NMS on different sides (seen once: boxes clipped to the same image corner, 1.5 px apart, scores 1e-7 apart). The set matcher accepts a detection whose partner could have suppressed it (same label, equal score, IoU ≥ 0.5), caps that at 2% of the set, and always prints a NOTE. (2) *A gradient that is zero in exact arithmetic can't be compared relatively*: the aggregator's `ref_fc_embed.bias` gradient vanishes by softmax shift-invariance, so both sides report ~1e-6 of roundoff. The harness asserts it *is* roundoff instead, which also proves the softmax runs over the right axis. Both rules were mutation-tested, and that testing caught a docstring-vs-code gap: the "exact" memory-bank artifacts had been compared with a float tolerance.
- **Losses are compared on their gradient too, not just their value.** A loss can be numerically right and still train wrong if `weight` is applied after reduction instead of before; only `d(loss)/d(pred)` notices. Two fixture traps cost real time and are documented in the harness docstrings: `torch.softmax` drifts ~1.8e-7 between torch versions (so don't build test inputs with it), and exactly-tied NMS scores come back in either order from mmcv vs torchvision (so make the scores tie-free). Both initially looked exactly like porting bugs.

## Revised roadmap (2026-09-16, after the audit)

Order changed from *data → training → reproduce* to **evaluation first**. A released checkpoint scoring its published mAP through the new code (M1) validates Phases 3–5 on real data in one shot, before any training compute is spent. Milestones (**M**) are the checkpoints to hold before moving on.

### Phase 5 — Evaluation, then data
- [x] **5a VID evaluator** — `vfe/evaluation/vid.py`, `vfe/datasets/{cocovid,imagenet_vid}.py` (annotation half of the dataset). Parity: `tools/checks/parity_vid_eval.py` on the full val set (176,126 frames, 431,511 synthetic detections). **Frame order, all ground truth, the motion table and the four headline metrics are bit-exact**; three per-class APs differ by one float64 ULP (numpy 1.23 → 2.5 summation order). Runs 2.4× faster than the original (47 s vs 115 s).
  - The motion-IoU `.mat` is converted once (`tools/convert_motion_iou.py`, legacy env) to `vfe/evaluation/vid_motion_iou.npz` (flat values + offsets), so **no `scipy`** and no ragged object arrays, which numpy ≥ 1.24 refuses.
  - Quirks reproduced deliberately, since the published numbers depend on them: IoU with +1 applied twice, in float32; fractional false positives weighted by `empty_weight`; and **one placeholder motion IoU of 0 for each of the 4,046 frames without objects**, which counts toward `empty_weight`.
  - **Frame order:** the original's shuffle drew from Python's global `random`, reseeded to 10 at import (`random.seed(10)` in `imagenet_vid_dataset.py`). The port uses a private `random.Random(10)` and reproduces the order exactly.
- [x] **5b Test-time data path** — `vfe/datasets/pipelines/` (loading, the `Seq*` transforms, formatting; a `PIPELINES` registry so data configs run unchanged), `vfe/datasets/imagenet_vid.py` (all four reference samplers, `__getitem__`), `vfe/datasets/collate.py`. Parity: `tools/checks/parity_vid_pipeline.py` on real val videos at 6 resolutions (12 samples, 90 decoded frames): **model-facing structure, every meta and every pixel bit-exact** against mmdet's `collate` + `scatter`.
  - Image operations make the same OpenCV calls mmcv made (decode, `int(x + 0.5)` rescale rounding, float32 normalise via `cv2.subtract/multiply`, bottom/right pad) instead of numpy equivalents; both envs ship OpenCV 4.14.0.
  - `DataContainer` is gone: metas stay plain objects and `collate_video_test` produces the nesting `simple_test` was written against (`img=[Tensor(1,3,H,W)]`, `img_metas=[[meta]]`, `ref_img_metas=[[[meta, ...]]]`), checked against the real mmcv output.
  - Training-only paths (box resizing, actual flips, multi-scale) raise `NotImplementedError` until 5d. The no-flip test path still draws `np.random.choice` once per sample, as the original did, so numpy's RNG stream stays aligned.
- [ ] **5c Test driver**, single- and multi-GPU (replaces `tools/test.py` / `dist_test.sh`), including the original's per-video sharding across GPUs (`DistributedVideoSampler`).
- [ ] **M1:** released MAMBA checkpoint on the full VID val set, run with vfe on Isambard → **AP50 83.8**, within the protocol's own run-to-run randomness (target ±0.2).
- [ ] **5d Train-time data path:** VID + DET concat, `bilateral_uniform` reference sampling, `SeqLoadAnnotations`, flip, format bundle, group/distributed sampler. Parity: **exact** under fixed seeds on sample videos and images.

### Phase 6 — Training
- [ ] **6a Optimiser construction:** SGD, and AdamW with mmcv's `paramwise_cfg.custom_keys` (`decay_mult`) semantics. Parity: **exact** parameter groups.
- [ ] **6b LR schedule:** per-iteration linear warmup + per-epoch steps. Parity: **exact** value at every iteration.
- [ ] **6c One full training step** (forward, backward, clip, update) on CPU, with released weights and a real batch. Parity: parameters after the step. This also closes the model-level backward gap.
- [ ] **6d Loop:** DDP via `torchrun`/`srun`, per-epoch sampler seeding, checkpoint/resume, an mmcv-compatible JSON log (so curves overlay the originals), and an eval hook. Gradient accumulation lets one 4-GPU Isambard node reproduce batch 8 exactly (all BatchNorm is frozen, so 4×2 ≡ 8×1).
- [ ] **M2:** short run on Isambard; loss and LR curves overlaid on the original logs.
- [ ] **M3:** full MAMBA 6x training → **AP50 83.8** (target ±0.5, typical run-to-run variance).

### Phase 7 — STPN
- [ ] `STPNSwinTransformer` (~1.0k lines), DVP predictor, `STPN` detector (~1.3k lines total). Parity uses the released STPN checkpoint on real frames, plus Swin in train mode (DropPath) on CPU.
- [ ] **M1′:** released STPN checkpoint → AP50 85.2. **M3′:** full 9x training.

### Phase 8 — Consolidate
- [ ] Freeze legacy harness outputs as golden files; a pytest suite that runs without the legacy env (locally, on Isambard, in CI).
- [ ] Remove the legacy `mmdet/` tree and legacy install files; new README; `uv.lock`.
- [ ] Isambard env: editable-install `vfe` (Phase 1b leftover), drop the stale `stash@{0}`.

**Out of scope for now:** TDViT and EOVOD (no code in this repo: new model work, after M3′). **Dropped:** SELSA; fp16/AMP (neither original recipe used it; revisit only as a speed-up after M3).

### Reproduction targets (ImageNet VID)
| Model | Backbone | AP50 (published) | AP50 (original log) | Checkpoint (HF `guanxiongsun/vfe.pytorch`) |
|---|---|---|---|---|
| MAMBA | ResNet-101-DC5 | 83.8 | 83.82 | `work_dirs/mamba_r101_dc5_6x/epoch_6_model.pth` — loads into vfe with 0 missing/unexpected keys |
| STPN  | Swin-T | 85.2 | 85.15 | `work_dirs/stpn_swint_adam_9x/epoch_9_model.pth` |

---

## Decisions (answered 2026-09-16)
1. **Data:** the dataset lives on the local USB drive `/media/guanxiong/tony/data/` and is uploaded to Isambard for reproduction; see *Data* below.
2. **MAMBA epochs 1–3:** no log exists. M3 assumes the published config (8×1 images, lr 1e-3, from scratch) and runs on Isambard.
3. **Isambard budget:** approved: M1 ≈ 2 GPU-hours, M3 ≈ 20–30 GPU-hours, STPN similar.
4. **Acceptance:** M1 within ±0.2 AP50; M3 within ±0.5.

## Data

**Local** (`/media/guanxiong/tony/data/`, 1.8 TB NTFS USB drive):
- Archives, identical to Hugging Face `guanxiongsun/imagenetvid`: `ILSVRC2015_VID.tar.gz` (92.06 GB), `ILSVRC2017_DET.tar.gz.aa` + `.ab` (60.86 GB), `ILSVRC2015/annotations.tar.gz` (57 MB).
- Extracted `ILSVRC2015/`: `Data/VID/{train,val,test,snippets}`, `Annotations/` (VID XMLs **and** the three JSONs), `ImageSets/VID`. **DET images are not extracted locally**, only archived.
- The repo uses it in place, since the local disk (135 GB free) could not hold a copy. `data/ILSVRC/` (git-ignored) holds symlinks in the config layout: `Data/VID`, `Annotations`, `ImageSets`, and `annotations` → the drive's `Annotations/`. Works only while the drive is mounted. Phase 5d's exact-parity checks will need a few DET images, extracted from the archive when needed.

**Isambard** (`/projects/b5cs/imagenet_vid/`, the project's shared-dataset area):
- `archives/`: the four files above, uploaded with resumable `rsync` (≈25 MB/s; parallel streams don't help, the uplink is the limit).
- `ILSVRC/`: extracted by `tools/isambard/stage_imagenet_vid.sbatch`, which (1) verifies SHA-256 against the Hugging Face checksums, proving the upload intact and identical to the HF copy, (2) merges VID and DET into the config layout, skipping `Data/VID/snippets` and the unlabelled `Data/{VID,DET}/test`, and (3) runs `tools/isambard/check_imagenet_vid.py` to confirm every image referenced by the three JSONs exists (vid_val 176,126; det_30plus1cls 349,721; vid_train). It writes `MANIFEST.txt` and `CHECK.txt` next to them.
- **Not a backup:** Isambard storage is working storage, deleted at project end. The durable copies are the local drive and the Hugging Face dataset repo.

## Open questions / risks
- **Stochastic evaluation.** Shuffled frame order and random memory sampling mean eval-level parity needs Python `random` and torch's CPU RNG seeded identically, and single-process runs.
- **Provenance of the published MAMBA run** (resumed at epoch 3; see Reproduction facts).
- **`DataContainer` semantics** (padded collation of variable-size images and metas) must be replicated in the new collation.
- **Library versions differ for the data path:** numpy 1.23 (legacy) vs 2.5, Pillow 10.4 vs 12.3. OpenCV is 4.14.0 in both. Exact parity in 5b/5d will show whether any of it matters.
- **The legacy CUDA stack mis-computes some gradients on the RTX 4060.** Gradient parity stays on CPU.
- **Checkpoint reading uses `weights_only=True`.** Fine for released `*_model.pth`; resuming a *full* mmcv training checkpoint (optimizer state, meta) may need `weights_only=False`.

## Progress log
- **2026-09-16 (data)** — Decisions answered (see above). The dataset was found on the local USB drive, archives included; they are identical in size to the Hugging Face copies. The repo reads it in place through `data/ILSVRC/` symlinks. Upload of the four archives to `/projects/b5cs/imagenet_vid/archives/` started (≈1.7 h at the uplink's ≈25 MB/s; Hugging Face → Isambard measured the same speed, so uploading costs nothing extra). Added `tools/isambard/stage_imagenet_vid.sbatch` (checksum → extract → completeness check) and `check_imagenet_vid.py`; both were tested locally, the tar exclude/strip logic on a mock archive.
- **2026-09-16 (audit)** — Stopped before Phase 5 to double-check the work and rethink the plan. All harnesses rerun from scratch, 17/17 pass. The released MAMBA checkpoint loads with 0 missing/unexpected keys and matches legacy on a 3-frame video. No mmlab module is loaded at runtime. From the original training logs: 8× A100 at batch 8, SGD/AdamW recipes, no fp16, ~38 min/epoch; the published MAMBA run was resumed at epoch 3; the eval protocol is stochastic (shuffled frames, random memory). Roadmap reordered to evaluation first (M1: released checkpoint → 83.8 on VID val), STPN moved after MAMBA's reproduction, SELSA and fp16 dropped, golden-file tests planned before the legacy tree is removed. Open decisions listed under *Decisions needed*.
- **2026-09-16 (late night)** — Phase 4f complete: MAMBA — the primary reproduction target — matches mmdet across ~650 artifacts on CPU and CUDA, including its stateful multi-frame inference. The full MAMBA model path (ResNet-101-DC5 → ChannelMapper → RPN → MAMBA RoI head with memory) is now pure PyTorch and verified. Remaining for Phase 4: STPN (+ `STPNSwinTransformer`, DVP predictor); SELSA optional. Next: Phase 5 (data pipeline + eval), which MAMBA's reproduction needs before STPN does.
- **2026-09-16 (night)** — Phase 4e complete: `FasterRCNN` built from the real MAMBA/STPN configs matches mmdet at initialisation, in training losses and in detections, on CPU and CUDA. The most consequential find of the day came from checking `init_weights()` rather than inference: vfe's `ResNet` never loaded its ImageNet checkpoint. It is fixed now, and every inference-only parity check had passed right over it. Next: 4f, MAMBA itself.
- **2026-09-16 (evening)** — Phases 4c + 4d complete: `RPNHead` and `StandardRoIHead` ported and verified (120 and 268 artifacts). Two findings change how parity is judged: CPU `F.linear` differs by ~9e-7 between the torch versions' MKL builds, and the **legacy torch 1.10/cu113 stack returns FC gradients up to 1.3% off on the RTX 4060** — proven against a float64 reference with a mmdet-free replay, so CUDA parity now covers forward passes only and gradients are judged on CPU. Also found that `ruff check tools/checks/` had never linted anything: the `"tools/*.py"` exclude glob matched the harness directory too. Fixed in `pyproject.toml`, and the one finding it had hidden is fixed. Next: 4e, the two-stage detector.
- **2026-09-16 (afternoon)** — Phases 4a + 4b complete. `vfe/core/` (anchors, bbox coder/assigner/sampler/transforms, `multiclass_nms`) is **bit-exact vs mmdet across 118 artifacts** on CPU and CUDA; `vfe/models/losses/` passes 45 value-and-gradient artifacts, with divergence only in the fused CE/BCE kernels (≤9.5e-7, ~6e-8 relative). Two apparent "bugs" turned out to be bad test *fixtures* — softmax drift and NMS tie ordering — both now documented in the harnesses so the next person doesn't re-derive them. Next: 4c, the RPN head.
- **2026-09-16 (later still)** — Phase 3 complete. ResNet (incl. DC5), Swin-T, FPN, `ChannelMapper` and the checkpoint loader ported to `vfe/models/`; supporting layers in `vfe/layers/`. `tools/checks/parity_backbone.py` (9 cases) passes on CPU and CUDA, ResNet/FPN bit-exact. The harness caught a real bug (`ConvModule` coercing `act_cfg=None` to ReLU, silently breaking FPN) — the case for writing the check before trusting the port. `STPNSwinTransformer` deferred; MAMBA's backbone path is done, so Phase 4 proceeds on MAMBA first.
- **2026-09-16 (later)** — Phases 1–2 complete. `vfe/config.py` + `vfe/registry.py` written; config loader is **bit-exact vs `mmcv.Config`** on all 4 VID configs. `vfe/ops/` written; **bit-exact vs compiled `mmcv.ops`** across 15 cases on CPU *and* CUDA. Parity-harness pattern established for the remaining phases.
- **2026-09-16** — Versions pinned (torch 2.10.0+cu128 / torchvision 0.25.0+cu128 / py3.12); `pyproject.toml` + `vfe/` package skeleton created; legacy `setup.py`/`requirements.txt` renamed aside. Local `vfe-torch` env built and GPU-verified on RTX 4060. Phase 1b done: uv installed on Isambard, repo cloned to `~/code/vfe.pytorch`, `.venv` built, `tools/checks/torch_smoke.py` passes on a GH200 (sm_90) — all `torchvision.ops` we need work on both arches. Caught and fixed the PyPI-aarch64-is-CPU-only trap.
- **2026-09-15** — Phase 0 complete: Miniforge installed; `vfe` legacy env built & GPU-verified (MAMBA/STPN build; torch + mmcv ops run on RTX 4060 via PTX-JIT). mmlab dependency surface inventoried. Plan drafted.
