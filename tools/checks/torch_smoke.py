"""GPU smoke test for the new pure-PyTorch ``vfe-torch`` env.

Exercises exactly the ops the rewrite depends on, so a pass here means the
mmcv.ops -> torchvision.ops swap (Phase 2) has a working backend on this
machine. Runs on both x86_64/sm_89 (local RTX 4060) and aarch64/sm_90 (GH200).

Usage:
    .venv/bin/python tools/checks/torch_smoke.py
    # on Isambard:
    srun --account=brics.b5cs --gpus=1 --ntasks=1 --time=00:05:00 \
        .venv/bin/python tools/checks/torch_smoke.py
"""

import platform

import torch
import torchvision

print(f"python      {platform.python_version()}  ({platform.machine()})")
print(f"torch       {torch.__version__}")
print(f"torchvision {torchvision.__version__}")
print(f"cuda(build) {torch.version.cuda}   available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device      {torch.cuda.get_device_name(0)}  sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"arch_list   {torch.cuda.get_arch_list()}")
print("-" * 70)

if not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible -- run this on a GPU node")

dev = "cuda"
failures = 0


def check(name, fn):
    global failures
    try:
        fn()
        torch.cuda.synchronize()
        print(f"OK    {name}")
    except Exception as e:  # noqa: BLE001 - smoke test reports, never raises
        failures += 1
        print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:200]}")


def _boxes(n=8):
    x1y1 = torch.rand(n, 2, device=dev) * 50
    return torch.cat([x1y1, x1y1 + 10 + torch.rand(n, 2, device=dev) * 20], dim=1)


def _amp():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return torch.nn.Conv2d(3, 8, 3).to(dev)(torch.randn(1, 3, 32, 32, device=dev))


def _roi_align():
    feat = torch.randn(1, 4, 32, 32, device=dev)
    rois = torch.tensor([[0.0, 0, 0, 16, 16]], device=dev)
    return torchvision.ops.roi_align(feat, rois, output_size=7, spatial_scale=1.0, sampling_ratio=2, aligned=True)


def _dcn():
    x = torch.randn(1, 4, 16, 16, device=dev)
    offset = torch.randn(1, 2 * 3 * 3, 16, 16, device=dev)
    weight = torch.randn(4, 4, 3, 3, device=dev)
    return torchvision.ops.deform_conv2d(x, offset, weight, padding=1)


def _assert(cond):
    if not cond:
        raise RuntimeError("not available")


check("matmul (cublas)", lambda: torch.randn(1024, 1024, device=dev) @ torch.randn(1024, 1024, device=dev))
check("conv2d (cudnn)", lambda: torch.nn.Conv2d(3, 16, 3, padding=1).to(dev)(torch.randn(2, 3, 64, 64, device=dev)))
check("bf16 matmul", lambda: torch.randn(512, 512, device=dev, dtype=torch.bfloat16) @ torch.randn(512, 512, device=dev, dtype=torch.bfloat16))
check("autocast amp", _amp)
check("torchvision.ops.nms", lambda: torchvision.ops.nms(_boxes(), torch.rand(8, device=dev), 0.5))
check("torchvision.ops.batched_nms", lambda: torchvision.ops.batched_nms(_boxes(), torch.rand(8, device=dev), torch.randint(0, 3, (8,), device=dev), 0.5))
check("torchvision.ops.roi_align", _roi_align)
check("torchvision.ops.deform_conv2d", _dcn)
check("nccl available", lambda: _assert(torch.distributed.is_nccl_available()))

print("-" * 70)
print("ALL OK" if failures == 0 else f"{failures} FAILURE(S)")
raise SystemExit(1 if failures else 0)
