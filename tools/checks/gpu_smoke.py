"""GPU smoke test for the legacy ``vfe`` env: run core torch ops + mmcv compiled ops on CUDA.

Confirms kernels actually execute on the local GPU. Relevant because torch 1.10 / mmcv 1.3.17
were built for cu113 (max sm_86) while the RTX 4060 is sm_89 (Ada) — these run via forward
PTX-JIT, which this script verifies for the ops MAMBA/STPN actually use.

Usage:
    conda run -n vfe --no-capture-output python tools/checks/gpu_smoke.py
"""
import torch

print("torch", torch.__version__, "| cap", torch.cuda.get_device_capability(0))


def check(name, fn):
    try:
        fn()
        torch.cuda.synchronize()
        print(f"OK    {name}")
    except Exception as e:
        print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:160]}")


# 1) basic elementwise + matmul (cublas)
check("matmul", lambda: (torch.randn(512, 512, device="cuda") @ torch.randn(512, 512, device="cuda")))


# 2) conv2d (cudnn)
def conv():
    x = torch.randn(2, 3, 64, 64, device="cuda")
    w = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
    return w(x)


check("conv2d/cudnn", conv)


# 3) mmcv compiled op: NMS (the real risk on sm_89)
def mmcv_nms():
    from mmcv.ops import nms
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float32, device="cuda")
    scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
    return nms(boxes, scores, 0.5)


check("mmcv.ops.nms (cuda)", mmcv_nms)


# 4) mmcv RoIAlign (used by the detectors)
def mmcv_roialign():
    from mmcv.ops import RoIAlign
    feat = torch.randn(1, 4, 16, 16, device="cuda")
    rois = torch.tensor([[0, 0, 0, 8, 8]], dtype=torch.float32, device="cuda")
    return RoIAlign(output_size=7, spatial_scale=1.0, sampling_ratio=2)(feat, rois)


check("mmcv.ops.RoIAlign (cuda)", mmcv_roialign)
