"""Parity check: ``vfe.models`` backbones/necks vs the ``mmdet`` oracle.

Same two-process pattern as ``parity_ops.py``. Two things are compared:

1. **``state_dict`` key sets** -- if these differ, released checkpoints will
   load partially and silently. This is the check that actually protects the
   port; the numeric one below only confirms the forward pass.
2. **Output feature maps**, with both sides loaded from the *same* synthetic
   weights. Weights are derived per-key from ``crc32(key)`` rather than from a
   sequential generator, so they do not depend on ``state_dict`` ordering or on
   either torch version's RNG.

Usage:
    conda run -n vfe       --no-capture-output python tools/checks/parity_backbone.py --impl mmdet --out ~/bb_mmdet.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_backbone.py --impl vfe   --out ~/bb_vfe.pt
    conda run -n vfe-torch --no-capture-output python tools/checks/parity_backbone.py --compare ~/bb_mmdet.pt ~/bb_vfe.pt

``--device cuda`` compares the CUDA path. TF32 is disabled there: it is on by
default for convolutions in both torch versions, and its ~10-bit mantissa
produces ~1e-4 relative differences that swamp anything the port itself could
be doing wrong. Disabling it makes the CUDA comparison as sharp as the CPU one.
"""

import argparse
import sys
import zlib
from collections import OrderedDict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

# ResNet-101 DC5: stride-16 output, layer4 dilated. The backbone of the MAMBA
# and SELSA configs (configs/_base_/models/vid/faster_rcnn_r50_dc5.py).
R101_DC5 = dict(
    type="ResNet",
    depth=101,
    num_stages=4,
    out_indices=(3,),
    strides=(1, 2, 2, 1),
    dilations=(1, 1, 1, 2),
    frozen_stages=1,
    norm_cfg=dict(type="BN", requires_grad=True),
    norm_eval=True,
    style="pytorch",
)
R50_DC5 = dict(R101_DC5, depth=50)
# Plain ResNet-50: no dilation, all four levels out. Exercises the code path
# where mmdet and torchvision agree, as a control for the DC5 cases.
R50_PLAIN = dict(
    type="ResNet", depth=50, num_stages=4, out_indices=(0, 1, 2, 3), frozen_stages=1,
    norm_cfg=dict(type="BN", requires_grad=True), norm_eval=True, style="pytorch",
)

# Swin-T, as the STPN configs configure it (minus the prompt tokens).
SWIN_T = dict(
    type="SwinTransformer",
    embed_dims=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.2,
    patch_norm=True,
    with_cp=False,
    convert_weights=True,
)

CHANNEL_MAPPER = dict(type="ChannelMapper", in_channels=[2048], out_channels=512, kernel_size=3)
FPN_SWIN = dict(type="FPN", in_channels=[96, 192, 384, 768], out_channels=256, num_outs=5)

CASES = [
    # (name, backbone_cfg, neck_cfg, train_mode)
    ("r101_dc5", R101_DC5, None, False),
    ("r101_dc5+mapper", R101_DC5, CHANNEL_MAPPER, False),
    # norm_eval + frozen_stages must keep BN in eval even under .train().
    ("r101_dc5/train_mode", R101_DC5, CHANNEL_MAPPER, True),
    ("r50_dc5", R50_DC5, None, False),
    ("r50_plain", R50_PLAIN, None, False),
    ("fpn_swin_shapes", None, FPN_SWIN, False),
    ("swin_t", SWIN_T, None, False),
    ("swin_t+fpn", SWIN_T, FPN_SWIN, False),
    # drop_path_rate=0 because DropPath draws from the global RNG in train
    # mode, which no seeding can align across two torch versions. What this
    # case checks is `_freeze_stages` -- recorded as the frozen-parameter list
    # below, not as output values.
    ("swin_t/frozen", dict(SWIN_T, drop_path_rate=0.0, frozen_stages=2), None, True),
]

# 224x400 is deliberately not a multiple of the window size at every stage, so
# ShiftWindowMSA's padding path is exercised.
IMAGE_SHAPE = (1, 3, 224, 400)

# Swin-T's output shapes for IMAGE_SHAPE. Fed to FPN directly in the
# `fpn_swin_shapes` case, so the neck is exercised independently of the backbone.
FPN_INPUT_SHAPES = [(1, 96, 152, 100), (1, 192, 76, 50), (1, 384, 38, 25), (1, 768, 19, 13)]


def seeded_like(key, tensor, seed=0):
    """A deterministic tensor for ``key``, independent of iteration order."""
    if not tensor.dtype.is_floating_point:
        return tensor.clone()
    g = torch.Generator().manual_seed((seed * 2654435761 + zlib.crc32(key.encode())) % (2**31))
    value = torch.randn(tensor.shape, generator=g) * 0.05
    if key.endswith("running_var"):
        # Variance must be positive, and near 1 so activations stay in range.
        value = value.abs() + 1.0
    return value.to(tensor.dtype)


def synth_state_dict(module, seed=0):
    return OrderedDict(
        (k, seeded_like(k, v, seed)) for k, v in module.state_dict().items()
    )


def make_input(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def run(impl, device):
    if device.startswith("cuda"):
        # Compare exact fp32, not two different TF32 roundings -- see module docstring.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if impl == "mmdet":
        from mmdet.models import build_backbone, build_neck
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from vfe.models import build_backbone, build_neck

    results = {}
    for name, bb_cfg, neck_cfg, train_mode in CASES:
        entry = {}
        modules = []
        if bb_cfg is not None:
            backbone = build_backbone(dict(bb_cfg))
            modules.append(("backbone", backbone))
        if neck_cfg is not None:
            neck = build_neck(dict(neck_cfg))
            modules.append(("neck", neck))

        for tag, module in modules:
            sd = synth_state_dict(module)
            module.load_state_dict(sd, strict=True)
            module.to(device)
            module.train(train_mode)
            entry[f"{tag}_keys"] = sorted(sd)
            # Which parameters `frozen_stages` actually froze, and which
            # submodules it forced out of train mode -- both are structural, so
            # they are compared as sets rather than numerically.
            entry[f"{tag}_frozen"] = sorted(
                n for n, p in module.named_parameters() if not p.requires_grad
            )
            entry[f"{tag}_eval_modules"] = sorted(
                n for n, m in module.named_modules() if n and not m.training
            )

        with torch.no_grad():
            if bb_cfg is not None:
                feats = dict(modules)["backbone"](make_input(IMAGE_SHAPE, 11).to(device))
            else:
                feats = tuple(
                    make_input(shape, 12 + i).to(device)
                    for i, shape in enumerate(FPN_INPUT_SHAPES)
                )
            if neck_cfg is not None:
                feats = dict(modules)["neck"](feats)

        for i, t in enumerate(feats):
            entry[f"out{i}"] = t.float().cpu()
        print(f"RAN   {name}  ->  {[tuple(t.shape) for t in feats]}")
        results[name] = entry
    return results


def compare(path_a, path_b, atol, rtol):
    a = torch.load(path_a, map_location="cpu", weights_only=False)
    b = torch.load(path_b, map_location="cpu", weights_only=False)

    failures = 0
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            failures += 1
            print(f"FAIL  {name}: only in {'A' if name in a else 'B'}")
            continue

        notes = []
        ok = True
        for key in sorted(set(a[name]) | set(b[name])):
            va, vb = a[name].get(key), b[name].get(key)
            if va is None or vb is None:
                ok = False
                notes.append(f"{key}: only in {'A' if vb is None else 'B'}")
            elif key.endswith(("_keys", "_frozen", "_eval_modules")):
                only_a, only_b = sorted(set(va) - set(vb)), sorted(set(vb) - set(va))
                if only_a or only_b:
                    ok = False
                    notes.append(f"{key}: {len(only_a)} only in A {only_a[:5]}, "
                                 f"{len(only_b)} only in B {only_b[:5]}")
            elif va.shape != vb.shape:
                ok = False
                notes.append(f"{key}: shape {tuple(va.shape)} != {tuple(vb.shape)}")
            elif not torch.allclose(va, vb, atol=atol, rtol=rtol):
                ok = False
                notes.append(
                    f"{key}: max|diff| = {(va - vb).abs().max().item():.3e} "
                    f"(scale {va.abs().max().item():.3e})"
                )

        if ok:
            print(f"OK    {name}")
        else:
            failures += 1
            print(f"FAIL  {name}")
            for note in notes:
                print(f"        {note}")

    print("-" * 70)
    print("BACKBONE PARITY OK" if failures == 0 else f"{failures} case(s) differ")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["mmdet", "vfe"])
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare, atol=args.atol, rtol=args.rtol)
    elif args.impl and args.out:
        torch.save(run(args.impl, args.device), args.out)
        print(f"saved -> {args.out}")
    else:
        ap.error("pass either --impl/--out or --compare")
