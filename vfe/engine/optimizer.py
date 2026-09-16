"""Optimiser construction. Port of mmcv's ``DefaultOptimizerConstructor``.

Without ``paramwise_cfg`` every parameter shares the optimiser's settings
(MAMBA: SGD). With it, each parameter's lr and weight decay are resolved
separately (STPN: AdamW with no decay on norms and position tables):

* ``custom_keys``: the first key that is a *substring* of the dotted parameter
  name wins, longest keys tried first (ties alphabetical). ``"norm"`` therefore
  matches ``backbone.stages.0.blocks.0.norm1.weight``.
* otherwise ``bias_lr_mult`` / ``bias_decay_mult`` / ``norm_decay_mult`` /
  ``dwconv_decay_mult`` apply, as in mmcv.

mmcv built one param group per parameter. Parameters with identical settings
are merged here, which changes nothing for SGD or AdamW (their updates are
per-parameter) but keeps the optimiser step batched.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

__all__ = ["build_optimizer", "paramwise_settings"]

OPTIMIZERS = {"SGD": torch.optim.SGD, "Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}
PARAMWISE_KEYS = {"custom_keys", "bias_lr_mult", "bias_decay_mult", "norm_decay_mult",
                  "dwconv_decay_mult", "bypass_duplicate"}


def paramwise_settings(model: nn.Module, base_lr: float, base_wd: float | None,
                       paramwise_cfg: dict) -> list[tuple[nn.Parameter, dict[str, float]]]:
    """``(parameter, overrides)`` in mmcv's traversal order; an empty dict means
    the optimiser's defaults."""
    unknown = set(paramwise_cfg) - PARAMWISE_KEYS
    if unknown:
        raise NotImplementedError(f"paramwise_cfg keys not ported: {sorted(unknown)}")
    custom_keys = paramwise_cfg.get("custom_keys", {})
    sorted_keys = sorted(sorted(custom_keys), key=len, reverse=True)
    bias_lr_mult = paramwise_cfg.get("bias_lr_mult", 1.0)
    bias_decay_mult = paramwise_cfg.get("bias_decay_mult", 1.0)
    norm_decay_mult = paramwise_cfg.get("norm_decay_mult", 1.0)
    dwconv_decay_mult = paramwise_cfg.get("dwconv_decay_mult", 1.0)
    bypass_duplicate = paramwise_cfg.get("bypass_duplicate", False)

    out: list[tuple[nn.Parameter, dict[str, float]]] = []
    seen: set[int] = set()

    def visit(module: nn.Module, prefix: str) -> None:
        is_norm = isinstance(module, (_BatchNorm, nn.GroupNorm, nn.LayerNorm))
        is_dwconv = isinstance(module, nn.Conv2d) and module.in_channels == module.groups
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                out.append((param, {}))
                continue
            if bypass_duplicate and id(param) in seen:
                continue
            seen.add(id(param))
            overrides: dict[str, float] = {}
            key = next((k for k in sorted_keys if k in f"{prefix}.{name}"), None)
            if key is not None:
                overrides["lr"] = base_lr * custom_keys[key].get("lr_mult", 1.0)
                if base_wd is not None:
                    overrides["weight_decay"] = base_wd * custom_keys[key].get("decay_mult", 1.0)
            else:
                if name == "bias" and not is_norm:
                    overrides["lr"] = base_lr * bias_lr_mult
                if base_wd is not None:
                    if is_norm:
                        overrides["weight_decay"] = base_wd * norm_decay_mult
                    elif is_dwconv:
                        overrides["weight_decay"] = base_wd * dwconv_decay_mult
                    elif name == "bias":
                        overrides["weight_decay"] = base_wd * bias_decay_mult
            out.append((param, overrides))
        for child_name, child in module.named_children():
            visit(child, f"{prefix}.{child_name}" if prefix else child_name)

    visit(model, "")
    return out


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """``cfg`` is a config's ``optimizer`` dict: ``type``, the optimiser's own
    arguments, and optionally ``paramwise_cfg``."""
    cfg = dict(cfg)
    optim_type = cfg.pop("type")
    paramwise_cfg = cfg.pop("paramwise_cfg", None)
    if optim_type not in OPTIMIZERS:
        raise NotImplementedError(f"optimizer {optim_type!r} is not ported")
    optim_cls = OPTIMIZERS[optim_type]
    if hasattr(model, "module"):  # DDP
        model = model.module

    if not paramwise_cfg:
        return optim_cls(model.parameters(), **cfg)

    # Merge parameters whose resolved settings agree, keeping first-seen order.
    groups: dict[tuple, dict[str, Any]] = {}
    for param, overrides in paramwise_settings(model, cfg["lr"], cfg.get("weight_decay"),
                                               paramwise_cfg):
        key = tuple(sorted(overrides.items()))
        groups.setdefault(key, {"params": [], **overrides})["params"].append(param)
    return optim_cls(list(groups.values()), **cfg)
