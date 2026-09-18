"""``build_{conv,norm,activation}_layer`` -- the ``mmcv.cnn`` brick builders.

Only the layer types this repo's configs reference are registered. Adding a new
one is a single ``NORM_LAYERS[...]`` entry, but leaving the table small keeps
it obvious what the models can actually contain.

The naming rule matters for checkpoint compatibility: ``build_norm_layer``
returns ``(name, layer)`` where ``name`` is an abbreviation plus a postfix
(``bn1``, ``bn2``, ...). mmdet's ResNet registers its norms under those names,
which is why its ``state_dict`` keys line up with torchvision's ``conv1/bn1/
layer1.0.bn1/...`` and why released checkpoints load without remapping.
"""

from __future__ import annotations

from typing import Any

from torch import nn

__all__ = ["build_conv_layer", "build_norm_layer", "build_activation_layer"]

CONV_LAYERS: dict[str, type[nn.Module]] = {
    "Conv": nn.Conv2d,
    "Conv1d": nn.Conv1d,
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
}

# (class, abbreviation) -- the abbreviation becomes the module's attribute name.
NORM_LAYERS: dict[str, tuple[type[nn.Module], str]] = {
    "BN": (nn.BatchNorm2d, "bn"),
    "BN1d": (nn.BatchNorm1d, "bn"),
    "BN2d": (nn.BatchNorm2d, "bn"),
    "BN3d": (nn.BatchNorm3d, "bn"),
    "SyncBN": (nn.SyncBatchNorm, "bn"),
    "GN": (nn.GroupNorm, "gn"),
    "LN": (nn.LayerNorm, "ln"),
    "IN": (nn.InstanceNorm2d, "in"),
}

ACTIVATION_LAYERS: dict[str, type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "PReLU": nn.PReLU,
    "ReLU6": nn.ReLU6,
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "SiLU": nn.SiLU,
}


def build_conv_layer(cfg: dict | None, *args: Any, **kwargs: Any) -> nn.Module:
    """Build a conv layer; ``cfg=None`` means a plain ``nn.Conv2d``."""
    if cfg is None:
        return nn.Conv2d(*args, **kwargs)
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict or None, got {type(cfg)}")
    cfg_ = dict(cfg)
    layer_type = cfg_.pop("type", None)
    if layer_type is None:
        raise KeyError("conv cfg must contain a 'type' key")
    if layer_type not in CONV_LAYERS:
        raise KeyError(
            f"unsupported conv type {layer_type!r}; known: {sorted(CONV_LAYERS)}. "
            "Deformable convs are not ported -- no config here uses them."
        )
    return CONV_LAYERS[layer_type](*args, **kwargs, **cfg_)


def build_norm_layer(
    cfg: dict, num_features: int, postfix: int | str = ""
) -> tuple[str, nn.Module]:
    """Build a norm layer, returning ``(name, layer)``.

    ``requires_grad`` in ``cfg`` sets the layer's affine parameters' grad flag
    (mmcv semantics); it is not a constructor argument. ``eps`` defaults to
    1e-5 to match mmcv, which matters for numerical parity with checkpoints
    trained under it.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg)}")
    cfg_ = dict(cfg)
    layer_type = cfg_.pop("type", None)
    if layer_type is None:
        raise KeyError("norm cfg must contain a 'type' key")
    if layer_type not in NORM_LAYERS:
        raise KeyError(f"unsupported norm type {layer_type!r}; known: {sorted(NORM_LAYERS)}")

    norm_cls, abbr = NORM_LAYERS[layer_type]
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)

    if layer_type == "GN":
        if "num_groups" not in cfg_:
            raise KeyError("GroupNorm requires 'num_groups'")
        layer = norm_cls(num_channels=num_features, **cfg_)
    elif layer_type == "LN":
        layer = norm_cls(num_features, **cfg_)
    else:
        layer = norm_cls(num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def build_activation_layer(cfg: dict) -> nn.Module:
    """Build an activation layer from ``dict(type='ReLU', inplace=True)``."""
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg)}")
    cfg_ = dict(cfg)
    layer_type = cfg_.pop("type", None)
    if layer_type is None:
        raise KeyError("activation cfg must contain a 'type' key")
    if layer_type not in ACTIVATION_LAYERS:
        raise KeyError(
            f"unsupported activation {layer_type!r}; known: {sorted(ACTIVATION_LAYERS)}"
        )
    return ACTIVATION_LAYERS[layer_type](**cfg_)
