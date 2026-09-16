"""Minimal replacement for ``mmcv.utils.Registry`` + ``build_from_cfg``.

Same contract as mmcv's: a config dict with a ``type`` key names a registered
class, remaining keys become its kwargs. Dropped from mmcv's version: registry
*scopes*/parents (a single-project codebase has no use for them), the
``build_func`` indirection, and deferred string-import resolution.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

__all__ = ["Registry", "build_from_cfg"]

T = TypeVar("T")


class Registry:
    """Name -> class mapping, populated by ``@REGISTRY.register_module()``."""

    def __init__(self, name: str):
        self._name = name
        self._module_dict: dict[str, type] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> dict[str, type]:
        return self._module_dict

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: object) -> bool:
        return key in self._module_dict

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={sorted(self._module_dict)})"

    def get(self, key: str) -> type:
        try:
            return self._module_dict[key]
        except KeyError:
            raise KeyError(
                f"{key!r} is not in the {self._name} registry. "
                f"Registered: {sorted(self._module_dict)}"
            ) from None

    def register_module(
        self,
        name: str | None = None,
        force: bool = False,
        module: type | None = None,
    ) -> Callable[[type[T]], type[T]] | type:
        """Usable as ``@reg.register_module()``, ``@reg.register_module('Alias')``
        or imperatively as ``reg.register_module(module=Cls)``."""
        if module is not None:
            self._register(module, name, force)
            return module

        def decorator(cls: type[T]) -> type[T]:
            self._register(cls, name, force)
            return cls

        return decorator

    def _register(self, module: type, name: str | None, force: bool) -> None:
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError(f"module must be a class or function, got {type(module)}")
        key = name or module.__name__
        if not force and key in self._module_dict:
            existing = self._module_dict[key]
            raise KeyError(
                f"{key!r} is already registered in {self._name} "
                f"({existing.__module__}.{existing.__qualname__})"
            )
        self._module_dict[key] = module


def build_from_cfg(cfg: dict, registry: Registry, default_args: dict | None = None) -> Any:
    """Instantiate ``registry[cfg['type']](**rest)``.

    ``cfg['type']`` may also be a class directly, which keeps hand-written
    Python configs from needing a registration round-trip.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg)}")
    if "type" not in cfg:
        raise KeyError(f"cfg must contain a 'type' key, got keys {sorted(cfg)}")

    args = dict(cfg)
    for key, value in (default_args or {}).items():
        args.setdefault(key, value)

    obj_type = args.pop("type")
    obj_cls = registry.get(obj_type) if isinstance(obj_type, str) else obj_type
    if not (inspect.isclass(obj_cls) or inspect.isfunction(obj_cls)):
        raise TypeError(f"type must be a str or class, got {type(obj_type)}")

    try:
        return obj_cls(**args)
    except Exception as e:
        raise type(e)(f"{obj_cls.__name__}: {e}") from e
