"""Minimal replacement for ``mmcv.Config``.

Only the features the VID configs in ``configs/`` actually use are supported:

* Python config files evaluated as modules; UPPER/lower-case top-level names are
  all kept, dunders are dropped.
* ``_base_`` inheritance (a single path or a list), resolved relative to the
  config file, with recursive dict merging.
* ``_delete_ = True`` inside a dict to replace rather than merge that subtree.
* Attribute access (``cfg.model.backbone.depth``) and dict access alike.

Deliberately *not* supported (unused by this repo): ``.yaml``/``.json`` configs,
``{{ fileDirname }}`` predefined variables, ``${...}`` env substitution, and
``Config.fromstring``. Keeping the surface this small is the point -- mmcv's
version is ~700 lines and pulls in ``addict`` plus a temp-dir import hack.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any

__all__ = ["ConfigDict", "Config", "merge_dicts"]

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"


class ConfigDict(dict):
    """A dict whose keys are also reachable as attributes, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}") from None
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, _wrap(value))

    def copy(self) -> ConfigDict:
        return copy.deepcopy(self)

    def __deepcopy__(self, memo: dict) -> ConfigDict:
        out = type(self)()
        memo[id(self)] = out
        for key, value in self.items():
            out[copy.deepcopy(key, memo)] = copy.deepcopy(value, memo)
        return out

    def to_dict(self) -> dict:
        """Plain-``dict`` view, for serialisation or ``**kwargs`` splatting."""
        return {k: _unwrap(v) for k, v in self.items()}


def _wrap(value: Any) -> Any:
    if isinstance(value, ConfigDict):
        return value
    if isinstance(value, dict):
        return ConfigDict({k: _wrap(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return type(value)(_wrap(v) for v in value)
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_unwrap(v) for v in value)
    return value


def merge_dicts(base: dict, override: dict) -> ConfigDict:
    """Recursively merge ``override`` into ``base``; ``override`` wins.

    A dict in ``override`` carrying ``_delete_ = True`` replaces the
    corresponding subtree outright instead of merging into it. This is how the
    configs swap e.g. a whole ``optimizer_config`` rather than patching it.
    """
    merged = ConfigDict(_wrap(base))
    for key, value in override.items():
        if isinstance(value, dict) and value.pop(DELETE_KEY, False):
            merged[key] = _wrap(value)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = _wrap(value)
    return merged


def _exec_py_config(path: Path) -> dict:
    """Import ``path`` as a throwaway module and return its public globals."""
    mod_name = f"_vfe_cfg_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load config {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
        return {k: v for k, v in vars(module).items() if not k.startswith("__")}
    finally:
        sys.modules.pop(mod_name, None)


class Config:
    """A loaded config file: ``Config.fromfile(path)``."""

    def __init__(self, cfg_dict: dict | None = None, filename: str | Path | None = None):
        self._cfg_dict = ConfigDict(_wrap(cfg_dict or {}))
        self._filename = str(filename) if filename is not None else None

    # -- construction -----------------------------------------------------
    @classmethod
    def fromfile(cls, filename: str | Path) -> Config:
        path = Path(filename).resolve()
        if path.suffix != ".py":
            raise NotImplementedError(f"only .py configs are supported, got {path.name}")
        return cls(cls._load(path), filename=path)

    @staticmethod
    def _load(path: Path) -> ConfigDict:
        raw = _exec_py_config(path)
        bases = raw.pop(BASE_KEY, [])
        if isinstance(bases, (str, Path)):
            bases = [bases]

        merged = ConfigDict()
        for base in bases:
            base_path = (path.parent / base).resolve()
            merged = merge_dicts(merged, Config._load(base_path))
        return merge_dicts(merged, raw)

    # -- access -----------------------------------------------------------
    @property
    def filename(self) -> str | None:
        return self._filename

    def __getattr__(self, name: str) -> Any:
        try:
            return getattr(self._cfg_dict, name)
        except AttributeError:
            raise AttributeError(f"Config has no attribute {name!r}") from None

    def __getitem__(self, name: str) -> Any:
        return self._cfg_dict[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self._cfg_dict[name] = value

    def __contains__(self, name: object) -> bool:
        return name in self._cfg_dict

    def __iter__(self):
        return iter(self._cfg_dict)

    def get(self, name: str, default: Any = None) -> Any:
        return self._cfg_dict.get(name, default)

    def keys(self):
        return self._cfg_dict.keys()

    def items(self):
        return self._cfg_dict.items()

    def to_dict(self) -> dict:
        return self._cfg_dict.to_dict()

    def merge_from_dict(self, options: dict) -> None:
        """Apply ``{'model.backbone.depth': 101}``-style dotted overrides."""
        patch: dict = {}
        for dotted, value in options.items():
            node = patch
            *parents, leaf = dotted.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value
        self._cfg_dict = merge_dicts(self._cfg_dict, patch)

    def __repr__(self) -> str:
        return f"Config (path: {self._filename}): {self._cfg_dict!r}"
