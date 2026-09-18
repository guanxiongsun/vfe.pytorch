"""``AutoAugment``: pick one of several augmentation policies per sample. Port
of ``mmdet.datasets.pipelines.auto_augment.AutoAugment`` (the policy wrapper
only; mmdet's individual AutoAugment operations are unused by the configs)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from vfe.datasets.builder import PIPELINES

__all__ = ["AutoAugment"]


@PIPELINES.register_module()
class AutoAugment:
    """``policies``: a list of policies, each a non-empty list of transform
    configs. Each call applies one policy chosen uniformly with numpy's global
    generator (``np.random.randint(n)``, the same draw as the original's
    ``np.random.choice`` over the policies)."""

    def __init__(self, policies: list[list[dict]]):
        from vfe.datasets.pipelines import Compose

        if not isinstance(policies, list) or not policies:
            raise ValueError("policies must be a non-empty list")
        for policy in policies:
            if not isinstance(policy, list) or not policy:
                raise ValueError("each policy must be a non-empty list of transforms")
            for transform in policy:
                if not isinstance(transform, dict) or "type" not in transform:
                    raise ValueError("each transform must be a dict with a 'type'")
        self.policies = copy.deepcopy(policies)
        self.transforms = [Compose(policy) for policy in self.policies]

    def __call__(self, results: Any) -> Any:
        return self.transforms[np.random.randint(len(self.transforms))](results)
