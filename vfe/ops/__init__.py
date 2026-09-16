"""Replacements for the ``mmcv.ops`` entry points this project uses.

Everything here delegates to ``torchvision.ops``, so there is no compiled
extension to build and the package installs from wheels on both x86_64 and
aarch64. The signatures and return values match mmcv's so call sites port over
unchanged; behaviour is pinned by ``tools/checks/parity_ops.py``, which diffs
against the legacy ``vfe`` env.
"""

from .nms import batched_nms, nms
from .roi_align import RoIAlign, roi_align

__all__ = ["nms", "batched_nms", "roi_align", "RoIAlign"]
