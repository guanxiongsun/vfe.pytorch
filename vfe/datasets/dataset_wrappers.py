"""Dataset wrappers. Port of ``mmdet.datasets.dataset_wrappers.ConcatDataset``."""

from __future__ import annotations

import numpy as np
from torch.utils.data import ConcatDataset as _ConcatDataset

__all__ = ["ConcatDataset"]


class ConcatDataset(_ConcatDataset):
    """``torch``'s ``ConcatDataset`` that also concatenates the members'
    aspect-ratio ``flag`` arrays, so the group sampler sees one index space.

    The VID configs train on a list of two datasets, ImageNet VID then the DET
    30-class subset, which ``build_dataset`` turns into one of these.
    """

    def __init__(self, datasets):
        super().__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        if hasattr(datasets[0], "flag"):
            self.flag = np.concatenate([ds.flag for ds in datasets])
