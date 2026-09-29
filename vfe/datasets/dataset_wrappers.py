"""Dataset wrappers. Port of ``mmdet.datasets.dataset_wrappers.{ConcatDataset,
MultiImageMixDataset}``."""

from __future__ import annotations

import bisect
import collections.abc
import copy

import numpy as np
from torch.utils.data import ConcatDataset as _ConcatDataset

from vfe.datasets.builder import DATASETS, PIPELINES
from vfe.registry import build_from_cfg

__all__ = ["ConcatDataset", "MultiImageMixDataset"]


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

    def get_ann_info(self, idx: int):
        """The ground truth of item ``idx``, from the member dataset holding it."""
        if idx < 0:
            idx += len(self)
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx - (self.cumulative_sizes[dataset_idx - 1] if dataset_idx else 0)
        return self.datasets[dataset_idx].get_ann_info(sample_idx)


@DATASETS.register_module()
class MultiImageMixDataset:
    """A dataset whose pipeline may mix in other items (Mosaic, MixUp): a
    transform with ``get_indexes`` receives those items' results as
    ``results['mix_results']``. ``update_skip_type_keys`` switches transforms
    off by type, as YOLOX does for its last epochs.

    ``dataset`` may be a config (or a list of them), built here.
    """

    def __init__(self, dataset, pipeline, dynamic_scale=None, skip_type_keys=None):
        if dynamic_scale is not None:
            raise ValueError("dynamic_scale is deprecated in mmdet; resize in the pipeline")
        if isinstance(dataset, (dict, list, tuple)):
            from vfe.datasets.builder import build_dataset

            dataset = build_dataset(dataset)
        self._skip_type_keys = skip_type_keys
        self.pipeline, self.pipeline_types = [], []
        for transform in pipeline:
            if not isinstance(transform, dict):
                raise TypeError("pipeline must be a list of dicts")
            self.pipeline_types.append(transform["type"])
            self.pipeline.append(build_from_cfg(transform, PIPELINES))
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        if hasattr(dataset, "flag"):
            self.flag = dataset.flag
        self.num_samples = len(dataset)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for transform, transform_type in zip(self.pipeline, self.pipeline_types, strict=True):
            if self._skip_type_keys is not None and transform_type in self._skip_type_keys:
                continue
            if hasattr(transform, "get_indexes"):
                indexes = transform.get_indexes(self.dataset)
                if not isinstance(indexes, collections.abc.Sequence):
                    indexes = [indexes]
                results["mix_results"] = [copy.deepcopy(self.dataset[i]) for i in indexes]
            results = transform(results)
            if results is None:
                # A transform dropped the sample (FilterAnnotations with
                # keep_empty): draw another, as the wrapped datasets do.
                return self[np.random.randint(len(self))]
            results.pop("mix_results", None)
        return results

    def update_skip_type_keys(self, skip_type_keys) -> None:
        if not all(isinstance(k, str) for k in skip_type_keys):
            raise TypeError("skip_type_keys must be strings")
        self._skip_type_keys = skip_type_keys
