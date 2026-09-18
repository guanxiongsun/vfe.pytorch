"""RoI feature extraction. Port of
``mmdet.models.roi_heads.roi_extractors.{base_roi_extractor,single_level_roi_extractor}``.

Turns a list of proposals into a fixed-size feature per proposal by RoIAlign-ing
out of the feature pyramid. With one input level (MAMBA's DC5 setup) that is all
it does; with several (STPN's FPN) each RoI is first routed to the level whose
resolution best matches its scale.

``GenericRoIExtractor`` (the SoftRoIPooling variant) is not ported -- nothing
here configures it.
"""

from __future__ import annotations

import torch
from torch import nn

from vfe import ops
from vfe.models.builder import ROI_EXTRACTORS

__all__ = ["BaseRoIExtractor", "SingleRoIExtractor"]


class BaseRoIExtractor(nn.Module):
    """Holds one RoI layer per pyramid level, each with its own spatial scale."""

    def __init__(self, roi_layer: dict, out_channels: int, featmap_strides: list[int]):
        super().__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides

    @property
    def num_inputs(self) -> int:
        return len(self.featmap_strides)

    def build_roi_layers(self, layer_cfg: dict, featmap_strides: list[int]) -> nn.ModuleList:
        """One layer per level. ``spatial_scale = 1 / stride`` is what converts
        RoI coordinates, which live in image space, into feature-map space."""
        cfg = dict(layer_cfg)
        layer_type = cfg.pop("type")
        if not hasattr(ops, layer_type):
            raise KeyError(f"vfe.ops has no RoI layer {layer_type!r}")
        layer_cls = getattr(ops, layer_type)
        return nn.ModuleList([layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])

    def roi_rescale(self, rois: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """Grow or shrink each RoI about its own centre."""
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = (rois[:, 3] - rois[:, 1]) * scale_factor
        h = (rois[:, 4] - rois[:, 2]) * scale_factor
        return torch.stack(
            (rois[:, 0], cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5), dim=-1
        )

    def forward(self, feats, rois, roi_scale_factor=None):
        raise NotImplementedError


@ROI_EXTRACTORS.register_module()
class SingleRoIExtractor(BaseRoIExtractor):
    """Each RoI is pooled from exactly one pyramid level.

    Args:
        roi_layer: e.g. ``dict(type='RoIAlign', output_size=7, sampling_ratio=2)``.
        out_channels: channels of the pooled feature.
        featmap_strides: stride of each input level w.r.t. the input image.
        finest_scale: the RoI size, in pixels, that maps to level 0.
    """

    def __init__(self, roi_layer, out_channels, featmap_strides, finest_scale: int = 56):
        super().__init__(roi_layer, out_channels, featmap_strides)
        self.finest_scale = finest_scale

    def map_roi_levels(self, rois: torch.Tensor, num_levels: int) -> torch.Tensor:
        """Assign each RoI a level from its ``sqrt(w * h)``, doubling per level:
        ``< 2 * finest_scale`` → 0, ``< 4 *`` → 1, and so on, clamped to range.

        The 1e-6 inside the log is mmdet's guard against ``log2(0)`` on a
        degenerate RoI; it is kept because removing it changes the level of
        boundary-sized RoIs.
        """
        scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        return target_lvls.clamp(min=0, max=num_levels - 1).long()

    def forward(self, feats, rois: torch.Tensor, roi_scale_factor: float | None = None):
        """``rois`` is ``(n, 5)`` = ``[batch_index, x1, y1, x2, y2]``."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(rois.size(0), self.out_channels, *out_size)

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            inds = (target_lvls == i).nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                roi_feats[inds] = self.roi_layers[i](feats[i], rois[inds])
            else:
                # A level that won the scale lottery for no RoI would otherwise
                # be missing from the graph, and under DDP that means its
                # gradients never arrive and the all-reduce hangs. Adding a
                # zero-valued term keeps every level connected.
                roi_feats = roi_feats + sum(
                    x.view(-1)[0] for x in self.parameters()
                ) * 0.0 + feats[i].sum() * 0.0
        return roi_feats
