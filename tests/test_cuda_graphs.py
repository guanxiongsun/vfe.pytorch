"""CUDA-graph replay of the fixed-shape parts of inference (GPU only)."""

import pytest
import torch

from vfe.engine.cuda_graphs import GraphedCallable, enable_cuda_graphs
from vfe.models.builder import build_detector

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

DETECTOR = dict(
    type="FCOS",
    backbone=dict(type="ResNet", depth=18, num_stages=4, out_indices=(0, 1, 2, 3),
                  frozen_stages=1, norm_cfg=dict(type="BN", requires_grad=False),
                  norm_eval=True, style="pytorch"),
    neck=dict(type="FPN", in_channels=[64, 128, 256, 512], out_channels=32, start_level=1,
              add_extra_convs="on_output", num_outs=5, relu_before_extra_convs=True),
    bbox_head=dict(type="FCOSHead", num_classes=30, in_channels=32, stacked_convs=2,
                   feat_channels=32, strides=[8, 16, 32, 64, 128],
                   norm_cfg=dict(type="GN", num_groups=4, requires_grad=True)),
    test_cfg=dict(nms_pre=200, min_bbox_size=0, score_thr=0.01,
                  nms=dict(type="nms", iou_threshold=0.5), max_per_img=20),
)


def test_graphed_callable_replays_per_shape_and_falls_back_with_grad():
    calls = []

    def fn(x):
        calls.append(1)
        return x * 2 + 1

    g = GraphedCallable(fn, warmup=1)
    a = torch.randn(4, device="cuda")
    with torch.no_grad():
        assert torch.equal(g(a), a * 2 + 1)
        n = len(calls)
        b = torch.randn(4, device="cuda")
        assert torch.equal(g(b), b * 2 + 1) and len(calls) == n  # replayed, not re-run
        assert torch.equal(g(torch.ones(3, device="cuda")), torch.full((3,), 3.0, device="cuda"))
        assert len(g.graphs) == 2
    out = g(a.requires_grad_())
    assert out.requires_grad  # with gradients on, the eager function runs


YOLOX = dict(
    type="YOLOX",
    input_size=(96, 128),
    backbone=dict(type="CSPDarknet", deepen_factor=0.33, widen_factor=0.125),
    neck=dict(type="YOLOXPAFPN", in_channels=[32, 64, 128], out_channels=32, num_csp_blocks=1),
    bbox_head=dict(type="YOLOXHead", num_classes=30, in_channels=32, feat_channels=32),
    test_cfg=dict(score_thr=0.0, nms=dict(type="nms", iou_threshold=0.65), max_per_img=20),
)


@pytest.mark.parametrize("config", [DETECTOR, YOLOX], ids=["fcos", "yolox"])
def test_graphed_detector_matches_eager(config):
    torch.manual_seed(0)
    detector = build_detector(config).cuda().eval()
    img = torch.randn(1, 3, 96, 128, device="cuda")
    new = torch.randn_like(img)
    metas = [dict(img_shape=(96, 128, 3), scale_factor=[1.0] * 4)]
    with torch.no_grad():
        eager_new = detector.extract_feat(new)
        feats = detector.extract_feat(img)
        eager = detector.bbox_head(feats)
        eager_dets = detector.bbox_head.simple_test(feats, metas)
        other = [f.flip(-1) for f in feats]
        eager_mixed = detector.bbox_head(feats, reg_feats=other)
    enable_cuda_graphs(detector)
    with torch.no_grad():
        for _ in range(2):  # capture, then replay
            graphed_feats = detector.extract_feat(img)
            graphed = detector.bbox_head(graphed_feats)
            graphed_dets = detector.bbox_head.simple_test(graphed_feats, metas)
        graphed_mixed = detector.bbox_head(graphed_feats, reg_feats=other)
        # a replay must read the new input; backbone and neck replay one graph each
        assert all(torch.equal(a, b) for a, b in zip(detector.extract_feat(new), eager_new,
                                                     strict=True))
        assert isinstance(detector.backbone.forward, GraphedCallable)
        assert len(detector.backbone.forward.graphs) == 1
    assert all(torch.equal(a, b) for a, b in zip(feats, graphed_feats, strict=True))
    for eager_out, graphed_out in ((eager, graphed), (eager_mixed, graphed_mixed)):
        assert all(torch.equal(a, b) for xs, ys in zip(eager_out, graphed_out, strict=True)
                   for a, b in zip(xs, ys, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(eager_dets[0], graphed_dets[0], strict=True))
