"""EOVOD: the location and size priors, the pixel memory, the aggregator,
and the detector's training and stateful inference on a tiny model."""

from pathlib import Path

import pytest
import torch

from vfe.config import Config
from vfe.models.builder import build_model
from vfe.models.vid.eovod import (
    EOVOD,
    PixelAggregator,
    PixelMemory,
    boxes_to_level_masks,
    scale_boxes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scale_boxes_about_the_centre():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0]])
    assert scale_boxes(boxes, 0.8).tolist() == [[1.0, 2.0, 9.0, 18.0]]
    assert scale_boxes(boxes, 1.0) is boxes


def test_boxes_to_level_masks_marks_cell_centres_inside_and_never_loses_a_box():
    boxes = torch.tensor([[8.0, 8.0, 40.0, 24.0]])
    masks = boxes_to_level_masks(boxes, [(4, 8), (2, 4)], [8, 16])
    # Stride 8: centres x in {12, 20, 28, 36}, y in {12, 20}.
    assert masks[0].nonzero().tolist() == [[1, 1], [1, 2], [1, 3], [1, 4],
                                           [2, 1], [2, 2], [2, 3], [2, 4]]
    # Stride 16: centres x in {8, 24, 40}, y in {8, 24}.
    assert masks[1].sum().item() == 6
    tiny = boxes_to_level_masks(torch.tensor([[30.0, 30.0, 31.0, 31.0]]), [(2, 4)], [16])[0]
    assert tiny.nonzero().tolist() == [[1, 1]]  # no centre inside: the cell holding the box
    empty = boxes_to_level_masks(torch.zeros(0, 4), [(2, 4)], [16])[0]
    assert empty.shape == (2, 4) and not empty.any()


def test_pixel_memory_caps_writes_and_replaces_at_random():
    torch.manual_seed(0)
    memory = PixelMemory(num_levels=2, capacity=10, num_keys=4, write_per_frame=6)
    assert memory.sample(0) is None
    memory.write(0, torch.arange(8.0).view(8, 1))  # 8 pixels -> 6 kept
    assert memory.sizes() == [6, 0]
    memory.write(0, torch.full((6, 1), 100.0))  # full: 4 old survive, 6 new
    assert memory.sizes() == [10, 0]
    assert (memory.banks[0] == 100).sum().item() == 6
    assert memory.sample(0).shape == (4, 1)
    memory.write(1, torch.zeros(0, 1))  # nothing to write
    assert memory.sample(1) is None
    memory.reset()
    assert memory.sizes() == [0, 0]
    with pytest.raises(ValueError):
        PixelMemory(1, capacity=4, write_per_frame=8)


def test_pixel_aggregator_is_residual_attention():
    torch.manual_seed(0)
    agg = PixelAggregator(channels=32, num_heads=4)
    x, ref = torch.randn(5, 32), torch.randn(7, 32)
    assert agg(x, ref).shape == (5, 32)
    torch.nn.init.zeros_(agg.fc.weight)
    torch.nn.init.zeros_(agg.fc.bias)
    assert torch.equal(agg(x, ref), x)  # with the output projection zeroed, identity
    with pytest.raises(ValueError):
        PixelAggregator(channels=30, num_heads=4)


TINY_DETECTOR = dict(
    type="FCOS",
    backbone=dict(type="ResNet", depth=18, num_stages=4, out_indices=(0, 1, 2, 3),
                  frozen_stages=1, norm_cfg=dict(type="BN", requires_grad=False),
                  norm_eval=True, style="pytorch"),
    neck=dict(type="FPN", in_channels=[64, 128, 256, 512], out_channels=32, start_level=1,
              add_extra_convs="on_output", num_outs=5, relu_before_extra_convs=True),
    bbox_head=dict(type="FCOSHead", num_classes=30, in_channels=32, stacked_convs=1,
                   feat_channels=32, strides=[8, 16, 32, 64, 128],
                   norm_cfg=dict(type="GN", num_groups=4, requires_grad=True)),
    train_cfg=dict(allowed_border=-1, pos_weight=-1, debug=False),
    test_cfg=dict(nms_pre=200, min_bbox_size=0, score_thr=0.01,
                  nms=dict(type="nms", iou_threshold=0.5), max_per_img=20),
)


def tiny_eovod(**kwargs):
    torch.manual_seed(0)
    return EOVOD(TINY_DETECTOR, aggregator=dict(num_heads=4, shared=True), **kwargs)


def meta(frame_id, shape=(96, 128, 3)):
    return dict(img_shape=shape, ori_shape=shape, pad_shape=shape,
                scale_factor=torch.tensor([1.0, 1.0, 1.0, 1.0]), frame_id=frame_id)


def test_eovod_rejects_unknown_options_and_two_stage_detectors():
    with pytest.raises(ValueError, match="location_prior"):
        tiny_eovod(location_prior=dict(threshold=0.5))
    mamba_cfg = Config.fromfile(REPO_ROOT / "configs/vid/mamba/mamba_r101_dc5_3x.py").model
    with pytest.raises(TypeError):
        EOVOD(mamba_cfg["detector"])


def test_eovod_trains_with_ground_truth_priors_and_reference_keys():
    model = tiny_eovod(location_prior=dict(train_jitter=0.2))
    model.train()
    img = torch.randn(1, 3, 96, 128)
    refs = torch.randn(1, 2, 3, 96, 128)
    gt = [torch.tensor([[10.0, 10.0, 60.0, 50.0], [70.0, 20.0, 120.0, 90.0]])]
    ref_gt = [torch.tensor([[0.0, 12.0, 8.0, 58.0, 52.0], [1.0, 68.0, 22.0, 118.0, 88.0]])]
    losses = model.forward_train(img, [meta(3)], gt, [torch.tensor([2, 7])], refs,
                                 [[meta(1), meta(5)]], ref_gt_bboxes=ref_gt,
                                 ref_gt_labels=[torch.tensor([[0, 2], [1, 7]])])
    assert set(losses) == {"loss_cls", "loss_bbox", "loss_centerness"}
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    agg = model.aggregators[0]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in agg.parameters())
    # Reference frames without boxes still exercise the aggregator (random keys).
    model.zero_grad()
    sum(model.forward_train(img, [meta(3)], gt, [torch.tensor([2, 7])], refs,
                            [[meta(1), meta(5)]], ref_gt_bboxes=[torch.zeros(0, 5)]).values()
        ).backward()
    assert agg.fc.weight.grad is not None
    with pytest.raises(ValueError):
        model.forward_train(torch.randn(2, 3, 96, 128), [meta(0)] * 2, gt * 2, None, refs, None)


def test_eovod_stateful_inference_carries_priors_and_memory_across_frames():
    # score_thr 0 validates every detection, so both priors and the memory
    # engage even with random weights.
    model = tiny_eovod(location_prior=dict(score_thr=0.0), size_prior=dict(interval=3),
                       memory=dict(capacity=256, num_keys=64, write_per_frame=32),
                       ref_chunk_size=2).eval()
    refs = torch.randn(1, 3, 3, 96, 128)  # the test pipeline's [Tensor(1, R, C, H, W)]
    ref_metas = [[[meta(0), meta(4), meta(8)]]]
    with torch.no_grad():
        out0 = model.simple_test(torch.randn(1, 3, 96, 128), [meta(0)], ref_img=[refs],
                                 ref_img_metas=ref_metas, rescale=True)
    assert len(out0) == 1 and len(out0[0]) == 30
    assert all(arr.shape[1] == 5 for arr in out0[0])
    sizes_after_seed = model.memory.sizes()
    assert sum(sizes_after_seed) > 0
    assert model._prev_boxes is not None and model._prev_boxes.shape[1] == 4
    # Frame 0 was a full frame: the size prior chose a suffix of the levels.
    assert model._active_levels is not None
    assert model._active_levels == list(range(model._active_levels[0], 5))
    assert model._frames_until_full == 2

    seen_levels = []
    original = model.detector.bbox_head.simple_test

    def spy(feats, img_metas, rescale=False, level_ids=None, with_levels=False):
        seen_levels.append(list(level_ids) if level_ids is not None else list(range(5)))
        return original(feats, img_metas, rescale=rescale, level_ids=level_ids,
                        with_levels=with_levels)

    model.detector.bbox_head.simple_test = spy
    with torch.no_grad():
        for frame_id in (1, 2, 3):
            out = model.simple_test(torch.randn(1, 3, 96, 128), [meta(frame_id)], rescale=True)
            assert len(out[0]) == 30
    # Frames 1 and 2 ran the restricted levels; frame 3 was full again.
    assert seen_levels[0] == seen_levels[1] and seen_levels[2] == list(range(5))
    assert model.memory.sizes() >= sizes_after_seed

    # A new video resets everything.
    with torch.no_grad():
        model.simple_test(torch.randn(1, 3, 96, 128), [meta(0)], rescale=True)
    assert model._frames_until_full in (0, 2)  # full frame, then possibly re-armed
    with pytest.raises(KeyError):
        model.simple_test(torch.randn(1, 3, 96, 128), [dict(meta(0), frame_id=-1)])


def test_eovod_runs_through_the_trainer_with_accumulation():
    """Two micro-batches through ``train_step`` with per-virtual-rank streams:
    the model's loss dict, the random keys and the jitter all fit the loop."""
    from vfe.engine.trainer import RngStreams, train_step

    model = tiny_eovod()
    model.train()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    def micro_batch(seed):
        g = torch.Generator().manual_seed(seed)
        return dict(
            img=torch.randn(1, 3, 96, 128, generator=g), img_metas=[meta(seed)],
            gt_bboxes=[torch.tensor([[10.0, 10.0, 60.0, 50.0]])], gt_labels=[torch.tensor([2])],
            gt_instance_ids=[torch.tensor([1])],
            ref_img=torch.randn(1, 2, 3, 96, 128, generator=g),
            ref_img_metas=[[meta(seed + 1), meta(seed + 2)]],
            ref_gt_bboxes=[torch.tensor([[0.0, 12.0, 8.0, 58.0, 52.0]])],
            ref_gt_labels=[torch.tensor([[0, 2]])], ref_gt_instance_ids=[torch.tensor([[0, 1]])],
        )

    torch.manual_seed(0)
    streams = RngStreams(2, torch.device("cpu"))
    log_vars = train_step(model, [micro_batch(1), micro_batch(2)], optimizer,
                          torch.device("cpu"), rng=streams)
    assert {"loss", "loss_cls", "loss_bbox", "loss_centerness"} <= set(log_vars)
    assert all(torch.isfinite(torch.tensor(v)) for v in log_vars.values())
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    assert all(p.grad is not None for p in model.aggregators.parameters())


def test_eovod_without_a_prior_detects_plainly():
    model = tiny_eovod(location_prior=dict(score_thr=1.1), size_prior=None).eval()
    with torch.no_grad():
        model.simple_test(torch.randn(1, 3, 96, 128), [meta(0)], rescale=False)
        model.simple_test(torch.randn(1, 3, 96, 128), [meta(1)], rescale=False)
    assert model.memory.sizes() == [0] * 5 and len(model._prev_boxes) == 0
    assert model._levels_to_run() == (list(range(5)), True)


@pytest.mark.parametrize("config, total, trainable", [
    # FCOS R-50-FPN is 32.18M in mmdet; the shared aggregator adds 263,168.
    ("configs/vid/eovod/eovod_fcos_r50_fpn_3x.py", 32_443_240, 32_167_720),
    ("configs/vid/eovod/eovod_fcos_r101_fpn_3x.py", 51_435_368, 51_107_624),
    ("configs/vid/eovod/eovod_fcos_r101_fpn_9x.py", 51_435_368, 51_107_624),
])
def test_eovod_configs_build(config, total, trainable):
    model = build_model(Config.fromfile(REPO_ROOT / config).model)
    assert type(model).__name__ == "EOVOD"
    assert sum(p.numel() for p in model.parameters()) == total
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == trainable
    assert model.size_prior_interval == 7 and model.box_ratio == 0.8
