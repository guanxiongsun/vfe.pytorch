"""EOVOD: the location and size priors, the key-pixel memory, the aggregator,
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


def test_pixel_memory_fixed_key_set_keeps_everything():
    memory = PixelMemory(num_levels=2)  # the paper's default: no caps, no updates
    assert memory.sample(0) is None and memory.update is False
    memory.write(0, torch.arange(8.0).view(8, 1))
    memory.write(0, torch.arange(5.0).view(5, 1))
    assert memory.sizes() == [13, 0]
    assert torch.equal(memory.sample(0), memory.banks[0])  # every key, in order
    memory.write(1, torch.zeros(0, 1))  # nothing to write
    assert memory.sample(1) is None
    memory.reset()
    assert memory.sizes() == [0, 0]


def test_pixel_memory_bank_caps_writes_and_replaces_at_random():
    torch.manual_seed(0)
    memory = PixelMemory(num_levels=2, update=True, capacity=10, num_keys=4, write_per_frame=6)
    memory.write(0, torch.arange(8.0).view(8, 1))  # 8 pixels -> 6 kept
    assert memory.sizes() == [6, 0]
    memory.write(0, torch.full((6, 1), 100.0))  # full: 4 old survive, 6 new
    assert memory.sizes() == [10, 0]
    assert (memory.banks[0] == 100).sum().item() == 6
    assert memory.sample(0).shape == (4, 1)
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
    kwargs.setdefault("aggregator", dict(num_heads=4, shared=True))
    return EOVOD(TINY_DETECTOR, **kwargs)


def meta(frame_id, shape=(96, 128, 3)):
    return dict(img_shape=shape, ori_shape=shape, pad_shape=shape,
                scale_factor=torch.tensor([1.0, 1.0, 1.0, 1.0]), frame_id=frame_id)


def test_eovod_rejects_unknown_options_and_two_stage_detectors():
    with pytest.raises(ValueError, match="location_prior"):
        tiny_eovod(location_prior=dict(threshold=0.5))
    with pytest.raises(ValueError, match="size_prior"):
        tiny_eovod(size_prior=dict(interval=-1))
    with pytest.raises(ValueError, match="validate_on"):
        tiny_eovod(location_prior=dict(validate_on="cls"))
    mamba_cfg = Config.fromfile(REPO_ROOT / "configs/vid/mamba/mamba_r101_dc5_3x.py").model
    with pytest.raises(TypeError):
        EOVOD(mamba_cfg["detector"])


def test_eovod_trains_with_ground_truth_priors_and_reference_keys():
    model = tiny_eovod(location_prior=dict(train_jitter=0.2),
                       aggregator=dict(num_heads=4, shared=False, query_chunk=7))
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
    assert len(model.aggregators) == 5  # one per level
    for agg in model.aggregators:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in agg.parameters())
    # Support frames without boxes still exercise the aggregator (random keys).
    model.zero_grad()
    sum(model.forward_train(img, [meta(3)], gt, [torch.tensor([2, 7])], refs,
                            [[meta(1), meta(5)]], ref_gt_bboxes=[torch.zeros(0, 5)]).values()
        ).backward()
    assert model.aggregators[0].fc.weight.grad is not None
    with pytest.raises(ValueError):
        model.forward_train(torch.randn(2, 3, 96, 128), [meta(0)] * 2, gt * 2, None, refs, None)


def test_training_prior_options_miss_move_and_add_boxes():
    gt = torch.tensor([[10.0, 10.0, 60.0, 50.0], [70.0, 20.0, 120.0, 90.0]])
    shape = (96, 128, 3)
    plain = tiny_eovod()
    assert torch.equal(plain._training_prior(gt, shape), scale_boxes(gt, 0.8))  # the paper's
    only_distractors = tiny_eovod(location_prior=dict(train_drop_prob=1.0, train_distractors=3))
    counts = set()
    for seed in range(40):
        torch.manual_seed(seed)
        boxes = only_distractors._training_prior(gt, shape)
        counts.add(len(boxes))
        # Inside the image, and each the size of a ground-truth box (times r).
        centre = (boxes[:, :2] + boxes[:, 2:]) / 2
        half = (boxes[:, 2:] - boxes[:, :2]) / 0.8 / 2
        assert ((centre - half) >= -1e-4).all() and (centre[:, 0] + half[:, 0] <= 128 + 1e-4).all()
        assert (centre[:, 1] + half[:, 1] <= 96 + 1e-4).all()
        sizes = {tuple(s) for s in (2 * half).round().tolist()}
        assert sizes <= {(50.0, 40.0), (50.0, 70.0)}
    assert counts == {0, 1, 2, 3}
    for seed in range(20):  # no ground truth: random sizes of 10-50% of the image per side
        torch.manual_seed(seed)
        boxes = only_distractors._training_prior(torch.zeros(0, 4), shape)
        frac = (boxes[:, 2:] - boxes[:, :2]) / 0.8 / torch.tensor([128.0, 96.0])
        assert ((frac >= 0.1 - 1e-4) & (frac <= 0.5 + 1e-4)).all()
    with pytest.raises(ValueError, match="train_plain_prob"):
        tiny_eovod(location_prior=dict(train_plain_prob=1.5))
    with pytest.raises(ValueError, match="train_distractors"):
        tiny_eovod(location_prior=dict(train_distractors=-1))


@pytest.mark.parametrize("location_prior", [
    dict(train_plain_prob=1.0),  # a step without a prior
    dict(train_drop_prob=1.0),  # a prior whose boxes were all dropped
])
def test_steps_without_aggregation_keep_the_aggregators_in_the_graph(location_prior):
    """When nothing is aggregated the aggregators get zero gradients rather
    than none, so DDP sees every parameter used."""
    model = tiny_eovod(location_prior=location_prior)
    model.train()
    gt = [torch.tensor([[10.0, 10.0, 60.0, 50.0]])]
    losses = model.forward_train(torch.randn(1, 3, 96, 128), [meta(3)], gt, [torch.tensor([2])],
                                 torch.randn(1, 2, 3, 96, 128), [[meta(1), meta(5)]],
                                 ref_gt_bboxes=[torch.tensor([[0.0, 12.0, 8.0, 58.0, 52.0]])])
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    for p in model.aggregators.parameters():
        assert p.grad is not None and not p.grad.any()
    assert model.detector.bbox_head.conv_cls.weight.grad.abs().sum() > 0


def test_query_chunking_does_not_change_the_result():
    model = tiny_eovod().eval()
    feats = [torch.randn(1, 32, 12, 16)]
    masks = [torch.zeros(12, 16, dtype=torch.bool)]
    masks[0][2:9, 3:14] = True
    keys = [torch.randn(40, 32)]
    with torch.no_grad():
        model.query_chunk = None
        whole = model._enhance(feats, masks, keys)[0]
        model.query_chunk = 5
        chunked = model._enhance(feats, masks, keys)[0]
    assert torch.allclose(whole, chunked, atol=1e-6)
    assert torch.equal(whole[0][:, ~masks[0]], feats[0][0][:, ~masks[0]])  # background untouched


def _reference_enhance(model, feats, masks, keys):
    """The per-level boolean-mask version _enhance replaced (several syncs per level)."""
    out = []
    for level, x in enumerate(feats):
        mask = None if masks is None else masks[level]
        key = keys[level]
        if mask is None or key is None or len(key) == 0 or not bool(mask.any()):
            out.append(x)
            continue
        queries = x[0][:, mask].t()
        chunks = queries.split(model.query_chunk) if model.query_chunk else (queries,)
        enhanced = torch.cat([model._aggregator(level)(q, key) for q in chunks], dim=0)
        x = x.clone()
        x[0][:, mask] = enhanced.t()
        out.append(x)
    return out


def test_enhance_with_one_sync_matches_the_per_level_masks():
    model = tiny_eovod(aggregator=dict(num_heads=4, shared=False, query_chunk=5))
    torch.manual_seed(1)
    shapes = [(12, 16), (6, 8), (3, 4), (2, 2), (1, 1)]
    masks = [torch.rand(h, w) > 0.6 for h, w in shapes]
    masks[2][:] = False  # a level whose mask is empty
    keys = [torch.randn(9, 32), None, torch.randn(4, 32), torch.randn(3, 32), torch.randn(0, 32)]
    for grad in (False, True):
        feats = [torch.randn(1, 32, h, w, requires_grad=grad) for h, w in shapes]
        new = model._enhance(feats, masks, keys)
        ref = _reference_enhance(model, feats, masks, keys)
        assert all(torch.equal(a, b) for a, b in zip(new, ref, strict=True))
        if grad:
            g_new = torch.autograd.grad(sum(o.square().sum() for o in new), feats)
            g_ref = torch.autograd.grad(sum(o.square().sum() for o in ref), feats)
            assert all(torch.equal(a, b) for a, b in zip(g_new, g_ref, strict=True))
    assert model._enhance(feats, None, keys) == feats  # no prior: every level as it was


def test_size_prior_rule_follows_the_paper_with_the_superset_as_an_option():
    det_bboxes = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.9], [0.0, 0.0, 50.0, 50.0, 0.7],
                               [0.0, 0.0, 5.0, 5.0, 0.2]])
    det_levels = torch.tensor([3, 1, 0])  # the low-scoring level-0 box is not validated
    model = tiny_eovod(size_prior=dict(interval=7))
    model._after_frame([], det_bboxes, det_levels, full=True)
    assert model._active_levels == [1, 3] and model._frames_until_full == 7
    assert model._levels_to_run() == ([1, 3], False) and model._frames_until_full == 6
    assert torch.equal(model._prev_boxes, det_bboxes[:2, :4])

    superset = tiny_eovod(size_prior=dict(interval=0, keep_higher_levels=True))
    superset._after_frame([], det_bboxes, det_levels, full=True)
    assert superset._active_levels == [1, 2, 3, 4]
    assert superset._levels_to_run() == ([0, 1, 2, 3, 4], True)  # T = 0: every frame is full

    neighbours = tiny_eovod(size_prior=dict(interval=7, margin=1))
    neighbours._after_frame([], det_bboxes, det_levels, full=True)
    assert neighbours._active_levels == [0, 1, 2, 3, 4]  # 1 -> 0..2, 3 -> 2..4
    one = tiny_eovod(size_prior=dict(interval=7, margin=1))
    one._after_frame([], det_bboxes[:1], det_levels[:1], full=True)
    assert one._active_levels == [2, 3, 4]
    floor = tiny_eovod(size_prior=dict(interval=7, margin=1, margin_min_level=1))
    floor._after_frame([], det_bboxes, det_levels, full=True)
    assert floor._active_levels == [1, 2, 3, 4]  # level 0 is never added, only kept
    kept = tiny_eovod(size_prior=dict(interval=7, margin=1, margin_min_level=1))
    kept._after_frame([], det_bboxes[:1].repeat(2, 1), torch.tensor([3, 0]), full=True)
    assert kept._active_levels == [0, 1, 2, 3, 4]  # a recorded level 0 still runs, as do its upper neighbour
    only_p3 = tiny_eovod(size_prior=dict(interval=7, margin=1, margin_min_level=1))
    only_p3._after_frame([], det_bboxes[:1], torch.tensor([2]), full=True)
    assert only_p3._active_levels == [1, 2, 3]
    down = tiny_eovod(size_prior=dict(interval=7, margin=1, margin_up=0))
    down._after_frame([], det_bboxes, det_levels, full=True)
    assert down._active_levels == [0, 1, 2, 3]  # 1 -> 0..1, 3 -> 2..3
    with pytest.raises(ValueError, match="margin"):
        tiny_eovod(size_prior=dict(margin=-1))
    with pytest.raises(ValueError, match="margin"):
        tiny_eovod(size_prior=dict(margin=1, margin_up=-1))

    nothing = tiny_eovod(size_prior=dict(interval=7))
    nothing._after_frame([], det_bboxes[2:], det_levels[2:], full=True)
    assert nothing._active_levels is None  # no validated box: keep every level


def test_reference_keys_and_the_first_frame_prior():
    refs = torch.randn(3, 3, 96, 128)
    metas = [meta(0), meta(4), meta(8)]
    paper = tiny_eovod(location_prior=dict(score_thr=0.0), ref_chunk_size=2).eval()
    with torch.no_grad():
        paper._gather_reference_keys(refs, metas)
    assert all(size > 0 for size in paper.memory.sizes())
    assert paper._prev_boxes is None  # the paper skips aggregation on the first frame

    bootstrapped = tiny_eovod(location_prior=dict(score_thr=0.0, bootstrap_first_frame=True)).eval()
    with torch.no_grad():
        bootstrapped._gather_reference_keys(refs, metas)
    assert bootstrapped._prev_boxes is not None and bootstrapped._prev_boxes.shape[1] == 4


def test_classification_only_aggregation_keeps_the_boxes_plain():
    """branches='cls': the regression tower reads the original maps, so the
    aggregation moves scores but never the boxes of the same candidates."""
    model = tiny_eovod(location_prior=dict(score_thr=0.0), size_prior=None,
                       aggregator=dict(num_heads=4, branches="cls")).eval()
    head = model.detector.bbox_head
    seen = {}
    original = head.simple_test

    def spy(feats, img_metas, **kwargs):
        seen["feats"], seen["reg"] = feats, kwargs.get("reg_feats")
        return original(feats, img_metas, **kwargs)

    head.simple_test = spy
    refs = torch.randn(1, 2, 3, 96, 128)
    with torch.no_grad():
        model.simple_test(torch.randn(1, 3, 96, 128), [meta(0)], ref_img=[refs],
                          ref_img_metas=[[[meta(0), meta(4)]]])
        model.simple_test(torch.randn(1, 3, 96, 128), [meta(1)])
        assert seen["reg"] is not None and any(
            not torch.equal(f, r) for f, r in zip(seen["feats"], seen["reg"], strict=True))
        boxes_mixed = head(seen["feats"], reg_feats=seen["reg"])[1]
        boxes_plain = head(seen["reg"])[1]
    assert all(torch.equal(a, b) for a, b in zip(boxes_mixed, boxes_plain, strict=True))
    # Training passes the plain maps to the regression tower too.
    train = tiny_eovod(aggregator=dict(num_heads=4, branches="cls"))
    train.train()
    losses = train.forward_train(torch.randn(1, 3, 96, 128), [meta(3)],
                                 [torch.tensor([[10.0, 10.0, 60.0, 50.0]])], [torch.tensor([2])],
                                 torch.randn(1, 2, 3, 96, 128), [[meta(1), meta(5)]],
                                 ref_gt_bboxes=[torch.tensor([[0.0, 12.0, 8.0, 58.0, 52.0]])])
    sum(losses.values()).backward()
    assert train.aggregators[0].fc.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="branches"):
        tiny_eovod(aggregator=dict(branches="reg"))


def test_validation_on_the_detection_or_the_class_score():
    det_bboxes = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.3], [20.0, 20.0, 40.0, 40.0, 0.6]])
    cls_scores = torch.tensor([0.55, 0.7])  # before centerness
    levels = torch.tensor([0, 1])
    for validate_on, expected in (("score", 1), ("cls_score", 2)):
        model = tiny_eovod(location_prior=dict(validate_on=validate_on), size_prior=None)
        model._after_frame([], det_bboxes, levels, full=True,
                           valid_scores=model._validation_scores(det_bboxes, cls_scores))
        assert len(model._prev_boxes) == expected


def test_eovod_stateful_inference_carries_priors_across_frames():
    # score_thr 0 validates every detection, so both priors engage even with
    # random weights.
    model = tiny_eovod(location_prior=dict(score_thr=0.0), size_prior=dict(interval=2),
                       ref_chunk_size=2).eval()
    refs = torch.randn(1, 3, 3, 96, 128)  # the test pipeline's [Tensor(1, R, C, H, W)]
    ref_metas = [[[meta(0), meta(4), meta(8)]]]
    with torch.no_grad():
        out0 = model.simple_test(torch.randn(1, 3, 96, 128), [meta(0)], ref_img=[refs],
                                 ref_img_metas=ref_metas, rescale=True)
    assert len(out0) == 1 and len(out0[0]) == 30
    assert all(arr.shape[1] == 5 for arr in out0[0])
    key_sizes = model.memory.sizes()
    assert sum(key_sizes) > 0
    assert model._prev_boxes is not None and model._prev_boxes.shape[1] == 4
    # Frame 0 was a full frame: the size prior recorded the validated levels.
    assert model._active_levels == sorted(set(model._active_levels))
    assert model._frames_until_full == 2

    seen_levels = []
    original = model.detector.bbox_head.simple_test

    def spy(feats, img_metas, rescale=False, level_ids=None, **kwargs):
        seen_levels.append(list(level_ids) if level_ids is not None else list(range(5)))
        return original(feats, img_metas, rescale=rescale, level_ids=level_ids, **kwargs)

    model.detector.bbox_head.simple_test = spy
    with torch.no_grad():
        for frame_id in (1, 2, 3):
            out = model.simple_test(torch.randn(1, 3, 96, 128), [meta(frame_id)], rescale=True)
            assert len(out[0]) == 30
    # T = 2: frames 1 and 2 ran the restricted levels; frame 3 was full again.
    assert seen_levels[0] == seen_levels[1] and len(seen_levels[0]) <= 5
    assert seen_levels[2] == list(range(5))
    assert model.memory.sizes() == key_sizes  # the paper's key set is fixed for the video

    # A new video resets everything.
    with torch.no_grad():
        model.simple_test(torch.randn(1, 3, 96, 128), [meta(0)], rescale=True)
    assert model.memory.sizes() == [0] * 5  # no reference frames given this time
    with pytest.raises(KeyError):
        model.simple_test(torch.randn(1, 3, 96, 128), [dict(meta(0), frame_id=-1)])


def test_eovod_memory_can_update_from_every_frame():
    model = tiny_eovod(location_prior=dict(score_thr=0.0), size_prior=None,
                       memory=dict(update=True, capacity=64, num_keys=16, write_per_frame=8)).eval()
    with torch.no_grad():
        model.simple_test(torch.randn(1, 3, 96, 128), [meta(0)])
        after_first = model.memory.sizes()
        model.simple_test(torch.randn(1, 3, 96, 128), [meta(1)])
    assert sum(after_first) > 0 and model.memory.sizes() >= after_first
    assert all(size <= 64 for size in model.memory.sizes())


def test_eovod_runs_through_the_trainer_with_accumulation():
    """Two micro-batches through ``train_step`` with per-virtual-rank streams:
    the model's loss dict, the random keys and the training-prior options all
    fit the loop."""
    from vfe.engine.trainer import RngStreams, train_step

    model = tiny_eovod(location_prior=dict(train_jitter=0.1, train_plain_prob=0.5,
                                           train_drop_prob=0.3, train_distractors=2))
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
    assert model.size_prior_interval == 7 and model.level_margin == 1 and model.margin_up == 1
    assert model.box_ratio == 0.8 and model.score_thr == 0.3 and model.validate_on == "score"
    assert model.memory.update is False and model.memory.num_keys == 4096
    assert model.bootstrap_first_frame is False and model.aggregate_reg is False
    assert (model.train_plain_prob, model.train_drop_prob, model.train_jitter,
            model.train_distractors) == (0.25, 0.3, 0.1, 2)
