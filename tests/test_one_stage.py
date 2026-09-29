"""The one-stage machinery FCOS is built from: grid points, the distance
coder, focal and IoU losses, and the head's targets, loss and decoding."""

import math

import pytest
import torch

from vfe.core import DistancePointBBoxCoder, MlvlPointGenerator, bbox2distance, distance2bbox
from vfe.models.dense_heads import FCOSHead
from vfe.models.losses import FocalLoss, IoULoss


def test_point_generator_centres_strides_and_valid_flags():
    gen = MlvlPointGenerator([8, 16])
    pts = gen.grid_priors([(2, 3), (1, 2)], device="cpu")
    assert pts[0].tolist() == [[4, 4], [12, 4], [20, 4], [4, 12], [12, 12], [20, 12]]
    assert pts[1].tolist() == [[8, 8], [24, 8]]
    with_stride = gen.grid_priors([(2, 3), (1, 2)], device="cpu", with_stride=True)
    assert with_stride[1].tolist() == [[8, 8, 16, 16], [24, 8, 16, 16]]
    # A 9x9 padded image covers ceil(9/8) = 2 cells at stride 8 and 1 at 16.
    flags = gen.valid_flags([(2, 3), (1, 2)], pad_shape=(9, 9), device="cpu")
    assert flags[0].tolist() == [True, True, False, True, True, False]
    assert flags[1].tolist() == [True, False]
    assert gen.sparse_priors(torch.tensor([0, 4]), (2, 3), 0, device="cpu").tolist() == [
        [4, 4], [12, 12]]


def test_distance_coder_round_trips_and_clips():
    points = torch.tensor([[10.0, 10.0], [50.0, 30.0]])
    boxes = torch.tensor([[2.0, 4.0, 20.0, 18.0], [30.0, 10.0, 90.0, 70.0]])
    coder = DistancePointBBoxCoder()
    distances = coder.encode(points, boxes)
    assert distances.tolist() == [[8.0, 6.0, 10.0, 8.0], [20.0, 20.0, 40.0, 40.0]]
    assert torch.equal(coder.decode(points, distances), boxes)
    assert torch.equal(bbox2distance(points, boxes, max_dis=30.0, eps=0.1)[1],
                       torch.tensor([20.0, 20.0, 29.9, 29.9]))
    clipped = distance2bbox(points, distances, max_shape=(40, 60, 3))
    assert clipped.tolist() == [[2.0, 4.0, 20.0, 18.0], [30.0, 10.0, 60.0, 40.0]]
    assert torch.equal(DistancePointBBoxCoder(clip_border=False).decode(
        points, distances, max_shape=(40, 60)), boxes)


def test_focal_loss_matches_the_formula_and_background_labels():
    torch.manual_seed(0)
    pred = torch.randn(4, 3)
    labels = torch.tensor([0, 2, 3, 1])  # 3 == background: an all-zero row
    onehot = torch.tensor([[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 1, 0]], dtype=torch.float)
    p = pred.sigmoid()
    pt = (1 - p) * onehot + p * (1 - onehot)
    w = (0.25 * onehot + 0.75 * (1 - onehot)) * pt**2
    expected = (torch.nn.functional.binary_cross_entropy_with_logits(
        pred, onehot, reduction="none") * w).sum()
    loss = FocalLoss(reduction="sum")
    assert loss(pred, labels).item() == pytest.approx(expected.item(), rel=1e-6)
    assert loss(pred, onehot).item() == pytest.approx(expected.item(), rel=1e-6)
    assert FocalLoss()(pred, labels, avg_factor=8.0).item() == pytest.approx(
        expected.item() / 8, rel=1e-6)
    with pytest.raises(NotImplementedError):
        FocalLoss(use_sigmoid=False)


def test_iou_loss_modes_and_zero_weights():
    pred = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]])
    target = torch.tensor([[1.0, 0.0, 3.0, 2.0], [0.0, 0.0, 2.0, 2.0]])  # IoU 1/3 and 1
    assert IoULoss(reduction="none")(pred, target).tolist() == pytest.approx(
        [math.log(3), 0.0], abs=1e-6)
    assert IoULoss(mode="linear", reduction="none")(pred, target).tolist() == pytest.approx(
        [2 / 3, 0.0], abs=1e-6)
    assert IoULoss()(pred, target, weight=torch.zeros(2)).item() == 0.0
    weighted = IoULoss()(pred, target, weight=torch.tensor([1.0, 0.0]), avg_factor=2.0)
    assert weighted.item() == pytest.approx(math.log(3) / 2, rel=1e-6)


def make_head(**kwargs):
    return FCOSHead(
        num_classes=3, in_channels=8, feat_channels=8, stacked_convs=1, strides=[8, 16],
        regress_ranges=((-1, 64), (64, 1e8)),
        norm_cfg=dict(type="GN", num_groups=2, requires_grad=True),
        test_cfg=dict(nms_pre=100, min_bbox_size=0, score_thr=0.01,
                      nms=dict(type="nms", iou_threshold=0.5), max_per_img=10),
        **kwargs,
    )


def test_fcos_targets_assign_points_inside_the_box_to_the_level_of_its_size():
    head = make_head()
    points = head.prior_generator.grid_priors([(4, 4), (2, 2)], device="cpu")
    gt = torch.tensor([[0.0, 0.0, 20.0, 20.0]])  # 32x32 image, one small box
    labels, targets = head.get_targets(points, [gt], [torch.tensor([1])])
    # Stride 8: centres 4 and 12 are inside, 20 is on the edge (not inside).
    assert labels[0].tolist() == [1, 1, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    assert targets[0][0].tolist() == [4.0, 4.0, 16.0, 16.0]
    # Stride 16 is for sizes above 64: everything is background there.
    assert labels[1].tolist() == [3, 3, 3, 3]
    assert head.centerness_target(targets[0][[0]]).item() == pytest.approx(math.sqrt(1 / 16))


def test_fcos_loss_decoding_and_level_subsets():
    torch.manual_seed(0)
    head = make_head().eval()
    head.init_weights()
    feats = [torch.randn(2, 8, 4, 4), torch.randn(2, 8, 2, 2)]
    metas = [dict(img_shape=(32, 32, 3), scale_factor=torch.tensor([2.0, 2.0, 2.0, 2.0]))] * 2
    gt_bboxes = [torch.tensor([[0.0, 0.0, 20.0, 20.0]]), torch.tensor([[2.0, 2.0, 30.0, 30.0]])]
    gt_labels = [torch.tensor([1]), torch.tensor([0])]

    losses = head.forward_train(feats, metas, gt_bboxes, gt_labels)
    assert set(losses) == {"loss_cls", "loss_bbox", "loss_centerness"}
    assert all(torch.isfinite(v) for v in losses.values())
    sum(losses.values()).backward()
    assert head.conv_cls.weight.grad is not None

    with torch.no_grad():
        results = head.simple_test(feats, metas, rescale=True)
        assert len(results) == 2 and all(len(r) == 2 for r in results)
        det_bboxes, det_labels = results[0]
        assert det_bboxes.shape[1] == 5 and det_bboxes[:, :4].max() <= 16.0  # rescaled by 2
        # Only the stride-16 level: every detection reports level 1.
        partial = head.simple_test([feats[1]], metas, level_ids=[1], with_levels=True)
        det_bboxes, det_labels, det_levels = partial[0]
        assert det_bboxes.shape[0] == det_levels.shape[0] and (det_levels == 1).all()
        # The class score before centerness: never below the detection score.
        det_bboxes, det_labels, det_levels, det_cls = head.simple_test(
            feats, metas, with_levels=True, with_cls_scores=True)[0]
        assert det_cls.shape == det_levels.shape and len(det_cls)
        assert (det_cls + 1e-6 >= det_bboxes[:, 4]).all()
        with pytest.raises(ValueError):
            head(feats, level_ids=[0])  # two feature maps for one level
        # One synchronisation per image: the same detections as per level,
        # whether fewer scores than nms_pre pass the threshold or more.
        for score_thr, nms_pre in ((0.3, 1000), (0.0001, 7)):
            cfg = dict(head.test_cfg, score_thr=score_thr, nms_pre=nms_pre)
            outs = head(feats)
            per_level = head.get_bboxes(*outs, img_metas=metas, cfg=cfg, with_levels=True,
                                        with_cls_scores=True)
            head.one_sync_postprocess = True
            one_sync = head.get_bboxes(*outs, img_metas=metas, cfg=cfg, with_levels=True,
                                       with_cls_scores=True)
            head.one_sync_postprocess = False
            for a, b in zip(per_level, one_sync, strict=True):
                assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
            pre = head.get_bboxes(*outs, img_metas=metas, cfg=cfg, with_nms=False)
            head.one_sync_postprocess = True
            pre_one = head.get_bboxes(*outs, img_metas=metas, cfg=cfg, with_nms=False)
            head.one_sync_postprocess = False
            assert all(torch.equal(x, y) for a, b in zip(pre, pre_one, strict=True)
                       for x, y in zip(a, b, strict=True))
        # Separate regression inputs: boxes follow reg_feats, scores follow feats.
        other = [f.flip(-1) for f in feats]
        mixed = head(feats, reg_feats=other)
        plain, swapped = head(feats), head(other)
        assert all(torch.equal(a, b) for a, b in zip(mixed[0], plain[0], strict=True))
        assert all(torch.equal(a, b) for a, b in zip(mixed[2], plain[2], strict=True))
        assert all(torch.equal(a, b) for a, b in zip(mixed[1], swapped[1], strict=True))
        with pytest.raises(ValueError):
            head(feats, reg_feats=other[:1])
