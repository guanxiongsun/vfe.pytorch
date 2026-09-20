"""STPN's training transforms: multi-scale resize, crops, max-size pad, AutoAugment."""

import numpy as np
import pytest

from vfe.datasets.pipelines import AutoAugment, RandomCrop, Resize, SeqMaxSizePad, SeqResize2

SCALES = [(480, 1333), (512, 1333), (544, 1333)]


def frame(h, w, bboxes=((10, 10, 50, 40),), labels=(3,)):
    return {
        "img": np.full((h, w, 3), 7, dtype=np.uint8),
        "img_fields": ["img"],
        "img_shape": (h, w, 3),
        "bbox_fields": ["gt_bboxes"],
        "gt_bboxes": np.array(bboxes, dtype=np.float32).reshape(-1, 4),
        "gt_labels": np.array(labels, dtype=np.int64),
    }


def test_multiscale_resize_draws_like_mmdet():
    resize = Resize(img_scale=SCALES, multiscale_mode="value", keep_ratio=True)
    np.random.seed(3)
    picks = [resize(frame(720, 1280))["scale_idx"] for _ in range(20)]
    np.random.seed(3)
    assert picks == [np.random.randint(len(SCALES)) for _ in range(20)]
    out = resize(frame(720, 1280))
    assert out["img"].shape[0] == SCALES[out["scale_idx"]][0]


def test_single_scale_resize_does_not_draw():
    np.random.seed(0)
    Resize(img_scale=(1000, 600), keep_ratio=True)(frame(720, 1280))
    after = np.random.rand()
    np.random.seed(0)
    assert after == np.random.rand()


def test_range_mode_is_not_ported():
    with pytest.raises(NotImplementedError):
        Resize(img_scale=SCALES, multiscale_mode="range")


def test_random_crop_bounds_boxes_and_labels():
    crop = RandomCrop(crop_size=(384, 600), crop_type="absolute_range", allow_negative_crop=True)
    np.random.seed(0)
    for _ in range(50):
        results = frame(500, 900, bboxes=((0, 0, 100, 100), (850, 450, 899, 499)), labels=(1, 2))
        out = crop(results)
        h, w = out["img"].shape[:2]
        assert 384 <= h <= 500 and 384 <= w <= 600
        assert out["img_shape"] == out["img"].shape
        boxes = out["gt_bboxes"]
        assert len(boxes) == len(out["gt_labels"])
        assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
        assert (boxes[:, 0::2] >= 0).all() and (boxes[:, 0::2] <= w).all()
        assert (boxes[:, 1::2] >= 0).all() and (boxes[:, 1::2] <= h).all()


def test_random_crop_rejects_empty_crops_unless_allowed():
    tiny_box = ((0, 0, 1, 1),)
    np.random.seed(1)
    strict = RandomCrop(crop_size=(10, 10), crop_type="absolute", allow_negative_crop=False)
    outcomes = [strict(frame(200, 200, bboxes=tiny_box, labels=(0,))) for _ in range(20)]
    assert any(out is None for out in outcomes)
    lenient = RandomCrop(crop_size=(10, 10), crop_type="absolute", allow_negative_crop=True)
    out = lenient(frame(200, 200, bboxes=((150, 150, 190, 190),), labels=(0,)))
    assert out is not None and len(out["gt_bboxes"]) == len(out["gt_labels"])


def test_seq_max_size_pad():
    frames = [frame(300, 400), frame(350, 380), frame(320, 420)]
    out = SeqMaxSizePad()(frames)
    for f in out:
        assert f["img"].shape == (350, 420, 3)
        assert f["pad_shape"] == (350, 420, 3)
    assert (out[0]["img"][300:] == 0).all() and (out[0]["img"][:300, :400] == 7).all()


def test_seq_resize2_draws_a_new_shared_scale():
    frames = [frame(400, 600), frame(400, 600)]
    for f in frames:
        f["scale"], f["scale_factor"] = (1000, 600), np.ones(4, dtype=np.float32)
    np.random.seed(5)
    out = SeqResize2(img_scale=SCALES, multiscale_mode="value", keep_ratio=True,
                     share_params=True)(frames)
    np.random.seed(5)
    expected = SCALES[np.random.randint(len(SCALES))]
    assert out[0]["scale"] == out[1]["scale"] == expected
    assert out[0]["img"].shape == out[1]["img"].shape


def test_auto_augment_picks_policies_with_numpy():
    augment = AutoAugment(policies=[[{"type": "SeqNormalize", "mean": [0, 0, 0],
                                      "std": [1, 1, 1], "to_rgb": False}],
                                    [{"type": "SeqPad", "size_divisor": 32}]])
    augment.transforms = [lambda r: "first", lambda r: "second"]
    np.random.seed(11)
    chosen = [augment([]) for _ in range(30)]
    np.random.seed(11)
    assert chosen == [("first", "second")[np.random.randint(2)] for _ in range(30)]


def test_auto_augment_validates_policies():
    for bad in ([], [[]], [[{"no_type": 1}]]):
        with pytest.raises(ValueError):
            AutoAugment(policies=bad)
