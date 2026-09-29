"""YOLOX in ``vfe``: the head's EOVOD extensions, the detector's multi-scale
step, the YOLOX learning-rate policy and training hooks, the mixed-image
transforms, the checkpoint converter, and the whole recipe through the
trainer on a tiny model. Model parity with mmdet is
``tools/checks/parity_yolox.py``; the training pipeline's is
``tools/checks/parity_yolox_pipeline.py``."""

import copy
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vfe.config import Config
from vfe.models.builder import build_detector, build_model
from vfe.models.vid.eovod import EOVOD

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "checks"))

TINY_YOLOX = dict(
    type="YOLOX",
    input_size=(128, 160),
    random_size_range=(3, 5),
    random_size_interval=2,
    backbone=dict(type="CSPDarknet", deepen_factor=0.33, widen_factor=0.125),
    neck=dict(type="YOLOXPAFPN", in_channels=[32, 64, 128], out_channels=32, num_csp_blocks=1),
    bbox_head=dict(type="YOLOXHead", num_classes=30, in_channels=32, feat_channels=32),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    test_cfg=dict(score_thr=0.0, nms=dict(type="nms", iou_threshold=0.65), max_per_img=20),
)


def meta(frame_id=0, shape=(128, 160, 3), scale=1.0):
    return dict(img_shape=shape, ori_shape=shape, pad_shape=shape, frame_id=frame_id,
                scale_factor=np.full(4, scale, dtype=np.float32))


def tiny_yolox():
    torch.manual_seed(0)
    model = build_detector(copy.deepcopy(TINY_YOLOX))
    model.init_weights()
    return model


def gts():
    return ([torch.tensor([[10.0, 12.0, 70.0, 90.0], [80.0, 20.0, 150.0, 60.0]]),
             torch.tensor([[30.0, 30.0, 100.0, 110.0]])],
            [torch.tensor([3, 7]), torch.tensor([12])])


# ---- the head's EOVOD extensions ---------------------------------------------------------

def test_head_runs_a_subset_of_levels_and_reports_levels_and_class_scores():
    model = tiny_yolox().eval()
    feats = model.extract_feat(torch.randn(1, 3, 128, 160))
    head = model.bbox_head
    with torch.no_grad():
        full = head.simple_test(feats, [meta()], with_levels=True, with_cls_scores=True)[0]
        part = head.simple_test([feats[1], feats[2]], [meta()], level_ids=[1, 2],
                                with_levels=True, with_cls_scores=True)[0]
    det_bboxes, det_labels, det_levels, det_cls = full
    assert det_bboxes.shape[1] == 5 and len(det_labels) == len(det_levels) == len(det_cls)
    assert set(det_levels.tolist()) <= {0, 1, 2}
    assert (det_cls >= det_bboxes[:, 4] - 1e-6).all()  # the score is class x objectness
    assert set(part[2].tolist()) <= {1, 2}
    with pytest.raises(ValueError, match="levels"):
        head(feats[:2])  # two maps for three levels
    # reg_feats feed the box and objectness branch only.
    other = [f + 1.0 for f in feats]
    with torch.no_grad():
        cls_a, reg_a, obj_a = head(feats)
        cls_b, reg_b, obj_b = head(feats, reg_feats=other)
    assert all(torch.equal(a, b) for a, b in zip(cls_a, cls_b, strict=True))
    assert not torch.equal(reg_a[0], reg_b[0]) and not torch.equal(obj_a[0], obj_b[0])


def test_an_image_without_detections_gives_five_column_boxes():
    model = tiny_yolox().eval()
    model.bbox_head.test_cfg = dict(score_thr=1.1, nms=dict(type="nms", iou_threshold=0.65))
    with torch.no_grad():
        det_bboxes, det_labels = model.bbox_head.simple_test(
            model.extract_feat(torch.randn(1, 3, 128, 160)), [meta()])[0]
    assert det_bboxes.shape == (0, 5) and det_labels.shape == (0,)


def test_training_losses_and_the_l1_term():
    model = tiny_yolox().train()
    gt_bboxes, gt_labels = gts()
    losses = model.forward_train(torch.randn(2, 3, 128, 160), [meta()] * 2, gt_bboxes, gt_labels)
    assert set(losses) == {"loss_cls", "loss_bbox", "loss_obj"}
    model.bbox_head.use_l1 = True
    losses = model.forward_train(torch.randn(2, 3, 128, 160), [meta()] * 2, *gts())
    assert set(losses) == {"loss_cls", "loss_bbox", "loss_obj", "loss_l1"}
    sum(losses.values()).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_multi_scale_steps_resize_the_batch_and_its_boxes():
    model = tiny_yolox().train()
    gt_bboxes, gt_labels = gts()
    model._input_size = (64, 96)
    img, boxes = model._preprocess(torch.randn(2, 3, 128, 160), [b.clone() for b in gt_bboxes])
    assert img.shape[-2:] == (64, 96)
    assert torch.allclose(boxes[0], gt_bboxes[0] * torch.tensor([0.6, 0.5, 0.6, 0.5]))
    model._input_size = model._default_input_size
    sizes = []
    for _ in range(4):
        model.forward_train(torch.randn(1, 3, 128, 160), [meta()], *[x[:1] for x in gts()])
        sizes.append(model._input_size)
    # A new size is drawn after every second step (random_size_interval=2).
    assert all(h % 32 == 0 and 96 <= h <= 160 for h, _ in sizes)


# ---- EOVOD on YOLOX -------------------------------------------------------------------

def tiny_eovod(**kwargs):
    torch.manual_seed(0)
    kwargs.setdefault("aggregator", dict(num_heads=4, shared=True, branches="cls"))
    model = EOVOD(copy.deepcopy(TINY_YOLOX), **kwargs)
    model.init_weights()
    return model


def test_eovod_trains_yolox_on_still_images_and_keeps_the_aggregators_in_the_graph():
    model = tiny_eovod().train()
    gt_bboxes, gt_labels = gts()
    losses = model.forward_train(torch.randn(2, 3, 128, 160), [meta()] * 2, gt_bboxes, gt_labels,
                                 ref_img=None, ref_img_metas=None)
    sum(losses.values()).backward()
    assert all(p.grad is not None for p in model.aggregators.parameters())
    assert all(p.grad is not None for p in model.detector.parameters() if p.requires_grad)


def test_eovod_with_yolox_runs_both_priors_on_a_video():
    model = tiny_eovod(location_prior=dict(score_thr=0.0), size_prior=dict(interval=2)).eval()
    refs = torch.randn(1, 3, 3, 128, 160)
    with torch.no_grad():
        out = model.simple_test(torch.randn(1, 3, 128, 160), [meta(0)], ref_img=[refs],
                                ref_img_metas=[[[meta(0), meta(4), meta(8)]]], rescale=True)
        assert len(out[0]) == 30
        assert sum(model.memory.sizes()) > 0 and model._prev_boxes is not None
        for fid in (1, 2, 3):
            out = model.simple_test(torch.randn(1, 3, 128, 160), [meta(fid)], rescale=True)
    assert len(out[0]) == 30


def test_eovod_trains_the_location_prior_on_yolox():
    model = tiny_eovod(location_prior=dict(train_plain_prob=0.0)).train()
    gt_bboxes, gt_labels = gts()
    losses = model.forward_train(
        torch.randn(1, 3, 128, 160), [meta()], gt_bboxes[:1], gt_labels[:1],
        ref_img=torch.randn(1, 2, 3, 128, 160), ref_img_metas=[[meta(1), meta(2)]],
        ref_gt_bboxes=[torch.tensor([[0.0, 12.0, 14.0, 72.0, 88.0], [1.0, 78.0, 22.0, 148.0, 62.0]])])
    sum(losses.values()).backward()
    assert model.aggregators[0].fc.weight.grad.abs().sum() > 0


# ---- schedule and hooks -----------------------------------------------------------------

def test_yolox_lr_policy():
    from vfe.engine.lr_scheduler import build_lr_scheduler

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    sched = build_lr_scheduler(optimizer, dict(
        policy="YOLOX", warmup="exp", by_epoch=False, warmup_by_epoch=True, warmup_ratio=1,
        warmup_iters=2, num_last_epochs=2, min_lr_ratio=0.05), iters_per_epoch=10, max_epochs=10)

    def lr_at(i):
        sched.before_iter(i)
        return optimizer.param_groups[0]["lr"]

    assert lr_at(0) == pytest.approx(0.01 * (1 / 20) ** 2)
    assert lr_at(19) == pytest.approx(0.01)  # end of the two warmup epochs
    # Cosine from 20 over 60 iterations: halfway at p = i + 1 = 50.
    assert lr_at(49) == pytest.approx(0.0005 + 0.5 * (0.01 - 0.0005) * (math.cos(math.pi / 2) + 1))
    assert lr_at(79) == pytest.approx(0.0005) and lr_at(99) == pytest.approx(0.0005)
    with pytest.raises(ValueError):
        build_lr_scheduler(optimizer, dict(policy="YOLOX", num_last_epochs=1))


def test_ema_hook_averages_and_swaps_at_epoch_boundaries():
    from vfe.engine.hooks import ExpMomentumEMAHook

    model = torch.nn.Linear(2, 1)
    hook = ExpMomentumEMAHook(momentum=0.1, total_iter=1)
    hook.before_run(model)
    assert {"ema_weight", "ema_bias"} <= set(dict(model.named_buffers()))
    hook.before_train_epoch(0, 2, model, None, None)  # swap: both equal at the start
    start = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    hook.after_train_iter(0)
    m = hook.momentum_at(0)
    assert m == pytest.approx(0.9 * math.exp(-1) + 0.1)
    expected_ema = start * (1 - m) + (start + 1) * m
    assert torch.allclose(model.ema_weight, expected_ema)
    hook.after_train_epoch(0, model)  # the model now holds the average
    assert torch.allclose(model.weight, expected_ema)
    assert torch.allclose(model.ema_weight, start + 1)
    hook.before_train_epoch(1, 2, model, None, None)  # and back
    assert torch.allclose(model.weight, start + 1)


def test_mode_switch_hook():
    from vfe.engine.hooks import YOLOXModeSwitchHook

    class Dataset:
        skip = None

        def update_skip_type_keys(self, keys):
            self.skip = keys

    model = tiny_eovod()
    dataset = Dataset()
    hook = YOLOXModeSwitchHook(num_last_epochs=2)
    logger = logging.getLogger("test")
    hook.before_train_epoch(6, 10, model, dataset, logger)  # epoch 7 of 10: not yet
    assert dataset.skip is None and not model.detector.bbox_head.use_l1
    hook.before_train_epoch(7, 10, model, dataset, logger)  # epoch 8 == 10 - 2
    assert dataset.skip == ("Mosaic", "RandomAffine", "MixUp")
    assert model.detector.bbox_head.use_l1


# ---- transforms and the converter -------------------------------------------------------

def test_square_pad_self_deciding_flip_and_the_format_bundle():
    from vfe.datasets.pipelines import DefaultFormatBundle, Pad, RandomFlip

    img = np.full((50, 80, 3), 7, dtype=np.uint8)
    results = dict(img=img, img_fields=["img"], bbox_fields=["gt_bboxes"],
                   gt_bboxes=np.array([[1.0, 2.0, 30.0, 40.0]], np.float32),
                   gt_labels=np.array([4]), img_shape=img.shape)
    results = Pad(pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0)))(results)
    assert results["img"].shape == (80, 80, 3)
    assert (results["img"][60, 10] == 114).all()  # every channel padded
    np.random.seed(0)
    flips = [RandomFlip(flip_ratio=0.5)(dict(copy.deepcopy(results)))["flip"]
             for _ in range(20)]
    assert 0 < sum(flips) < 20
    out = DefaultFormatBundle()(results)
    assert out["img"].dtype == torch.float32 and out["img"].shape == (3, 80, 80)
    assert out["pad_shape"] == (80, 80, 3)


def test_mixed_dataset_skips_transforms_by_type():
    from parity_yolox_pipeline import PIPELINE, SyntheticDataset

    from vfe.datasets import MultiImageMixDataset

    mixed = MultiImageMixDataset(SyntheticDataset(), copy.deepcopy(PIPELINE))
    np.random.seed(0)
    sample = mixed[0]
    assert sample["img"].shape == (3, 640, 640)
    assert len(sample["gt_bboxes"]) == len(sample["gt_labels"])
    mixed.update_skip_type_keys(("Mosaic", "RandomAffine", "MixUp"))
    plain = mixed[1]
    assert plain["img"].shape == (3, 640, 640)


def test_converter_maps_megvii_names():
    from convert_yolox_megvii import convert_key

    assert convert_key("backbone.backbone.stem.conv.bn.weight") == "backbone.stem.conv.bn.weight"
    assert convert_key("backbone.backbone.dark2.0.conv.weight") == "backbone.stage1.0.conv.weight"
    assert convert_key("backbone.backbone.dark3.1.conv2.bn.bias") == \
        "backbone.stage2.1.short_conv.bn.bias"
    assert convert_key("backbone.backbone.dark4.1.m.2.conv1.conv.weight") == \
        "backbone.stage3.1.blocks.2.conv1.conv.weight"
    assert convert_key("backbone.backbone.dark5.1.conv2.conv.weight") == \
        "backbone.stage4.1.conv2.conv.weight"  # SPP keeps its names
    assert convert_key("backbone.backbone.dark5.2.conv3.conv.weight") == \
        "backbone.stage4.2.final_conv.conv.weight"
    assert convert_key("backbone.C3_n4.m.0.conv2.bn.running_var") == \
        "neck.bottom_up_blocks.1.blocks.0.conv2.bn.running_var"
    assert convert_key("backbone.lateral_conv0.conv.weight") == "neck.reduce_layers.0.conv.weight"
    assert convert_key("head.stems.2.bn.bias") == "neck.out_convs.2.bn.bias"
    assert convert_key("head.cls_preds.1.weight") == "bbox_head.multi_level_conv_cls.1.weight"
    assert convert_key("head.reg_convs.0.1.conv.weight") == \
        "bbox_head.multi_level_reg_convs.0.1.conv.weight"
    with pytest.raises(KeyError):
        convert_key("head.unknown.weight")


# ---- the recipe through the trainer --------------------------------------------------------

def test_the_yolox_recipe_runs_through_the_trainer(tmp_path):
    """Two epochs of three steps on synthetic images: the mixed-image
    pipeline, batches of two, the YOLOX schedule, the mode switch at the last
    epoch, the norm sync and the EMA, and a checkpoint holding the average."""
    from parity_yolox_pipeline import SyntheticDataset

    from vfe.datasets import MultiImageMixDataset
    from vfe.engine.trainer import train_detector

    cfg = Config.fromfile(REPO_ROOT / "configs/vid/eovod/eovod_yolox_m_10e.py")
    # Square, as the real 640 x 640: the letterbox then always fills one side,
    # so every padded image is the same even size (YOLOX's Focus stem halves it).
    cfg.model.detector = dict(copy.deepcopy(TINY_YOLOX), input_size=(160, 160))
    cfg.model.aggregator = dict(num_heads=4, shared=True, branches="cls")
    cfg.data.samples_per_gpu = 2
    cfg.data.workers_per_gpu = 0
    cfg.lr_config.num_last_epochs = 1
    for hook in cfg.custom_hooks:
        if "num_last_epochs" in hook:
            hook["num_last_epochs"] = 1
    pipeline = copy.deepcopy(cfg.data.train.pipeline)
    for t in pipeline:  # the tiny model's input size
        if "img_scale" in t:
            t["img_scale"] = (160, 160)
        if t["type"] == "RandomAffine":
            t["border"] = (-80, -80)
    dataset = MultiImageMixDataset(SyntheticDataset(), pipeline)
    model = build_model(cfg.model)
    model.init_weights()
    torch.manual_seed(0)
    np.random.seed(0)
    train_detector(model, dataset, cfg, work_dir=str(tmp_path), timestamp="t", meta={},
                   logger=logging.getLogger("test"), device=torch.device("cpu"), seed=0,
                   distributed=False, validate=False, max_epochs=2, max_iters_per_epoch=3)
    assert model.detector.bbox_head.use_l1  # the last epoch switched
    assert dataset._skip_type_keys == ("Mosaic", "RandomAffine", "MixUp")
    checkpoint = torch.load(tmp_path / "epoch_2.pth", weights_only=False)
    state = checkpoint["state_dict"]
    assert any(k.startswith("ema_") for k in state)
    # After the epoch the model holds the average, the ema_ buffer the live weights.
    name = "detector.bbox_head.multi_level_conv_cls.0.weight"
    assert not torch.equal(state[name], state["ema_" + name.replace(".", "_")])


def test_freeze_norm_keeps_batch_statistics_while_training():
    from torch.nn.modules.batchnorm import _BatchNorm

    model = tiny_eovod(freeze_norm=True).train()
    norms = [m for m in model.detector.modules() if isinstance(m, _BatchNorm)]
    assert norms and not any(m.training for m in norms)
    assert model.aggregators.training and model.detector.bbox_head.training
    assert all(p.requires_grad for m in norms for p in m.parameters())
    before = norms[0].running_mean.clone()
    model.forward_train(torch.randn(1, 3, 128, 160), [meta()], *[x[:1] for x in gts()],
                        ref_img=torch.randn(1, 2, 3, 128, 160),
                        ref_img_metas=[[meta(1), meta(2)]],
                        ref_gt_bboxes=[torch.tensor([[0.0, 12.0, 14.0, 72.0, 88.0]])])
    assert torch.equal(norms[0].running_mean, before)


@pytest.mark.parametrize("config", ["eovod_yolox_m_80e.py", "eovod_yolox_m_10e.py",
                                    "eovod_yolox_m_lpn_3e.py"])
def test_yolox_configs_build(config):
    cfg = Config.fromfile(REPO_ROOT / "configs/vid/eovod" / config)
    model = build_model(cfg.model)
    assert type(model.detector).__name__ == "YOLOX"
    # YOLOX-M with 30 classes (25,326,495 with COCO's 80), and one aggregator
    # of 192 channels.
    assert sum(p.numel() for p in model.parameters()) == 25_445_769
    assert model.freeze_norm == (config == "eovod_yolox_m_lpn_3e.py")
