"""YOLOX's training recipe on clips: ``SeqShared`` gives a key frame and its
support frames the same random draws (mosaic layout, warp, mixed-in clip,
colour, flip), ``MultiImageMixDataset`` mixes whole clips, and EOVOD trains on
batches of clips with YOLOX's multi-scale step."""

import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "checks"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from parity_yolox_pipeline import SyntheticDataset  # noqa: E402
from test_yolox import TINY_YOLOX, gts, meta  # noqa: E402

import vfe.datasets.pipelines  # noqa: E402,F401  (registers the transforms)
from vfe.config import Config  # noqa: E402
from vfe.datasets import MultiImageMixDataset  # noqa: E402
from vfe.datasets.pipelines import Mosaic, SeqShared  # noqa: E402
from vfe.models.builder import build_model  # noqa: E402
from vfe.models.vid.eovod import EOVOD  # noqa: E402


class SyntheticClips:
    """Clips of a key frame and two support frames: the same image, its boxes
    moved by a few pixels from frame to frame -- as a dataset with
    ``ref_img_sampler`` and ``SeqLoadAnnotations`` would hand the mixer."""

    CLASSES = SyntheticDataset.CLASSES

    def __init__(self, n=10, identical=False):
        self.images = SyntheticDataset(n)
        self.flag = self.images.flag
        self.identical = identical

    def __len__(self):
        return len(self.images)

    def get_ann_info(self, idx):
        return self.images.get_ann_info(idx)

    def __getitem__(self, idx):
        key = self.images[idx]
        clip = []
        for t in range(3):
            frame = copy.deepcopy(key)
            if not self.identical and t:
                frame["gt_bboxes"] = frame["gt_bboxes"] + 2.0 * t
                frame["img"] = np.roll(frame["img"], 2 * t, axis=(0, 1))
            frame["img_info"] = dict(frame["img_info"], frame_id=t)
            clip.append(frame)
        return clip


IMG = (640, 640)
PAD = dict(img=(114.0, 114.0, 114.0))


def shared(transform):
    return dict(type="SeqShared", transform=transform)


CLIP_PIPELINE = [
    shared(dict(type="Mosaic", img_scale=IMG, pad_val=114.0)),
    shared(dict(type="RandomAffine", scaling_ratio_range=(0.1, 2), border=(-320, -320))),
    shared(dict(type="MixUp", img_scale=IMG, ratio_range=(0.8, 1.6), pad_val=114.0)),
    shared(dict(type="YOLOXHSVRandomAug")),
    shared(dict(type="RandomFlip", flip_ratio=0.5)),
    shared(dict(type="Resize", img_scale=IMG, keep_ratio=True)),
    shared(dict(type="Pad", pad_to_square=True, pad_val=PAD)),
    shared(dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False)),
    dict(type="VideoCollect", keys=["img", "gt_bboxes", "gt_labels"]),
    dict(type="ConcatVideoReferences"),
    dict(type="SeqDefaultFormatBundle", ref_prefix="ref"),
]


def test_every_frame_gets_the_same_draws_and_frame_0_the_single_image_ones():
    clips = SyntheticClips()
    clip = clips[0]
    mixes = [clips[i] for i in (3, 5, 7)]
    np.random.seed(11)
    single = Mosaic(img_scale=IMG, pad_val=114.0)(
        dict(copy.deepcopy(clip[0]), mix_results=copy.deepcopy([m[0] for m in mixes])))
    np.random.seed(11)
    clip[0]["mix_results"] = mixes
    out = SeqShared(dict(type="Mosaic", img_scale=IMG, pad_val=114.0))(clip)
    assert np.array_equal(out[0]["img"], single["img"])
    assert np.array_equal(out[0]["gt_bboxes"], single["gt_bboxes"])
    # The same layout for every frame: the boxes, moved 2 px a frame, move by
    # the same scaled offset in every frame's mosaic.
    assert len(out) == 3 and all(len(o["gt_bboxes"]) == len(single["gt_bboxes"]) for o in out)
    step = out[1]["gt_bboxes"] - out[0]["gt_bboxes"]
    assert np.allclose(out[2]["gt_bboxes"] - out[1]["gt_bboxes"], step, atol=1e-3)
    assert "mix_results" not in out[0]


def test_identical_frames_stay_identical_through_the_whole_pipeline():
    mixed = MultiImageMixDataset(SyntheticClips(identical=True), copy.deepcopy(CLIP_PIPELINE))
    for i in range(4):
        np.random.seed(i)
        sample = mixed[i]
        assert sample["img"].dtype == torch.float32 and sample["img"].shape == (3, 640, 640)
        assert sample["ref_img"].shape == (2, 3, 640, 640)
        for r in range(2):
            assert torch.equal(sample["ref_img"][r], sample["img"])
            refs = sample["ref_gt_bboxes"][sample["ref_gt_bboxes"][:, 0] == r, 1:]
            assert torch.equal(refs, sample["gt_bboxes"])


def test_skipping_by_the_wrapped_transform_type():
    mixed = MultiImageMixDataset(SyntheticClips(), copy.deepcopy(CLIP_PIPELINE))
    assert mixed.pipeline_types[:3] == ["Mosaic", "RandomAffine", "MixUp"]
    mixed.update_skip_type_keys(("Mosaic", "RandomAffine", "MixUp"))
    np.random.seed(0)
    sample = mixed[2]
    assert sample["img"].shape == (3, 640, 640) and sample["ref_img"].shape == (2, 3, 640, 640)


def tiny_eovod(**kwargs):
    torch.manual_seed(0)
    kwargs.setdefault("aggregator", dict(num_heads=4, position="backbone", levels=[1, 2],
                                         shared=False, branches="cls",
                                         backbone_strides=(8, 16, 32)))
    kwargs.setdefault("location_prior", dict(queries="all", train_keys="random",
                                             train_random_keys=40))
    model = EOVOD(copy.deepcopy(TINY_YOLOX), **kwargs)
    model.init_weights()
    return model


def clips_batch(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt_bboxes, gt_labels = gts()
    return dict(img=torch.randn(n, 3, 128, 160, generator=g), img_metas=[meta(3)] * n,
                gt_bboxes=[gt_bboxes[i % 2].clone() for i in range(n)],
                gt_labels=[gt_labels[i % 2].clone() for i in range(n)],
                ref_img=torch.randn(n, 2, 3, 128, 160, generator=g),
                ref_img_metas=[[meta(1), meta(2)]] * n,
                ref_gt_bboxes=[torch.tensor([[0.0, 12.0, 14.0, 72.0, 88.0]])] * n)


def test_eovod_trains_on_a_batch_of_clips_with_batch_statistics():
    model = tiny_eovod().train()
    assert any(m.training for m in model.detector.modules()
               if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
    losses = model.forward_train(**clips_batch(3))
    sum(losses.values()).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    with pytest.raises(ValueError, match="support frames"):
        batch = clips_batch(2)
        model.forward_train(**dict(batch, ref_img=batch["ref_img"][:1]))


def test_plain_steps_are_drawn_per_key_frame():
    model = tiny_eovod(location_prior=dict(queries="all", train_keys="random",
                                           train_random_keys=40, train_plain_prob=0.5)).train()
    calls = []
    original = model._enhance

    def spy(feats, masks, keys):
        calls.append(1)
        return original(feats, masks, keys)

    model._enhance = spy
    torch.manual_seed(3)
    for _ in range(6):
        model.forward_train(**clips_batch(4))
    # Every step's four key frames decide alone: neither all nor none aggregated.
    assert 0 < len(calls) < 24


def test_clip_multiscale_resizes_frames_and_boxes_and_advances_the_schedule():
    from test_eovod import TINY_DETECTOR  # FCOS: no multi-scale schedule

    with pytest.raises(ValueError, match="clip_multiscale"):
        EOVOD(copy.deepcopy(TINY_DETECTOR), clip_multiscale=True)
    model = tiny_eovod(clip_multiscale=True).train()
    batch = clips_batch(2)
    model.detector._input_size = (96, 128)  # from the default (128, 160)
    img, refs, gt, ref_gt, metas = model._resize_clips(
        batch["img"], batch["ref_img"], batch["gt_bboxes"], batch["ref_gt_bboxes"],
        batch["img_metas"])
    assert img.shape[-2:] == (96, 128) and refs.shape == (2, 2, 3, 96, 128)
    scale = torch.tensor([0.8, 0.75, 0.8, 0.75])  # x: 128 / 160, y: 96 / 128
    assert torch.allclose(gt[0], batch["gt_bboxes"][0] * scale)
    assert torch.allclose(ref_gt[0][:, 1:], batch["ref_gt_bboxes"][0][:, 1:] * scale)
    assert torch.equal(ref_gt[0][:, 0], batch["ref_gt_bboxes"][0][:, 0])
    assert metas[0]["img_shape"][:2] == (96, 128)
    progress = model.detector._progress_in_iter
    model.forward_train(**clips_batch(2))
    assert model.detector._progress_in_iter == progress + 1


def test_the_clip_recipe_runs_through_the_trainer(tmp_path):
    """The config's recipe on synthetic clips and a tiny YOLOX: mixed clips,
    batches of two with BatchNorm training, the prior, YOLOX's schedule, the
    switch at the last epoch, and the EMA."""
    from vfe.engine.trainer import train_detector

    cfg = Config.fromfile(REPO_ROOT / "configs/vid/eovod/eovod_yolox_m_clips_10e.py")
    cfg.model.detector = dict(copy.deepcopy(TINY_YOLOX), input_size=(160, 160))
    cfg.model.aggregator = dict(num_heads=4, position="backbone", levels=[1, 2],
                                shared=False, branches="cls", backbone_strides=(8, 16, 32))
    cfg.model.location_prior.train_random_keys = 40
    cfg.data.samples_per_gpu = 2
    cfg.data.workers_per_gpu = 0
    cfg.lr_config.num_last_epochs = 1
    for hook in cfg.custom_hooks:
        if "num_last_epochs" in hook:
            hook["num_last_epochs"] = 1
    pipeline = copy.deepcopy(cfg.data.train.pipeline)
    for t in pipeline:
        inner = t.get("transform", t)
        if "img_scale" in inner:
            inner["img_scale"] = (160, 160)
        if inner["type"] == "RandomAffine":
            inner["border"] = (-80, -80)
    dataset = MultiImageMixDataset(SyntheticClips(), pipeline)
    model = build_model(cfg.model)
    model.init_weights()
    torch.manual_seed(0)
    np.random.seed(0)
    train_detector(model, dataset, cfg, work_dir=str(tmp_path), timestamp="t", meta={},
                   logger=logging.getLogger("test"), device=torch.device("cpu"), seed=0,
                   distributed=False, validate=False, max_epochs=2, max_iters_per_epoch=2)
    assert model.detector.bbox_head.use_l1
    assert dataset._skip_type_keys == ("Mosaic", "RandomAffine", "MixUp")
    assert model.detector._progress_in_iter == 4  # the clip multi-scale schedule ran
    state = torch.load(tmp_path / "epoch_2.pth", weights_only=False)["state_dict"]
    assert any(k.startswith("ema_aggregators") for k in state)
